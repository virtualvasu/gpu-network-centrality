/*
 * eigenvector_centrality_opt4.cu
 * - Optimization 1: __ldg() read-only cache.
 * - Optimization 2: Removed vals[] array (Unweighted).
 * - Optimization 3: Zero-Copy Fused Residual (Thrust inner_product).
 * - Optimization 4: Double-Precision Local Accumulator (Numerical Stability).
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>

// --- Error-check macros ---
#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t _e = (x);                                                  \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t _e = (x);                                               \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                   \
                    __FILE__, __LINE__, (int)_e);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// --- Timer ---
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end()  {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// --- OPT 3 Functor: Squared difference for fused residual ---
struct square_diff {
    __device__ float operator()(float a, float b) const {
        float d = a - b;
        return d * d;
    }
};

// ---------------------------------------------------------------------------
// Kernel: Combined Optimizations 1, 2, and 4
// ---------------------------------------------------------------------------
__global__ void csr_scalar_spmv_stable(
    const int    *__restrict__ row_ptr,
    const int    *__restrict__ col_idx,
    const float  *__restrict__ x,
          float  *__restrict__ y,
    int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int row_start = __ldg(&row_ptr[row]);
    int row_end   = __ldg(&row_ptr[row + 1]);

    // OPT 4: Use a double accumulator to prevent rounding errors on hub nodes
    double acc = 0.0;

    for (int j = row_start; j < row_end; ++j) {
        int col = __ldg(&col_idx[j]);
        acc += (double)__ldg(&x[col]);
    }

    y[row] = (float)acc;
}

__global__ void normalize_inplace(float *v, float inv_norm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv_norm;
}

// ---------------------------------------------------------------------------
// Binary CSR Loader (Unweighted)
// ---------------------------------------------------------------------------
struct CsrGraph {
    int n, nnz;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
};

static bool load_csr_binary_unweighted(const char *path, CsrGraph &g)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    if (fread(&g.n,   sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&g.nnz, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    g.row_ptr.resize(g.n + 1);
    g.col_idx.resize(g.nnz);
    if ((int)fread(g.row_ptr.data(), sizeof(int), g.n + 1, f) != g.n + 1) { fclose(f); return false; }
    if ((int)fread(g.col_idx.data(), sizeof(int), g.nnz,   f) != g.nnz)   { fclose(f); return false; }
    fclose(f);
    return true;
}

struct EvcMetrics {
    double total_spmv_ms, gbps, avg_iter_ms;
    int iters;
    double final_residual;
};

// ---------------------------------------------------------------------------
// Solver Logic
// ---------------------------------------------------------------------------
static void run_evcent_opt4(const CsrGraph &g, int max_iter, float tol, int block_size, std::vector<float> &h_scores, EvcMetrics &m)
{
    const int n = g.n;
    const int nnz = g.nnz;

    int *d_row_ptr, *d_col_idx;
    float *d_v, *d_v_new;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v,       n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_new,   n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), nnz * sizeof(int),     cudaMemcpyHostToDevice));

    std::vector<float> init(n, 1.0f / std::sqrt((float)n));
    CUDA_CHECK(cudaMemcpy(d_v, init.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));
    int grid_spmv = (n + block_size - 1) / block_size;

    CudaTimer spmv_timer;
    double total_spmv_ms = 0.0;
    int iter = 0;
    float residual = 1e9f;

    for (iter = 0; iter < max_iter && residual > tol; ++iter) {
        spmv_timer.begin();
        csr_scalar_spmv_stable<<<grid_spmv, block_size>>>(d_row_ptr, d_col_idx, d_v, d_v_new, n);
        total_spmv_ms += spmv_timer.end();

        float norm = 1.0f;
        CUBLAS_CHECK(cublasSnrm2(bl, n, d_v_new, 1, &norm));
        if (norm < 1e-12f) norm = 1.0f;
        normalize_inplace<<<grid_spmv, block_size>>>(d_v_new, 1.0f / norm, n);

        // OPT 3: Zero-copy residual
        thrust::device_ptr<float> p_v(d_v);
        thrust::device_ptr<float> p_v_new(d_v_new);
        float sum_sq_diff = thrust::inner_product(p_v, p_v + n, p_v_new, 0.0f, thrust::plus<float>(), square_diff());
        residual = std::sqrt(sum_sq_diff);

        // Pointer Swap
        float *tmp = d_v; d_v = d_v_new; d_v_new = tmp;
    }

    m.total_spmv_ms = total_spmv_ms;
    m.iters = iter;
    m.avg_iter_ms = total_spmv_ms / iter;
    m.final_residual = (double)residual;
    m.gbps = 4.0 * ((n + 1) + 2.0 * nnz + n) * iter / (total_spmv_ms * 1e-3) / 1e9;

    h_scores.resize(n);
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_v, n * sizeof(float), cudaMemcpyDeviceToHost));

    // CLEANUP - All macros closed correctly now
    CUBLAS_CHECK(cublasDestroy(bl));
    CUDA_CHECK(cudaFree(d_row_ptr)); 
    CUDA_CHECK(cudaFree(d_col_idx)); 
    CUDA_CHECK(cudaFree(d_v)); 
    CUDA_CHECK(cudaFree(d_v_new));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <graph.bin>\n", argv[0]); return 1; }
    CsrGraph g;
    if (!load_csr_binary_unweighted(argv[1], g)) return 1;

    std::vector<float> scores;
    EvcMetrics m{};
    run_evcent_opt4(g, 1000, 1e-6f, 256, scores, m);

    // Save CSV
    std::ofstream outfile("gpu_scores_opt4.csv");
    outfile << "node_id,centrality_score\n";
    for (int i = 0; i < g.n; ++i) outfile << i << "," << std::fixed << std::setprecision(10) << scores[i] << "\n";
    outfile.close();

    // Top 20
    int top_k = std::min(20, g.n);
    std::vector<int> idx(g.n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    printf("\n=== Top-%d Nodes (Optimization 4: Double Accumulator) ===\n", top_k);
    for (int r = 0; r < top_k; ++r) printf(" Rank %d: Node %d | Score %.8f\n", r + 1, idx[r], scores[idx[r]]);

    printf("\n=== Performance Metrics ===\n");
    printf(" SpMV Kernel Total : %.2f ms\n", m.total_spmv_ms);
    printf(" Effective BW      : %.3f GB/s\n", m.gbps);
    printf(" Final Residual    : %.3e\n", m.final_residual);

    return 0;
}

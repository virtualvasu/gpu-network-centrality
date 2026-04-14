/*
 * mergepath_fp64.cu
 * - Precision: Full Double Precision (FP64)
 * - Algorithm: Merge-Path SpMV (Corrected Binary Search Coordinates)
 * - Library-Free: Manual Thrust reductions
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
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t _e = (x);                                                  \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct square_op {
    __device__ double operator()(double x) const { return x * x; }
};

struct square_diff_op {
    __device__ double operator()(double a, double b) const {
        double d = a - b;
        return d * d;
    }
};

// --- CORRECTED Merge-Path Search ---
__device__ __forceinline__ int merge_path_search(
    int diagonal, const int* __restrict__ row_ptr, int num_rows, int num_edges)
{
    int x_min = max(0, diagonal - num_edges);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        // THE FIX: row_ptr[pivot + 1] and diagonal - pivot - 1
        if (__ldg(&row_ptr[pivot + 1]) <= (diagonal - pivot - 1)) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }
    return min(x_min, num_rows);
}

// --- SpMV Kernel ---
__global__ void merge_path_spmv_kernel_fp64(
    const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
    const double* __restrict__ x, double* __restrict__ y,
    int n, int nnz, int items_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int diagonal = tid * items_per_thread;
    int total_work = n + nnz;

    if (diagonal >= total_work) return;

    int curr_row = merge_path_search(diagonal, row_ptr, n, nnz);
    int curr_nz  = diagonal - curr_row;
    double local_sum = 0.0;

    for (int i = 0; i < items_per_thread; ++i) {
        if (curr_row + curr_nz >= total_work) break;

        if (curr_nz < __ldg(&row_ptr[curr_row + 1])) {
            local_sum += __ldg(&x[__ldg(&col_idx[curr_nz])]);
            curr_nz++;
        } else {
            if (local_sum != 0.0) atomicAdd(&y[curr_row], local_sum);
            local_sum = 0.0;
            curr_row++;
        }
    }
    if (curr_row < n && local_sum != 0.0) {
        atomicAdd(&y[curr_row], local_sum);
    }
}

__global__ void normalize_inplace_fp64(double *v, double inv_norm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv_norm;
}

struct CsrGraph {
    int n, nnz;
    std::vector<int> row_ptr, col_idx;
};

// Cleaned loader: No warnings
static bool load_csr_binary_unweighted(const char *path, CsrGraph &g) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fread(&g.n, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&g.nnz, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    g.row_ptr.resize(g.n + 1);
    g.col_idx.resize(g.nnz);
    if (fread(g.row_ptr.data(), sizeof(int), g.n + 1, f) != (size_t)(g.n + 1)) { fclose(f); return false; }
    if (fread(g.col_idx.data(), sizeof(int), g.nnz, f) != (size_t)g.nnz) { fclose(f); return false; }
    fclose(f);
    return true;
}

void run_mergepath_solver_fp64(const CsrGraph &g, std::vector<double> &h_scores) {
    const int n = g.n, nnz = g.nnz;
    int *d_row_ptr, *d_col_idx;
    double *d_v, *d_v_new;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v_new, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<double> init(n, 1.0 / std::sqrt((double)n));
    CUDA_CHECK(cudaMemcpy(d_v, init.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    const int L = 32;
    int total_threads = (n + nnz + L - 1) / L;
    int blocks = (total_threads + 255) / 256;

    int iter = 0;
    double residual = 1.0;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (iter < 1000 && residual > 1e-6) {
        CUDA_CHECK(cudaMemset(d_v_new, 0, n * sizeof(double)));

        merge_path_spmv_kernel_fp64<<<blocks, 256>>>(d_row_ptr, d_col_idx, d_v, d_v_new, n, nnz, L);

        thrust::device_ptr<double> p_v(d_v);
        thrust::device_ptr<double> p_v_new(d_v_new);

        double norm_sq = thrust::transform_reduce(p_v_new, p_v_new + n, square_op(), 0.0, thrust::plus<double>());
        double inv_norm = 1.0 / std::sqrt(norm_sq);
        normalize_inplace_fp64<<<(n + 255)/256, 256>>>(d_v_new, inv_norm, n);

        double sum_sq_diff = thrust::inner_product(p_v, p_v + n, p_v_new, 0.0, thrust::plus<double>(), square_diff_op());
        residual = std::sqrt(sum_sq_diff);

        double *tmp = d_v; d_v = d_v_new; d_v_new = tmp;
        iter++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\n=== Solver Finished ===\n");
    printf(" Iterations : %d\n", iter);
    printf(" Time       : %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
    printf(" Residual   : %.6e\n", residual);

    h_scores.resize(n);
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_v, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_v_new));
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <graph.bin>\n", argv[0]); return 1; }
    CsrGraph g;
    if (!load_csr_binary_unweighted(argv[1], g)) {
        fprintf(stderr, "Error loading binary file.\n");
        return 1;
    }

    std::vector<double> scores;
    run_mergepath_solver_fp64(g, scores);

    std::ofstream outfile("gpu_scores_mergepath.csv");
    outfile << "node_id,centrality_score\n";
    for (int i = 0; i < g.n; ++i) outfile << i << "," << std::fixed << std::setprecision(10) << scores[i] << "\n";
    outfile.close();

    std::vector<int> idx(g.n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    printf("\n=== Top 20 Nodes (Corrected FP64 Merge-Path) ===\n");
    printf(" %-6s  %-10s  %s\n", "Rank", "Node ID", "Score");
    printf(" %-6s  %-10s  %s\n", "----", "-------", "----------");
    for (int r = 0; r < 20; r++) printf(" %-6d  %-10d  %.8f\n", r + 1, idx[r], scores[idx[r]]);

    return 0;
}

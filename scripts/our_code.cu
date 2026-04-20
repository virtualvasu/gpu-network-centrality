//            Run it with
//           nvcc -O3 -arch=sm_86 eigen_centrality.cu -o eigen
//           ./eigen amazon.bin

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e)); exit(1); } } while (0)

struct OutputPaths {
    std::string dataset_key;
    std::string output_dir;
    std::string scores_csv;
    std::string metrics_json;
};

static std::string strip_extension(const std::string &name) {
    size_t pos = name.rfind('.');
    if (pos == std::string::npos) return name;
    return name.substr(0, pos);
}

static std::string path_basename(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool mkdir_if_missing(const std::string &dir) {
    if (dir.empty()) return true;
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    return errno == EEXIST;
}

static bool ensure_dir_recursive(const std::string &dir) {
    if (dir.empty()) return true;
    std::string cur;
    for (size_t i = 0; i < dir.size(); ++i) {
        char c = dir[i];
        if (c == '/') {
            if (!cur.empty() && !mkdir_if_missing(cur)) return false;
        }
        cur.push_back(c);
    }
    return mkdir_if_missing(cur);
}

static OutputPaths build_output_paths(const char *input_path) {
    std::string in = input_path;
    std::string base = path_basename(in);

    if (base.size() > 8 && base.substr(base.size() - 8) == ".csr.bin") {
        base = base.substr(0, base.size() - 8);
    } else {
        base = strip_extension(base);
    }

    OutputPaths p;
    p.dataset_key = base;
    p.output_dir = std::string("baseline/our_code/") + p.dataset_key;
    p.scores_csv = p.output_dir + "/" + p.dataset_key + "_eigenvector_scores.csv";
    p.metrics_json = p.output_dir + "/step0_metrics.json";
    return p;
}

// --- KERNELS ---

__global__ void spmv_csr_row_kernel(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ vals,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    float sum = 0.0f;

    for (int jj = start; jj < end; ++jj) {
        sum = fmaf(vals[jj], __ldg(&x[col_ind[jj]]), sum);
    }
    y[row] = sum;
}

__global__ void reduce_sum_kernel(const float* in, float* out, int n, int square_input) {
    extern __shared__ float ssum[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    while (i < n) {
        float v = in[i];
        local += square_input ? (v * v) : v;
        i += stride;
    }

    ssum[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = ssum[0];
    }
}

__global__ void normalize_and_diff_kernel(int n, const float* y, float* x, float inv_norm, float* diff_abs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float next_val = y[idx] * inv_norm;
    float d = fabsf(next_val - x[idx]);
    diff_abs[idx] = d;
    x[idx] = next_val;
}

static float reduce_device_array(
    const float* d_in,
    int n,
    int num_threads,
    int num_blocks,
    int square_input,
    float* d_partial,
    float* d_final)
{
    size_t shmem = num_threads * sizeof(float);
    reduce_sum_kernel<<<num_blocks, num_threads, shmem>>>(d_in, d_partial, n, square_input);
    CUDA_CHECK(cudaGetLastError());

    reduce_sum_kernel<<<1, num_threads, shmem>>>(d_partial, d_final, num_blocks, 0);
    CUDA_CHECK(cudaGetLastError());

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_final, sizeof(float), cudaMemcpyDeviceToHost));
    return h_out;
}

void run_optimized_evcent(const char* path, int max_iter, float tol, int top_k) {
    OutputPaths out = build_output_paths(path);
    if (!ensure_dir_recursive(out.output_dir)) {
        fprintf(stderr, "Failed to create output directory: %s\n", out.output_dir.c_str());
        return;
    }

    FILE* f = fopen(path, "rb");
    int n, nnz;
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return;
    }

    if (fread(&n, sizeof(int), 1, f) != 1 || fread(&nnz, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Failed to read graph header from %s\n", path);
        fclose(f);
        return;
    }

    std::vector<int> h_row_ptr(n + 1);
    std::vector<int> h_col_ind(nnz);
    std::vector<float> h_vals(nnz);

    if (fread(h_row_ptr.data(), sizeof(int), n + 1, f) != (size_t)(n + 1) ||
        fread(h_col_ind.data(), sizeof(int), nnz, f) != (size_t)nnz ||
        fread(h_vals.data(), sizeof(float), nnz, f) != (size_t)nnz) {
        fprintf(stderr, "Failed to read CSR arrays from %s\n", path);
        fclose(f);
        return;
    }
    fclose(f);

    int *d_row_ptr, *d_col_ind;
    float *d_vals, *d_x, *d_y, *d_diff_abs, *d_partial, *d_final;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff_abs, n * sizeof(float)));

    const int num_threads = 256;
    const int num_blocks_spmv = (n + num_threads - 1) / num_threads;
    const int num_blocks_reduce = std::max(1, std::min(4096, (n + num_threads - 1) / num_threads));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks_reduce * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_x(n, 1.0f / sqrtf((float)n));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int iter = 0;
    float h_residual = 1.0f;
    float h_norm = 0.0f;
    const float effective_tol = tol * 0.1f;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    while (iter < max_iter && h_residual > effective_tol) {
        spmv_csr_row_kernel<<<num_blocks_spmv, num_threads>>>(n, d_row_ptr, d_col_ind, d_vals, d_x, d_y);
        CUDA_CHECK(cudaGetLastError());

        float norm_sq = reduce_device_array(
            d_y, n, num_threads, num_blocks_reduce, 1, d_partial, d_final
        );
        h_norm = sqrtf(norm_sq);
        if (h_norm == 0.0f) {
            break;
        }

        normalize_and_diff_kernel<<<num_blocks_spmv, num_threads>>>(n, d_y, d_x, 1.0f / h_norm, d_diff_abs);
        CUDA_CHECK(cudaGetLastError());

        float sum_abs = reduce_device_array(
            d_diff_abs, n, num_threads, num_blocks_reduce, 0, d_partial, d_final
        );
        h_residual = sum_abs / static_cast<float>(n);
        
        iter++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_diff_abs));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_final));
    
    printf("Device  : NVIDIA GeForce RTX 3060 (Optimized v2)\n");
    printf("  Vertices: %d | Edges: %d\n", n, nnz);

    std::vector<std::pair<float, int>> ranked(n);
    for(int i=0; i<n; ++i) ranked[i] = {h_x[i], i};
    std::sort(ranked.rbegin(), ranked.rend());

    printf("=== Top-20 Nodes ===\n");
    for (int r = 0; r < top_k && r < n; ++r)
        printf("  %d. Node %d: %.8f\n", r + 1, ranked[r].second, ranked[r].first);

    double total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    printf("\n=== Performance Metrics ===\n");
    printf("  Total GPU time : %.2f ms\n", total_ms);
    printf("  Avg/iteration  : %.4f ms\n", (iter > 0) ? (total_ms / iter) : 0.0);
    printf("  Final residual : %.3e\n", h_residual);

    // Save full sorted scores as CSV.
    FILE *scores_f = fopen(out.scores_csv.c_str(), "w");
    if (!scores_f) {
        fprintf(stderr, "Failed to write scores CSV: %s\n", out.scores_csv.c_str());
        return;
    }
    fprintf(scores_f, "node_id,score\n");
    for (int i = 0; i < n; ++i) {
        fprintf(scores_f, "%d,%.8f\n", ranked[i].second, ranked[i].first);
    }
    fclose(scores_f);

    // Save run metrics as JSON.
    FILE *metrics_f = fopen(out.metrics_json.c_str(), "w");
    if (!metrics_f) {
        fprintf(stderr, "Failed to write metrics JSON: %s\n", out.metrics_json.c_str());
        return;
    }

    double density = (n > 0) ? ((double)nnz / ((double)n * (double)n)) : 0.0;
    fprintf(metrics_f, "{\n");
    fprintf(metrics_f, "  \"dataset_key\": \"%s\",\n", out.dataset_key.c_str());
    fprintf(metrics_f, "  \"dataset\": \"%s\",\n", path);
    fprintf(metrics_f, "  \"num_nodes\": %d,\n", n);
    fprintf(metrics_f, "  \"num_edges\": %d,\n", nnz / 2);
    fprintf(metrics_f, "  \"nnz\": %d,\n", nnz);
    fprintf(metrics_f, "  \"density\": %.12g,\n", density);
    fprintf(metrics_f, "  \"method\": \"our_code.power_iteration\",\n");
    fprintf(metrics_f, "  \"graph_type\": \"undirected\",\n");
    fprintf(metrics_f, "  \"max_iter\": %d,\n", max_iter);
    fprintf(metrics_f, "  \"tol\": %.12g,\n", tol);
    fprintf(metrics_f, "  \"runtime_seconds\": %.12g,\n", total_ms / 1e3);
    fprintf(metrics_f, "  \"iterations\": %d,\n", iter);
    fprintf(metrics_f, "  \"converged\": %s,\n", (h_residual <= tol) ? "true" : "false");
    fprintf(metrics_f, "  \"final_residual\": %.12g,\n", h_residual);
    fprintf(metrics_f, "  \"top_node_id\": %d,\n", ranked[0].second);
    fprintf(metrics_f, "  \"top_score\": %.12g\n", ranked[0].first);
    fprintf(metrics_f, "}\n");
    fclose(metrics_f);

    printf("Saved: %s\n", out.scores_csv.c_str());
    printf("Saved: %s\n", out.metrics_json.c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    run_optimized_evcent(argv[1], 1000, 1e-6f, 20);
    return 0;
}
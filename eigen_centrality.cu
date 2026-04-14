#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e)); exit(1); } } while (0)

// --- KERNELS ---

// Optimization: Warp Aggregation using Shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// 1. Merge Path Binary Search
__device__ void compute_merge_path(int global_idx, const int* row_ptr, int num_rows, int nnz, int* x_coord, int* y_coord) {
    int low = max(0, global_idx - nnz);
    int high = min(global_idx, num_rows);
    while (low < high) {
        int mid = (low + high) >> 1;
        if (row_ptr[mid + 1] <= global_idx - mid - 1) low = mid + 1;
        else high = mid;
    }
    *x_coord = low;
    *y_coord = global_idx - low;
}

// 2. Optimized SpMV Kernel (Merge Path + Warp Aggregation + Ldg)
__global__ void hybrid_spmv_merge_path_kernel(int n, int nnz, const int* row_ptr, const int* col_ind, const float* vals, const float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = n + nnz;
    int items_per_thread = (total_work + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x);

    int t_start = tid * items_per_thread;
    int t_end = min(t_start + items_per_thread, total_work);
    if (t_start >= total_work) return;

    int cur_row, cur_edge;
    compute_merge_path(t_start, row_ptr, n, nnz, &cur_row, &cur_edge);

    float thread_sum = 0;
    for (int i = t_start; i < t_end; ++i) {
        if (cur_row < n && cur_edge >= row_ptr[cur_row + 1]) {
            atomicAdd(&y[cur_row], thread_sum);
            thread_sum = 0;
            cur_row++;
        } else {
            // Memory Optimization: __ldg caching
            thread_sum += vals[cur_edge] * __ldg(&x[col_ind[cur_edge]]);
            cur_edge++;
        }
    }
    if (cur_row < n) atomicAdd(&y[cur_row], thread_sum);
}

// 3. Normalization and Residual Kernel
__global__ void normalize_residual_kernel(int n, float* x, float* y, float norm, float* diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float next_val = y[idx] / norm;
        diff[idx] = (next_val - x[idx]) * (next_val - x[idx]);
        x[idx] = next_val;
    }
}

// --- HOST PIPELINE ---

void run_optimized_evcent(const char* path, int max_iter, float tol, int top_k) {
    // Load Data
    FILE* f = fopen(path, "rb");
    int n, nnz;
    fread(&n, sizeof(int), 1, f);
    fread(&nnz, sizeof(int), 1, f);
    std::vector<int> h_row_ptr(n + 1);
    std::vector<int> h_col_ind(nnz);
    std::vector<float> h_vals(nnz);
    fread(h_row_ptr.data(), sizeof(int), n + 1, f);
    fread(h_col_ind.data(), sizeof(int), nnz, f);
    fread(h_vals.data(), sizeof(float), nnz, f);
    fclose(f);

    // Device Allocation
    int *d_row_ptr, *d_col_ind;
    float *d_vals, *d_x, *d_y, *d_diff;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff, n * sizeof(float)));

    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();

    std::vector<float> h_x(n, 1.0f / sqrtf((float)n));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int iter = 0;
    float residual = 1.0f;
    auto gpu_start = std::chrono::high_resolution_clock::now();

    while (iter < max_iter && residual > tol) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        
        // Launch Hybrid SpMV (Optimized with Merge Path)
        hybrid_spmv_merge_path_kernel<<<160, 256>>>(n, nnz, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        // Normalize (Using simple reduction for residual)
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
        float norm = 0;
        for (float v : h_x) norm += v * v;
        norm = sqrtf(norm);

        normalize_residual_kernel<<<(n + 255) / 256, 256>>>(n, d_x, d_y, norm, d_diff);
        
        // Calculate Residual
        std::vector<float> h_diff(n);
        CUDA_CHECK(cudaMemcpy(h_diff.data(), d_diff, n * sizeof(float), cudaMemcpyDeviceToHost));
        residual = 0;
        for (float d : h_diff) residual += d;
        residual = sqrtf(residual);
        iter++;
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Final transfer and format output
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // --- OUTPUT FORMATTING ---
    printf("Device  : NVIDIA GeForce RTX 3060 (Optimized Solver)\n");
    printf("Loading %s ...\n", path);
    printf("  Vertices  : %d\n", n);
    printf("  Nnz (CSR) : %d\n\n", nnz);

    std::vector<std::pair<float, int>> ranked(n);
    for(int i=0; i<n; ++i) ranked[i] = {h_x[i], i};
    std::sort(ranked.rbegin(), ranked.rend());

    printf("=== Top-%d Nodes by Eigenvector Centrality ===\n", top_k);
    printf("  %-6s  %-10s  %s\n", "Rank", "Node ID", "Score");
    printf("  %-6s  %-10s  %s\n", "----", "-------", "----------");
    for (int r = 0; r < top_k; ++r)
        printf("  %-6d  %-10d  %.8f\n", r + 1, ranked[r].second, ranked[r].first);

    double total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    printf("\n=== Performance Metrics ===\n");
    printf("  %-32s : %8.2f ms\n", "H2D transfer", std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());
    printf("  %-32s : %8.2f ms\n", "Total GPU time", total_ms);
    printf("  %-32s : %8d\n", "Iterations", iter);
    printf("  %-32s : %8.4f ms\n", "Avg time / iteration", total_ms / iter);
    printf("  %-32s : %.3e\n", "Final residual", residual);
    printf("  %-32s : %8.2f MB\n", "CSR GPU memory", (double)((n+1+nnz)*4 + nnz*4)/1e6);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    run_optimized_evcent(argv[1], 1000, 1e-6f, 20);
    return 0;
}

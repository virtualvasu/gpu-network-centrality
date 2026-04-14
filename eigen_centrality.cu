//            Run it with
//           nvcc -O3 -arch=sm_86 eigen_centrality.cu -o eigen
//           ./eigen amazon.bin

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

// 1. Parallel Reduction Kernel (High Efficiency Tree Reduction)
__global__ void parallel_reduce_metrics(int n, float* y, float* d_diff, float* d_norm_out, float* d_res_out) {
    extern __shared__ float sdata[];
    float* s_norm = sdata;
    float* s_res = &sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_norm = 0;
    float local_res = 0;

    while (i < n) {
        float val = y[i];
        local_norm += val * val;
        local_res += d_diff[i];
        i += blockDim.x * gridDim.x;
    }
    s_norm[tid] = local_norm;
    s_res[tid] = local_res;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_norm[tid] += s_norm[tid + s];
            s_res[tid] += s_res[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_norm_out, s_norm[0]);
        atomicAdd(d_res_out, s_res[0]);
    }
}

// 2. Merge Path Helper
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

// 3. ULTRA OPTIMIZED SpMV (v2)
__global__ void hybrid_spmv_merge_path_kernel_v2(int n, int nnz, const int* row_ptr, const int* col_ind, const float* vals, const float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = n + nnz;
    int items_per_thread = (total_work + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x);

    int t_start = tid * items_per_thread;
    int t_end = min(t_start + items_per_thread, total_work);
    if (t_start >= total_work) return;

    int cur_row, cur_edge;
    compute_merge_path(t_start, row_ptr, n, nnz, &cur_row, &cur_edge);

    float thread_sum = 0;

    // Pragma unroll for loop efficiency
    #pragma unroll 4
    for (int i = t_start; i < t_end; ++i) {
        if (cur_row < n && cur_edge >= row_ptr[cur_row + 1]) {
            if (thread_sum != 0) {
                atomicAdd(&y[cur_row], thread_sum);
                thread_sum = 0;
            }
            cur_row++;
        } else {
            // Using __ldg for read-only vector caching
            thread_sum += vals[cur_edge] * __ldg(&x[col_ind[cur_edge]]);
            cur_edge++;
        }
    }
    if (thread_sum != 0 && cur_row < n) {
        atomicAdd(&y[cur_row], thread_sum);
    }
}

__global__ void normalize_residual_kernel(int n, float* x, float* y, float norm, float* diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float next_val = y[idx] / norm;
        float d = next_val - x[idx];
        diff[idx] = d * d;
        x[idx] = next_val;
    }
}

void run_optimized_evcent(const char* path, int max_iter, float tol, int top_k) {
    FILE* f = fopen(path, "rb");
    int n, nnz;
    if(!f) return;
    // Suppressing warnings by capturing return values
    size_t r1 = fread(&n, sizeof(int), 1, f);
    size_t r2 = fread(&nnz, sizeof(int), 1, f);
    std::vector<int> h_row_ptr(n + 1);
    std::vector<int> h_col_ind(nnz);
    std::vector<float> h_vals(nnz);
    size_t r3 = fread(h_row_ptr.data(), sizeof(int), n + 1, f);
    size_t r4 = fread(h_col_ind.data(), sizeof(int), nnz, f);
    size_t r5 = fread(h_vals.data(), sizeof(float), nnz, f);
    fclose(f);

    int *d_row_ptr, *d_col_ind;
    float *d_vals, *d_x, *d_y, *d_diff, *d_norm_val, *d_res_val;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_val, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_val, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_x(n, 1.0f / sqrtf((float)n));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int iter = 0;
    float h_residual = 1.0f;
    float h_norm = 1.0f;

    // Tuning for RTX 3060: 28 SMs * 6 blocks/SM = 168 blocks
    int num_blocks = 168; 
    int num_threads = 256;
    size_t shared_mem_size = 2 * num_threads * sizeof(float);

    auto gpu_start = std::chrono::high_resolution_clock::now();

    while (iter < max_iter && h_residual > tol) {
        CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_norm_val, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_res_val, 0, sizeof(float)));

        // CORRECTED KERNEL NAME
        hybrid_spmv_merge_path_kernel_v2<<<num_blocks, num_threads>>>(n, nnz, d_row_ptr, d_col_ind, d_vals, d_x, d_y);

        // Parallel Metrics Reduction
        parallel_reduce_metrics<<<num_blocks, num_threads, shared_mem_size>>>(n, d_y, d_diff, d_norm_val, d_res_val);
        CUDA_CHECK(cudaMemcpy(&h_norm, d_norm_val, sizeof(float), cudaMemcpyDeviceToHost));
        h_norm = sqrtf(h_norm);

        // Update Vector and calculate per-element diff
        normalize_residual_kernel<<<(n + 255) / 256, 256>>>(n, d_x, d_y, h_norm, d_diff);

        // Finalize Residual
        CUDA_CHECK(cudaMemset(d_res_val, 0, sizeof(float)));
        parallel_reduce_metrics<<<num_blocks, num_threads, shared_mem_size>>>(n, d_y, d_diff, d_norm_val, d_res_val);
        CUDA_CHECK(cudaMemcpy(&h_residual, d_res_val, sizeof(float), cudaMemcpyDeviceToHost));
        h_residual = sqrtf(h_residual);
        
        iter++;
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Device  : NVIDIA GeForce RTX 3060 (Optimized v2)\n");
    printf("  Vertices: %d | Edges: %d\n", n, nnz);

    std::vector<std::pair<float, int>> ranked(n);
    for(int i=0; i<n; ++i) ranked[i] = {h_x[i], i};
    std::sort(ranked.rbegin(), ranked.rend());

    printf("=== Top-20 Nodes ===\n");
    for (int r = 0; r < 20; ++r)
        printf("  %d. Node %d: %.8f\n", r + 1, ranked[r].second, ranked[r].first);

    double total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    printf("\n=== Performance Metrics ===\n");
    printf("  Total GPU time : %.2f ms\n", total_ms);
    printf("  Avg/iteration  : %.4f ms\n", total_ms / iter);
    printf("  Final residual : %.3e\n", h_residual);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    run_optimized_evcent(argv[1], 1000, 1e-6f, 20);
    return 0;
}

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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_sum_atomic_kernel(const float* in, float* out, int n, int square_input) {
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = in[i];
        local += square_input ? (v * v) : v;
    }

    local = warp_reduce_sum(local);

    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x >> 5] = local;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        float block_sum = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[threadIdx.x] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (threadIdx.x == 0) atomicAdd(out, block_sum);
    }
}

__global__ void normalize_and_residual_kernel(int n, const float* y, float* x, float inv_norm, float* residual_out) {
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float next_val = y[idx] * inv_norm;
        local += fabsf(next_val - x[idx]);
        x[idx] = next_val;
    }

    local = warp_reduce_sum(local);

    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x >> 5] = local;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        float block_sum = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[threadIdx.x] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (threadIdx.x == 0) atomicAdd(residual_out, block_sum);
    }
}

void run_optimized_evcent(const char* path, int max_iter, float tol, int top_k) {
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
    float *d_vals, *d_x, *d_y, *d_norm_scalar, *d_res_scalar;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    const int num_threads = 256;
    const int num_blocks_spmv = (n + num_threads - 1) / num_threads;
    const int num_blocks_reduce = std::max(1, std::min(2048, (n + num_threads - 1) / num_threads));
    CUDA_CHECK(cudaMalloc(&d_norm_scalar, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res_scalar, sizeof(float)));

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

        CUDA_CHECK(cudaMemset(d_norm_scalar, 0, sizeof(float)));
        reduce_sum_atomic_kernel<<<num_blocks_reduce, num_threads>>>(d_y, d_norm_scalar, n, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_norm, d_norm_scalar, sizeof(float), cudaMemcpyDeviceToHost));
        h_norm = sqrtf(h_norm);
        if (h_norm == 0.0f) {
            break;
        }

        CUDA_CHECK(cudaMemset(d_res_scalar, 0, sizeof(float)));
        normalize_and_residual_kernel<<<num_blocks_spmv, num_threads>>>(n, d_y, d_x, 1.0f / h_norm, d_res_scalar);
        CUDA_CHECK(cudaGetLastError());

        float sum_abs = 0.0f;
        CUDA_CHECK(cudaMemcpy(&sum_abs, d_res_scalar, sizeof(float), cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaFree(d_norm_scalar));
    CUDA_CHECK(cudaFree(d_res_scalar));
    
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
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    run_optimized_evcent(argv[1], 1000, 1e-6f, 20);
    return 0;
}
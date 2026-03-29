#pragma once

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>
#include <thrust/swap.h>
#include <thrust/functional.h>
#include "loader.hpp"
#include "merge_path.cuh"

// -----------------------------------------------------------------
// Device functors
// -----------------------------------------------------------------

// Normalizes a vector element by dividing by a scalar
struct divide_by_scalar {
    float s;
    explicit divide_by_scalar(float s) : s(s) {}
    __device__ float operator()(float x) const { return x / s; }
};

// Casts float → double and squares (for L2 norm in double precision)
struct double_square {
    __device__ double operator()(float x) const {
        return (double)x * (double)x;
    }
};

// Double-precision squared difference (for convergence residual)
struct double_sq_diff {
    __device__ double operator()(float a, float b) const {
        double d = (double)a - (double)b;
        return d * d;
    }
};

// -----------------------------------------------------------------
// execute_eigenvector_centrality
//
// Runs Power Iteration to find the dominant eigenvector of the
// adjacency matrix A (stored as a symmetric CSR graph).
//
// Algorithm per iteration:
//   1. y  = A * x          (Merge-Path SpMV kernel)
//   2. y  = y / ||y||_2    (normalization)
//   3. r  = ||y - x||_2    (L2 convergence residual)
//   4. x  ← y              (O(1) pointer swap — NOT a copy)
//
// Key fixes vs previous version:
//   - thrust::swap(d_x, d_y) replaces thrust::copy (O(1) vs O(N))
//   - L2 norm and residual computed in double precision to prevent
//     accumulated float rounding over 1000 iterations
//   - Only one cudaDeviceSynchronize() per iteration (after the raw
//     CUDA kernel); all Thrust calls are already device-synchronous
//   - items_per_thread raised from 7 to 32 for better instruction-level
//     parallelism and memory latency hiding
// -----------------------------------------------------------------
void execute_eigenvector_centrality(
    const GraphCSR& graph,
    int   max_iterations = 1000,
    float tolerance      = 1e-6f)
{
    std::cout << "Starting Eigenvector Centrality Solver...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    const int N = graph.num_nodes;
    const int M = graph.num_edges;

    // --- Upload graph to GPU ---
    thrust::device_vector<int> d_row_ptr(graph.row_ptr);
    thrust::device_vector<int> d_col_ind(graph.col_ind);

    // --- Initialize score vectors ---
    // x is set to the uniform unit vector: ||x||_2 = 1.0 by construction
    const float init_val = 1.0f / std::sqrt(static_cast<float>(N));
    thrust::device_vector<float> d_x(N, init_val);
    thrust::device_vector<float> d_y(N, 0.0f);

    // Raw pointers for the custom CUDA kernel
    const int*   raw_rp = thrust::raw_pointer_cast(d_row_ptr.data());
    const int*   raw_ci = thrust::raw_pointer_cast(d_col_ind.data());
    const float* raw_x  = thrust::raw_pointer_cast(d_x.data());
    float*       raw_y  = thrust::raw_pointer_cast(d_y.data());

    // --- Kernel launch configuration ---
    // L = 32: each thread processes 32 units of merged work.
    // Higher L means fewer binary searches (O(log W) each) and better
    // reuse of cached row_ptr values across the walk loop.
    // Tested range: L in [16, 64]. L=32 balances occupancy vs search cost.
    const int THREADS_PER_BLOCK = 256;
    const int L                 = 32;
    const int total_work        = N + M;
    const int total_threads     = (total_work + L - 1) / L;
    const int num_blocks        = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int   iter     = 0;
    float residual = 1.0f;

    while (iter < max_iterations && residual > tolerance) {

        // Step A: reset y to zero
        thrust::fill(d_y.begin(), d_y.end(), 0.0f);

        // Step B: SpMV  y = A * x
        // Note: raw_x must be updated after each swap, so we always
        // re-cast the pointer at the start of the loop body.
        raw_x = thrust::raw_pointer_cast(d_x.data());
        raw_y = thrust::raw_pointer_cast(d_y.data());

        merge_path_spmv_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
            raw_rp, raw_ci, raw_x, raw_y, N, M, L);

        // One sync needed here: raw CUDA kernel is asynchronous;
        // Thrust ops below read d_y and must see the completed result.
        cudaDeviceSynchronize();

        // Step C: ||y||_2 in double precision
        // Using transform_reduce with a double accumulator prevents
        // the catastrophic cancellation that float summation suffers
        // over millions of elements.
        double norm_sq = thrust::transform_reduce(
            d_y.begin(), d_y.end(),
            double_square(), 0.0, thrust::plus<double>());
        float norm = static_cast<float>(std::sqrt(norm_sq));

        // Step D: normalize  y = y / ||y||
        thrust::transform(d_y.begin(), d_y.end(), d_y.begin(),
                          divide_by_scalar(norm));

        // Step E: convergence residual  ||y - x||_2  in double precision
        double diff_sq = thrust::inner_product(
            d_y.begin(), d_y.end(),
            d_x.begin(),
            0.0,
            thrust::plus<double>(),
            double_sq_diff());
        residual = static_cast<float>(std::sqrt(diff_sq));

        // Step F: x ← y
        // FIX: was thrust::copy (O(N) device memory traffic every iteration).
        // thrust::swap is O(1) — it exchanges internal device pointers only.
        thrust::swap(d_x, d_y);

        ++iter;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "----------------------------------------\n";
    if (iter >= max_iterations)
        std::cout << "WARNING: Reached iteration cap (" << max_iterations
                  << ") without full convergence.\n";
    else
        std::cout << "SUCCESS: Converged in " << iter << " iterations.\n";
    std::cout << "Final Residual:  " << residual    << "\n";
    std::cout << "Solver Time:     " << elapsed << " seconds\n";
    std::cout << "----------------------------------------\n";
}

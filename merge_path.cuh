#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// -----------------------------------------------------------------
// merge_path_search
//
// Finds the starting (row, nz) coordinate for a thread whose work
// begins at 'diagonal' in the merged (rows + nonzeros) sequence.
//
// The 2D grid:  x-axis = row index (0..M),  y-axis = nz index (0..NNZ)
// A thread at diagonal d must satisfy:  row + nz = d
// We binary-search for the largest 'row' such that row_ptr[row] <= d - row,
// which is the condition for the thread to be inside (or at the start of)
// that row's nonzero segment.
//
// FIX vs original: the condition was row_ptr[pivot] <= (diagonal - pivot - 1)
// which is an off-by-one error. The correct condition is:
//     row_ptr[pivot] <= (diagonal - pivot)
// -----------------------------------------------------------------
__device__ __forceinline__ int merge_path_search(
    int diagonal,
    const int* __restrict__ row_ptr,
    int num_rows,
    int num_nonzeros)
{
    int x_min = max(0, diagonal - num_nonzeros);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;

        // __ldg: load via read-only (texture) cache — row_ptr never changes
        // during kernel execution, so L1 texture cache hits are guaranteed.
        if (__ldg(&row_ptr[pivot]) <= (diagonal - pivot)) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }
    return min(x_min, num_rows);
}

// -----------------------------------------------------------------
// merge_path_spmv_kernel
//
// Computes y = A * x for an unweighted CSR matrix A.
//
// Each thread:
//   1. Binary-searches its starting (row, nz) coordinate on its diagonal.
//   2. Walks L steps (items_per_thread), alternating between:
//        Horizontal step — consume a nonzero edge (accumulate x[col])
//        Vertical step   — cross a row boundary (flush accumulator via atomicAdd)
//   3. Flushes any remaining accumulator at the end.
//
// Design choices:
//   - double local accumulator: prevents float precision loss inside a row
//     when a hub node has thousands of contributing threads.
//   - __ldg on col_ind and x: both are read-only during SpMV, so the
//     L1 texture cache is used instead of the general L1, doubling
//     effective read bandwidth on heavily accessed hub columns.
//   - Unweighted assumption: edge weight = 1.0 implicitly; the values[]
//     array is omitted entirely, cutting global memory traffic by ~33%.
//   - atomicAdd only at row boundaries: each thread does at most one
//     atomic write per row it spans, not one per nonzero.
// -----------------------------------------------------------------
__global__ void merge_path_spmv_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_ind,
    const float* __restrict__ x,
    float*       __restrict__ y,
    int num_rows,
    int num_nonzeros,
    int items_per_thread)
{
    int thread_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int diagonal   = thread_id * items_per_thread;
    int total_work = num_rows + num_nonzeros;

    if (diagonal >= total_work) return;

    // Find this thread's starting position in the merged sequence
    int curr_row = merge_path_search(diagonal, row_ptr, num_rows, num_nonzeros);
    int curr_nz  = diagonal - curr_row;

    double local_sum = 0.0;

    for (int step = 0; step < items_per_thread; ++step) {
        // Have we consumed all work?
        if ((curr_row + curr_nz) >= total_work) break;

        // Decide: are we still inside the current row (horizontal),
        // or have we exhausted its nonzeros (vertical / row boundary)?
        if (curr_row < num_rows &&
            curr_nz  < __ldg(&row_ptr[curr_row + 1]))
        {
            // --- Horizontal step: process one nonzero edge ---
            // Both col_ind and x are read-only; __ldg routes them through
            // the L1 texture cache for better bandwidth on repeated column hits.
            local_sum += (double)__ldg(&x[__ldg(&col_ind[curr_nz])]);
            ++curr_nz;
        }
        else
        {
            // --- Vertical step: crossed a row boundary ---
            if (curr_row < num_rows && local_sum != 0.0) {
                // atomicAdd is required: another thread may hold the other
                // half of this row's sum.
                atomicAdd(&y[curr_row], (float)local_sum);
            }
            local_sum = 0.0;
            ++curr_row;
        }
    }

    // Final flush: thread exited the loop still holding a partial sum
    if (curr_row < num_rows && local_sum != 0.0) {
        atomicAdd(&y[curr_row], (float)local_sum);
    }
}

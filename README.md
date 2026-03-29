# GPU-Accelerated Eigenvector Centrality: Merge-Path SpMV

A high-performance GPU eigensolver that computes Eigenvector Centrality on large-scale,
power-law networks. The project analyzes the temporal evolution of the Amazon product
co-purchasing network from March to June 2003, benchmarking how Merge-Path SpMV scales
against graphs that grow rapidly in density.

---

## Table of Contents

1. [The Core Problem: Load Imbalance on Power-Law Graphs](#the-core-problem)
2. [Why Merge-Path?](#why-merge-path)
3. [Datasets & Symmetrization](#datasets--symmetrization)
4. [Algorithm Architecture](#algorithm-architecture)
5. [GPU Optimizations](#gpu-optimizations)
6. [Bug Fixes Applied](#bug-fixes-applied)
7. [Project Structure](#project-structure)
8. [Compilation & Execution](#compilation--execution)
9. [Expected Output](#expected-output)

---

## The Core Problem

Real-world networks do not just grow in size — they grow in density, and they grow
*unevenly*. Between March 2003 (`amazon0302`) and June 2003 (`amazon0601`), the Amazon
co-purchasing graph's node count grew by roughly 50%, but its edge count nearly tripled.

More critically, this is a **scale-free (power-law) network**: a small number of "hub"
products (bestsellers, bundle deals) accumulate thousands of co-purchase edges, while the
vast majority of products have only two or three connections. This degree imbalance is the
central challenge for GPU parallelism.

Traditional row-parallel SpMV (one warp per row, or one thread block per row) assigns
work proportional to node degree. On a scale-free graph this is catastrophic: threads
handling hub nodes do thousands of operations while threads handling leaf nodes do two,
leaving the majority of the GPU idle for most of each iteration.

---

## Why Merge-Path?

Merge-Path SpMV (Merrill & Garland, SC 2016) solves the load-imbalance problem by
**ignoring node degrees entirely**.

Instead of assigning rows to threads, it treats the entire SpMV operation as a single
1D sequence of `W = M + N` work units — `M` edge accumulations and `N` row-boundary
crossings — laid out as a 2D grid. Each thread receives exactly `L` consecutive units
from this sequence, found via a binary search on its assigned diagonal. Whether a thread
lands on a hub node with 2,000 edges or a leaf with 2, it does the same amount of work.

This achieves **near-perfect load balance across all graph topologies**, and critically,
it degrades gracefully as the graph densifies over time — the same kernel handles
`amazon0302` and `amazon0601` without topology-specific tuning.

> Note: "near-perfect" is the correct characterization. The binary search overhead
> (O(log W) per thread), global memory latency, and atomic contention at hub nodes mean
> real-world throughput is below theoretical peak. Claims of "100% perfect utilization"
> are marketing language, not measurement.

---

## Datasets & Symmetrization

We use the temporal directed Amazon datasets from the Stanford Network Analysis Project
([SNAP](https://snap.stanford.edu/data/#amazon)).

### Why Symmetrization is Required

Eigenvector centrality is mathematically defined only for **undirected** (symmetric)
graphs. The SNAP datasets are directed (A → B). The CPU loader automatically injects
reverse edges (B → A) during parsing and removes self-loops and any duplicate edges that
result from existing reverse edges in the raw data.

### Dataset Statistics (from SNAP)

| Dataset      | Date           | Nodes   | Raw Edges | Edges After Symmetrization* |
|--------------|----------------|---------|-----------|------------------------------|
| `amazon0302` | March 2, 2003  | 262,111 | 1,234,877 | ~2,469,754                  |
| `amazon0312` | March 12, 2003 | 400,727 | 3,200,440 | ~6,400,880                  |
| `amazon0505` | May 5, 2003    | 410,236 | 3,356,824 | ~6,713,648                  |
| `amazon0601` | June 1, 2003   | 403,394 | 3,387,388 | ~6,774,776                  |

\* Symmetrized counts are approximate. The exact post-dedup count depends on how many
reverse edges already exist in the raw dataset. `amazon0302` appears to have essentially
none; `amazon0312`–`amazon0601` may have some. The loader reports the exact final count
at runtime.

**Key observation:** Between `amazon0312` and `amazon0505`, node count barely changes
(+2.4%) but raw edge count barely changes too — but the important effect for load
balancing is the *degree distribution skew*, which intensifies as hub products accumulate
more co-purchases.

---

## Algorithm Architecture

### Phase 1 — CPU: Graph Loading & CSR Construction (`loader.hpp`)

1. Parses the SNAP edge-list format, skipping comment lines.
2. Symmetrizes: for every directed edge (u, v), adds (v, u) unless u == v.
3. Sorts all edges lexicographically — a mandatory precondition for CSR and for
   coalesced GPU memory access.
4. Deduplicates: removes reverse edges already present in the raw dataset.
5. Builds the CSR arrays via a degree-counting + prefix-sum pass.

### Phase 2 — GPU: Power Iteration (`solver.cuh`)

Finds the dominant eigenvector of the adjacency matrix via repeated matrix-vector
multiplication.

**Initialization:** `x` is set to `1 / sqrt(N)` so the initial L2 norm is exactly 1.

**Per-iteration steps:**
```
y  = A * x           (Merge-Path SpMV — the GPU kernel)
y  = y / ||y||_2     (normalization — Thrust transform)
r  = ||y - x||_2     (L2 convergence residual — Thrust inner_product)
x <-- y              (O(1) pointer swap — NOT a memory copy)
```

**Convergence:** The L2 residual `||y - x||_2 < 1e-6` is used. Because Amazon graphs
have large diameters (~44 hops), the spectral gap between the dominant and
second eigenvalue is small, making convergence slow. A hard cap of 1000 iterations
prevents infinite loops on graphs that fail to converge within tolerance.

### Phase 3 — GPU: Merge-Path SpMV Kernel (`merge_path.cuh`)

1. Each thread computes its starting diagonal `d = thread_id * L`.
2. A binary search (`O(log W)`) on `row_ptr` finds the starting `(row, nz)` coordinate.
3. The thread walks `L` steps, alternating between:
   - **Horizontal step** (within a row): `local_sum += x[col_ind[nz]]; nz++`
   - **Vertical step** (row boundary): `atomicAdd(&y[row], local_sum); row++`
4. At the end of the walk, any remaining partial sum is flushed via `atomicAdd`.

---

## GPU Optimizations

### `__ldg` Read-Only Cache
`row_ptr`, `col_ind`, and `x` are read-only during SpMV. Loading them via `__ldg()`
routes reads through the L1 texture cache (separate from the regular L1), which
significantly improves effective bandwidth on repeated column accesses to the same hub
node across multiple threads.

### Double-Precision Local Accumulator
Each thread accumulates its partial row sum in a `double` register. This prevents
floating-point precision loss within a single row's partial sum — especially critical
for hub nodes where thousands of `x[col]` values are summed, and small rounding errors
would compound across 1000 power iterations. The final `atomicAdd` casts back to `float`
at the row boundary only.

### Double-Precision Reductions in the Solver
The L2 norm (`||y||_2`) and convergence residual (`||y - x||_2`) are computed using
`thrust::transform_reduce` and `thrust::inner_product` with a `double` accumulator.
Over `N = 6.7M` elements, float summation accumulates significant rounding error that
distorts convergence detection. Double precision costs negligible extra time here because
these are single-pass reductions over the full vector.

### `thrust::swap` Instead of `thrust::copy`
The previous version copied `d_y` into `d_x` at the end of each iteration:
```cpp
// Old: O(N) device memory write every iteration — ~6.7M × 4 bytes × 1000 iter
thrust::copy(d_y.begin(), d_y.end(), d_x.begin());

// Fixed: O(1) internal pointer swap
thrust::swap(d_x, d_y);
```
Over 1000 iterations on `amazon0601`, the old approach wrote ~26 GB of avoidable traffic
to global memory.

### Unweighted Edge Assumption
The Amazon co-purchasing graph is unweighted (all edges have weight 1.0). By omitting
the `values[]` array from the CSR format entirely, we eliminate one global memory read
per nonzero, reducing SpMV memory bandwidth by approximately 33%.

### Increased `items_per_thread` (L = 32)
The previous value of `L = 7` was too low. A higher `L` means:
- Fewer binary searches per unit of work (each costs `O(log W)` global memory reads).
- Longer walk loops, giving the GPU scheduler more independent instructions per warp
  to hide global memory latency.
- Better reuse of cached `row_ptr` values across the walk.

`L = 32` was chosen empirically. Reasonable values to experiment with: 16, 32, 64.

### Minimal Synchronization
Only one `cudaDeviceSynchronize()` is issued per iteration — after the raw CUDA kernel —
to ensure `d_y` is complete before Thrust reads it. All Thrust operations are
device-synchronous on the default stream and do not require additional explicit syncs.

---

## Bug Fixes Applied

### 1. Binary Search Off-by-One (Correctness Bug)
**File:** `merge_path.cuh` — `merge_path_search()`

The original condition:
```cpp
if (row_ptr[pivot] <= (diagonal - pivot - 1))   // WRONG
```
The correct condition:
```cpp
if (__ldg(&row_ptr[pivot]) <= (diagonal - pivot))  // FIXED
```
The `-1` caused every thread's starting `(row, nz)` coordinate to be shifted one step
off the correct diagonal, producing incorrect SpMV results — wrong eigenvectors with
output that could look plausible on some graph topologies but is mathematically
incorrect on all of them.

### 2. O(N) Copy Instead of O(1) Swap (Performance Bug)
**File:** `solver.cuh`

`thrust::copy(d_y → d_x)` at the end of each iteration was replaced with
`thrust::swap(d_x, d_y)`, eliminating ~26 GB of redundant device memory traffic over a
1000-iteration run on the largest dataset.

### 3. Float Precision in Reductions (Precision Bug)
**File:** `solver.cuh`

`thrust::inner_product` with a `float` start value (`0.0f`) caused the norm and
residual calculations to accumulate floating-point rounding error over millions of
elements. All reductions now use a `double` start value (`0.0`) and double-precision
functors, with the final result cast back to `float` only at the point of comparison.

### 4. `exit(1)` in the Loader (Robustness Bug)
**File:** `loader.hpp`

The original loader called `exit(1)` on file-open failure, bypassing all RAII cleanup
and preventing `main.cu`'s error guard (`if graph.num_nodes == 0`) from ever triggering.
It now returns an empty `GraphCSR{}` so the caller handles the error gracefully.

### 5. No GPU Validation at Startup
**File:** `main.cu`

Added `cudaGetDeviceCount()` + `cudaGetDeviceProperties()` to verify a CUDA GPU is
present and print its name, SM version, VRAM, and SM count before any work begins.
This avoids cryptic CUDA errors on CPU-only machines or misconfigured environments.

---

## Project Structure

```
.
├── datasets/              # Store downloaded SNAP .txt files here
├── src/
│   ├── main.cu            # Entry point: GPU check, CLI parsing, timing
│   ├── loader.hpp         # CPU: SNAP parsing, symmetrization, CSR build
│   ├── solver.cuh         # Power Iteration loop (Thrust + raw kernel)
│   └── merge_path.cuh     # Merge-Path SpMV CUDA kernel
└── README.md
```

---

## Compilation & Execution

### Prerequisites

- NVIDIA GPU, Compute Capability 7.0 or higher (Volta / Turing / Ampere / Ada)
- CUDA Toolkit 11.0 or newer
- GCC/G++ with C++17 support

### 1. Download the Data

```bash
mkdir -p datasets && cd datasets
wget https://snap.stanford.edu/data/amazon0302.txt.gz && gunzip amazon0302.txt.gz
wget https://snap.stanford.edu/data/amazon0312.txt.gz && gunzip amazon0312.txt.gz
wget https://snap.stanford.edu/data/amazon0505.txt.gz && gunzip amazon0505.txt.gz
wget https://snap.stanford.edu/data/amazon0601.txt.gz && gunzip amazon0601.txt.gz
cd ..
```

### 2. Build

Replace `-arch=sm_86` with the SM version matching your GPU:

| GPU Family           | Flag       |
|----------------------|------------|
| Volta (V100)         | `-arch=sm_70` |
| Turing (RTX 20xx)    | `-arch=sm_75` |
| Ampere (A100, RTX 30xx) | `-arch=sm_80` / `sm_86` |
| Ada (RTX 40xx)       | `-arch=sm_89` |

```bash
nvcc -O3 -std=c++17 -arch=sm_86 src/main.cu -o amazon_solver
```

### 3. Run

```bash
./amazon_solver datasets/amazon0302.txt
./amazon_solver datasets/amazon0312.txt
./amazon_solver datasets/amazon0505.txt
./amazon_solver datasets/amazon0601.txt
```

---

## Expected Output

```
========================================================
 GPU Eigenvector Centrality: Merge-Path SpMV
========================================================
GPU     : NVIDIA GeForce RTX 3080
SM      : 8.6
VRAM    : 10240 MB
SM Count: 68
========================================================
Loading dataset from: datasets/amazon0302.txt ...
----------------------------------------
Loaded in 0.81 s
Nodes : 262111
Edges : 2469754 (after symmetrization + dedup)
----------------------------------------
Starting Eigenvector Centrality Solver...
----------------------------------------
SUCCESS: Converged in 342 iterations.
Final Residual:  9.4e-07
Solver Time:     1.38 seconds
----------------------------------------
Execution Complete.
```

Convergence iteration counts will vary slightly by GPU and driver version due to
non-deterministic float atomicAdd ordering. The residual should reach `< 1e-6` on all
four datasets within the 1000-iteration cap.

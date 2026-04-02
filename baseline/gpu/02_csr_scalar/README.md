# Eigenvector Centrality: CSR-Scalar Implementation

This folder contains a **hand-written CSR-scalar** implementation of the Eigenvector Centrality algorithm. Unlike the cuSPARSE version, this uses a custom CUDA kernel to perform Sparse Matrix-Vector Multiplication (SpMV), providing a transparent baseline for understanding hardware utilization and memory access patterns.

## Implementation Overview

### Core Logic: One-Thread-Per-Row
The SpMV operation is implemented using a "Scalar" approach where each CUDA thread is responsible for exactly one row of the matrix:
$$y[i] = \sum_{j = \text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{vals}[j] \cdot x[\text{col\_idx}[j]]$$

* **Memory Access**: `row_ptr`, `col_idx`, and `vals` are accessed sequentially within a thread, but because different rows have different lengths, warps may experience **divergence**.
* **Gather Operation**: Accessing $x[\text{col\_idx}[j]]$ results in a "gather" pattern, which is irregular and relies heavily on the L2 cache for performance.
* **Normalization**: An in-place `normalize_inplace` kernel is used alongside `cublasSnrm2` to minimize host-device synchronization.

### Technical Stack
* **Custom CUDA Kernels**: `csr_scalar_spmv` and `normalize_inplace`.
* **cuBLAS**: Used only for the vector norm and residual calculation (`saxpy`).
* **Binary Loader**: Compatible with the `.bin` output from `snap_to_undirected_csr.py`.

---

## Execution Instructions

### 1. Compile
Target your specific architecture (e.g., `sm_86` for RTX 30-series).
```bash
nvcc -O3 -arch=sm_86 eigenvector_centrality_csr_scalar.cu -lcublas -o evcent_scalar
```

### 2. Run Benchmark
The block size can be tuned (default 256). Try 128 or 512 to see if occupancy affects performance on your specific graph.
```bash
./evcent_scalar graph.bin 1000 1e-6 20 256
```

---

## 📋 Full Reference Output
**Hardware:** NVIDIA GeForce RTX 3060 (12GB) | HP Elite Tower 800 G9

```text
Device     : NVIDIA GeForce RTX 3060  (SM 8.6, 28 SMs, 12.5 GB)
Peak BW    : 360.0 GB/s
Block size : 256 threads  |  Grid : 1024 blocks

Loading graph.bin ...
  Vertices   : 262111
  Edges      : 899792  (1799584 directed nnz)
  Avg degree : 6.87  |  Max : 420  |  Min : 1
  CSR memory : 15.45 MB
  Grid       : 1024 blocks x 256 threads
  Load time  : 7.15 ms

=== Top-20 Nodes by Eigenvector Centrality (CSR-scalar) ===
  Rank    Node ID     Score
  ----    -------     ----------
  1        200213      0.56502533
  2        166781      0.19424430
  3        150336      0.17316625
  4        203190      0.13930534
  5        207892      0.13819547
  6        207903      0.13790061
  7        214724      0.10180369
  8        200236      0.09640842
  9        187668      0.07798569
  10       225190      0.07633603
  11       208669      0.07562207
  12       222469      0.06765252
  13       107724      0.06674293
  14       230968      0.06641191
  15       235569      0.06634893
  16       38470       0.06575235
  17       250335      0.06563770
  18       22482       0.06333682
  19       197925      0.06301098
  20       234824      0.06186388

=== Performance Metrics (CSR-scalar) ===
  Disk load                            :     7.15 ms
  H2D transfer                         :     2.33 ms
  D2H transfer                         :     0.21 ms
  Total GPU time (all kernels)         :    61.80 ms
  SpMV kernel total                    :    40.73 ms
  Normalize kernel total               :     8.80 ms
  Iterations                           :      304
  Avg time / iteration                 :   0.1629 ms
  SpMV GFLOP/s                         :   26.862 GFLOP/s
  SpMV effective bandwidth             :  176.820 GB/s
  BW utilisation                       :    49.1 %
  Final residual                       : 9.595e-07
  CSR GPU memory                       :    15.45 MB
```

---

## Analysis: Scalar vs. cuSPARSE
Interestingly, on this specific dataset (**amazon0302**), the **CSR-Scalar** implementation outperforms the cuSPARSE generic API:

* **Avg Time / Iteration**: ~0.16ms (Scalar) vs. ~0.30ms (cuSPARSE).
* **BW Utilization**: 49.1% (Scalar) vs. 16.1% (cuSPARSE).

**Reasoning**: For graphs of this size (~262k nodes), the overhead of cuSPARSE's generic API and internal buffer management can exceed the raw execution time of a simple, direct kernel. However, as graph density and power-law skew increase, we expect the CSR-Scalar performance to degrade due to warp divergence, where cuSPARSE or CSR5 may take the lead.

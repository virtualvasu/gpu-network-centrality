# GPU Graph Analytics: Eigenvector Centrality Benchmarking

This is a high-performance benchmarking suite designed to evaluate different strategies for computing **Eigenvector Centrality** on NVIDIA GPUs. 

We compare a variety of implementations—ranging from hand-written scalar kernels to industry-standard libraries and advanced tiling formats—to measure throughput, memory efficiency, and convergence stability.

## 📊 Implementation Comparison

| Baseline | Input Complexity | Data Structures Required | Memory Overhead | Description |
| :--- | :--- | :--- | :--- | :--- |
| **CSR-Scalar** | Low | Plain CSR (row, col, val) | Minimal (0%) | Custom-written row-parallel CUDA kernel. |
| **cuSPARSE** | Medium | CSR Descriptors + Buffer | Low (Variable) | Uses NVIDIA's optimized generic SpMV API. |
| **CSR5** | High | CSR + Tile Descriptors | Medium | Advanced 2D tiling format for load balancing. |
| **cuGraph** | Low (API) | Edge List (COO) or Graph | High | RAPIDS framework-level graph object. |

---

## 📁 Repository Structure

* **`/scripts`**: Contains `snap_to_undirected_csr.py` for preprocessing raw SNAP text files into optimized binary CSR formats.
* **`/data`**: Directory for storing `.bin` graph files and raw datasets.
* **`/01_cusparse`**: The baseline implementation using NVIDIA cuSPARSE.
* **`/02_csr_scalar`**: Hand-written CUDA kernel (row-parallel approach).
* **`/03_csr5`**: Implementation utilizing the CSR5 storage format.
* **`/04_cugraph`**: High-level implementation using the NVIDIA RAPIDS cuGraph library.

---

## 🛠️ General Workflow

### 1. Preprocessing
All implementations (except potentially cuGraph) consume a **Binary CSR** format to minimize disk I/O and string parsing bottlenecks during benchmarking.
```bash
python3 scripts/snap_to_undirected_csr.py data/input.txt data/graph.bin
```

### 2. Running Algos
Navigate to the specific implementation folder and follow the local `README.md` for build and execution instructions. Each folder provides detailed telemetry including:
* **Effective Bandwidth (GB/s)**
* **GFLOP/s**
* **Iteration-level timing**

---

## 💻 System Requirements
* **OS:** Linux (Ubuntu 22.04+ recommended)
* **GPU:** NVIDIA Ampere architecture or newer (Compute Capability 8.0+)
* **Toolkit:** CUDA 12.x+, cuSPARSE, cuBLAS
* **Python:** 3.8+ with `numpy` (for scripts)

---

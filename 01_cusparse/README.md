# Eigenvector Centrality: cuSPARSE Baseline

This folder contains the **cuSPARSE** implementation of the Eigenvector Centrality algorithm. This version serves as the "Gold Standard" for performance and accuracy comparisons against custom-built kernels.

## Implementation Overview

The implementation follows the **Power Iteration** method to find the dominant eigenvector of the graph's adjacency matrix.

### Core Logic
1.  **Initialization**: The score vector $v$ is initialized to $1/\sqrt{n}$.
2.  **SpMV (Matrix-Vector Multiplication)**: Computes $v_{next} = A \times v$ using the `cusparseSpMV` generic API.
3.  **Normalization**: The resulting vector is normalized using the $L_2$ norm:
    $$v_{next} = \frac{v_{next}}{\|v_{next}\|_2}$$
    This is handled via `cublasSnrm2` and `cublasSscal`.
4.  **Convergence**: The process repeats until the residual $\|v_{next} - v\|_2 < \text{tolerance}$ or the maximum iteration count is reached.

### Technical Stack
* **cuSPARSE**: Manages the CSR descriptors (`cusparseSpMatDescr_t`) and provides highly optimized SpMV kernels.
* **cuBLAS**: Used for high-performance vector reductions (norms) and scaling.
* **Binary Loader**: Uses a custom C++ struct to `fread` the `.bin` file directly into GPU-ready host vectors.

---

## Execution Instructions

### 1. Prepare Binary CSR Input
If your dataset is already in CSR arrays (`indptr.txt`, `indices.txt`, `data.txt`), you can pack it directly to `graph.bin` without any undirected-conversion step:

```bash
python3 - <<'PY'
import struct
from pathlib import Path

csr_dir = Path('../dataset/Amazon0601_csr')
out = csr_dir / 'graph.bin'

row_ptr = [int(x) for x in (csr_dir / 'indptr.txt').read_text().split()]
col_idx = [int(x) for x in (csr_dir / 'indices.txt').read_text().split()]
vals = [float(x) for x in (csr_dir / 'data.txt').read_text().split()]

n = len(row_ptr) - 1
nnz = len(col_idx)
assert len(vals) == nnz and row_ptr[-1] == nnz

with out.open('wb') as f:
  f.write(struct.pack('<ii', n, nnz))
  f.write(struct.pack(f'<{n+1}i', *row_ptr))
  f.write(struct.pack(f'<{nnz}i', *col_idx))
  f.write(struct.pack(f'<{nnz}f', *vals))

print(f'Wrote {out} with n={n}, nnz={nnz}')
PY
```

### 2. Compile
```bash
nvcc -O3 -arch=sm_86 eigenvector_centrality_cusparse.cu -lcusparse -lcublas -o evcent
```

### 3. Run Benchmark
```bash
./evcent ../dataset/Amazon0601_csr/graph.bin 1000 1e-6 20
```

---

## Full Reference Output
**Hardware:** NVIDIA GeForce RTX 3060 (12GB) | HP Elite Tower 800 G9

```text
Device  : NVIDIA GeForce RTX 3060  (SM 8.6, 28 SMs, 12.5 GB)
Peak BW : 360.0 GB/s

Loading graph.bin ...
  Vertices  : 262111
  Nnz (CSR) : 1799584
  Load time : 7.82 ms

=== Top-20 Nodes by Eigenvector Centrality ===
  Rank    Node ID    Score
  ----    -------    ----------
  1        200213      0.56502545
  2        166781      0.19424430
  3        150336      0.17316617
  4        203190      0.13930535
  5        207892      0.13819544
  6        207903      0.13790059
  7        214724      0.10180366
  8        200236      0.09640837
  9        187668      0.07798601
  10       225190      0.07633601
  11       208669      0.07562204
  12       222469      0.06765251
  13       107724      0.06674292
  14       230968      0.06641189
  15       235569      0.06634892
  16       38470       0.06575233
  17       250335      0.06563770
  18       22482       0.06333380
  19       197925      0.06301096
  20       234824      0.06186387

=== Performance Metrics ===
  Disk load                        :     7.82 ms
  H2D transfer                     :     3.17 ms
  D2H transfer                     :     0.22 ms
  Total GPU time                   :   104.73 ms
  SpMV+normalize only              :    91.59 ms
  Iterations                       :      302
  Avg time / iteration             :   0.3033 ms
  Effective GFLOP/s                :   11.867 GFLOP/s
  Effective bandwidth              :   57.840 GB/s
  BW utilisation                   :    16.1 %
  Final residual                   : 9.917e-07
  CSR GPU memory                   :    15.45 MB
```

---

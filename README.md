# Merge-Path CSR: Load-Balanced SpMV for Irregular Graphs

## 1. Overview
**Merge-Path CSR** is a high-performance algorithm for Sparse Matrix-Vector Multiplication (SpMV) on parallel architectures (GPUs). Unlike standard CSR implementations that assign work based on rows, Merge-Path partitions work based on **total units of work**, ensuring 100% thread utilization even in highly irregular "power-law" graphs.



---

## 2. The Problem: Load Imbalance in Standard CSR
In real-world graphs (e.g., social networks), node degrees vary significantly. 
* **Standard Approach:** Assigning one thread per row leads to **Warp Divergence**. 
* **The Bottleneck:** Threads assigned to "short" rows finish instantly and sit idle, while threads assigned to "long" rows (celebrity nodes) bottleneck the entire GPU.



---

## 3. The Concept: The Merge-Path Strategy
Merge-Path reimagines the SpMV operation as a 2D search problem. It "merges" two logical sequences into a single work-stream:
1.  **Non-Zero Elements ($NNZ$):** The multiplication operations.
2.  **Row Boundaries ($M$):** The points where a row sum is finalized and written to the output vector.

**Total Work Units ($W$):** $M + NNZ$.

### The 2D Search Space
Imagine a grid where the X-axis is the list of non-zeros and the Y-axis is the list of row pointers. The algorithm traces a "path" from $(0,0)$ to $(NNZ, M)$.



---

## 4. Implementation Details
The algorithm ensures every thread receives exactly $L = W / (\text{Total Threads})$ units of work.

### Step A: Independent Partitioning (Binary Search)
To find its starting point without inter-thread communication, each thread $k$ performs a **binary search** along a diagonal in the 2D grid.
* It searches for the intersection of the diagonal $k \times L$ and the logical "Merge Path."
* **Result:** Each thread independently identifies its starting `(row_idx, nz_idx)` in $O(\log W)$ time.

### Step B: The Execution Loop (The Walk)
Each thread "walks" its assigned segment of the path:
* **Horizontal Step:** Processes a non-zero element (Multiply-Accumulate).
* **Vertical Step:** Hits a row boundary, writes the local accumulator to the output vector $y[row]$, and resets for the next row.

---

## 5. Application: Eigenvector Centrality
For computing Eigenvector Centrality, this kernel is typically used within a **Power Iteration** loop:
1.  **Initialize:** $x_0$ vector.
2.  **Iterate:** * Execute **Merge-Path SpMV**: $x_{k+1} = Ax_k$.
    * **Normalize** $x_{k+1}$.
    * Check for convergence.
3.  **Result:** The principal eigenvector represents the centrality scores.

---

## 6. Comparison Summary

| Metric | Standard CSR | Segmented Scan | Merge-Path CSR |
| :--- | :--- | :--- | :--- |
| **Load Balancing** | Poor (Row-based) | Good (Element-based) | **Perfect** (Work-based) |
| **GPU Efficiency** | High Divergence | High Bandwidth Overhead | **Maximized Throughput** |
| **Complexity** | Simple | Medium | High (Binary Search) |
| **Data Format** | Plain CSR | Plain CSR + Flags | **Plain CSR** |

---

> **Key Takeaway:** Merge-Path CSR is the optimal choice for GPU-accelerated graph algorithms where the graph structure is unpredictable or highly skewed.
---

## Pseudocode:

### Phase 1: The Binary Search
This runs at the very beginning of the kernel. Every thread $k$ calculates its unique starting point $(i, j)$ in the $M \times NNZ$ work-grid.

```python
# Constants
M = number_of_rows
NNZ = number_of_non_zeros
W = M + NNZ  # Total units of work
T = total_threads
L = ceil(W / T)  # Work units per thread

# Thread-specific logic
k = thread_id
K = k * L # My starting work index

# Range for binary search on the diagonal
# We are searching for the row index 'i'
low = max(0, K - NNZ)
high = min(K, M)

while low < high:
    mid = (low + high + 1) // 2
    # Check if the row boundary at 'mid' occurs AFTER our work index 'K'
    if row_ptr[mid] <= K - mid:
        low = mid
    else:
        high = mid - 1

# Resulting Coordinates
start_row = low
start_nz  = K - low
```

---

### Phase 2: The Streaming Walk
Now that the thread knows exactly which row and which non-zero it starts at, it consumes its $L$ units of work.

```python
curr_row = start_row
curr_nz  = start_nz
accumulator = 0.0

for step in range(L):
    # Check if we have exceeded total work (for the last thread)
    if (curr_row + curr_nz) >= W:
        break

    # Determine if the next unit of work is a Non-Zero or a Row Boundary
    # We look at the next row's starting position
    if curr_row < M and curr_nz < row_ptr[curr_row + 1]:
        # --- HORIZONTAL STEP (Process Non-Zero) ---
        col = col_indices[curr_nz]
        val = values[curr_nz]
        accumulator += val * x[col]
        curr_nz += 1
    else:
        # --- VERTICAL STEP (Hit Row Boundary) ---
        # We finished a segment of a row. 
        # Use atomicAdd because another thread might have 
        # processed a different part of this same row.
        if curr_row < M:
            atomicAdd(y[curr_row], accumulator)
            
        accumulator = 0.0
        curr_row += 1

# Final Cleanup: If the thread ends while still holding an accumulation
if accumulator != 0 and curr_row < M:
    atomicAdd(y[curr_row], accumulator)
```

---

### Critical "Foolproof" Tips

1.  **The $M+NNZ$ Logic:** Remember that the total path is $M$ vertical steps and $NNZ$ horizontal steps. If your `row_ptr` has $M+1$ elements, the last element `row_ptr[M]` should equal $NNZ$.
2.  **Atomics:** While `atomicAdd` has a tiny performance penalty, it is the "foolproof" way to handle rows that are split across two threads. More advanced versions use shared memory "reductions," but that adds 200 lines of complex code.
3.  **Boundary Conditions:** Notice the `low` and `high` calculation in the binary search. It ensures that threads don't search outside the bounds of the matrix, even if they are the very first or very last thread.
4.  **Zero-Length Rows:** This pseudocode naturally handles empty rows. If `row_ptr[i] == row_ptr[i+1]`, the "Horizontal Step" condition will fail immediately, the thread will perform a "Vertical Step," write $0$ (or nothing) to that row, and move on.

---

### How to use this for Centrality
In your **Power Iteration** loop, you would launch this kernel, then launch a second, much simpler "Normalization" kernel that finds the norm of $y$ and divides every element by it before the next iteration begins.

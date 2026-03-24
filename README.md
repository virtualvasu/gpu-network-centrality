# 📄 Power Iteration vs. Lanczos for Eigenvector Centrality (K = 1)

---

## 📌 Overview

This document provides a formal comparison between **Power Iteration** and the **Lanczos Method** for computing **eigenvector centrality** on large-scale sparse graphs.

The goal is to justify the algorithmic choice for this project:

> 🔑 **When K = 1 (only the principal eigenvector is required), Power Iteration is the most efficient and scalable method, while Lanczos does not provide meaningful advantages and introduces additional overhead.**

---

## 🎯 Problem Statement

Given a large sparse adjacency matrix ( A \in \mathbb{R}^{n \times n} ), we aim to compute:

[
A x = \lambda x
]

where:

* ( x ) = principal eigenvector (eigenvector centrality)
* ( \lambda ) = largest eigenvalue
* ( K = 1 )

---

## ⚙️ Methods Compared

### 1. Power Iteration

### 2. Lanczos Method (Krylov Subspace Method)

---

## 🟢 Power Iteration

### Algorithm

[
x_{k+1} = \frac{A x_k}{|A x_k|}
]

### Characteristics

* Uses a **single vector**
* Requires **1 Sparse Matrix-Vector Multiplication (SpMV)** per iteration
* Memory complexity: **O(n)**
* Highly compatible with **CSR format and GPU acceleration**

---

### Advantages

✔ Minimal computational overhead
✔ Fully parallelizable (SpMV-dominated)
✔ Low memory footprint
✔ Numerically stable (due to normalization)
✔ Ideal for GPU execution (cuSPARSE optimized)

---

### Limitation

* Convergence rate depends on eigenvalue gap:
  [
  \frac{\lambda_2}{\lambda_1}
  ]

---

## 🔵 Lanczos Method

### Core Idea

Lanczos constructs a **K-dimensional Krylov subspace**:

[
\mathcal{K}_K(A, v) = \text{span}{v, Av, A^2v, ..., A^{K-1}v}
]

It projects ( A ) into a small tridiagonal matrix ( T ), and solves:

[
T y = \theta y
]

---

### Intended Benefits

✔ Efficient computation of **top-K eigenpairs simultaneously**
✔ Faster convergence for **multiple eigenvectors (K > 1)**

---

## 🔴 Limitations of Lanczos for K = 1

### 1. No Subspace Advantage

For K = 1:

* Krylov subspace degenerates to a single direction
* Projection matrix ( T ) becomes scalar (1×1)
* No meaningful speedup over power iteration

---

### 2. Additional Memory Overhead

Lanczos requires storing:

```text
K basis vectors → O(nK)
```

Even for small K:

* Higher memory consumption
* Increased memory bandwidth usage

---

### 3. Orthogonality Maintenance Cost

Lanczos suffers from **loss of orthogonality** due to floating-point errors.

Mitigation:

```text
Reorthogonalization → O(nK²)
```

For K = 1:

* This overhead is unnecessary
* Power iteration avoids this entirely

---

### 4. Increased Computational Complexity

Each Lanczos iteration involves:

* SpMV
* Vector projections (dot products)
* Orthogonalization steps
* Tridiagonal matrix updates

Compared to power iteration:

```text
Power iteration = 1 SpMV + normalization
```

---

### 5. Reduced GPU Efficiency

Lanczos introduces:

* Frequent synchronization points
* Reduction-heavy operations
* Irregular memory access patterns

Power iteration:

* Dominated by SpMV (highly optimized on GPU)

---

## ⚖️ Comparative Summary

| Feature                   | Power Iteration | Lanczos  |
| ------------------------- | --------------- | -------- |
| Target Use Case           | K = 1           | K > 1    |
| Memory Usage              | O(n)            | O(nK)    |
| Per Iteration Cost        | Low             | Medium   |
| GPU Efficiency            | High            | Medium   |
| Numerical Stability       | High            | Moderate |
| Implementation Complexity | Low             | High     |
| Benefit for K=1           | ✅ Yes           | ❌ No     |

---

## 🚀 Practical Optimization: Accelerated Power Iteration

To address slow convergence, we employ acceleration techniques such as:

```text
y = A x
y = y + β (x - x_old)
x = normalize(y)
```

### Benefits

✔ Faster convergence (empirically 2–5×)
✔ No additional memory overhead
✔ Fully GPU-compatible
✔ Retains simplicity of power iteration

---

## 🧠 Key Insight

> For K = 1, the problem is not algorithmic sophistication,
> but **minimizing cost per iteration and improving convergence rate**.

Power iteration achieves:

* Minimal per-iteration cost
* Maximum GPU efficiency
* Optimal memory usage

---

## 🏁 Final Recommendation

For **large-scale eigenvector centrality computation (K = 1)**:

> ✅ Use **Power Iteration with CSR + SpMV (optionally accelerated)**
> ❌ Avoid Lanczos and block methods

---

## 📌 Implementation Stack

* **Matrix format**: merge path CSR (Compressed Sparse Row)
* **SpMV**: cuSPARSE
* **Vector operations**: cuBLAS
* **Optional optimization**: Merge Path for load balancing

---

## 📎 Conclusion

While Lanczos is a powerful method for computing multiple eigenpairs, it does not provide advantages for single eigenvector computation. Power iteration remains the most efficient, scalable, and practical approach for eigenvector centrality in large graphs.

---

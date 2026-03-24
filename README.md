# Power Iteration vs. Lanczos for Eigenvector Centrality (K = 1)

## Overview
This document provides a formal comparison between Power Iteration and the Lanczos Method for computing eigenvector centrality on large-scale sparse graphs. The goal is to justify the algorithmic choice for this project:

When $K = 1$ (only the principal eigenvector is required), Power Iteration is the most efficient and scalable method, while Lanczos does not provide meaningful advantages and introduces additional overhead.

## Problem Statement
Given a large sparse adjacency matrix $A \in \mathbb{R}^{n \times n}$, we aim to compute:

$$Ax = \lambda x$$

where:
* $x$ = principal eigenvector (eigenvector centrality)
* $\lambda$ = largest eigenvalue
* $K = 1$

---

## Methods Compared

### 1. Power Iteration
**Algorithm**
$$x_{k+1} = \frac{Ax_k}{\|Ax_k\|}$$

**Characteristics**
* Uses a single vector
* Requires 1 Sparse Matrix-Vector Multiplication (SpMV) per iteration
* Memory complexity: $O(n)$
* Highly compatible with CSR format and GPU acceleration

**Advantages**
* Minimal computational overhead
* Fully parallelizable (SpMV-dominated)
* Low memory footprint
* Numerically stable (due to normalization)
* Ideal for GPU execution (cuSPARSE optimized)

**Limitation**
Convergence rate depends on the eigenvalue gap: $\frac{\lambda_2}{\lambda_1}$



### 2. Lanczos Method (Krylov Subspace Method)
**Core Idea**
Lanczos constructs a K-dimensional Krylov subspace:
$$\mathcal{K}_K(A, v) = \text{span}\{v, Av, A^2v, ..., A^{K-1}v\}$$

It projects $A$ into a small tridiagonal matrix $T$, and solves:
$$Ty = \theta y$$

**Intended Benefits**
* Efficient computation of top-K eigenpairs simultaneously
* Faster convergence for multiple eigenvectors ($K > 1$)

---

## Limitations of Lanczos for K = 1

1. **No Subspace Advantage**: For $K = 1$, the Krylov subspace degenerates to a single direction. The projection matrix $T$ becomes a scalar ($1 \times 1$). There is no meaningful speedup over power iteration.
2. **Additional Memory Overhead**: Lanczos requires storing $K$ basis vectors $\rightarrow O(nK)$. Even for small $K$, this results in higher memory consumption and increased memory bandwidth usage.
3. **Orthogonality Maintenance Cost**: Lanczos suffers from loss of orthogonality due to floating-point errors. Mitigation requires Reorthogonalization $\rightarrow O(nK^2)$. For $K = 1$, this overhead is unnecessary; Power Iteration avoids this entirely.
4. **Increased Computational Complexity**: Each Lanczos iteration involves SpMV, vector projections (dot products), orthogonalization steps, and tridiagonal matrix updates. Power Iteration involves only 1 SpMV + normalization.
5. **Reduced GPU Efficiency**: Lanczos introduces frequent synchronization points, reduction-heavy operations, and irregular memory access patterns. Power Iteration is dominated by SpMV, which is highly optimized on GPUs.

---

## Comparative Summary

| Feature | Power Iteration | Lanczos |
| :--- | :--- | :--- |
| Target Use Case | $K = 1$ | $K > 1$ |
| Memory Usage | $O(n)$ | $O(nK)$ |
| Per Iteration Cost | Low | Medium |
| GPU Efficiency | High | Medium |
| Numerical Stability | High | Moderate |
| Implementation Complexity | Low | High |
| Benefit for $K=1$ | Yes | No |

---

## Practical Optimization: Accelerated Power Iteration
To address slow convergence, we employ acceleration techniques:

1. $y = Ax_k$
2. $y = y + \beta (x_k - x_{k-1})$
3. $x_{k+1} = \frac{y}{\|y\|}$

**Benefits**
* Faster convergence (empirically 2–5x)
* No additional memory overhead
* Fully GPU-compatible
* Retains simplicity of power iteration

---

## Key Insight
For $K = 1$, the problem is not algorithmic sophistication, but minimizing cost per iteration and improving convergence rate. Power iteration achieves minimal per-iteration cost, maximum GPU efficiency, and optimal memory usage.

## Conclusion
While Lanczos is a powerful method for computing multiple eigenpairs, it does not provide advantages for single eigenvector computation. Power iteration remains the most efficient, scalable, and practical approach for eigenvector centrality in large graphs.

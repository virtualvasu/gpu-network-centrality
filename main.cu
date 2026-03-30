/**
 * ============================================================================
 * Eigenvector Centrality via Mixed-Precision Lanczos Algorithm on CUDA
 * ============================================================================
 * Based on: "A Mixed Precision, Multi-GPU Design for Large-scale Top-K
 *            Sparse Eigenproblems" — Sgherzi, Parravicini, Santambrogio
 *            IEEE ISCAS 2022
 *
 * INPUT:  Any SNAP edge-list file (.txt / .tsv), e.g.:
 *           https://snap.stanford.edu/data/
 *         Supported formats:
 *           - Tab-separated:   "0\t1"
 *           - Space-separated: "0 1"
 *           - Lines starting with '#' or '%' are comments (skipped)
 *           - Self-loops are skipped
 *           - Both 0-based and 1-based node IDs (auto-detected & remapped)
 *           - Directed graphs are symmetrized automatically
 *
 * OUTPUT: TSV file  →  node_id \t eigenvector_centrality
 *
 * ALGORITHM (paper §III):
 *   Phase A [GPU]  Lanczos iteration builds Krylov basis V and
 *                  tridiagonal scalars α[], β[]  (Algorithm 1)
 *   Phase B [CPU]  Power iteration on the tiny K×K tridiagonal T
 *                  (paper Fig. 1 ④ — too small to saturate GPU SMs)
 *   Phase C [GPU]  Ritz vector  x = V · t  via cuBLAS SAXPY
 *
 * MIXED-PRECISION STRATEGY (FDF = Float-Double-Float, paper §III-A):
 *   ┌──────────────────────────────┬──────────┬──────────────────────────┐
 *   │ Operation                    │ Precision│ Reason                   │
 *   ├──────────────────────────────┼──────────┼──────────────────────────┤
 *   │ Matrix + Lanczos vectors     │ float32  │ 2× memory, higher BW     │
 *   │ SpMV output                  │ float32  │ fast, sufficient          │
 *   │ Dot products α, β            │ float64  │ numerical stability       │
 *   │ Vector recurrence update     │ float32  │ dominated by memory BW   │
 *   └──────────────────────────────┴──────────┴──────────────────────────┘
 *   Paper result: FDF is 50% faster than pure float64 and
 *                 12× more accurate than pure float32.
 *
 * BUILD:
 *   nvcc -O3 -arch=sm_75 -o eigcentrality main.cu -lcusparse -lcublas -lm
 *   Arch flags: sm_70=V100  sm_80=A100  sm_86=RTX3090  sm_89=RTX4090
 *
 * RUN:
 *   ./eigcentrality <snap_graph.txt> [output.tsv] [max_iter] [top_k]
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

/* ═══════════════════════════ Error-check macros ════════════════════════════ */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA ERROR] %s:%d — %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t _s = (call);                                           \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[cuSPARSE ERROR] %s:%d — status %d\n",            \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[cuBLAS ERROR] %s:%d — status %d\n",              \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ═══════════════════════════ Compile-time defaults ════════════════════════ */

#define DEFAULT_MAX_ITER    100       /* Lanczos iterations                   */
#define DEFAULT_TOP_K       20        /* Top nodes printed to console         */
#define REORTH_INTERVAL     2         /* Re-orthogonalize every N iters       */
#define POWER_MAX_ITER      2000      /* CPU power-iter limit on tridiag      */
#define POWER_TOL           1e-12     /* CPU power-iter convergence threshold */
#define LANCZOS_TOL         1e-9      /* β threshold → invariant subspace     */

/* ═══════════════════════════ Wall-clock timer ══════════════════════════════ */

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ═══════════════════════════ CSR matrix (host) ═════════════════════════════ */

typedef struct {
    int    n;        /* number of nodes (= n_rows = n_cols)                   */
    long   nnz;      /* number of stored non-zeros                            */
    int   *row_ptr;  /* length n+1                                            */
    int   *col_idx;  /* length nnz                                            */
    float *values;   /* length nnz  (all 1.0f, unweighted adjacency)         */
} CSRMatrix;

/* ═══════════════════════════ SNAP loader ═══════════════════════════════════ */
/*
 * Handles every SNAP edge-list variant:
 *   - web-Google.txt, wiki-Talk.txt, soc-LiveJournal1.txt, roadNet-CA.txt …
 *   - Comments lines: # or %
 *   - Delimiters: TAB or SPACE (or both)
 *   - 0-based or 1-based node IDs  (auto-detected via min_id)
 *   - Directed input → symmetrized to undirected adjacency matrix
 *   - Duplicate/reverse edges are stored as-is (CSR naturally handles them;
 *     cuSPARSE SpMV sums duplicates, which is harmless for unweighted graphs)
 */

static int parse_edge_line(const char *line, long *src, long *dst) {
    const char *p = line;
    /* skip leading whitespace */
    while (*p && (*p == ' ' || *p == '\t')) p++;
    /* skip comment lines */
    if (*p == '#' || *p == '%' || *p == '\0' || *p == '\n') return 0;
    char *end;
    *src = strtol(p, &end, 10);
    if (end == p) return 0;          /* not a number */
    p = end;
    while (*p && (*p == ' ' || *p == '\t' || *p == ',')) p++;
    *dst = strtol(p, &end, 10);
    if (end == p) return 0;
    return 1;
}

CSRMatrix *load_snap(const char *filename) {
    printf("[IO] Opening SNAP dataset: %s\n", filename);

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[IO ERROR] Cannot open '%s': ", filename);
        perror(""); exit(EXIT_FAILURE);
    }

    /* ── Pass 1: count valid edges and find node ID range ── */
    char line[1024];
    long n_edges = 0;
    long min_id  = LONG_MAX;
    long max_id  = LONG_MIN;
    long src, dst;

    while (fgets(line, sizeof(line), fp)) {
        if (!parse_edge_line(line, &src, &dst)) continue;
        if (src == dst) continue;
        n_edges++;
        if (src < min_id) min_id = src;
        if (dst < min_id) min_id = dst;
        if (src > max_id) max_id = src;
        if (dst > max_id) max_id = dst;
    }

    if (n_edges == 0) {
        fprintf(stderr, "[IO ERROR] No valid edges found in '%s'\n", filename);
        fclose(fp); exit(EXIT_FAILURE);
    }

    /* Remap IDs to [0, n) */
    int  base = (int)min_id;
    int  n    = (int)(max_id - min_id + 1);

    printf("[IO] Node IDs: min=%ld max=%ld → n=%d nodes (base=%d)\n",
           min_id, max_id, n, base);
    printf("[IO] Valid directed edges (no self-loops): %ld\n", n_edges);
    printf("[IO] Symmetrizing for undirected centrality...\n");

    /* ── Allocate COO (2 × n_edges for symmetrization) ── */
    long  coo_cap = n_edges * 2;
    int  *coo_r   = (int*) malloc(coo_cap * sizeof(int));
    int  *coo_c   = (int*) malloc(coo_cap * sizeof(int));
    if (!coo_r || !coo_c) {
        fprintf(stderr, "[IO ERROR] OOM: cannot allocate %ld-entry COO\n", coo_cap);
        exit(EXIT_FAILURE);
    }

    /* ── Pass 2: fill COO ── */
    rewind(fp);
    long idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (!parse_edge_line(line, &src, &dst)) continue;
        if (src == dst) continue;
        int s = (int)(src - base);
        int d = (int)(dst - base);
        coo_r[idx] = s; coo_c[idx] = d; idx++;   /* forward edge  */
        coo_r[idx] = d; coo_c[idx] = s; idx++;   /* reverse edge  */
    }
    fclose(fp);
    long total_coo = idx;
    printf("[IO] Symmetric COO entries: %ld\n", total_coo);

    /* ── Count non-zeros per row (for CSR row_ptr) ── */
    long *row_cnt = (long*) calloc(n, sizeof(long));
    if (!row_cnt) { fprintf(stderr, "[IO ERROR] OOM (row_cnt)\n"); exit(1); }
    for (long i = 0; i < total_coo; i++) row_cnt[coo_r[i]]++;

    /* ── Build CSR ── */
    CSRMatrix *M  = (CSRMatrix*) malloc(sizeof(CSRMatrix));
    M->n          = n;
    M->nnz        = total_coo;
    M->row_ptr    = (int*)   malloc((n + 1) * sizeof(int));
    M->col_idx    = (int*)   malloc(total_coo * sizeof(int));
    M->values     = (float*) malloc(total_coo * sizeof(float));

    if (!M->row_ptr || !M->col_idx || !M->values) {
        fprintf(stderr, "[IO ERROR] OOM: cannot allocate CSR arrays\n");
        exit(EXIT_FAILURE);
    }

    M->row_ptr[0] = 0;
    for (int i = 0; i < n; i++)
        M->row_ptr[i + 1] = M->row_ptr[i] + (int)row_cnt[i];

    /* ── Scatter col_idx ── */
    long *fill = (long*) calloc(n, sizeof(long));
    for (long i = 0; i < total_coo; i++) {
        int  r   = coo_r[i];
        long pos = (long)M->row_ptr[r] + fill[r];
        M->col_idx[pos] = coo_c[i];
        M->values[pos]  = 1.0f;
        fill[r]++;
    }

    free(row_cnt); free(fill); free(coo_r); free(coo_c);

    double mat_mb = (M->nnz * (sizeof(int) + sizeof(float))
                  + (M->n + 1) * sizeof(int)) / 1e6;
    printf("[IO] CSR ready: n=%d  nnz=%ld  (%.1f MB host RAM)\n",
           M->n, M->nnz, mat_mb);
    return M;
}

/* ═══════════════════════════ CUDA kernels ══════════════════════════════════ */

/* v[i] /= norm */
__global__ void k_normalize(float *v, float norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] /= norm;
}

/*
 * Lanczos recurrence (FDF mixed precision, paper §III-A):
 *   v_{k+1} = vtmp  -  alpha * v_k  -  beta * v_{k-1}
 * beta = 0.0 on iteration 0 (v_prev is ignored via beta=0).
 */
__global__ void k_lanczos_update(float       *vnxt,
                                 const float *vtmp,
                                 const float *vi,
                                 const float *vi_prev,
                                 float        alpha,
                                 float        beta,
                                 int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        vnxt[i] = vtmp[i] - alpha * vi[i] - beta * vi_prev[i];
}

/*
 * Gram-Schmidt orthogonalization step (Algorithm 1 lines 12-18):
 *   v -= dot * vj
 */
__global__ void k_axpy_neg(float *v, const float *vj, float dot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] -= dot * vj[i];
}

/*
 * Upcast float32 → float64 element-wise.
 * Used before cublasDdot to achieve mixed-precision α, β computation.
 */
__global__ void k_f2d(const float *src, double *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}

/* ═══════════════════════════ Lanczos eigensolver ═══════════════════════════ */

void run_lanczos(const CSRMatrix *M,
                 float           *h_eigvec,   /* output: n floats */
                 int              max_iter,
                 double           tol)
{
    int  n   = M->n;
    long nnz = M->nnz;

    printf("\n[Lanczos] n=%d  nnz=%ld  max_iter=%d  reorth_every=%d\n",
           n, nnz, max_iter, REORTH_INTERVAL);

    /* ── Handles ── */
    cusparseHandle_t sp;  CUSPARSE_CHECK(cusparseCreate(&sp));
    cublasHandle_t   bl;  CUBLAS_CHECK(cublasCreate(&bl));

    /* ── Upload CSR to GPU ── */
    int   *d_rp, *d_ci;
    float *d_val;
    CUDA_CHECK(cudaMalloc(&d_rp,  (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci,  nnz     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, nnz     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rp,  M->row_ptr, (n+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,  M->col_idx, nnz  *sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, M->values,  nnz  *sizeof(float), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t sp_mat;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &sp_mat, n, n, nnz,
        d_rp, d_ci, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    /* ── Allocate Krylov basis V[0..max_iter-1] on GPU (float32) ── */
    float **d_V = (float**) malloc(max_iter * sizeof(float*));
    for (int i = 0; i < max_iter; i++)
        CUDA_CHECK(cudaMalloc(&d_V[i], n * sizeof(float)));

    /* ── Temporary buffers ── */
    float  *d_vtmp;
    double *d_da, *d_db;
    CUDA_CHECK(cudaMalloc(&d_vtmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_da,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_db,   n * sizeof(double)));

    /* ── Kernel config ── */
    const int THR = 256;
    int BLK = (n + THR - 1) / THR;

    /* ── Initialize v_0: L2-normalized uniform-random vector ── */
    {
        float *h_v0 = (float*) malloc(n * sizeof(float));
        srand(20220101u);
        double nrm = 0.0;
        for (int i = 0; i < n; i++) {
            h_v0[i] = (float)rand() / RAND_MAX - 0.5f;
            nrm += (double)h_v0[i] * h_v0[i];
        }
        nrm = sqrt(nrm);
        for (int i = 0; i < n; i++) h_v0[i] /= (float)nrm;
        CUDA_CHECK(cudaMemcpy(d_V[0], h_v0, n*sizeof(float), cudaMemcpyHostToDevice));
        free(h_v0);
    }

    /* ── Tridiagonal scalars ── */
    double *alpha = (double*) calloc(max_iter, sizeof(double));
    double *beta  = (double*) calloc(max_iter, sizeof(double));

    /* ── SpMV workspace ── */
    void  *d_spmv_buf = NULL;
    size_t spmv_bufsz = 0;
    float  one_f = 1.0f, zero_f = 0.0f;

    double t0     = wall_time();
    int    n_iter = max_iter;   /* set to actual count at convergence */

    /* ════════════════════ Lanczos main loop ════════════════════════════════
     *
     *  Implements Algorithm 1 of Sgherzi et al. 2022
     *
     *  ① β_k  = ||v_k||₂                   (float64 norm via cublasDdot)
     *  ② Normalize: v_k ← v_k / β_k
     *  ③ vtmp  = M · v_k                    (cuSPARSE SpMV, float32)
     *  ④ α_k   = v_k · vtmp                 (float64 dot — mixed precision)
     *  ⑤ v_{k+1} = vtmp - α·v_k - β·v_{k-1} (float32 kernel)
     *  ⑥ Re-orthogonalize every REORTH_INTERVAL iters (Gram-Schmidt)
     *
     ═══════════════════════════════════════════════════════════════════════ */

    for (int k = 0; k < max_iter; k++) {

        /* ── ① Compute β_k and normalize (skip first iteration) ── */
        if (k > 0) {
            /* d_V[k] was written as raw vnxt in the previous iteration */
            k_f2d<<<BLK, THR>>>(d_V[k], d_da, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            double bb;
            CUBLAS_CHECK(cublasDdot(bl, n, d_da, 1, d_da, 1, &bb));
            beta[k] = sqrt(bb);

            if (beta[k] < 1e-14) {
                printf("[Lanczos] Invariant subspace at k=%d (β=%.2e)\n",
                       k, beta[k]);
                n_iter = k;
                break;
            }
            k_normalize<<<BLK, THR>>>(d_V[k], (float)beta[k], n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        /* ── ③ vtmp = M · v_k  (SpMV, float32) ── */
        cusparseDnVecDescr_t dv_vi, dv_vt;
        CUSPARSE_CHECK(cusparseCreateDnVec(&dv_vi, n, d_V[k], CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dv_vt, n, d_vtmp, CUDA_R_32F));

        /* Query buffer size on first use */
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one_f, sp_mat, dv_vi, &zero_f, dv_vt,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_bufsz));
        if (spmv_bufsz > 0 && d_spmv_buf == NULL)
            CUDA_CHECK(cudaMalloc(&d_spmv_buf, spmv_bufsz));

        CUSPARSE_CHECK(cusparseSpMV(
            sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one_f, sp_mat, dv_vi, &zero_f, dv_vt,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUSPARSE_CHECK(cusparseDestroyDnVec(dv_vi));
        CUSPARSE_CHECK(cusparseDestroyDnVec(dv_vt));

        /* ── ④ α_k = v_k · vtmp  (float64 mixed-precision dot) ── */
        k_f2d<<<BLK, THR>>>(d_V[k], d_da, n);
        k_f2d<<<BLK, THR>>>(d_vtmp, d_db, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUBLAS_CHECK(cublasDdot(bl, n, d_da, 1, d_db, 1, &alpha[k]));

        /* ── ⑤ v_{k+1} = vtmp - α_k·v_k - β_k·v_{k-1}  (float32) ── */
        if (k < max_iter - 1) {
            float        af     = (float)alpha[k];
            float        bf     = (k > 0) ? (float)beta[k] : 0.0f;
            const float *v_prev = (k > 0) ? d_V[k-1] : d_vtmp; /* β=0 → unused */

            k_lanczos_update<<<BLK, THR>>>(d_V[k+1], d_vtmp, d_V[k], v_prev, af, bf, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            /* ── ⑥ Re-orthogonalization (paper Algorithm 1 lines 12-18) ── */
            if ((k + 1) % REORTH_INTERVAL == 0) {
                for (int j = 0; j <= k; j++) {
                    float o = 0.0f;
                    CUBLAS_CHECK(cublasSdot(bl, n, d_V[j], 1, d_V[k+1], 1, &o));
                    if (fabsf(o) > 1e-12f) {
                        k_axpy_neg<<<BLK, THR>>>(d_V[k+1], d_V[j], o, n);
                        CUDA_CHECK(cudaDeviceSynchronize());
                    }
                }
            }
        }

        /* ── Convergence check ── */
        if (k > 1 && beta[k] < tol) {
            printf("[Lanczos] Converged: k=%d  β=%.2e < tol=%.2e\n",
                   k, beta[k], tol);
            n_iter = k + 1;
            break;
        }

        if ((k + 1) % 10 == 0 || k == 0)
            printf("[Lanczos]  iter %3d  α=% .6f  β=%.6f\n",
                   k + 1, alpha[k], (k > 0 ? beta[k] : 0.0));
    }

    printf("[Lanczos] GPU phase: %.3f s  (%d iters)\n",
           wall_time() - t0, n_iter);

    /* ════════════════════ Phase B: CPU tridiagonal solve ═══════════════════
     *
     * Paper §III-B / Fig. 1 ④:
     * "The small tridiagonal matrices the Lanczos algorithm outputs cannot
     *  saturate the stream processors of a modern GPU. Instead, we achieve
     *  better execution time by performing this step on a CPU."
     *
     * We run power iteration on the n_iter × n_iter symmetric tridiagonal T
     * to find its dominant eigenvector t.
     ═══════════════════════════════════════════════════════════════════════ */

    int m = n_iter;
    if (m < 1) m = 1;
    printf("[CPU-Tridiag] Solving %d×%d tridiagonal...\n", m, m);

    double *t_cur = (double*) calloc(m, sizeof(double));
    double *t_new = (double*) calloc(m, sizeof(double));
    t_cur[0] = 1.0;

    double lambda = 0.0;
    for (int it = 0; it < POWER_MAX_ITER; it++) {
        /* Symmetric tridiagonal multiply: t_new = T · t_cur */
        for (int i = 0; i < m; i++) {
            t_new[i] = alpha[i] * t_cur[i];
            if (i > 0)     t_new[i] += beta[i]     * t_cur[i - 1];
            if (i < m - 1) t_new[i] += beta[i + 1] * t_cur[i + 1];
        }
        /* Rayleigh quotient and normalize */
        double rq = 0.0, nrm = 0.0;
        for (int i = 0; i < m; i++) {
            rq  += t_new[i] * t_cur[i];
            nrm += t_new[i] * t_new[i];
        }
        nrm = sqrt(nrm);
        for (int i = 0; i < m; i++) t_new[i] /= nrm;

        double err = fabs(rq - lambda);
        lambda = rq;
        memcpy(t_cur, t_new, m * sizeof(double));

        if (err < POWER_TOL && it > 5) {
            printf("[CPU-Tridiag] Converged at iter %d  λ=%.8f\n", it, lambda);
            break;
        }
    }
    printf("[CPU-Tridiag] Dominant eigenvalue λ = %.8f\n", lambda);

    /* ════════════════════ Phase C: Ritz vector on GPU ══════════════════════
     *
     * x = V · t  (Ritz vector ≈ dominant eigenvector of M)
     * Implemented as m cuBLAS SAXPY operations.
     ═══════════════════════════════════════════════════════════════════════ */
    printf("[Ritz] Reconstructing eigenvector of M: x = V·t ...\n");

    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));

    for (int j = 0; j < m; j++) {
        float tj = (float)t_cur[j];
        CUBLAS_CHECK(cublasSaxpy(bl, n, &tj, d_V[j], 1, d_x, 1));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Normalize to unit L2 */
    float xnrm;
    CUBLAS_CHECK(cublasSnrm2(bl, n, d_x, 1, &xnrm));
    if (xnrm > 0.0f)
        k_normalize<<<BLK, THR>>>(d_x, xnrm, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download + take abs (eigenvector sign is arbitrary) */
    CUDA_CHECK(cudaMemcpy(h_eigvec, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) h_eigvec[i] = fabsf(h_eigvec[i]);

    printf("[Lanczos] Total wall time: %.3f s\n", wall_time() - t0);

    /* ── Cleanup ── */
    free(alpha); free(beta); free(t_cur); free(t_new);
    for (int i = 0; i < max_iter; i++) cudaFree(d_V[i]);
    free(d_V);
    cudaFree(d_vtmp); cudaFree(d_da);   cudaFree(d_db);
    cudaFree(d_x);    cudaFree(d_rp);   cudaFree(d_ci);   cudaFree(d_val);
    if (d_spmv_buf) cudaFree(d_spmv_buf);
    cusparseDestroySpMat(sp_mat);
    cusparseDestroy(sp);
    cublasDestroy(bl);
}

/* ═══════════════════════════ Output ════════════════════════════════════════ */

static void write_output(const char *path, const float *c, int n) {
    FILE *fp = fopen(path, "w");
    if (!fp) { perror("[IO ERROR] Cannot open output"); exit(EXIT_FAILURE); }
    fprintf(fp, "node_id\teigenvector_centrality\n");
    for (int i = 0; i < n; i++)
        fprintf(fp, "%d\t%.10f\n", i, c[i]);
    fclose(fp);
    printf("[IO] Results written → %s\n", path);
}

static void print_top_nodes(const float *c, int n, int k) {
    int top = (k < n) ? k : n;
    int *idx = (int*) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;

    /* Partial selection sort O(n·top), fine for top ≤ 100 */
    for (int i = 0; i < top; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (c[idx[j]] > c[idx[best]]) best = j;
        int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║      Top-%d Nodes by Eigenvector Centrality           ║\n", top);
    printf("╠═══════════╦══════════════╦═══════════════════════════╣\n");
    printf("║  Rank     ║  Node ID     ║  Centrality               ║\n");
    printf("╠═══════════╬══════════════╬═══════════════════════════╣\n");
    for (int i = 0; i < top; i++)
        printf("║  %-8d ║  %-11d ║  %-25.10f  ║\n",
               i + 1, idx[i], c[idx[i]]);
    printf("╚═══════════╩══════════════╩═══════════════════════════╝\n\n");

    free(idx);
}

/* ═══════════════════════════ Main ══════════════════════════════════════════ */

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr,
            "\n"
            "  Eigenvector Centrality — Mixed-Precision Lanczos on CUDA\n"
            "  Based on Sgherzi et al., IEEE ISCAS 2022\n\n"
            "  Usage:\n"
            "    %s <snap_graph.txt>  [output.tsv]  [max_iter]  [top_k]\n\n"
            "  Arguments:\n"
            "    snap_graph.txt   SNAP edge-list (tab- or space-separated)\n"
            "                     Lines starting with # or %% are skipped\n"
            "    output.tsv       Output file with centrality scores   [default: centrality.tsv]\n"
            "    max_iter         Lanczos iterations                   [default: %d]\n"
            "    top_k            Number of top nodes to print         [default: %d]\n\n"
            "  Examples:\n"
            "    %s web-Google.txt\n"
            "    %s wiki-Talk.txt centrality.tsv 80 30\n"
            "    %s soc-LiveJournal1.txt out.tsv 120 50\n\n"
            "  SNAP datasets: https://snap.stanford.edu/data/\n\n",
            argv[0], DEFAULT_MAX_ITER, DEFAULT_TOP_K,
            argv[0], argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    const char *in_file  =  argv[1];
    const char *out_file = (argc >= 3) ? argv[2] : "centrality.tsv";
    int max_iter         = (argc >= 4) ? atoi(argv[3]) : DEFAULT_MAX_ITER;
    int top_k            = (argc >= 5) ? atoi(argv[4]) : DEFAULT_TOP_K;

    if (max_iter < 5) {
        fprintf(stderr, "[ERROR] max_iter must be >= 5 (got %d)\n", max_iter);
        return EXIT_FAILURE;
    }

    /* ── GPU discovery ── */
    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu == 0) {
        fprintf(stderr, "[ERROR] No CUDA GPU detected. Is the driver installed?\n");
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n[CUDA] Using GPU 0: %s  (%.2f GB VRAM  |  CC %d.%d  |  %d SMs)\n",
           prop.name, prop.totalGlobalMem / 1e9,
           prop.major, prop.minor, prop.multiProcessorCount);
    if (ngpu > 1)
        printf("[CUDA] %d additional GPU(s) present — single-GPU mode\n", ngpu - 1);

    /* ── Load SNAP dataset ── */
    double t0 = wall_time();
    CSRMatrix *M = load_snap(in_file);
    printf("[IO] Load time: %.3f s\n", wall_time() - t0);

    /* ── Memory estimate and warning ── */
    size_t est_gpu =
          (size_t)M->nnz * (sizeof(int) + sizeof(float))  /* CSR matrix      */
        + (size_t)(M->n + 1) * sizeof(int)                /* CSR row_ptr     */
        + (size_t)max_iter * M->n * sizeof(float)         /* Krylov basis V  */
        + (size_t)M->n * (sizeof(float) + 2 * sizeof(double)); /* tmp buffers */

    printf("[MEM] Estimated GPU RAM: %.2f GB  (%.2f GB available on GPU 0)\n",
           est_gpu / 1e9, prop.totalGlobalMem / 1e9);

    if (est_gpu > prop.totalGlobalMem * 0.90) {
        int safe_iter = (int)((prop.totalGlobalMem * 0.80
                               - (size_t)M->nnz * (sizeof(int) + sizeof(float))
                               - (size_t)(M->n + 1) * sizeof(int)
                               - (size_t)M->n * (sizeof(float) + 2 * sizeof(double)))
                              / ((size_t)M->n * sizeof(float)));
        safe_iter = (safe_iter < 10) ? 10 : safe_iter;
        fprintf(stderr,
            "[MEM WARNING] Estimated usage exceeds 90%% of VRAM.\n"
            "              Reduce max_iter. Suggested safe value: %d\n"
            "              Re-run: %s %s %s %d %d\n",
            safe_iter, argv[0], in_file, out_file, safe_iter, top_k);
        /* Continue anyway — CUDA will fail cleanly if we actually OOM */
    }

    /* ── Run eigensolver ── */
    float *h_centrality = (float*) malloc(M->n * sizeof(float));
    if (!h_centrality) {
        fprintf(stderr, "[ERROR] OOM allocating output vector (%d floats)\n", M->n);
        return EXIT_FAILURE;
    }

    double t_solve = wall_time();
    run_lanczos(M, h_centrality, max_iter, LANCZOS_TOL);
    printf("[PERF] Eigensolver wall time: %.3f s\n", wall_time() - t_solve);

    /* ── Print and save ── */
    print_top_nodes(h_centrality, M->n, top_k);
    write_output(out_file, h_centrality, M->n);

    /* ── Summary stats ── */
    double csum = 0.0, cmax = 0.0;
    for (int i = 0; i < M->n; i++) {
        if (h_centrality[i] > cmax) cmax = h_centrality[i];
        csum += h_centrality[i];
    }
    printf("[STATS] n=%d  max_centrality=%.8f  mean_centrality=%.8f\n",
           M->n, cmax, csum / M->n);
    printf("[DONE] Eigenvector Centrality complete.\n\n");

    /* ── Cleanup ── */
    free(M->row_ptr); free(M->col_idx); free(M->values); free(M);
    free(h_centrality);
    return EXIT_SUCCESS;
}

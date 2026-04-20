/*
 * eigenvector_centrality_csr_scalar.cu
 *
 * Eigenvector centrality via power iteration using a hand-written
 * CSR-scalar SpMV kernel: one CUDA thread per matrix row.
 *
 * Reads the binary CSR file produced by snap_to_undirected_csr.py.
 *
 * Binary CSR layout (little-endian):
 *   int32   n
 *   int32   nnz
 *   int32   row_ptr[n+1]
 *   int32   col_idx[nnz]
 *   float32 vals[nnz]
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 eigenvector_centrality_csr_scalar.cu -lcublas -o evcent_scalar
 *
 * Usage:
 *   ./evcent_scalar <graph.bin> [max_iter=1000] [tol=1e-6] [top_k=20] [block_size=256]
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x)                                                         \
    do {                                                                      \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(x)                                                       \
    do {                                                                      \
        cublasStatus_t _e = (x);                                              \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                 \
                    __FILE__, __LINE__, (int)_e);                             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA event timer
// ---------------------------------------------------------------------------
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start); }
    float end()  {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// CSR-scalar SpMV kernel
//
// One thread per row i:
//   y[i] = sum_{j = row_ptr[i]}^{row_ptr[i+1]-1}  vals[j] * x[col_idx[j]]
//
// Each thread:
//   1. Reads row_ptr[i] and row_ptr[i+1]  (2 int loads)
//   2. Loops over the row's non-zeros      (nnz_i iterations)
//   3. Each iteration: 1 int load (col_idx) + 1 float load (vals)
//                    + 1 gather load (x[col]) + 1 FMA
//   4. Writes one float (y[i])
//
// Performance characteristic:
//   - Irregular memory access pattern on x[] (random gather).
//   - Threads in the same warp handle different rows and thus
//     access entirely different columns => no coalescing on x.
//   - vals[] and col_idx[] ARE accessed sequentially within a row,
//     but adjacent threads in a warp start at different offsets
//     => partial coalescing only for regular (equal-degree) graphs.
//   - Works well when all rows have similar length (regular graphs).
//   - Degrades on power-law graphs where a few rows are very long
//     (those threads become the bottleneck; others sit idle).
// ---------------------------------------------------------------------------
__global__ void csr_scalar_spmv(
    const int   *__restrict__ row_ptr,   // [n+1]
    const int   *__restrict__ col_idx,   // [nnz]
    const float *__restrict__ vals,      // [nnz]
    const float *__restrict__ x,         // input vector  [n]
          float *__restrict__ y,         // output vector [n]
    int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int   row_start = row_ptr[row];
    int   row_end   = row_ptr[row + 1];
    float acc       = 0.f;

    for (int j = row_start; j < row_end; ++j)
        acc += vals[j] * x[col_idx[j]];

    y[row] = acc;
}

// ---------------------------------------------------------------------------
// L2-normalise kernel  (in-place, called after cuBLAS nrm2)
//
// Avoids a separate cublasSscal call — saves one kernel launch.
// ---------------------------------------------------------------------------
__global__ void normalize_inplace(float *v, float inv_norm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= inv_norm;
}

// ---------------------------------------------------------------------------
// CSR loader
// ---------------------------------------------------------------------------
struct CsrGraph {
    int n, nnz;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<float> vals;
};

static bool load_csr_binary(const char *path, CsrGraph &g)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return false; }

    if (fread(&g.n,   sizeof(int), 1, f) != 1) { fclose(f); return false; }
    if (fread(&g.nnz, sizeof(int), 1, f) != 1) { fclose(f); return false; }

    g.row_ptr.resize(g.n + 1);
    g.col_idx.resize(g.nnz);
    g.vals.resize(g.nnz);

    if ((int)fread(g.row_ptr.data(), sizeof(int),   g.n + 1, f) != g.n + 1) { fclose(f); return false; }
    if ((int)fread(g.col_idx.data(), sizeof(int),   g.nnz,   f) != g.nnz)   { fclose(f); return false; }
    if ((int)fread(g.vals.data(),    sizeof(float), g.nnz,   f) != g.nnz)   { fclose(f); return false; }

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
struct EvcMetrics {
    double load_ms;
    double h2d_ms;
    double d2h_ms;
    double total_gpu_ms;
    double total_spmv_ms;      // SpMV kernel time only
    double total_norm_ms;      // normalize kernel time only
    double avg_iter_ms;
    double gflops;             // SpMV GFLOP/s
    double gbps;               // SpMV effective bandwidth (GB/s)
    double bw_util_pct;
    int    iters;
    double final_residual;
};

struct OutputPaths {
    std::string dataset_key;
    std::string output_dir;
    std::string scores_csv;
    std::string metrics_json;
};

static std::string strip_extension(const std::string &name)
{
    size_t pos = name.rfind('.');
    if (pos == std::string::npos) return name;
    return name.substr(0, pos);
}

static std::string path_basename(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool mkdir_if_missing(const std::string &dir)
{
    if (dir.empty()) return true;
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    return errno == EEXIST;
}

static bool ensure_dir_recursive(const std::string &dir)
{
    if (dir.empty()) return true;
    std::string cur;

    for (size_t i = 0; i < dir.size(); ++i) {
        char c = dir[i];
        if (c == '/') {
            if (!cur.empty() && !mkdir_if_missing(cur)) return false;
        }
        cur.push_back(c);
    }
    return mkdir_if_missing(cur);
}

static OutputPaths build_output_paths(const char *input_path)
{
    std::string in = input_path;
    std::string base = path_basename(in);

    if (base.size() > 8 && base.substr(base.size() - 8) == ".csr.bin") {
        base = base.substr(0, base.size() - 8);
    } else {
        base = strip_extension(base);
    }

    OutputPaths p;
    p.dataset_key = base;
    p.output_dir = std::string("baseline/csr_scalar/") + p.dataset_key;
    p.scores_csv = p.output_dir + "/" + p.dataset_key + "_eigenvector_scores.csv";
    p.metrics_json = p.output_dir + "/step0_metrics.json";
    return p;
}

static bool write_scores_csv(const std::string &path, const std::vector<int> &sorted_idx,
                             const std::vector<float> &scores)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "node_id,score\n");
    for (size_t i = 0; i < sorted_idx.size(); ++i) {
        int node = sorted_idx[i];
        fprintf(f, "%d,%.8f\n", node, scores[node]);
    }
    fclose(f);
    return true;
}

static bool write_metrics_json(const std::string &path, const OutputPaths &out,
                               const char *input_path, const CsrGraph &g,
                               int max_iter, float tol, int top_k,
                               const EvcMetrics &m, double density,
                               double peak_bw, double bw_util,
                               int top_node_id, double top_score)
{
    FILE *f = fopen(path.c_str(), "w");
    if (!f) return false;

    fprintf(f, "{\n");
    fprintf(f, "  \"dataset_key\": \"%s\",\n", out.dataset_key.c_str());
    fprintf(f, "  \"dataset\": \"%s\",\n", input_path);
    fprintf(f, "  \"num_nodes\": %d,\n", g.n);
    fprintf(f, "  \"num_edges\": %d,\n", g.nnz / 2);
    fprintf(f, "  \"nnz\": %d,\n", g.nnz);
    fprintf(f, "  \"density\": %.12g,\n", density);
    fprintf(f, "  \"method\": \"csr_scalar.power_iteration\",\n");
    fprintf(f, "  \"graph_type\": \"undirected\",\n");
    fprintf(f, "  \"max_iter\": %d,\n", max_iter);
    fprintf(f, "  \"tol\": %.12g,\n", tol);
    fprintf(f, "  \"top_k\": %d,\n", top_k);
    fprintf(f, "  \"runtime_seconds\": %.12g,\n", m.total_gpu_ms / 1e3);
    fprintf(f, "  \"iterations\": %d,\n", m.iters);
    fprintf(f, "  \"converged\": %s,\n", (m.final_residual <= tol) ? "true" : "false");
    fprintf(f, "  \"final_residual\": %.12g,\n", m.final_residual);
    fprintf(f, "  \"h2d_ms\": %.12g,\n", m.h2d_ms);
    fprintf(f, "  \"d2h_ms\": %.12g,\n", m.d2h_ms);
    fprintf(f, "  \"spmv_ms\": %.12g,\n", m.total_spmv_ms);
    fprintf(f, "  \"normalize_ms\": %.12g,\n", m.total_norm_ms);
    fprintf(f, "  \"avg_iter_ms\": %.12g,\n", m.avg_iter_ms);
    fprintf(f, "  \"effective_gflops\": %.12g,\n", m.gflops);
    fprintf(f, "  \"effective_bandwidth_gbps\": %.12g,\n", m.gbps);
    fprintf(f, "  \"peak_bandwidth_gbps\": %.12g,\n", peak_bw);
    fprintf(f, "  \"bw_util_percent\": %.12g,\n", bw_util);
    fprintf(f, "  \"top_node_id\": %d,\n", top_node_id);
    fprintf(f, "  \"top_score\": %.12g\n", top_score);
    fprintf(f, "}\n");

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------
static void run_evcent_scalar(
    const CsrGraph     &g,
    int                 max_iter,
    float               tol,
    int                 block_size,
    std::vector<float> &h_scores,
    EvcMetrics         &m)
{
    const int n   = g.n;
    const int nnz = g.nnz;

    // ---- Device allocations ----
    int   *d_row_ptr, *d_col_idx;
    float *d_vals;
    float *d_v, *d_v_new, *d_diff;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,  nnz     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,     nnz     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,        n       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_new,    n       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff,     n       * sizeof(float)));

    // ---- H2D ----
    CudaTimer h2d_timer;
    h2d_timer.begin();
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (n + 1) * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(),  nnz     * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,    g.vals.data(),     nnz     * sizeof(float), cudaMemcpyHostToDevice));
    m.h2d_ms = h2d_timer.end();

    // ---- Initialise v = 1/sqrt(n) ----
    {
        std::vector<float> init(n, 1.0f / std::sqrt((float)n));
        CUDA_CHECK(cudaMemcpy(d_v, init.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ---- cuBLAS for nrm2 and residual ----
    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));

    // ---- Grid/block for SpMV and normalize ----
    int grid_spmv = (n + block_size - 1) / block_size;

    // ---- Power iteration ----
    CudaTimer total_timer, spmv_timer, norm_timer;
    double total_spmv_ms = 0.0, total_norm_ms = 0.0;
    int   iter     = 0;
    float residual = 1e9f;

    total_timer.begin();

    for (iter = 0; iter < max_iter && residual > tol; ++iter) {

        // -- SpMV: v_new = A * v --
        spmv_timer.begin();
        csr_scalar_spmv<<<grid_spmv, block_size>>>(
            d_row_ptr, d_col_idx, d_vals, d_v, d_v_new, n);
        total_spmv_ms += spmv_timer.end();

        // -- Normalize v_new in-place --
        norm_timer.begin();
        float norm = 1.0f;
        CUBLAS_CHECK(cublasSnrm2(bl, n, d_v_new, 1, &norm));
        if (norm < 1e-12f) norm = 1.0f;
        normalize_inplace<<<grid_spmv, block_size>>>(d_v_new, 1.0f / norm, n);
        total_norm_ms += norm_timer.end();

        // -- Residual: ||v_new - v|| --
        CUDA_CHECK(cudaMemcpy(d_diff, d_v_new, n * sizeof(float), cudaMemcpyDeviceToDevice));
        float neg1 = -1.0f;
        CUBLAS_CHECK(cublasSaxpy(bl, n, &neg1, d_v, 1, d_diff, 1));
        CUBLAS_CHECK(cublasSnrm2(bl, n, d_diff, 1, &residual));

        // -- Swap v <-> v_new --
        float *tmp = d_v; d_v = d_v_new; d_v_new = tmp;
    }

    m.total_gpu_ms  = total_timer.end();
    m.total_spmv_ms = total_spmv_ms;
    m.total_norm_ms = total_norm_ms;
    m.avg_iter_ms   = iter > 0 ? (total_spmv_ms + total_norm_ms) / iter : 0.0;
    m.iters         = iter;
    m.final_residual = (double)residual;

    // -- Performance metrics --
    // SpMV: 2 FLOPs per non-zero (multiply + add)
    m.gflops = 2.0 * nnz * iter / (total_spmv_ms * 1e-3) / 1e9;

    // SpMV bytes per iteration:
    //   row_ptr : 4 * (n+1)   -- two loads per row but amortised as ~1 per row
    //   col_idx : 4 * nnz
    //   vals    : 4 * nnz
    //   x read  : 4 * nnz     -- worst case: every col is distinct (random gather)
    //   y write : 4 * n
    double bytes_per_iter = 4.0 * ((n + 1) + 2.0 * nnz + nnz + n);
    m.gbps = bytes_per_iter * iter / (total_spmv_ms * 1e-3) / 1e9;

    // ---- D2H ----
    h_scores.resize(n);
    CudaTimer d2h_timer;
    d2h_timer.begin();
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_v, n * sizeof(float), cudaMemcpyDeviceToHost));
    m.d2h_ms = d2h_timer.end();

    // ---- Cleanup ----
    CUBLAS_CHECK(cublasDestroy(bl));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_v_new));
    CUDA_CHECK(cudaFree(d_diff));
}

// ---------------------------------------------------------------------------
// Degree statistics (host-side, for the summary)
// ---------------------------------------------------------------------------
static void degree_stats(const CsrGraph &g,
                         double &avg, int &mx, int &mn)
{
    mx = 0; mn = INT_MAX;
    long long sum = 0;
    for (int i = 0; i < g.n; ++i) {
        int d = g.row_ptr[i + 1] - g.row_ptr[i];
        sum += d;
        if (d > mx) mx = d;
        if (d < mn) mn = d;
    }
    avg = (double)sum / g.n;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <graph.bin> [max_iter=1000] [tol=1e-6] [top_k=20] [block_size=256]\n"
            "\n"
            "  graph.bin   -- binary CSR from snap_to_undirected_csr.py\n"
            "  max_iter    -- max power iterations          (default 1000)\n"
            "  tol         -- convergence tolerance         (default 1e-6)\n"
            "  top_k       -- top nodes to print            (default 20)\n"
            "  block_size  -- CUDA threads per block        (default 256)\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *bin_path  = argv[1];
    int   max_iter        = argc > 2 ? atoi(argv[2])        : 1000;
    float tol             = argc > 3 ? (float)atof(argv[3]) : 1e-6f;
    int   top_k           = argc > 4 ? atoi(argv[4])        : 20;
    int   block_size      = argc > 5 ? atoi(argv[5])        : 256;

    OutputPaths out = build_output_paths(bin_path);
    if (!ensure_dir_recursive(out.output_dir)) {
        fprintf(stderr, "Failed to create output directory: %s\n", out.output_dir.c_str());
        return EXIT_FAILURE;
    }

    // ---- Device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3
                   * (prop.memoryBusWidth / 8) / 1e9;
    printf("Device     : %s  (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    printf("Peak BW    : %.1f GB/s\n", peak_bw);
    printf("Block size : %d threads  |  Grid : %d blocks\n\n",
           block_size, (0 + block_size - 1) / block_size);  // placeholder; real grid printed below

    // ---- Load CSR ----
    CsrGraph g;
    printf("Loading %s ...\n", bin_path);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!load_csr_binary(bin_path, g)) {
        fprintf(stderr, "Failed to load %s\n", bin_path);
        return EXIT_FAILURE;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double avg_deg; int max_deg, min_deg;
    degree_stats(g, avg_deg, max_deg, min_deg);

    int grid = (g.n + block_size - 1) / block_size;
    double csr_mb = 4.0 * ((g.n + 1) + g.nnz + g.nnz) / 1e6;

    printf("  Vertices   : %d\n",     g.n);
    printf("  Edges      : %d  (%d directed nnz)\n", g.nnz / 2, g.nnz);
    printf("  Avg degree : %.2f  |  Max : %d  |  Min : %d\n",
           avg_deg, max_deg, min_deg);
    printf("  CSR memory : %.2f MB\n", csr_mb);
    printf("  Grid       : %d blocks x %d threads\n", grid, block_size);
    printf("  Load time  : %.2f ms\n\n", load_ms);

    // ---- Run ----
    std::vector<float> scores;
    EvcMetrics m{};
    m.load_ms = load_ms;
    run_evcent_scalar(g, max_iter, tol, block_size, scores, m);
    m.bw_util_pct = peak_bw > 0 ? m.gbps / peak_bw * 100.0 : 0.0;

    // ---- Top-k ----
    top_k = std::min(top_k, g.n);
    std::vector<int> idx(g.n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<int> sorted_idx(g.n);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b){ return scores[a] > scores[b]; });

    if (!write_scores_csv(out.scores_csv, sorted_idx, scores)) {
        fprintf(stderr, "Failed to write scores CSV: %s\n", out.scores_csv.c_str());
        return EXIT_FAILURE;
    }

    double density = g.n > 0 ? (double)g.nnz / ((double)g.n * (double)g.n) : 0.0;
    if (!write_metrics_json(out.metrics_json, out, bin_path, g, max_iter, tol, top_k,
                            m, density, peak_bw, m.bw_util_pct, idx[0], scores[idx[0]])) {
        fprintf(stderr, "Failed to write metrics JSON: %s\n", out.metrics_json.c_str());
        return EXIT_FAILURE;
    }

    printf("=== Top-%d Nodes by Eigenvector Centrality (CSR-scalar) ===\n", top_k);
    printf("  %-6s  %-10s  %s\n", "Rank", "Node ID", "Score");
    printf("  %-6s  %-10s  %s\n", "----", "-------", "----------");
    for (int r = 0; r < top_k; ++r)
        printf("  %-6d  %-10d  %.8f\n", r + 1, idx[r], scores[idx[r]]);

    // ---- Metrics ----
    printf("\n=== Performance Metrics (CSR-scalar) ===\n");
    printf("  %-36s : %8.2f ms\n",       "Disk load",                    m.load_ms);
    printf("  %-36s : %8.2f ms\n",       "H2D transfer",                 m.h2d_ms);
    printf("  %-36s : %8.2f ms\n",       "D2H transfer",                 m.d2h_ms);
    printf("  %-36s : %8.2f ms\n",       "Total GPU time (all kernels)", m.total_gpu_ms);
    printf("  %-36s : %8.2f ms\n",       "SpMV kernel total",            m.total_spmv_ms);
    printf("  %-36s : %8.2f ms\n",       "Normalize kernel total",       m.total_norm_ms);
    printf("  %-36s : %8d\n",             "Iterations",                   m.iters);
    printf("  %-36s : %8.4f ms\n",       "Avg time / iteration",         m.avg_iter_ms);
    printf("  %-36s : %8.3f GFLOP/s\n",  "SpMV GFLOP/s",                 m.gflops);
    printf("  %-36s : %8.3f GB/s\n",     "SpMV effective bandwidth",     m.gbps);
    printf("  %-36s : %7.1f %%\n",       "BW utilisation",               m.bw_util_pct);
    printf("  %-36s : %.3e\n",            "Final residual",               m.final_residual);
    printf("  %-36s : %8.2f MB\n",       "CSR GPU memory",               csr_mb);
    printf("\nSaved: %s\n", out.scores_csv.c_str());
    printf("Saved: %s\n", out.metrics_json.c_str());

    return EXIT_SUCCESS;
}
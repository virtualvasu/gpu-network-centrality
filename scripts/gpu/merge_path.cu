// nvcc -O3 -arch=sm_86 eigen_centrality.cu -o eigen
// ./eigen amazon.bin

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(_e),   \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =============================================================================
// Robust Binary Loader — supports both 32-bit and 64-bit nnz formats
// (ported from our_code.cu for compatibility with all dataset sizes)
// =============================================================================
struct CsrGraphHost {
  int n = 0;
  int nnz = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<float> vals;
};

static bool load_csr_binary_compat(const char *path, CsrGraphHost &g) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s\n", path);
    return false;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return false;
  }
  long long file_size = ftell(f);
  if (file_size < 0) {
    fclose(f);
    return false;
  }
  rewind(f);

  int32_t n32 = 0;
  if (fread(&n32, sizeof(int32_t), 1, f) != 1) {
    fclose(f);
    return false;
  }

  unsigned char nnz_probe[8] = {0};
  if (fread(nnz_probe, 1, sizeof(nnz_probe), f) != sizeof(nnz_probe)) {
    fclose(f);
    return false;
  }

  int32_t nnz32 = 0;
  int64_t nnz64 = 0;
  memcpy(&nnz32, nnz_probe, sizeof(int32_t));
  memcpy(&nnz64, nnz_probe, sizeof(int64_t));

  auto size_matches_32 = [&](int32_t n, int32_t nnz) {
    if (n <= 0 || nnz < 0)
      return false;
    __int128 e =
        4 + 4 + (__int128)(n + 1) * 4 + (__int128)nnz * 4 + (__int128)nnz * 4;
    return e == file_size;
  };
  auto size_matches_64 = [&](int32_t n, int64_t nnz) {
    if (n <= 0 || nnz < 0)
      return false;
    __int128 e =
        4 + 8 + (__int128)(n + 1) * 8 + (__int128)nnz * 4 + (__int128)nnz * 4;
    return e == file_size;
  };

  const bool is_64 = size_matches_64(n32, nnz64);
  const bool is_32 = size_matches_32(n32, nnz32);
  if (!is_64 && !is_32) {
    fprintf(stderr, "Unrecognized CSR binary format: %s (size=%lld)\n", path,
            file_size);
    fclose(f);
    return false;
  }

  rewind(f);
  if (is_64) {
    int64_t nnz_read = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1 ||
        fread(&nnz_read, sizeof(int64_t), 1, f) != 1) {
      fclose(f);
      return false;
    }
    if (nnz_read < 0 || nnz_read > (int64_t)std::numeric_limits<int>::max()) {
      fprintf(stderr, "nnz out of int range: %lld\n", (long long)nnz_read);
      fclose(f);
      return false;
    }
    std::vector<int64_t> rp64((size_t)n32 + 1);
    g.row_ptr.resize((size_t)n32 + 1);
    g.col_ind.resize((size_t)nnz_read);
    g.vals.resize((size_t)nnz_read);
    if (fread(rp64.data(), sizeof(int64_t), (size_t)n32 + 1, f) !=
            (size_t)n32 + 1 ||
        fread(g.col_ind.data(), sizeof(int), (size_t)nnz_read, f) !=
            (size_t)nnz_read ||
        fread(g.vals.data(), sizeof(float), (size_t)nnz_read, f) !=
            (size_t)nnz_read) {
      fclose(f);
      return false;
    }
    for (size_t i = 0; i < g.row_ptr.size(); ++i)
      g.row_ptr[i] = (int)rp64[i];
    g.n = n32;
    g.nnz = (int)nnz_read;
  } else {
    int32_t nnz_read = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1 ||
        fread(&nnz_read, sizeof(int32_t), 1, f) != 1) {
      fclose(f);
      return false;
    }
    g.n = n32;
    g.nnz = nnz_read;
    g.row_ptr.resize((size_t)g.n + 1);
    g.col_ind.resize((size_t)g.nnz);
    g.vals.resize((size_t)g.nnz);
    if (fread(g.row_ptr.data(), sizeof(int), (size_t)g.n + 1, f) !=
            (size_t)g.n + 1 ||
        fread(g.col_ind.data(), sizeof(int), (size_t)g.nnz, f) !=
            (size_t)g.nnz ||
        fread(g.vals.data(), sizeof(float), (size_t)g.nnz, f) !=
            (size_t)g.nnz) {
      fclose(f);
      return false;
    }
  }
  fclose(f);
  return true;
}

// =============================================================================
// Output path helpers
// =============================================================================
struct OutputPaths {
  std::string dataset_key, output_dir, scores_csv, metrics_json;
};

static std::string _basename(const std::string &p) {
  size_t pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}
static bool _mkdir_p(const std::string &dir) {
  if (dir.empty())
    return true;
  if (mkdir(dir.c_str(), 0755) == 0)
    return true;
  return errno == EEXIST;
}
static bool ensure_dir(const std::string &dir) {
  std::string cur;
  for (char c : dir) {
    if (c == '/') {
      if (!cur.empty() && !_mkdir_p(cur))
        return false;
    }
    cur.push_back(c);
  }
  return _mkdir_p(cur);
}
static OutputPaths build_paths(const char *input_path) {
  std::string base = _basename(input_path);
  if (base.size() > 8 && base.substr(base.size() - 8) == ".csr.bin")
    base = base.substr(0, base.size() - 8);
  else {
    size_t dot = base.rfind('.');
    if (dot != std::string::npos)
      base = base.substr(0, dot);
  }
  OutputPaths p;
  p.dataset_key = base;
  p.output_dir = "baseline/eigen_centrality/" + base;
  p.scores_csv = p.output_dir + "/" + base + "_eigenvector_scores.csv";
  p.metrics_json = p.output_dir + "/step0_metrics.json";
  return p;
}

// =============================================================================
// KERNELS
// =============================================================================

// 1. Parallel Reduction — computes norm² of y and sum of diff[] in one pass.
//    Two shared memory arrays avoid a second kernel launch.
__global__ void parallel_reduce_metrics(int n, const float *__restrict__ y,
                                        const float *__restrict__ d_diff,
                                        float *__restrict__ d_norm_out,
                                        float *__restrict__ d_res_out) {
  extern __shared__ float sdata[];
  float *s_norm = sdata;
  float *s_res = &sdata[blockDim.x];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  float local_norm = 0.0f, local_res = 0.0f;

  // Grid-stride loop: handles n > total number of threads
  while (i < (unsigned int)n) {
    float val = y[i];
    local_norm += val * val;
    local_res += d_diff[i];
    i += blockDim.x * gridDim.x;
  }
  s_norm[tid] = local_norm;
  s_res[tid] = local_res;
  __syncthreads();

  // Tree reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_norm[tid] += s_norm[tid + s];
      s_res[tid] += s_res[tid + s];
    }
    __syncthreads();
  }

  // One atomicAdd per block (168 calls max) — low contention
  if (tid == 0) {
    atomicAdd(d_norm_out, s_norm[0]);
    atomicAdd(d_res_out, s_res[0]);
  }
}

// 2. Merge Path binary search — maps linear work index to (row, edge) coords
__device__ void compute_merge_path(int global_idx,
                                   const int *__restrict__ row_ptr,
                                   int num_rows, int nnz, int *x_coord,
                                   int *y_coord) {
  int low = max(0, global_idx - nnz);
  int high = min(global_idx, num_rows);
  while (low < high) {
    int mid = (low + high) >> 1;
    if (row_ptr[mid + 1] <= global_idx - mid - 1)
      low = mid + 1;
    else
      high = mid;
  }
  *x_coord = low;
  *y_coord = global_idx - low;
}

// 3. Merge Path SpMV — equal work per thread regardless of row degree.
//    fmaf() for fused multiply-add (single rounding, faster than a*b+c).
//    __ldg() routes x[] reads through read-only (texture) cache.
__global__ void hybrid_spmv_merge_path_kernel_v2(
    int n, int nnz, const int *__restrict__ row_ptr,
    const int *__restrict__ col_ind, const float *__restrict__ vals,
    const float *__restrict__ x, float *__restrict__ y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_work = n + nnz;
  int items_per_thread =
      (total_work + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x);

  int t_start = tid * items_per_thread;
  int t_end = min(t_start + items_per_thread, total_work);
  if (t_start >= total_work)
    return;

  int cur_row, cur_edge;
  compute_merge_path(t_start, row_ptr, n, nnz, &cur_row, &cur_edge);

  float thread_sum = 0.0f;

#pragma unroll 4
  for (int i = t_start; i < t_end; ++i) {
    if (cur_row < n && cur_edge >= row_ptr[cur_row + 1]) {
      if (thread_sum != 0.0f) {
        atomicAdd(&y[cur_row], thread_sum);
        thread_sum = 0.0f;
      }
      cur_row++;
    } else {
      // __ldg: read-only cache for input vector x
      thread_sum += vals[cur_edge] * __ldg(&x[col_ind[cur_edge]]);
      cur_edge++;
    }
  }
  if (thread_sum != 0.0f && cur_row < n)
    atomicAdd(&y[cur_row], thread_sum);
}

// 4. Normalize & compute per-element squared diff in one pass.
__global__ void normalize_residual_kernel(int n, float *__restrict__ x,
                                          const float *__restrict__ y,
                                          float norm,
                                          float *__restrict__ diff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float next_val = y[idx] / norm;
    float d = next_val - x[idx];
    diff[idx] = d * d;
    x[idx] = next_val;
  }
}

// =============================================================================
// Main solver
// =============================================================================
void run_optimized_evcent(const char *path, int max_iter, float tol,
                          int top_k) {
  OutputPaths out = build_paths(path);
  if (!ensure_dir(out.output_dir)) {
    fprintf(stderr, "Failed to create output dir: %s\n",
            out.output_dir.c_str());
    return;
  }

  CsrGraphHost graph;
  if (!load_csr_binary_compat(path, graph)) {
    fprintf(stderr, "Failed to parse CSR binary: %s\n", path);
    return;
  }

  const int n = graph.n;
  const int nnz = graph.nnz;

  // --- GPU Allocations ---
  int *d_row_ptr, *d_col_ind;
  float *d_vals, *d_x, *d_y, *d_diff, *d_norm_val, *d_res_val;
  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_diff, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_norm_val, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_res_val, sizeof(float)));

  // --- Timing events ---
  cudaEvent_t ev_h2d_0, ev_h2d_1, ev_d2h_0, ev_d2h_1;
  cudaEvent_t ev_spmv_0, ev_spmv_1, ev_norm_0, ev_norm_1;
  cudaEventCreate(&ev_h2d_0);
  cudaEventCreate(&ev_h2d_1);
  cudaEventCreate(&ev_d2h_0);
  cudaEventCreate(&ev_d2h_1);
  cudaEventCreate(&ev_spmv_0);
  cudaEventCreate(&ev_spmv_1);
  cudaEventCreate(&ev_norm_0);
  cudaEventCreate(&ev_norm_1);

  cudaEventRecord(ev_h2d_0);
  CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_ind, graph.col_ind.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, graph.vals.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Initial vector: uniform 1/sqrt(n) ---
  std::vector<float> h_x(n, 1.0f / sqrtf((float)n));
  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(ev_h2d_1);

  // --- Dynamic block counts (replaces hardcoded 168) ---
  const int num_threads = 256;
  const size_t shared_mem = 2 * num_threads * sizeof(float);
  // SpMV: enough blocks to cover n+nnz work items
  const int num_blocks_spmv =
      std::max(1, std::min(4096, (n + nnz + num_threads - 1) / num_threads));
  // Reduce: enough blocks to cover n elements (grid-stride handles the rest)
  const int num_blocks_reduce =
      std::max(1, std::min(4096, (n + num_threads - 1) / num_threads));
  // Normalize: exact coverage of n elements
  const int num_blocks_norm = (n + num_threads - 1) / num_threads;

  int iter = 0;
  float h_residual = 1.0f;
  float h_norm = 1.0f;

  auto gpu_start = std::chrono::high_resolution_clock::now();

  float acc_spmv_ms = 0.0f, acc_norm_ms = 0.0f;

  while (iter < max_iter && h_residual > tol) {

    // Step 1: SpMV  d_y = A * d_x
    CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));
    cudaEventRecord(ev_spmv_0);
    hybrid_spmv_merge_path_kernel_v2<<<num_blocks_spmv, num_threads>>>(
        n, nnz, d_row_ptr, d_col_ind, d_vals, d_x, d_y);
    cudaEventRecord(ev_spmv_1);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Compute ||d_y||₂
    CUDA_CHECK(cudaMemset(d_norm_val, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_res_val, 0, sizeof(float)));
    parallel_reduce_metrics<<<num_blocks_reduce, num_threads, shared_mem>>>(
        n, d_y, d_diff, d_norm_val, d_res_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(
        cudaMemcpy(&h_norm, d_norm_val, sizeof(float), cudaMemcpyDeviceToHost));
    h_norm = sqrtf(h_norm);
    {
      float t = 0;
      cudaEventElapsedTime(&t, ev_spmv_0, ev_spmv_1);
      acc_spmv_ms += t;
    }

    if (h_norm == 0.0f)
      break;

    // Step 3: Normalize  d_x <- d_y / ||d_y||,  write squared diff to d_diff
    cudaEventRecord(ev_norm_0);
    normalize_residual_kernel<<<num_blocks_norm, num_threads>>>(n, d_x, d_y,
                                                                h_norm, d_diff);
    cudaEventRecord(ev_norm_1);
    CUDA_CHECK(cudaGetLastError());

    // Step 4: Residual = ||x_new - x_old||₂ / n
    CUDA_CHECK(cudaMemset(d_norm_val, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_res_val, 0, sizeof(float)));
    parallel_reduce_metrics<<<num_blocks_reduce, num_threads, shared_mem>>>(
        n, d_y, d_diff, d_norm_val, d_res_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&h_residual, d_res_val, sizeof(float),
                          cudaMemcpyDeviceToHost));
    h_residual = sqrtf(h_residual);
    {
      float t = 0;
      cudaEventElapsedTime(&t, ev_norm_0, ev_norm_1);
      acc_norm_ms += t;
    }

    iter++;
  }

  auto gpu_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

  // --- D2H transfer ---
  cudaEventRecord(ev_d2h_0);
  CUDA_CHECK(
      cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
  cudaEventRecord(ev_d2h_1);
  CUDA_CHECK(cudaEventSynchronize(ev_d2h_1));

  float h2d_ms = 0, d2h_ms = 0;
  cudaEventElapsedTime(&h2d_ms, ev_h2d_0, ev_h2d_1);
  cudaEventElapsedTime(&d2h_ms, ev_d2h_0, ev_d2h_1);

  // --- Cleanup GPU ---
  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_ind));
  CUDA_CHECK(cudaFree(d_vals));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_diff));
  CUDA_CHECK(cudaFree(d_norm_val));
  CUDA_CHECK(cudaFree(d_res_val));
  cudaEventDestroy(ev_h2d_0);
  cudaEventDestroy(ev_h2d_1);
  cudaEventDestroy(ev_d2h_0);
  cudaEventDestroy(ev_d2h_1);
  cudaEventDestroy(ev_spmv_0);
  cudaEventDestroy(ev_spmv_1);
  cudaEventDestroy(ev_norm_0);
  cudaEventDestroy(ev_norm_1);

  // --- Rank nodes ---
  std::vector<std::pair<float, int>> ranked(n);
  for (int i = 0; i < n; ++i)
    ranked[i] = {h_x[i], i};
  std::sort(ranked.rbegin(), ranked.rend());

  // --- Reconstruction error: ||Ax - lambda*x||_2 on CPU ---
  // lambda = Rayleigh quotient = x^T A x
  double lambda = 0.0;
  std::vector<double> Ax(n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; ++j)
      Ax[i] += (double)graph.vals[j] * (double)h_x[graph.col_ind[j]];
  for (int i = 0; i < n; ++i)
    lambda += (double)h_x[i] * Ax[i];
  double recon_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = Ax[i] - lambda * (double)h_x[i];
    recon_err += d * d;
  }
  recon_err = sqrt(recon_err);

  // --- Derived metrics ---
  double total_sec = total_ms / 1000.0;
  double density = (n > 0) ? ((double)nnz / ((double)n * (double)n)) : 0.0;
  double mteps = (iter > 0) ? ((double)nnz * iter / (total_sec * 1e6)) : 0.0;
  // Effective GFLOPS: 2*nnz (SpMV) + 2*n (normalize) + 2*n (reduce) per iter
  double gflops =
      (iter > 0) ? ((2.0 * nnz + 4.0 * n) * iter / (total_sec * 1e9)) : 0.0;
  // Effective BW: (nnz+n)*3*4 bytes per SpMV iter (vals, col_ind, x/y streams)
  double eff_bw_gbps =
      (iter > 0 && acc_spmv_ms > 0)
          ? ((double)(nnz + n) * 3 * 4 * iter / (acc_spmv_ms / 1000.0) / 1e9)
          : 0.0;
  double peak_bw_gbps = 360.04; // RTX 3060 spec
  double bw_util_pct =
      (peak_bw_gbps > 0) ? (eff_bw_gbps / peak_bw_gbps * 100.0) : 0.0;
  size_t mem_bytes = (size_t)(n + 1) * sizeof(int) + (size_t)nnz * sizeof(int) +
                     (size_t)nnz * sizeof(float) +
                     (size_t)n * sizeof(float) * 3 + 2 * sizeof(float);
  double avg_iter_ms = (iter > 0) ? (total_ms / iter) : 0.0;

  // --- Console output ---
  printf("Device  : NVIDIA GeForce RTX 3060 (Optimized v2)\n");
  printf("  Vertices: %d | Edges: %d\n", n, nnz / 2);
  printf("=== Top-%d Nodes ===\n", top_k);
  for (int r = 0; r < top_k && r < n; ++r)
    printf("  %d. Node %d: %.8f\n", r + 1, ranked[r].second, ranked[r].first);
  printf("\n=== Performance Metrics ===\n");
  printf("  Total GPU time  : %.2f ms\n", total_ms);
  printf("  SpMV total      : %.2f ms\n", acc_spmv_ms);
  printf("  Normalize total : %.2f ms\n", acc_norm_ms);
  printf("  Avg/iteration   : %.4f ms\n", avg_iter_ms);
  printf("  Iterations      : %d\n", iter);
  printf("  Final residual  : %.3e\n", h_residual);
  printf("  Recon error L2  : %.6e\n", recon_err);
  printf("  MTEPS           : %.2f\n", mteps);
  printf("  Eff. GFLOPS     : %.2f\n", gflops);
  printf("  Eff. BW (GB/s)  : %.2f\n", eff_bw_gbps);

  // --- CSV scores ---
  FILE *csv = fopen(out.scores_csv.c_str(), "w");
  if (csv) {
    fprintf(csv, "node_id,score\n");
    for (int i = 0; i < n; ++i)
      fprintf(csv, "%d,%.8f\n", ranked[i].second, ranked[i].first);
    fclose(csv);
    printf("Saved: %s\n", out.scores_csv.c_str());
  }

  // --- JSON metrics ---
  FILE *jf = fopen(out.metrics_json.c_str(), "w");
  if (jf) {
    fprintf(jf, "{\n");
    fprintf(jf, "  \"dataset_key\": \"%s\",\n", out.dataset_key.c_str());
    fprintf(jf, "  \"dataset\": \"%s\",\n", path);
    fprintf(jf, "  \"num_nodes\": %d,\n", n);
    fprintf(jf, "  \"num_edges\": %d,\n", nnz / 2);
    fprintf(jf, "  \"nnz\": %d,\n", nnz);
    fprintf(jf, "  \"density\": %.12g,\n", density);
    fprintf(jf, "  \"method\": \"merge_path.power_iteration\",\n");
    fprintf(jf, "  \"graph_type\": \"undirected\",\n");
    fprintf(jf, "  \"max_iter\": %d,\n", max_iter);
    fprintf(jf, "  \"tol\": %.12g,\n", (double)tol);
    fprintf(jf, "  \"top_k\": %d,\n", top_k);
    fprintf(jf, "  \"runtime_seconds\": %.12g,\n", total_sec);
    fprintf(jf, "  \"execution_time_seconds\": %.12g,\n", total_sec);
    fprintf(jf, "  \"execution_time_ms\": %.12g,\n", total_ms);
    fprintf(jf, "  \"mteps\": %.12g,\n", mteps);
    fprintf(jf, "  \"iterations\": %d,\n", iter);
    fprintf(jf, "  \"converged\": %s,\n",
            (h_residual <= tol) ? "true" : "false");
    fprintf(jf, "  \"final_residual\": %.12g,\n", (double)h_residual);
    fprintf(jf, "  \"reconstruction_error_l2\": %.12g,\n", recon_err);
    fprintf(jf, "  \"h2d_ms\": %.12g,\n", (double)h2d_ms);
    fprintf(jf, "  \"d2h_ms\": %.12g,\n", (double)d2h_ms);
    fprintf(jf, "  \"spmv_ms\": %.12g,\n", (double)acc_spmv_ms);
    fprintf(jf, "  \"normalize_ms\": %.12g,\n", (double)acc_norm_ms);
    fprintf(jf, "  \"avg_iter_ms\": %.12g,\n", avg_iter_ms);
    fprintf(jf, "  \"effective_gflops\": %.12g,\n", gflops);
    fprintf(jf, "  \"effective_bandwidth_gbps\": %.12g,\n", eff_bw_gbps);
    fprintf(jf, "  \"peak_bandwidth_gbps\": %.12g,\n", peak_bw_gbps);
    fprintf(jf, "  \"bw_util_percent\": %.12g,\n", bw_util_pct);
    fprintf(jf, "  \"memory_footprint_bytes\": %zu,\n", mem_bytes);
    fprintf(jf, "  \"memory_footprint_gb\": %.12g,\n", (double)mem_bytes / 1e9);
    fprintf(jf, "  \"global_memory_load_transactions\": null,\n");
    fprintf(jf, "  \"l2_cache_hit_rate_percent\": null,\n");
    fprintf(jf, "  \"unified_cache_hit_rate_percent\": null,\n");
    fprintf(jf, "  \"top_node_id\": %d,\n", ranked[0].second);
    fprintf(jf, "  \"top_score\": %.12g,\n", (double)ranked[0].first);
    fprintf(jf, "  \"vector_orthogonality_deg_avg\": null,\n");
    fprintf(jf, "  \"precision_mode\": \"float32\",\n");
    fprintf(jf, "  \"precision_tradeoff_note\": \"single_precision_only\"\n");
    fprintf(jf, "}\n");
    fclose(jf);
    printf("Saved: %s\n", out.metrics_json.c_str());
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <graph.bin>\n", argv[0]);
    return 1;
  }
  run_optimized_evcent(argv[1], 1000, 1e-6f, 20);
  return 0;
}
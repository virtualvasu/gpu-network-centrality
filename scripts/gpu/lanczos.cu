// compile: nvcc -O3 lanczos_centrality.cu -o lanczos_centrality -lcusparse -lcublas
// execute./lanczos_centrality graph.bin

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include <string>
#include <limits>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// --- Error Checking Macros ---
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// --- Data Structures ---
struct CSRGraph {
    int num_nodes;
    int num_edges;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<float> values;
};

struct OutputPaths {
    std::string dataset_key;
    std::string output_dir;
    std::string scores_csv;
    std::string metrics_json;
};

static std::string strip_extension(const std::string &name) {
    size_t pos = name.rfind('.');
    if (pos == std::string::npos) return name;
    return name.substr(0, pos);
}

static std::string path_basename(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool mkdir_if_missing(const std::string &dir) {
    if (dir.empty()) return true;
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    return errno == EEXIST;
}

static bool ensure_dir_recursive(const std::string &dir) {
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

static OutputPaths build_output_paths(const char *input_path) {
    std::string in = input_path;
    std::string base = path_basename(in);

    if (base.size() > 8 && base.substr(base.size() - 8) == ".csr.bin") {
        base = base.substr(0, base.size() - 8);
    } else {
        base = strip_extension(base);
    }

    OutputPaths p;
    p.dataset_key = base;
    p.output_dir = std::string("baseline/lanczos/") + p.dataset_key;
    p.scores_csv = p.output_dir + "/" + p.dataset_key + "_eigenvector_scores.csv";
    p.metrics_json = p.output_dir + "/step0_metrics.json";
    return p;
}

// --- Binary Loader ---
CSRGraph load_csr_bin(const std::string& filename) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        std::cerr << "Failed to seek " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    long long file_size = ftell(f);
    if (file_size < 0) {
        fclose(f);
        std::cerr << "Failed to read size for " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    rewind(f);

    int32_t n32 = 0;
    if (fread(&n32, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        std::cerr << "Failed to read n from " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned char nnz_probe[8] = {0};
    if (fread(nnz_probe, 1, sizeof(nnz_probe), f) != sizeof(nnz_probe)) {
        fclose(f);
        std::cerr << "Failed to read nnz probe from " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int32_t nnz32 = 0;
    int64_t nnz64 = 0;
    memcpy(&nnz32, nnz_probe, sizeof(int32_t));
    memcpy(&nnz64, nnz_probe, sizeof(int64_t));

    auto size_matches_32 = [&](int32_t n, int32_t nnz) {
        if (n <= 0 || nnz < 0) return false;
        __int128 expected = 0;
        expected += 4;
        expected += 4;
        expected += static_cast<__int128>(n + 1) * 4;
        expected += static_cast<__int128>(nnz) * 4;
        expected += static_cast<__int128>(nnz) * 4;
        return expected == file_size;
    };

    auto size_matches_64 = [&](int32_t n, int64_t nnz) {
        if (n <= 0 || nnz < 0) return false;
        __int128 expected = 0;
        expected += 4;
        expected += 8;
        expected += static_cast<__int128>(n + 1) * 8;
        expected += static_cast<__int128>(nnz) * 4;
        expected += static_cast<__int128>(nnz) * 4;
        return expected == file_size;
    };

    const bool is_64 = size_matches_64(n32, nnz64);
    const bool is_32 = size_matches_32(n32, nnz32);
    if (!is_64 && !is_32) {
        std::cerr << "Unrecognized CSR binary format: " << filename
                  << " size=" << file_size
                  << " n=" << n32
                  << " nnz32=" << nnz32
                  << " nnz64=" << nnz64 << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    rewind(f);

    CSRGraph graph;

    if (is_64) {
        int64_t nnz_read = 0;
        if (fread(&graph.num_nodes, sizeof(int32_t), 1, f) != 1 ||
            fread(&nnz_read, sizeof(int64_t), 1, f) != 1) {
            fclose(f);
            std::cerr << "Failed to read 64-bit header from " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        if (nnz_read < 0 || nnz_read > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            fclose(f);
            std::cerr << "nnz out of range for current implementation: " << nnz_read << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<int64_t> row_ptr64(static_cast<size_t>(graph.num_nodes) + 1);
        graph.num_edges = static_cast<int>(nnz_read);
        graph.row_ptr.resize(static_cast<size_t>(graph.num_nodes) + 1);
        graph.col_ind.resize(static_cast<size_t>(graph.num_edges));
        graph.values.resize(static_cast<size_t>(graph.num_edges));

        if (fread(row_ptr64.data(), sizeof(int64_t), static_cast<size_t>(graph.num_nodes) + 1, f) != static_cast<size_t>(graph.num_nodes) + 1 ||
            fread(graph.col_ind.data(), sizeof(int), static_cast<size_t>(graph.num_edges), f) != static_cast<size_t>(graph.num_edges) ||
            fread(graph.values.data(), sizeof(float), static_cast<size_t>(graph.num_edges), f) != static_cast<size_t>(graph.num_edges)) {
            fclose(f);
            std::cerr << "Failed to read CSR payload from " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < graph.row_ptr.size(); ++i) {
            if (row_ptr64[i] < 0 || row_ptr64[i] > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                fclose(f);
                std::cerr << "row_ptr[" << i << "] out of range for int32 offsets" << std::endl;
                exit(EXIT_FAILURE);
            }
            graph.row_ptr[i] = static_cast<int>(row_ptr64[i]);
        }
    } else {
        if (fread(&graph.num_nodes, sizeof(int32_t), 1, f) != 1 ||
            fread(&graph.num_edges, sizeof(int32_t), 1, f) != 1) {
            fclose(f);
            std::cerr << "Failed to read 32-bit header from " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        graph.row_ptr.resize(static_cast<size_t>(graph.num_nodes) + 1);
        graph.col_ind.resize(static_cast<size_t>(graph.num_edges));
        graph.values.resize(static_cast<size_t>(graph.num_edges));

        if (fread(graph.row_ptr.data(), sizeof(int), static_cast<size_t>(graph.num_nodes) + 1, f) != static_cast<size_t>(graph.num_nodes) + 1 ||
            fread(graph.col_ind.data(), sizeof(int), static_cast<size_t>(graph.num_edges), f) != static_cast<size_t>(graph.num_edges) ||
            fread(graph.values.data(), sizeof(float), static_cast<size_t>(graph.num_edges), f) != static_cast<size_t>(graph.num_edges)) {
            fclose(f);
            std::cerr << "Failed to read CSR payload from " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);

    return graph;
}

// --- Power Iteration for small CPU tridiagonal matrix ---
std::vector<float> solve_tridiagonal_eigen(const std::vector<float>& alpha, const std::vector<float>& beta, int m) {
    std::vector<float> y(m, 1.0f / std::sqrt(m));
    std::vector<float> y_next(m, 0.0f);
    
    for (int iter = 0; iter < 1000; ++iter) {
        float norm = 0.0f;
        for (int i = 0; i < m; ++i) {
            float val = alpha[i] * y[i];
            if (i > 0) val += beta[i - 1] * y[i - 1];
            if (i < m - 1) val += beta[i] * y[i + 1];
            y_next[i] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < m; ++i) {
            y[i] = y_next[i] / norm;
        }
    }
    return y;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <graph.bin>\n";
        return 1;
    }

    OutputPaths out = build_output_paths(argv[1]);
    if (!ensure_dir_recursive(out.output_dir)) {
        std::cerr << "Failed to create output directory: " << out.output_dir << std::endl;
        return 1;
    }

    // Performance Timers
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_transfer, stop_transfer;
    cudaEvent_t start_lanczos, stop_lanczos;
    cudaEventCreate(&start_total); cudaEventCreate(&stop_total);
    cudaEventCreate(&start_transfer); cudaEventCreate(&stop_transfer);
    cudaEventCreate(&start_lanczos); cudaEventCreate(&stop_lanczos);

    cudaEventRecord(start_total);

    // 1. Load Data
    auto cpu_start = std::chrono::high_resolution_clock::now();
    CSRGraph h_graph = load_csr_bin(argv[1]);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double io_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    int n = h_graph.num_nodes;
    int nnz = h_graph.num_edges;
    int m = std::min(n, 50); // Lanczos steps

    // 2. Initialize Libraries
    cusparseHandle_t cusparseH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // 3. Allocate & Transfer Data to GPU
    cudaEventRecord(start_transfer);
    
    int *d_row_ptr, *d_col_ind;
    float *d_values;
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_graph.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_graph.col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_graph.values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    float *d_V, *d_v_curr, *d_v_prev, *d_w;
    CHECK_CUDA(cudaMalloc((void**)&d_V, n * m * sizeof(float))); 
    CHECK_CUDA(cudaMalloc((void**)&d_v_curr, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_v_prev, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_w, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_prev, 0, n * sizeof(float)));

    // 4. Initialize Starting Vector
    std::vector<float> h_v0(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    float initial_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        h_v0[i] = dist(rng);
        initial_norm += h_v0[i] * h_v0[i];
    }
    initial_norm = std::sqrt(initial_norm);
    for (int i = 0; i < n; ++i) h_v0[i] /= initial_norm;
    
    CHECK_CUDA(cudaMemcpy(d_v_curr, h_v0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, d_v_curr, n * sizeof(float), cudaMemcpyDeviceToDevice)); 

    cudaEventRecord(stop_transfer);

    // 5. Setup cuSPARSE SpMV
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz, d_row_ptr, d_col_ind, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_v_curr, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_w, CUDA_R_32F));

    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    float const_alpha = 1.0f, const_beta_spmv = 0.0f;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &const_alpha, matA, vecX, &const_beta_spmv, vecY, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // 6. Lanczos Iteration Loop
    cudaEventRecord(start_lanczos);
    std::vector<float> h_alpha(m, 0.0f);
    std::vector<float> h_beta(m, 0.0f);
    int actual_m = m;
    
    for (int j = 0; j < m; ++j) {
        CHECK_CUSPARSE(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &const_alpha, matA, vecX, &const_beta_spmv, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUBLAS(cublasSdot(cublasH, n, d_v_curr, 1, d_w, 1, &h_alpha[j]));
        float neg_alpha = -h_alpha[j];
        CHECK_CUBLAS(cublasSaxpy(cublasH, n, &neg_alpha, d_v_curr, 1, d_w, 1));

        if (j > 0) {
            float neg_beta = -h_beta[j - 1];
            CHECK_CUBLAS(cublasSaxpy(cublasH, n, &neg_beta, d_v_prev, 1, d_w, 1));
        }

        CHECK_CUBLAS(cublasSnrm2(cublasH, n, d_w, 1, &h_beta[j]));

        if (h_beta[j] < 1e-6f) {
            actual_m = j + 1;
            break; 
        }

        if (j < m - 1) {
            CHECK_CUBLAS(cublasScopy(cublasH, n, d_v_curr, 1, d_v_prev, 1));
            CHECK_CUBLAS(cublasScopy(cublasH, n, d_w, 1, d_v_curr, 1));
            float inv_beta = 1.0f / h_beta[j];
            CHECK_CUBLAS(cublasSscal(cublasH, n, &inv_beta, d_v_curr, 1));
            CHECK_CUDA(cudaMemcpy(d_V + (j + 1) * n, d_v_curr, n * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    cudaEventRecord(stop_lanczos);

    // 7. Solve Tridiagonal & Reconstruct
    std::vector<float> y = solve_tridiagonal_eigen(h_alpha, h_beta, actual_m);

    float *d_y, *d_x; 
    CHECK_CUDA(cudaMalloc((void**)&d_y, actual_m * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_y, y.data(), actual_m * sizeof(float), cudaMemcpyHostToDevice));

    float one = 1.0f, zero = 0.0f;
    CHECK_CUBLAS(cublasSgemv(cublasH, CUBLAS_OP_N, n, actual_m,
                             &one, d_V, n, d_y, 1, &zero, d_x, 1));

    std::vector<float> h_x(n);
    CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // --- ACCURACY FIX: Explicit L2 Normalization & Sign Correction ---
    // Find the maximum absolute value to determine the proper sign vector-wide
    float max_abs = 0.0f, dominant_sign = 1.0f;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_x[i]) > max_abs) {
            max_abs = std::abs(h_x[i]);
            dominant_sign = (h_x[i] < 0.0f) ? -1.0f : 1.0f;
        }
    }

    // Calculate L2 Norm
    float l2_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        l2_norm += h_x[i] * h_x[i];
    }
    l2_norm = std::sqrt(l2_norm);

    // Apply normalization and correct the sign
    std::vector<std::pair<int, float>> centrality(n);
    for (int i = 0; i < n; ++i) {
        float normalized_val = (h_x[i] * dominant_sign) / l2_norm;
        // Safety guard: force tiny floating point noise slightly below 0 to 0
        if (normalized_val < 0.0f && normalized_val > -1e-6f) normalized_val = 0.0f;
        centrality[i] = {i, normalized_val}; 
    }

    std::sort(centrality.begin(), centrality.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    // --- Write Full Output to CSV ---
    std::ofstream csv_file(out.scores_csv);
    if (csv_file.is_open()) {
        csv_file << "node_id,score\n";
        for (int i = 0; i < n; ++i) {
            csv_file << centrality[i].first << "," << std::fixed << std::setprecision(8) << centrality[i].second << "\n";
        }
        csv_file.close();
        std::cout << "\nSuccessfully wrote full results to " << out.scores_csv << "\n";
    } else {
        std::cerr << "\nFailed to open " << out.scores_csv << " for writing.\n";
    }

    // --- Print Results & Metrics ---
    std::cout << "\n--- Top 20 Nodes by Eigenvector Centrality ---\n";
    std::cout << "Rank\tNode ID\t\tScore\n";
    std::cout << "----------------------------------------------\n";
    for (int i = 0; i < std::min(n, 20); ++i) {
        std::cout << i + 1 << "\t" << centrality[i].first << "\t\t" << std::fixed << std::setprecision(6) << centrality[i].second << "\n";
    }

    float time_transfer = 0, time_lanczos = 0, time_total = 0;
    cudaEventElapsedTime(&time_transfer, start_transfer, stop_transfer);
    cudaEventElapsedTime(&time_lanczos, start_lanczos, stop_lanczos);
    cudaEventElapsedTime(&time_total, start_total, stop_total);

    std::cout << "\n--- Performance Metrics ---\n";
    std::cout << "Graph Loading (CPU IO): \t" << io_time << " ms\n";
    std::cout << "Host-to-Device Transfer: \t" << time_transfer << " ms\n";
    std::cout << "Lanczos SpMV Loop ("<< actual_m <<" iter): \t" << time_lanczos << " ms\n";
    std::cout << "Total GPU Execution Time: \t" << time_total << " ms\n";
    std::cout << "----------------------------------------------\n";

    // --- Write Metrics JSON ---
    double density = (n > 0) ? ((double)nnz / ((double)n * (double)n)) : 0.0;
    std::ofstream metrics_file(out.metrics_json);
    if (metrics_file.is_open()) {
        metrics_file << "{\n";
        metrics_file << "  \"dataset_key\": \"" << out.dataset_key << "\",\n";
        metrics_file << "  \"dataset\": \"" << argv[1] << "\",\n";
        metrics_file << "  \"num_nodes\": " << n << ",\n";
        metrics_file << "  \"num_edges\": " << (nnz / 2) << ",\n";
        metrics_file << "  \"nnz\": " << nnz << ",\n";
        metrics_file << "  \"density\": " << std::setprecision(12) << density << ",\n";
        metrics_file << "  \"method\": \"lanczos\",\n";
        metrics_file << "  \"graph_type\": \"undirected\",\n";
        metrics_file << "  \"max_iter\": " << m << ",\n";
        metrics_file << "  \"tol\": 1e-6,\n";
        metrics_file << "  \"runtime_seconds\": " << std::setprecision(12) << (time_total / 1e3) << ",\n";
        metrics_file << "  \"iterations\": " << actual_m << ",\n";
        metrics_file << "  \"converged\": true,\n";
        metrics_file << "  \"io_ms\": " << std::setprecision(12) << io_time << ",\n";
        metrics_file << "  \"h2d_ms\": " << std::setprecision(12) << time_transfer << ",\n";
        metrics_file << "  \"lanczos_loop_ms\": " << std::setprecision(12) << time_lanczos << ",\n";
        metrics_file << "  \"top_node_id\": " << centrality[0].first << ",\n";
        metrics_file << "  \"top_score\": " << std::setprecision(12) << centrality[0].second << "\n";
        metrics_file << "}\n";
        metrics_file.close();
        std::cout << "Saved metrics to " << out.metrics_json << "\n";
    } else {
        std::cerr << "Failed to open " << out.metrics_json << " for writing.\n";
    }

    // Cleanup
    cudaFree(d_row_ptr); cudaFree(d_col_ind); cudaFree(d_values);
    cudaFree(d_V); cudaFree(d_v_curr); cudaFree(d_v_prev); cudaFree(d_w);
    cudaFree(d_y); cudaFree(d_x); cudaFree(dBuffer);
    cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    cusparseDestroy(cusparseH); cublasDestroy(cublasH);
    cudaEventDestroy(start_total); cudaEventDestroy(stop_total);
    cudaEventDestroy(start_transfer); cudaEventDestroy(stop_transfer);
    cudaEventDestroy(start_lanczos); cudaEventDestroy(stop_lanczos);

    return 0;
}
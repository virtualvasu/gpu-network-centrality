// compile: nvcc -O3 lanczos_centrality.cu -o lanczos_centrality -lcusparse -lcublas
// execute./lanczos_centrality graph.bin

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
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

// --- Binary Loader ---
CSRGraph load_csr_bin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    CSRGraph graph;
    file.read(reinterpret_cast<char*>(&graph.num_nodes), sizeof(int));
    file.read(reinterpret_cast<char*>(&graph.num_edges), sizeof(int));

    graph.row_ptr.resize(graph.num_nodes + 1);
    graph.col_ind.resize(graph.num_edges);
    graph.values.resize(graph.num_edges);

    file.read(reinterpret_cast<char*>(graph.row_ptr.data()), (graph.num_nodes + 1) * sizeof(int));
    file.read(reinterpret_cast<char*>(graph.col_ind.data()), graph.num_edges * sizeof(int));
    file.read(reinterpret_cast<char*>(graph.values.data()), graph.num_edges * sizeof(float));

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
    std::string csv_filename = "eigenvector_centrality.csv";
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        csv_file << "NodeID,Score\n";
        for (int i = 0; i < n; ++i) {
            csv_file << centrality[i].first << "," << std::fixed << std::setprecision(8) << centrality[i].second << "\n";
        }
        csv_file.close();
        std::cout << "\nSuccessfully wrote full results to " << csv_filename << "\n";
    } else {
        std::cerr << "\nFailed to open " << csv_filename << " for writing.\n";
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
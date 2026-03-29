#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "loader.hpp"
#include "solver.cuh"

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << " GPU Eigenvector Centrality: Merge-Path SpMV\n";
    std::cout << "========================================================\n";

    // 1. Verify a CUDA GPU is present and print its properties
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA-capable GPU detected. ("
                  << cudaGetErrorString(err) << ")\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU     : " << prop.name << "\n";
    std::cout << "SM      : " << prop.major << "." << prop.minor << "\n";
    std::cout << "VRAM    : " << (prop.totalGlobalMem >> 20) << " MB\n";
    std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
    std::cout << "========================================================\n";

    // 2. Parse CLI argument
    if (argc < 2) {
        std::cerr << "Usage:   " << argv[0] << " <path_to_dataset.txt>\n";
        std::cerr << "Example: " << argv[0] << " datasets/amazon0302.txt\n";
        return 1;
    }
    std::string dataset_path = argv[1];

    // 3. Load and symmetrize the graph into CSR format (CPU phase)
    GraphCSR graph = load_snap_graph_to_csr(dataset_path);
    if (graph.num_nodes == 0 || graph.num_edges == 0) {
        std::cerr << "Error: Graph is empty or failed to load correctly.\n";
        return 1;
    }

    // 4. Run the GPU eigenvector centrality solver
    //    max_iterations = 1000, convergence tolerance = 1e-6
    execute_eigenvector_centrality(graph, 1000, 1e-6f);

    std::cout << "Execution Complete.\n";
    return 0;
}

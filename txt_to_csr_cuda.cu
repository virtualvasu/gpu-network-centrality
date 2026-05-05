#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

using clock_type = std::chrono::steady_clock;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(_err));                 \
        }                                                                       \
    } while (0)

struct OutputPaths {
    std::string csr_path;
    std::string csr_bin_path;
};

std::string ensure_suffix(std::string path, const std::string& suffix) {
    if (path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return path;
    }
    return path + suffix;
}

OutputPaths resolve_output_paths(const std::string& out_path) {
    if (out_path.size() >= 8 && out_path.compare(out_path.size() - 8, 8, ".csr.bin") == 0) {
        return {out_path.substr(0, out_path.size() - 4), out_path};
    }
    if (out_path.size() >= 4 && out_path.compare(out_path.size() - 4, 4, ".csr") == 0) {
        return {out_path, out_path + ".bin"};
    }
    if (out_path.size() >= 4 && out_path.compare(out_path.size() - 4, 4, ".bin") == 0) {
        std::string csr_path = out_path.substr(0, out_path.size() - 4);
        if (csr_path.size() < 4 || csr_path.compare(csr_path.size() - 4, 4, ".csr") != 0) {
            csr_path = ensure_suffix(std::move(csr_path), ".csr");
        }
        return {csr_path, out_path};
    }
    std::string csr_path = ensure_suffix(out_path, ".csr");
    return {csr_path, csr_path + ".bin"};
}

bool parse_edge_line(const std::string& line, int32_t& src, int32_t& dst) {
    const char* begin = line.data();
    const char* end = begin + line.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    if (begin == end || *begin == '#') {
        return false;
    }

    auto first = std::from_chars(begin, end, src);
    if (first.ec != std::errc()) {
        return false;
    }

    begin = first.ptr;
    while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }

    auto second = std::from_chars(begin, end, dst);
    if (second.ec != std::errc()) {
        return false;
    }

    return true;
}

template <typename Fn>
void stream_edges(const std::string& input_path, Fn&& fn) {
    std::ifstream input(input_path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::string line;
    line.reserve(64);
    while (std::getline(input, line)) {
        int32_t src = 0;
        int32_t dst = 0;
        if (parse_edge_line(line, src, dst)) {
            fn(src, dst);
        }
    }
}

void write_binary(const std::string& path,
                  int32_t n,
                  int32_t nnz,
                  const std::vector<int32_t>& row_ptr,
                  const std::vector<int32_t>& col_idx) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    output.write(reinterpret_cast<const char*>(&n), sizeof(n));
    output.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    output.write(reinterpret_cast<const char*>(row_ptr.data()), static_cast<std::streamsize>(row_ptr.size() * sizeof(int32_t)));
    output.write(reinterpret_cast<const char*>(col_idx.data()), static_cast<std::streamsize>(col_idx.size() * sizeof(int32_t)));

    constexpr std::size_t kChunk = 1u << 20;
    std::vector<float> ones(kChunk, 1.0f);
    int32_t remaining = nnz;
    while (remaining > 0) {
        std::size_t count = static_cast<std::size_t>(std::min<int32_t>(remaining, static_cast<int32_t>(kChunk)));
        output.write(reinterpret_cast<const char*>(ones.data()), static_cast<std::streamsize>(count * sizeof(float)));
        remaining -= static_cast<int32_t>(count);
    }
}

void sort_rows_on_gpu(std::vector<int32_t>& col_idx, const std::vector<int32_t>& row_ptr) {
    const int32_t nnz = static_cast<int32_t>(col_idx.size());
    if (nnz <= 1) {
        return;
    }

    int32_t* d_in = nullptr;
    int32_t* d_out = nullptr;
    int32_t* d_offsets = nullptr;
    int32_t* d_sizes = nullptr;
    void* d_temp = nullptr;
    std::size_t temp_bytes = 0;

    CUDA_CHECK(cudaMalloc(&d_in, static_cast<std::size_t>(nnz) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(nnz) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_offsets, static_cast<std::size_t>(row_ptr.size()) * sizeof(int32_t)));

    std::vector<int32_t> sizes(row_ptr.size() - 1);
    for (std::size_t i = 0; i + 1 < row_ptr.size(); ++i) {
        sizes[i] = row_ptr[i + 1] - row_ptr[i];
    }

    CUDA_CHECK(cudaMemcpy(d_in, col_idx.data(), static_cast<std::size_t>(nnz) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, row_ptr.data(), row_ptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    const int32_t num_segments = static_cast<int32_t>(sizes.size());
    CUDA_CHECK(cudaMalloc(&d_sizes, static_cast<std::size_t>(num_segments) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_sizes, sizes.data(), static_cast<std::size_t>(num_segments) * sizeof(int32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr,
        temp_bytes,
        d_in,
        d_out,
        nnz,
        num_segments,
        d_offsets,
        d_offsets + 1));

    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp,
        temp_bytes,
        d_in,
        d_out,
        nnz,
        num_segments,
        d_offsets,
        d_offsets + 1));

    CUDA_CHECK(cudaMemcpy(col_idx.data(), d_out, static_cast<std::size_t>(nnz) * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_temp);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    cudaFree(d_in);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input.txt> <output.csr>\n";
            return 1;
        }

        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const OutputPaths paths = resolve_output_paths(output_path);

        const auto t0 = clock_type::now();

        std::cout << "Pass 1: scan max node id...\n";
        int64_t max_node = -1;
        std::size_t raw_edges = 0;
        stream_edges(input_path, [&](int32_t src, int32_t dst) {
            if (src == dst) {
                return;
            }
            max_node = std::max<int64_t>(max_node, std::max<int32_t>(src, dst));
            ++raw_edges;
        });

        if (max_node < 0) {
            throw std::runtime_error("No valid edges found.");
        }
        if (max_node + 1 > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("Node ids exceed int32 range.");
        }

        const int32_t n = static_cast<int32_t>(max_node + 1);
        std::cout << "  n = " << n << ", raw edges = " << raw_edges << "\n";

        std::cout << "Pass 2: count degrees...\n";
        std::vector<int32_t> degree(static_cast<std::size_t>(n), 0);
        std::size_t kept_edges = 0;
        stream_edges(input_path, [&](int32_t src, int32_t dst) {
            if (src == dst) {
                return;
            }
            ++degree[static_cast<std::size_t>(src)];
            ++degree[static_cast<std::size_t>(dst)];
            ++kept_edges;
        });

        std::vector<int32_t> row_ptr(static_cast<std::size_t>(n) + 1, 0);
        for (int32_t i = 0; i < n; ++i) {
            row_ptr[static_cast<std::size_t>(i) + 1] = row_ptr[static_cast<std::size_t>(i)] + degree[static_cast<std::size_t>(i)];
        }

        const int32_t nnz = row_ptr.back();
        std::cout << "  undirected edges = " << kept_edges << ", nnz = " << nnz << "\n";

        std::cout << "Pass 3: fill CSR indices...\n";
        std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
        std::vector<int32_t> cursor = row_ptr;
        stream_edges(input_path, [&](int32_t src, int32_t dst) {
            if (src == dst) {
                return;
            }
            col_idx[static_cast<std::size_t>(cursor[static_cast<std::size_t>(src)]++)] = dst;
            col_idx[static_cast<std::size_t>(cursor[static_cast<std::size_t>(dst)]++)] = src;
        });

        std::cout << "GPU sort rows with CUB segmented radix sort...\n";
        const auto sort_start = clock_type::now();
        sort_rows_on_gpu(col_idx, row_ptr);
        const auto sort_end = clock_type::now();

        const auto t1 = clock_type::now();
        std::cout << "Writing " << paths.csr_path << " and " << paths.csr_bin_path << "...\n";
        write_binary(paths.csr_path, n, nnz, row_ptr, col_idx);
        write_binary(paths.csr_bin_path, n, nnz, row_ptr, col_idx);
        const auto t2 = clock_type::now();

        const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double sort_ms = std::chrono::duration<double, std::milli>(sort_end - sort_start).count();
        const double write_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

        std::cout << "Done.\n";
        std::cout << "  n          = " << n << "\n";
        std::cout << "  nnz        = " << nnz << "\n";
        std::cout << "  build time = " << build_ms << " ms\n";
        std::cout << "  gpu sort   = " << sort_ms << " ms\n";
        std::cout << "  write time = " << write_ms << " ms\n";
        std::cout << "  total time = " << total_ms << " ms\n";

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "  GPU        = " << prop.name << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
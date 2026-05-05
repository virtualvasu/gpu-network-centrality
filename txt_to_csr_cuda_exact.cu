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
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

namespace {

using clock_type = std::chrono::steady_clock;

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

inline uint64_t pack_edge(uint32_t src, uint32_t dst) {
    return (static_cast<uint64_t>(src) << 32) | static_cast<uint64_t>(dst);
}

inline uint32_t unpack_src(uint64_t edge) {
    return static_cast<uint32_t>(edge >> 32);
}

inline uint32_t unpack_dst(uint64_t edge) {
    return static_cast<uint32_t>(edge & 0xffffffffu);
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
                  int64_t nnz,
                  const std::vector<int64_t>& row_ptr,
                  const std::vector<int32_t>& col_idx) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    output.write(reinterpret_cast<const char*>(&n), sizeof(n));
    output.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    output.write(reinterpret_cast<const char*>(row_ptr.data()), static_cast<std::streamsize>(row_ptr.size() * sizeof(int64_t)));
    output.write(reinterpret_cast<const char*>(col_idx.data()), static_cast<std::streamsize>(col_idx.size() * sizeof(int32_t)));

    constexpr std::size_t kChunk = 1u << 20;
    std::vector<float> ones(kChunk, 1.0f);
    int64_t remaining = nnz;
    while (remaining > 0) {
        std::size_t count = static_cast<std::size_t>(std::min<int64_t>(remaining, static_cast<int64_t>(kChunk)));
        output.write(reinterpret_cast<const char*>(ones.data()), static_cast<std::streamsize>(count * sizeof(float)));
        remaining -= static_cast<int64_t>(count);
    }
}

std::vector<uint64_t> gpu_sort_unique_edges(std::vector<uint64_t>& edges) {
    if (edges.empty()) {
        return {};
    }

    thrust::device_vector<uint64_t> d_edges(edges.begin(), edges.end());
    thrust::sort(d_edges.begin(), d_edges.end());
    auto new_end = thrust::unique(d_edges.begin(), d_edges.end());
    std::size_t unique_count = static_cast<std::size_t>(new_end - d_edges.begin());

    std::vector<uint64_t> unique_edges(unique_count);
    thrust::copy(d_edges.begin(), new_end, unique_edges.begin());
    return unique_edges;
}

std::vector<uint64_t> gpu_sort_directed_edges(std::vector<uint64_t>& edges) {
    if (edges.empty()) {
        return {};
    }

    // Use CPU sorting instead of GPU to avoid GPU out-of-memory for very large graphs
    std::sort(edges.begin(), edges.end());
    return edges;
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

        std::cout << "Pass 1: scanning max node id...\n";
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

        const int32_t max_node_i32 = static_cast<int32_t>(max_node);
        const std::size_t node_capacity = static_cast<std::size_t>(max_node_i32) + 1;
        std::cout << "  max node id = " << max_node_i32 << ", raw edges = " << raw_edges << "\n";

        std::cout << "Pass 2: collecting canonical undirected edges...\n";
        std::vector<uint64_t> canonical_edges;
        canonical_edges.reserve(raw_edges);
        std::vector<uint8_t> present(node_capacity, 0);
        stream_edges(input_path, [&](int32_t src, int32_t dst) {
            if (src == dst) {
                return;
            }
            present[static_cast<std::size_t>(src)] = 1;
            present[static_cast<std::size_t>(dst)] = 1;
            uint32_t a = static_cast<uint32_t>(std::min(src, dst));
            uint32_t b = static_cast<uint32_t>(std::max(src, dst));
            canonical_edges.push_back(pack_edge(a, b));
        });

        const auto gpu_unique_start = clock_type::now();
        std::vector<uint64_t> unique_edges = gpu_sort_unique_edges(canonical_edges);
        canonical_edges.clear();
        canonical_edges.shrink_to_fit();
        const auto gpu_unique_end = clock_type::now();

        int32_t n = 0;
        std::vector<int32_t> remap(node_capacity, -1);
        for (int32_t node = 0; node <= max_node_i32; ++node) {
            if (present[static_cast<std::size_t>(node)]) {
                remap[static_cast<std::size_t>(node)] = n++;
            }
        }

        std::cout << "  unique undirected edges = " << unique_edges.size() << "\n";
        std::cout << "  compact vertices        = " << n << "\n";

        std::cout << "Pass 3: building compact CSR degree counts...\n";
        std::vector<int64_t> degree(static_cast<std::size_t>(n), 0);
        for (uint64_t edge : unique_edges) {
            int32_t u = remap[static_cast<std::size_t>(unpack_src(edge))];
            int32_t v = remap[static_cast<std::size_t>(unpack_dst(edge))];
            ++degree[static_cast<std::size_t>(u)];
            ++degree[static_cast<std::size_t>(v)];
        }

        std::vector<int64_t> row_ptr(static_cast<std::size_t>(n) + 1, 0);
        for (int32_t i = 0; i < n; ++i) {
            row_ptr[static_cast<std::size_t>(i) + 1] = row_ptr[static_cast<std::size_t>(i)] + degree[static_cast<std::size_t>(i)];
        }

        const int64_t nnz = row_ptr.back();
        std::cout << "  nnz (directed) = " << nnz << "\n";

        std::cout << "Pass 4: expanding to directed edges...\n";
        std::vector<uint64_t> directed_edges;
        directed_edges.resize(static_cast<std::size_t>(nnz));
        int64_t write_idx = 0;
        for (uint64_t edge : unique_edges) {
            int32_t u = remap[static_cast<std::size_t>(unpack_src(edge))];
            int32_t v = remap[static_cast<std::size_t>(unpack_dst(edge))];
            directed_edges[write_idx++] = pack_edge(static_cast<uint32_t>(u), static_cast<uint32_t>(v));
            directed_edges[write_idx++] = pack_edge(static_cast<uint32_t>(v), static_cast<uint32_t>(u));
        }
        unique_edges.clear();
        unique_edges.shrink_to_fit();

        std::cout << "Pass 5: GPU global sort of directed CSR entries...\n";
        const auto gpu_sort_start = clock_type::now();
        std::vector<uint64_t> sorted_directed = gpu_sort_directed_edges(directed_edges);
        directed_edges.clear();
        directed_edges.shrink_to_fit();
        const auto gpu_sort_end = clock_type::now();

        std::vector<int32_t> col_idx(static_cast<std::size_t>(nnz));
        for (int32_t row = 0; row < n; ++row) {
            int64_t begin = row_ptr[static_cast<std::size_t>(row)];
            int64_t end = row_ptr[static_cast<std::size_t>(row) + 1];
            for (int64_t i = begin; i < end; ++i) {
                col_idx[static_cast<std::size_t>(i)] = static_cast<int32_t>(unpack_dst(sorted_directed[static_cast<std::size_t>(i)]));
            }
        }
        sorted_directed.clear();
        sorted_directed.shrink_to_fit();

        const auto t1 = clock_type::now();
        std::cout << "Writing " << paths.csr_path << " and " << paths.csr_bin_path << "...\n";
        write_binary(paths.csr_path, n, nnz, row_ptr, col_idx);
        write_binary(paths.csr_bin_path, n, nnz, row_ptr, col_idx);
        const auto t2 = clock_type::now();

        const double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        const double parse_ms = std::chrono::duration<double, std::milli>(gpu_unique_start - t0).count();
        const double unique_ms = std::chrono::duration<double, std::milli>(gpu_unique_end - gpu_unique_start).count();
        const double sort_ms = std::chrono::duration<double, std::milli>(gpu_sort_end - gpu_sort_start).count();
        const double write_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::cout << "Done.\n";
        std::cout << "  n                = " << n << "\n";
        std::cout << "  nnz               = " << nnz << "\n";
        std::cout << "  GPU unique stage  = " << unique_ms << " ms\n";
        std::cout << "  GPU directed sort = " << sort_ms << " ms\n";
        std::cout << "  write time        = " << write_ms << " ms\n";
        std::cout << "  total time        = " << total_ms << " ms\n";
        std::cout << "  parse+CPU prep    = " << parse_ms << " ms\n";

        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        std::cout << "  GPU               = " << prop.name << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}

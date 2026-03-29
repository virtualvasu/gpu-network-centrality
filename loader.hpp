#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// Compressed Sparse Row graph representation
struct GraphCSR {
    int num_nodes = 0;
    int num_edges = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
};

// Helper for sorting and deduplicating edges
struct Edge {
    int u, v;
    bool operator<(const Edge& o) const {
        return u != o.u ? u < o.u : v < o.v;
    }
    bool operator==(const Edge& o) const {
        return u == o.u && v == o.v;
    }
};

// Loads a SNAP-format directed edge-list, symmetrizes it (adds reverse edges),
// deduplicates, and returns a CSR graph. Returns an empty GraphCSR on failure.
GraphCSR load_snap_graph_to_csr(const std::string& filepath) {
    std::cout << "Loading dataset from: " << filepath << " ...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filepath << "\n";
        return GraphCSR{};   // caller checks num_nodes == 0
    }

    std::vector<Edge> edges;
    std::string line;
    int max_node_id = -1;

    // --- Pass 1: parse edges and symmetrize ---
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;

        max_node_id = std::max({max_node_id, u, v});

        edges.push_back({u, v});          // original directed edge
        if (u != v)
            edges.push_back({v, u});       // reverse edge (symmetrization)
    }
    infile.close();

    if (max_node_id < 0) {
        std::cerr << "Error: No valid edges found in file.\n";
        return GraphCSR{};
    }

    int num_nodes = max_node_id + 1;

    // --- Pass 2: sort and deduplicate ---
    // Sorting by (u, v) groups edges per source node, required for CSR build.
    // Dedup removes any reverse edges that already existed in the raw dataset.
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    int num_edges = static_cast<int>(edges.size());

    // --- Pass 3: build CSR arrays ---
    GraphCSR graph;
    graph.num_nodes = num_nodes;
    graph.num_edges = num_edges;
    graph.row_ptr.assign(num_nodes + 1, 0);
    graph.col_ind.resize(num_edges);

    // Count degree of each source node
    for (int i = 0; i < num_edges; ++i) {
        graph.row_ptr[edges[i].u + 1]++;
        graph.col_ind[i] = edges[i].v;
    }

    // Prefix-sum to finalize row_ptr
    for (int i = 0; i < num_nodes; ++i)
        graph.row_ptr[i + 1] += graph.row_ptr[i];

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "----------------------------------------\n";
    std::cout << "Loaded in " << elapsed << " s\n";
    std::cout << "Nodes : " << graph.num_nodes << "\n";
    std::cout << "Edges : " << graph.num_edges << " (after symmetrization + dedup)\n";
    std::cout << "----------------------------------------\n";

    return graph;
}

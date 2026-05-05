#!/usr/bin/env python3
"""
Convert text CSR adjacency list format to binary CSR format.
Also checks if graph is directed or undirected.
Uses ThreadPoolExecutor for parallel parsing and sorting.
"""
import sys
import struct
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from multiprocessing import cpu_count

def parse_line(line_data):
    """
    Parse a single line in text CSR format.
    Returns list of (u, v) edges and max node id from this line.
    """
    line_idx, line = line_data
    line = line.strip()
    if not line:
        return [], -1
    
    parts = line.split()
    if len(parts) < 1:
        return [], -1
    
    try:
        node_id = int(parts[0])
        max_node = node_id
        edges_local = []
        
        for neighbor_str in parts[1:]:
            neighbor = int(neighbor_str)
            max_node = max(max_node, neighbor)
            edges_local.append((node_id, neighbor))
        
        return edges_local, max_node
    except (ValueError, IndexError):
        return [], -1

def load_text_csr(path, max_sample=10000, num_processes=None):
    """
    Load text CSR format where each line is: node_id neighbor1 neighbor2 ...
    Uses ProcessPoolExecutor to parallelize line parsing (true parallelism, bypasses GIL).
    Returns n, edges as set of (u,v) tuples, and a sample for directionality check.
    """
    if num_processes is None:
        num_processes = cpu_count() or 4
    
    edges = set()
    max_node = -1
    line_count = 0
    
    print(f"Loading {path} with {num_processes} processes...")
    
    # Read all lines first (needed for parallel processing)
    with open(path, "r") as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"  Read {total_lines} lines from disk")
    
    # Parse lines in parallel using processes (true parallelism)
    print(f"  Parsing lines in parallel using {num_processes} processes...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all parsing tasks
        futures = {
            executor.submit(parse_line, (i, line)): i 
            for i, line in enumerate(lines)
        }
        
        # Collect results as they complete
        for future_idx, future in enumerate(as_completed(futures)):
            try:
                edges_local, node_max = future.result()
                if edges_local:
                    edges.update(edges_local)
                    max_node = max(max_node, node_max)
                    line_count += 1
            except Exception as e:
                print(f"Error parsing line: {e}")
            
            if future_idx % 100000 == 0 and future_idx > 0:
                print(f"  Parsed {future_idx} lines, {len(edges)} edges, max_node={max_node}")
    
    n = max_node + 1
    print(f"Total: {line_count} nodes, {len(edges)} directed edges, n={n}")
    
    # Check directionality on a sample
    is_undirected = True
    num_asymmetric = 0
    sample_edges = list(edges)[:min(max_sample, len(edges))]
    for u, v in sample_edges:
        if u != v and (v, u) not in edges:
            is_undirected = False
            num_asymmetric += 1
            if num_asymmetric <= 5:
                print(f"  Asymmetric: ({u}, {v}) exists but ({v}, {u}) does not")
    
    if is_undirected:
        print(f"✓ Graph is UNDIRECTED (sample of {len(sample_edges)} edges all symmetric)")
    else:
        print(f"✗ Graph is DIRECTED (found {num_asymmetric} asymmetric edges in sample)")
    
    return n, edges, is_undirected

def sort_adjacency_list(adj_node):
    """
    Sort a single adjacency list. Used for parallel sorting.
    """
    node_idx, neighbors = adj_node
    return node_idx, sorted(neighbors)

def edges_to_binary_csr(n, edges, output_path, num_processes=None):
    """
    Convert edge set to binary CSR format and save.
    Format: 
      int32 n, int32 nnz
      int32[n+1] row_ptr 
      int32[nnz] col_idx
      float32[nnz] vals
    Uses ProcessPoolExecutor to parallelize adjacency list sorting (true parallelism).
    """
    if num_processes is None:
        num_processes = cpu_count() or 4
    
    print(f"\nBuilding CSR matrix using {num_processes} processes...")
    
    # Build adjacency lists
    print("  Building adjacency lists...")
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < n and v < n:
            adj[u].append(v)
    
    # Sort each adjacency list in parallel using processes (true parallelism)
    print(f"  Sorting {n} adjacency lists in parallel...")
    adj_sorted = [None] * n
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(sort_adjacency_list, (i, adj[i])): i 
            for i in range(n) if adj[i]
        }
        
        completed_count = 0
        for future in as_completed(futures):
            node_idx, sorted_neighbors = future.result()
            adj_sorted[node_idx] = sorted_neighbors
            completed_count += 1
            if completed_count % 100000 == 0:
                print(f"    Sorted {completed_count} lists...")
    
    # Fill in empty adjacency lists
    for i in range(n):
        if adj_sorted[i] is None:
            adj_sorted[i] = []
    
    # Build CSR arrays
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_idx_list = []
    
    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + len(adj_sorted[i])
        col_idx_list.extend(adj_sorted[i])
    
    col_idx = np.array(col_idx_list, dtype=np.int32)
    vals = np.ones(len(col_idx), dtype=np.float32)
    nnz = len(col_idx)
    
    print(f"  n={n}, nnz={nnz}, density={nnz / (n*n):.2e}")
    
    # Write binary
    print(f"Writing to {output_path}...")
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<ii", n, nnz))
        
        # Arrays
        row_ptr.tofile(f)
        col_idx.tofile(f)
        vals.tofile(f)
    
    print(f"✓ Saved: {output_path}")
    return n, nnz

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".csr", ".csr.bin")
    else:
        input_path = "dataset/big_graphs/uk-2002_output.csr"
        output_path = "dataset/big_graphs/uk-2002_output.csr.bin"
    
    n, edges, is_undirected = load_text_csr(input_path, max_sample=10000, num_processes=cpu_count())
    
    if not is_undirected:
        print("\n⚠ Graph is DIRECTED. Your code only works for undirected graphs.")
        print("Options:")
        print("  1. Make it undirected by adding reverse edges: (u,v) → (u,v) and (v,u)")
        print("  2. Use only as-is and accept directed result")
        response = input("\nAdd reverse edges to make undirected? (y/n) [default: y]: ").strip().lower()
        if response != "n":
            print("\nCreating undirected version by adding reverse edges...")
            edges_undirected = set(edges)
            for u, v in list(edges):
                if u != v:  # skip self-loops
                    edges_undirected.add((v, u))
            print(f"  Original: {len(edges)} edges → Undirected: {len(edges_undirected)} edges")
            edges = edges_undirected
    
    n, nnz = edges_to_binary_csr(n, edges, output_path, num_processes=cpu_count())
    print(f"\n✓ Complete: {n} nodes, {nnz} non-zeros")

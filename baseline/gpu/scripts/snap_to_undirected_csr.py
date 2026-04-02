"""
snap_to_undirected_csr.py

Converts a SNAP directed edge-list (.txt) to an undirected CSR binary file
that the CUDA eigenvector centrality code reads directly.

SNAP edge-list format:
  # comment lines start with #
  <src> <dst>
  <src> <dst>
  ...
  (node IDs are arbitrary non-negative integers)

Output binary layout (little-endian int32 / float32):
  [0]          n      (int32) number of vertices
  [1]          nnz    (int32) number of non-zeros in CSR (= 2 * undirected edges)
  [2..n+2]     row_ptr (int32, length n+1)
  [n+3..n+3+nnz-1]  col_idx (int32, length nnz)
  [n+3+nnz..n+3+2*nnz-1]  vals (float32, length nnz, all 1.0 for unweighted)

Usage:
  python snap_to_undirected_csr.py <input.txt> <output.bin>

Prints summary stats and performance info to stdout.
"""

import sys
import time
import struct
import array
from collections import defaultdict


def load_snap_edges(path: str):
    """Read SNAP edge list, skipping comment lines. Returns list of (src, dst) int pairs."""
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src, dst = int(parts[0]), int(parts[1])
            edges.append((src, dst))
    return edges


def build_undirected_csr(edges):
    """
    Given directed edges, build an undirected CSR by symmetrising:
      - add both (u,v) and (v,u)
      - deduplicate
      - re-map node IDs to [0, n)
      - sort neighbours per row (required by cuSPARSE)
    Returns (n, nnz, row_ptr, col_idx, node_map)
      node_map[original_id] = compact_id
    """
    # Collect unique nodes and symmetrise
    node_set = set()
    edge_set = set()
    for u, v in edges:
        if u == v:          # skip self-loops
            continue
        node_set.add(u)
        node_set.add(v)
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Compact node ID mapping (sorted for reproducibility)
    node_list = sorted(node_set)
    node_map  = {orig: new for new, orig in enumerate(node_list)}
    n         = len(node_list)

    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edge_set:
        adj[node_map[u]].append(node_map[v])

    # Sort each adjacency list (cuSPARSE requires sorted col indices)
    for key in adj:
        adj[key].sort()

    # Build CSR arrays
    row_ptr = array.array("i", [0] * (n + 1))
    col_idx_list = []
    for i in range(n):
        neighbours = adj.get(i, [])
        row_ptr[i + 1] = row_ptr[i] + len(neighbours)
        col_idx_list.extend(neighbours)

    col_idx = array.array("i", col_idx_list)
    vals    = array.array("f", [1.0] * len(col_idx_list))
    nnz     = len(col_idx_list)

    return n, nnz, row_ptr, col_idx, vals, node_map


def write_binary(path: str, n, nnz, row_ptr, col_idx, vals):
    """Write CSR to a compact binary file."""
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<ii", n, nnz))
        # Arrays
        row_ptr.tofile(f)
        col_idx.tofile(f)
        vals.tofile(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python snap_to_undirected_csr.py <input.txt> <output.bin>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    # --- Load ---
    t0 = time.perf_counter()
    print(f"Loading edge list: {in_path}")
    edges = load_snap_edges(in_path)
    t1 = time.perf_counter()
    print(f"  Read {len(edges):,} directed edges in {(t1-t0)*1e3:.1f} ms")

    # --- Build CSR ---
    print("Building undirected CSR...")
    n, nnz, row_ptr, col_idx, vals, node_map = build_undirected_csr(edges)
    t2 = time.perf_counter()
    print(f"  {n:,} vertices, {nnz // 2:,} undirected edges ({nnz:,} directed nnz) "
          f"in {(t2-t1)*1e3:.1f} ms")

    # --- Stats ---
    degrees = [row_ptr[i+1] - row_ptr[i] for i in range(n)]
    avg_deg = sum(degrees) / n if n else 0
    max_deg = max(degrees) if degrees else 0
    min_deg = min(degrees) if degrees else 0
    density = nnz / (n * n) if n else 0

    print(f"\n=== Graph Statistics ===")
    print(f"  Vertices          : {n:,}")
    print(f"  Undirected edges  : {nnz//2:,}")
    print(f"  CSR nnz           : {nnz:,}")
    print(f"  Avg degree        : {avg_deg:.2f}")
    print(f"  Max degree        : {max_deg}")
    print(f"  Min degree        : {min_deg}")
    print(f"  Density           : {density:.2e}")

    # Memory estimate for GPU
    mem_bytes = 4 * ((n + 1) + nnz + nnz)   # row_ptr + col_idx + vals
    print(f"  Est. GPU memory   : {mem_bytes / 1e6:.2f} MB (CSR arrays only)")

    # --- Write ---
    write_binary(out_path, n, nnz, row_ptr, col_idx, vals)
    t3 = time.perf_counter()
    import os
    file_size = os.path.getsize(out_path)
    print(f"\nWrote binary CSR  : {out_path}  ({file_size / 1e6:.2f} MB)  "
          f"in {(t3-t2)*1e3:.1f} ms")
    print(f"Total time        : {(t3-t0)*1e3:.1f} ms")

    # Binary layout reminder for the CUDA reader
    print(f"\nBinary layout (all little-endian):")
    print(f"  [0]          n       = {n}  (int32)")
    print(f"  [1]          nnz     = {nnz}  (int32)")
    print(f"  [2..{n+2}]     row_ptr  (int32, {n+1} values)")
    print(f"  [{n+3}..{n+3+nnz-1}] col_idx  (int32, {nnz} values)")
    print(f"  [{n+3+nnz}..{n+3+2*nnz-1}] vals     (float32, {nnz} values)")


if __name__ == "__main__":
    main()

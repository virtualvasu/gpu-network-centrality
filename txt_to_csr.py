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
    python snap_to_undirected_csr.py <input.txt> <output.csr>

Prints summary stats and performance info to stdout.
"""

import sys
import time
import struct
import array
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


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


def ensure_suffix(path: str, suffix: str) -> str:
    """Ensure a path ends with a required suffix."""
    if path.endswith(suffix):
        return path
    return f"{path}{suffix}"


def resolve_output_paths(out_path: str):
    """Return output paths for both .csr and .csr.bin artifacts."""
    if out_path.endswith(".csr.bin"):
        return out_path[:-4], out_path
    if out_path.endswith(".csr"):
        return out_path, f"{out_path}.bin"
    if out_path.endswith(".bin"):
        csr_path = out_path[:-4]
        if not csr_path.endswith(".csr"):
            csr_path = ensure_suffix(csr_path, ".csr")
        return csr_path, out_path

    csr_path = ensure_suffix(out_path, ".csr")
    return csr_path, f"{csr_path}.bin"


def compute_eigenvector_scores(n, row_ptr, col_idx, max_iter=1000, tol=1e-6):
    """Compute leading-eigenvector scores using power iteration on CSR."""
    if n == 0:
        return np.array([], dtype=np.float64), True, 0.0

    t0 = time.perf_counter()
    data = np.ones(len(col_idx), dtype=np.float64)
    A = csr_matrix((data, np.asarray(col_idx, dtype=np.int64), np.asarray(row_ptr, dtype=np.int64)), shape=(n, n))

    x = np.full(n, 1.0 / math.sqrt(n), dtype=np.float64)
    converged = False

    for _ in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm == 0.0:
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            converged = True
            break
        x = x_new

    runtime = time.perf_counter() - t0
    return x, converged, runtime


def write_scores_csv(path: str, scores_rows):
    """Write eigenvector scores to CSV with columns node_id,score."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "score"])
        writer.writerows(scores_rows)


def write_metrics_json(path: str, metrics):
    """Write run and graph metrics as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python snap_to_undirected_csr.py <input.txt> <output.csr>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]
    csr_path, csr_bin_path = resolve_output_paths(out_path)

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

    # --- Write CSR outputs ---
    write_binary(csr_path, n, nnz, row_ptr, col_idx, vals)
    write_binary(csr_bin_path, n, nnz, row_ptr, col_idx, vals)
    t3 = time.perf_counter()
    csr_size = os.path.getsize(csr_path)
    csr_bin_size = os.path.getsize(csr_bin_path)
    print(f"\nWrote CSR file    : {csr_path}  ({csr_size / 1e6:.2f} MB)")
    print(f"Wrote CSR.BIN file: {csr_bin_path}  ({csr_bin_size / 1e6:.2f} MB)")
    print(f"Write time        : {(t3-t2)*1e3:.1f} ms")
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
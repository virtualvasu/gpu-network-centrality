"""
Step 0: NetworkX Baseline (Amazon0601)

Standalone Python-script equivalent of step0_networkx_eigen_baseline.ipynb.
It converts a raw edge-list to CSR arrays on disk, loads the graph with
NetworkX, computes eigenvector centrality, and writes baseline artifacts.

Usage:
    python3 step0_networkx_eigen_baseline.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Tuple
import json
import shutil

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# =============================================================================
# CONFIGURATION: dataset-specific baseline run
# =============================================================================
DATASET_KEY = "Amazon0601"
RAW_DATA_PATH = Path(f"dataset/{DATASET_KEY}.txt")
CSR_DIR = Path(f"dataset/{DATASET_KEY}_csr")
METADATA_PATH = CSR_DIR / "metadata.json"
INDPTR_PATH = CSR_DIR / "indptr.txt"
INDICES_PATH = CSR_DIR / "indices.txt"
DATA_PATH = CSR_DIR / "data.txt"

OUTPUT_DIR = Path(f"baseline/networkx/{DATASET_KEY}")
OUTPUT_PREFIX = DATASET_KEY

MAX_ITER = 1000
TOL = 1e-6
TOP_K = 20


def parse_edge_list(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Parse SNAP edge-list text file into undirected CSR arrays."""
    edges = set()
    nodes = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            if u == v:
                continue

            # Symmetrize and deduplicate for an undirected graph.
            edges.add((u, v))
            edges.add((v, u))
            nodes.add(u)
            nodes.add(v)

    node_list = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)

    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[node_to_idx[u]].append(node_to_idx[v])

    for neighbors in adjacency:
        neighbors.sort()

    indptr = np.zeros(n + 1, dtype=np.int64)
    indices_list = []
    for i, neighbors in enumerate(adjacency):
        indptr[i + 1] = indptr[i] + len(neighbors)
        indices_list.extend(neighbors)

    indices = np.asarray(indices_list, dtype=np.int64)
    data = np.ones(indices.shape[0], dtype=np.float64)

    undirected_edges = len(indices_list) // 2
    return indptr, indices, data, n, undirected_edges


def write_array(path: Path, values: Iterable) -> None:
    """Write a 1D array to text file, one value per line."""
    arr = np.asarray(list(values))
    np.savetxt(path, arr, fmt="%.18g")


def main() -> None:
    # Remove only current dataset artifacts for a clean rerun
    assert RAW_DATA_PATH.exists(), f"Missing raw dataset file: {RAW_DATA_PATH}"

    if CSR_DIR.exists():
        shutil.rmtree(CSR_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Convert the raw edge list to CSR
    indptr, indices, data, num_nodes, num_edges = parse_edge_list(RAW_DATA_PATH)
    CSR_DIR.mkdir(parents=True, exist_ok=True)
    write_array(INDPTR_PATH, indptr)
    write_array(INDICES_PATH, indices)
    write_array(DATA_PATH, data)

    metadata: Dict[str, object] = {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "indptr_length": int(len(indptr)),
        "indices_length": int(len(indices)),
        "data_length": int(len(data)),
        "input_file": str(RAW_DATA_PATH),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"CSR conversion complete: {CSR_DIR}")
    print(f"Nodes={num_nodes}, edges={num_edges}, nnz={len(data)}")

    assert CSR_DIR.exists(), f"Missing CSR directory: {CSR_DIR}"
    assert METADATA_PATH.exists(), f"Missing metadata file: {METADATA_PATH}"
    assert INDPTR_PATH.exists(), f"Missing indptr file: {INDPTR_PATH}"
    assert INDICES_PATH.exists(), f"Missing indices file: {INDICES_PATH}"
    assert DATA_PATH.exists(), f"Missing data file: {DATA_PATH}"

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    num_nodes = int(metadata["num_nodes"])
    num_edges = int(metadata["num_edges"])

    indptr = np.loadtxt(INDPTR_PATH, dtype=np.int64)
    indices = np.loadtxt(INDICES_PATH, dtype=np.int64)
    data = np.loadtxt(DATA_PATH, dtype=np.float64)

    A = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)

    print(f"Loaded CSR graph: nodes={num_nodes}, edges={num_edges}, nnz={A.nnz}")
    print(f"Graph type: {type(G).__name__}")

    # Verify that the graph is undirected
    print(f"\n{'=' * 70}")
    print("GRAPH TYPE VERIFICATION")
    print(f"{'=' * 70}")
    print(f"Is graph directed? {nx.is_directed(G)}")
    print(f"Graph type: {type(G).__name__}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"{'=' * 70}\n")

    t0 = perf_counter()
    converged = True

    try:
        centrality = nx.eigenvector_centrality(
            G,
            max_iter=MAX_ITER,
            tol=TOL,
            weight=None,
        )
    except nx.PowerIterationFailedConvergence as e:
        converged = False
        centrality = getattr(e, "eigenvector", None)
        if centrality is None:
            raise RuntimeError(
                f"Eigenvector centrality did not converge in {MAX_ITER} iterations and no partial vector was returned."
            ) from e

    runtime_seconds = perf_counter() - t0
    print(f"Centrality computed in {runtime_seconds:.3f} seconds | converged={converged}")

    scores_df = (
        pd.DataFrame(centrality.items(), columns=["node_id", "score"])
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    topk_df = scores_df.head(TOP_K).copy()
    topk_df.insert(0, "rank", range(1, len(topk_df) + 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_eigenvector_scores.csv"
    topk_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_top{TOP_K}.csv"
    metrics_path = OUTPUT_DIR / "step0_metrics.json"

    scores_df.to_csv(scores_path, index=False)
    topk_df.to_csv(topk_path, index=False)

    metrics = {
        "dataset_key": DATASET_KEY,
        "dataset": str(RAW_DATA_PATH),
        "csr_dir": str(CSR_DIR),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "nnz": int(A.nnz),
        "density": float(A.nnz / (num_nodes * num_nodes)),
        "method": "networkx.eigenvector_centrality",
        "graph_type": "undirected",
        "max_iter": int(MAX_ITER),
        "tol": float(TOL),
        "runtime_seconds": float(runtime_seconds),
        "converged": bool(converged),
        "top_score": float(scores_df.iloc[0]["score"]),
        "top_node_id": int(scores_df.iloc[0]["node_id"]),
        "score_sum": float(scores_df["score"].sum()),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"Saved: {scores_path}")
    print(f"Saved: {topk_path}")
    print(f"Saved: {metrics_path}")

    print("\nTop-K preview:")
    print(topk_df.to_string(index=False))
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # Optional: show existing com-orkut artifacts without modifying them
    orkut_dir = Path("baseline/networkx/com-orkut")
    orkut_metrics_path = orkut_dir / "step0_metrics.json"
    orkut_top20_path = orkut_dir / "com-orkut_top20.csv"

    if orkut_metrics_path.exists():
        with orkut_metrics_path.open("r", encoding="utf-8") as f:
            orkut_metrics = json.load(f)
        print("\nFound existing com-orkut metrics:")
        print(json.dumps(orkut_metrics, indent=2))
    else:
        print("\nNo existing com-orkut metrics found at baseline/networkx/com-orkut/step0_metrics.json")

    if orkut_top20_path.exists():
        print("Found existing com-orkut top-20:")
        print(pd.read_csv(orkut_top20_path).to_string(index=False))
    else:
        print("No existing com-orkut top-20 found at baseline/networkx/com-orkut/com-orkut_top20.csv")


if __name__ == "__main__":
    main()

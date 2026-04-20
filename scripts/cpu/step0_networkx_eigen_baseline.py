"""
Step 0: NetworkX Baseline (Amazon0601)

Standalone Python-script equivalent of step0_networkx_eigen_baseline.ipynb.
It converts a raw edge-list to CSR arrays on disk, loads the graph with
NetworkX, computes eigenvector centrality, and writes baseline artifacts.

Usage:
    python3 step0_networkx_eigen_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict
import json
import shutil
import struct

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# =============================================================================
# CONFIGURATION: dataset-specific baseline run
# =============================================================================
DEFAULT_DATASET_KEY = "roadNet-CA"

def build_paths(dataset_key: str):
    csr_dir = Path(f"dataset/{dataset_key}_csr")
    csr_bin_path = csr_dir / f"{dataset_key}.csr"
    output_dir = Path(f"baseline/networkx/{dataset_key}")
    output_prefix = dataset_key
    return csr_dir, csr_bin_path, output_dir, output_prefix

MAX_ITER = 1000
TOL = 1e-6
TOP_K = 20


def load_binary_csr(path: Path):
    """Load binary CSR file written by txt_to_csr.py (.csr or .csr.bin)."""
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"Invalid CSR file header: {path}")

        num_nodes, nnz = struct.unpack("<ii", header)
        indptr = np.fromfile(f, dtype=np.int32, count=num_nodes + 1).astype(np.int64)
        indices = np.fromfile(f, dtype=np.int32, count=nnz).astype(np.int64)
        data = np.fromfile(f, dtype=np.float32, count=nnz).astype(np.float64)

    if indptr.size != num_nodes + 1 or indices.size != nnz or data.size != nnz:
        raise RuntimeError(
            f"Incomplete CSR binary content in {path}. "
            f"Expected nodes={num_nodes}, nnz={nnz}."
        )

    num_edges = nnz // 2
    return indptr, indices, data, num_nodes, num_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NetworkX eigenvector baseline from existing binary CSR.")
    parser.add_argument(
        "--dataset-key",
        default=DEFAULT_DATASET_KEY,
        help="Dataset key used for paths: dataset/<key>_csr/<key>.csr",
    )
    args = parser.parse_args()

    dataset_key = args.dataset_key
    csr_dir, csr_bin_path, output_dir, output_prefix = build_paths(dataset_key)

    # Keep existing CSR input untouched; clean only output artifacts.
    assert csr_bin_path.exists(), f"Missing CSR binary file: {csr_bin_path}"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    indptr, indices, data, num_nodes, num_edges = load_binary_csr(csr_bin_path)
    print(f"Loaded existing CSR binary: {csr_bin_path}")
    print(f"Nodes={num_nodes}, edges={num_edges}, nnz={len(data)}")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"{output_prefix}_eigenvector_scores.csv"
    topk_path = output_dir / f"{output_prefix}_top{TOP_K}.csv"
    metrics_path = output_dir / "step0_metrics.json"

    scores_df.to_csv(scores_path, index=False)
    topk_df.to_csv(topk_path, index=False)

    metrics = {
        "dataset_key": dataset_key,
        "dataset": str(csr_bin_path),
        "csr_dir": str(csr_dir),
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

#!/usr/bin/env python3
"""
compare_accuracy.py

Compare the top-20 eigenvector centrality scores across all 6 implementations.
Uses the NetworkX CPU baseline as ground truth.

All implementations output CSV files with columns:
    node_id,score          (or node_id,centrality_score for merge-path)
    <int>,<float>

Usage:
    python3 compare_accuracy.py \
        --networkx    <path_to_networkx_scores.csv>  \
        --our-code    <path_to_our_code_scores.csv>   \
        --csr-scalar  <path_to_csr_scalar_scores.csv> \
        --cusparse    <path_to_cusparse_scores.csv>    \
        --lanczos     <path_to_lanczos_scores.csv>     \
        --merge-path  <path_to_mergepath_scores.csv>   \
        [--top-k 20]

Example (from the scripts/ directory after running all implementations on roadNet-CA):
    python3 compare_accuracy.py \
        --networkx    baseline/networkx/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --our-code    baseline/our_code/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --csr-scalar  baseline/csr_scalar/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --cusparse    baseline/cu_sparse/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --lanczos     baseline/lanczos/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --merge-path  gpu_scores_mergepath.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_scores(csv_path: str) -> pd.DataFrame:
    """Load a scores CSV (node_id, score) and return sorted descending by score."""
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    if "node_id" not in df.columns or "score" not in df.columns:
        print(f"ERROR: CSV must have 'node_id' and 'score' columns: {csv_path}",
              file=sys.stderr)
        sys.exit(1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def top_k_nodes(df: pd.DataFrame, k: int) -> List[int]:
    """Return list of top-k node IDs."""
    return df.head(k)["node_id"].astype(int).tolist()


def top_k_scores(df: pd.DataFrame, k: int) -> np.ndarray:
    """Return numpy array of top-k scores."""
    return df.head(k)["score"].values.astype(np.float64)


def node_to_score_map(df: pd.DataFrame) -> Dict[int, float]:
    """Build a dict node_id -> score from full dataframe."""
    return dict(zip(df["node_id"].astype(int), df["score"].astype(float)))


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def set_overlap(ref_nodes: List[int], test_nodes: List[int]) -> Tuple[float, int, int]:
    """
    Fraction of top-k nodes that appear in both sets.
    Returns (overlap_fraction, common_count, k).
    """
    ref_set = set(ref_nodes)
    test_set = set(test_nodes)
    common = ref_set & test_set
    k = len(ref_nodes)
    return len(common) / k, len(common), k


def rank_correlation(ref_nodes: List[int], test_nodes: List[int]) -> float:
    """
    Kendall-tau-style rank agreement metric.
    For nodes that appear in both top-k lists, measure how closely
    their ranks match. Returns a value in [0, 1] where 1 = perfect match.
    """
    ref_rank = {node: rank for rank, node in enumerate(ref_nodes)}
    test_rank = {node: rank for rank, node in enumerate(test_nodes)}
    common = set(ref_rank.keys()) & set(test_rank.keys())
    if len(common) < 2:
        return 0.0

    # Spearman-style: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    n = len(common)
    d_sq_sum = sum((ref_rank[node] - test_rank[node]) ** 2 for node in common)
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return rho


def score_mae(ref_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Mean Absolute Error of the top-k scores (positional comparison)."""
    k = min(len(ref_scores), len(test_scores))
    return float(np.mean(np.abs(ref_scores[:k] - test_scores[:k])))


def score_rmse(ref_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Root Mean Squared Error of the top-k scores (positional comparison)."""
    k = min(len(ref_scores), len(test_scores))
    return float(np.sqrt(np.mean((ref_scores[:k] - test_scores[:k]) ** 2)))


def score_max_abs_error(ref_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Maximum absolute error between corresponding top-k scores."""
    k = min(len(ref_scores), len(test_scores))
    return float(np.max(np.abs(ref_scores[:k] - test_scores[:k])))


def cosine_similarity(ref_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Cosine similarity between the top-k score vectors."""
    k = min(len(ref_scores), len(test_scores))
    a, b = ref_scores[:k], test_scores[:k]
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(dot / (na * nb))


def relative_score_error(ref_scores: np.ndarray, test_scores: np.ndarray) -> float:
    """Mean relative error: mean(|ref - test| / |ref|) for ref > 0."""
    k = min(len(ref_scores), len(test_scores))
    mask = np.abs(ref_scores[:k]) > 1e-15
    if not np.any(mask):
        return 0.0
    rel = np.abs(ref_scores[:k][mask] - test_scores[:k][mask]) / np.abs(ref_scores[:k][mask])
    return float(np.mean(rel))


# ──────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 90
THIN_SEP  = "-" * 90

def print_header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def compare_one(name: str,
                ref_df: pd.DataFrame,
                test_df: pd.DataFrame,
                k: int) -> Dict:
    """Compare a single implementation against the reference and return metrics dict."""
    ref_nodes = top_k_nodes(ref_df, k)
    test_nodes = top_k_nodes(test_df, k)
    ref_scores = top_k_scores(ref_df, k)
    test_scores = top_k_scores(test_df, k)

    overlap_frac, common_count, total = set_overlap(ref_nodes, test_nodes)
    rho = rank_correlation(ref_nodes, test_nodes)
    mae = score_mae(ref_scores, test_scores)
    rmse = score_rmse(ref_scores, test_scores)
    max_err = score_max_abs_error(ref_scores, test_scores)
    cos_sim = cosine_similarity(ref_scores, test_scores)
    rel_err = relative_score_error(ref_scores, test_scores)

    metrics = {
        "name": name,
        "set_overlap": f"{common_count}/{total} ({overlap_frac*100:.1f}%)",
        "rank_correlation": rho,
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_err,
        "cosine_similarity": cos_sim,
        "mean_relative_error": rel_err,
    }

    print(f"\n  Implementation: {name}")
    print(THIN_SEP)
    print(f"  {'Metric':<30s}  {'Value':>20s}")
    print(f"  {'------':<30s}  {'-----':>20s}")
    print(f"  {'Set Overlap (top-k nodes)':<30s}  {metrics['set_overlap']:>20s}")
    print(f"  {'Rank Correlation (Spearman)':<30s}  {rho:>20.6f}")
    print(f"  {'Score MAE':<30s}  {mae:>20.2e}")
    print(f"  {'Score RMSE':<30s}  {rmse:>20.2e}")
    print(f"  {'Max Absolute Error':<30s}  {max_err:>20.2e}")
    print(f"  {'Cosine Similarity':<30s}  {cos_sim:>20.8f}")
    print(f"  {'Mean Relative Error':<30s}  {rel_err:>20.2e}")

    # Per-node detail table: show the top-k side by side
    print(f"\n  {'Rank':>4s}  {'NX Node':>8s}  {'NX Score':>12s}  "
          f"{'GPU Node':>8s}  {'GPU Score':>12s}  {'Δ Score':>12s}  {'Match':>5s}")
    print(f"  {'----':>4s}  {'-------':>8s}  {'--------':>12s}  "
          f"{'--------':>8s}  {'---------':>12s}  {'-------':>12s}  {'-----':>5s}")

    for i in range(k):
        rn = ref_nodes[i]
        rs = ref_scores[i]
        tn = test_nodes[i]
        ts = test_scores[i]
        delta = ts - rs
        match = "✓" if rn == tn else "✗"
        print(f"  {i+1:>4d}  {rn:>8d}  {rs:>12.8f}  "
              f"{tn:>8d}  {ts:>12.8f}  {delta:>+12.2e}  {match:>5s}")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────

def print_summary_table(all_metrics: List[Dict]):
    """Print a compact summary comparison table."""
    print_header("SUMMARY COMPARISON TABLE")
    print()

    # Table header
    names = [m["name"] for m in all_metrics]
    col_w = max(18, max(len(n) for n in names) + 2)

    header = f"  {'Metric':<30s}"
    for m in all_metrics:
        header += f"  {m['name']:>{col_w}s}"
    print(header)
    print(f"  {'------':<30s}" + f"  {'-' * col_w}" * len(all_metrics))

    # Rows
    row_keys = [
        ("Set Overlap",          "set_overlap",         "s"),
        ("Rank Correlation",     "rank_correlation",    ".6f"),
        ("Score MAE",            "mae",                 ".2e"),
        ("Score RMSE",           "rmse",                ".2e"),
        ("Max Abs Error",        "max_abs_error",       ".2e"),
        ("Cosine Similarity",    "cosine_similarity",   ".8f"),
        ("Mean Relative Error",  "mean_relative_error", ".2e"),
    ]

    for label, key, fmt in row_keys:
        row = f"  {label:<30s}"
        for m in all_metrics:
            val = m[key]
            if isinstance(val, str):
                row += f"  {val:>{col_w}s}"
            else:
                row += f"  {val:>{col_w}{fmt}}"
        print(row)

    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare top-k eigenvector centrality accuracy across implementations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each CSV file must have columns: node_id,score
Rows should be sorted by score (descending), which is the default output
of all implementations in this project.

The NetworkX implementation is treated as the ground-truth reference.
All other implementations are compared against it.

Example:
    python3 compare_accuracy.py \\
        --networkx    baseline/networkx/roadNet-CA/roadNet-CA_eigenvector_scores.csv \\
        --our-code    baseline/our_code/roadNet-CA/roadNet-CA_eigenvector_scores.csv \\
        --csr-scalar  baseline/csr_scalar/roadNet-CA/roadNet-CA_eigenvector_scores.csv \\
        --cusparse    baseline/cu_sparse/roadNet-CA/roadNet-CA_eigenvector_scores.csv \\
        --lanczos     baseline/lanczos/roadNet-CA/roadNet-CA_eigenvector_scores.csv \\
        --merge-path  baseline/eigen_centrality/roadNet-CA/roadNet-CA_eigenvector_scores.csv
        """,
    )
    parser.add_argument("--networkx", required=True,
                        help="Path to NetworkX baseline scores CSV (ground truth)")
    parser.add_argument("--our-code", required=True,
                        help="Path to our_code.cu scores CSV")
    parser.add_argument("--csr-scalar", required=True,
                        help="Path to CSR-scalar + cuBLAS scores CSV")
    parser.add_argument("--cusparse", required=True,
                        help="Path to cuSPARSE + cuBLAS scores CSV")
    parser.add_argument("--lanczos", required=True,
                        help="Path to Lanczos scores CSV")
    parser.add_argument("--merge-path", required=True,
                        help="Path to Merge-Path FP32 scores CSV")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top nodes to compare (default: 20)")
    args = parser.parse_args()

    k = args.top_k

    # ── Load all CSVs ──
    print_header(f"EIGENVECTOR CENTRALITY ACCURACY COMPARISON (Top-{k})")
    print(f"\n  Ground truth : NetworkX CPU  ({args.networkx})")

    implementations = {
        "our_code.cu":              args.our_code,
        "CSR-scalar + cuBLAS":      args.csr_scalar,
        "cuSPARSE + cuBLAS":        args.cusparse,
        "Lanczos":                  args.lanczos,
        "Merge-Path FP32":          args.merge_path,
    }

    ref_df = load_scores(args.networkx)
    print(f"  Reference loaded: {len(ref_df)} nodes")

    test_dfs = {}
    for name, path in implementations.items():
        df = load_scores(path)
        test_dfs[name] = df
        print(f"  {name:25s} loaded: {len(df)} nodes  ({path})")

    # ── Show reference top-k ──
    print_header(f"REFERENCE: NetworkX Top-{k}")
    ref_nodes = top_k_nodes(ref_df, k)
    ref_scores = top_k_scores(ref_df, k)
    print(f"\n  {'Rank':>4s}  {'Node ID':>8s}  {'Score':>14s}")
    print(f"  {'----':>4s}  {'-------':>8s}  {'-----':>14s}")
    for i in range(k):
        print(f"  {i+1:>4d}  {ref_nodes[i]:>8d}  {ref_scores[i]:>14.8f}")

    # ── Compare each implementation ──
    all_metrics = []
    for name, df in test_dfs.items():
        print_header(f"COMPARISON: {name} vs NetworkX")
        metrics = compare_one(name, ref_df, df, k)
        all_metrics.append(metrics)

    # ── Summary table ──
    print_summary_table(all_metrics)

    # ── Overall verdict ──
    print_header("VERDICT")
    for m in all_metrics:
        cos = m["cosine_similarity"]
        rel = m["mean_relative_error"]
        overlap = m["set_overlap"]
        if cos >= 0.9999 and rel < 0.01:
            verdict = "✅ EXCELLENT — near-identical to NetworkX"
        elif cos >= 0.999 and rel < 0.05:
            verdict = "✅ GOOD — very close to NetworkX"
        elif cos >= 0.99 and rel < 0.10:
            verdict = "⚠️  ACCEPTABLE — minor deviations"
        elif cos >= 0.95:
            verdict = "⚠️  FAIR — noticeable deviations"
        else:
            verdict = "❌ POOR — significant accuracy loss"
        print(f"  {m['name']:25s} : {verdict}  (overlap={overlap}, cos={cos:.6f})")

    print(f"\n{SEPARATOR}\n")


if __name__ == "__main__":
    main()

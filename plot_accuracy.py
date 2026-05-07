#!/usr/bin/env python3
"""
plot_accuracy.py

Generate publication-quality plots comparing eigenvector centrality
accuracy across all GPU implementations vs the NetworkX CPU baseline.

Usage (same arguments as compare_accuracy.py):
    python3 plot_accuracy.py \
        --networkx   baseline/networkx/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --our-code   baseline/our_code/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --csr-scalar baseline/csr_scalar/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --cusparse   baseline/cu_sparse/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        --lanczos    baseline/lanczos/roadNet-CA/roadNet-CA_eigenvector_scores.csv \
        [--top-k 20] [--out-dir accuracy_plots]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colours & style ──────────────────────────────────────────────────
COLORS = {
    "our_code.cu":         "#4361ee",
    "CSR-scalar + cuBLAS": "#f72585",
    "cuSPARSE + cuBLAS":   "#7209b7",
    "Lanczos":             "#06d6a0",
}
REF_COLOR = "#ff9f1c"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Helpers ──────────────────────────────────────────────────────────
def load_scores(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: not found: {csv_path}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(path)
    return df.sort_values("score", ascending=False).reset_index(drop=True)

def top_k(df, k):
    sub = df.head(k)
    return sub["node_id"].astype(int).tolist(), sub["score"].values.astype(np.float64)

# ── Metric helpers ───────────────────────────────────────────────────
def set_overlap_frac(a, b):
    return len(set(a) & set(b)) / len(a)

def rank_corr(a, b):
    ra = {n: i for i, n in enumerate(a)}
    rb = {n: i for i, n in enumerate(b)}
    common = set(ra) & set(rb)
    if len(common) < 2: return 0.0
    n = len(common)
    d2 = sum((ra[c] - rb[c])**2 for c in common)
    return 1 - 6*d2 / (n*(n**2-1))

def cos_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dot/(na*nb) if na > 1e-15 and nb > 1e-15 else 0.0

# ── Plot functions ───────────────────────────────────────────────────

def plot_score_comparison(ref_nodes, ref_scores, impl_data, k, out_dir):
    """Bar chart: reference vs each implementation scores side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    x = np.arange(k)

    for ax, (name, (nodes, scores)) in zip(axes, impl_data.items()):
        w = 0.35
        ax.bar(x - w/2, ref_scores, w, label="NetworkX (ref)", color=REF_COLOR, alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, scores, w, label=name, color=COLORS[name], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Centrality Score")
        ax.set_title(f"{name} vs NetworkX")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in x], fontsize=8)
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    fig.suptitle(f"Top-{k} Eigenvector Centrality Scores: GPU vs NetworkX", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "score_comparison_bars.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_score_error(ref_scores, impl_data, k, out_dir):
    """Per-rank absolute error for each implementation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, k+1)

    for name, (_, scores) in impl_data.items():
        errors = np.abs(ref_scores[:k] - scores[:k])
        ax.plot(x, errors, "o-", label=name, color=COLORS[name], linewidth=2, markersize=5)

    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Absolute Score Error")
    ax.set_title(f"Per-Rank Absolute Score Error (Top-{k})")
    ax.set_xticks(x)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "per_rank_error.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_relative_error(ref_scores, impl_data, k, out_dir):
    """Per-rank relative error for each implementation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, k+1)

    for name, (_, scores) in impl_data.items():
        mask = np.abs(ref_scores[:k]) > 1e-15
        rel_err = np.zeros(k)
        rel_err[mask] = np.abs(ref_scores[:k][mask] - scores[:k][mask]) / np.abs(ref_scores[:k][mask]) * 100
        ax.plot(x, rel_err, "s-", label=name, color=COLORS[name], linewidth=2, markersize=5)

    ax.set_xlabel("Rank Position")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"Per-Rank Relative Score Error (Top-{k})")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "per_rank_relative_error.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_heatmap(ref_nodes, ref_scores, impl_data, k, out_dir):
    """Heatmap of key accuracy metrics across implementations."""
    names = list(impl_data.keys())
    metrics_labels = ["Set Overlap", "Rank Correlation", "Cosine Similarity",
                      "1 − MAE×1e3", "1 − RMSE×1e3"]
    data = np.zeros((len(names), len(metrics_labels)))

    for i, name in enumerate(names):
        nodes, scores = impl_data[name]
        data[i, 0] = set_overlap_frac(ref_nodes, nodes)
        data[i, 1] = rank_corr(ref_nodes, nodes)
        data[i, 2] = cos_sim(ref_scores[:k], scores[:k])
        mae = float(np.mean(np.abs(ref_scores[:k] - scores[:k])))
        rmse = float(np.sqrt(np.mean((ref_scores[:k] - scores[:k])**2)))
        data[i, 3] = max(0, 1 - mae * 1e3)
        data[i, 4] = max(0, 1 - rmse * 1e3)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics_labels)))
    ax.set_xticklabels(metrics_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(metrics_labels)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_title(f"Accuracy Heatmap vs NetworkX (Top-{k})", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score (1.0 = perfect)")
    fig.tight_layout()
    path = out_dir / "accuracy_heatmap.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_radar(ref_nodes, ref_scores, impl_data, k, out_dir):
    """Radar / spider chart of accuracy metrics."""
    categories = ["Set\nOverlap", "Rank\nCorrelation", "Cosine\nSimilarity",
                   "1−MAE\n(scaled)", "1−MaxErr\n(scaled)"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name, (nodes, scores) in impl_data.items():
        mae = float(np.mean(np.abs(ref_scores[:k] - scores[:k])))
        max_err = float(np.max(np.abs(ref_scores[:k] - scores[:k])))
        vals = [
            set_overlap_frac(ref_nodes, nodes),
            max(0, rank_corr(ref_nodes, nodes)),
            cos_sim(ref_scores[:k], scores[:k]),
            max(0, 1 - mae * 1e3),
            max(0, 1 - max_err * 1e3),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", label=name, color=COLORS[name], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy Radar (Top-{k} vs NetworkX)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    path = out_dir / "accuracy_radar.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_node_match_matrix(ref_nodes, impl_data, k, out_dir):
    """Binary heatmap: does each implementation match the reference at each rank?"""
    names = list(impl_data.keys())
    matrix = np.zeros((len(names), k))

    for i, name in enumerate(names):
        nodes, _ = impl_data[name]
        for j in range(k):
            matrix[i, j] = 1.0 if nodes[j] == ref_nodes[j] else 0.0

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(k))
    ax.set_xticklabels([str(i+1) for i in range(k)], fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Rank Position")
    ax.set_title(f"Node-ID Match at Each Rank (Green=Match, Red=Mismatch)", fontsize=13, fontweight="bold")

    for i in range(len(names)):
        for j in range(k):
            sym = "✓" if matrix[i, j] == 1 else "✗"
            color = "black" if matrix[i, j] == 1 else "white"
            ax.text(j, i, sym, ha="center", va="center", fontsize=10, color=color)

    fig.tight_layout()
    path = out_dir / "node_match_matrix.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_cumulative_overlap(ref_nodes, impl_data, out_dir):
    """Line plot: cumulative set overlap as we go from top-1 to top-k."""
    fig, ax = plt.subplots(figsize=(10, 6))
    k = len(ref_nodes)
    x = np.arange(1, k+1)

    for name, (nodes, _) in impl_data.items():
        overlaps = []
        for i in range(1, k+1):
            ov = len(set(ref_nodes[:i]) & set(nodes[:i])) / i
            overlaps.append(ov)
        ax.plot(x, overlaps, "o-", label=name, color=COLORS[name], linewidth=2, markersize=5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect match")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Set Overlap Fraction")
    ax.set_title("Cumulative Set Overlap vs NetworkX")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cumulative_overlap.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_bars(ref_nodes, ref_scores, impl_data, k, out_dir):
    """Grouped bar chart of key metrics side-by-side."""
    names = list(impl_data.keys())
    metrics = {}
    for name in names:
        nodes, scores = impl_data[name]
        metrics[name] = {
            "Set Overlap": set_overlap_frac(ref_nodes, nodes),
            "Rank Correlation": max(0, rank_corr(ref_nodes, nodes)),
            "Cosine Similarity": cos_sim(ref_scores[:k], scores[:k]),
        }

    metric_names = list(list(metrics.values())[0].keys())
    x = np.arange(len(metric_names))
    width = 0.18
    offsets = np.linspace(-width*1.5, width*1.5, len(names))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        vals = [metrics[name][m] for m in metric_names]
        bars = ax.bar(x + offsets[i], vals, width, label=name, color=COLORS[name], alpha=0.9, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Accuracy Metrics Summary (Top-{k} vs NetworkX)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "summary_bars.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy comparison graphs.")
    parser.add_argument("--networkx", required=True)
    parser.add_argument("--our-code", required=True)
    parser.add_argument("--csr-scalar", required=True)
    parser.add_argument("--cusparse", required=True)
    parser.add_argument("--lanczos", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="accuracy_plots",
                        help="Directory to save plots (default: accuracy_plots)")
    args = parser.parse_args()

    k = args.top_k
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ref_df = load_scores(args.networkx)
    ref_nodes, ref_scores = top_k(ref_df, k)

    impl_map = {
        "our_code.cu":         args.our_code,
        "CSR-scalar + cuBLAS": args.csr_scalar,
        "cuSPARSE + cuBLAS":   args.cusparse,
        "Lanczos":             args.lanczos,
    }

    impl_data = {}
    for name, path in impl_map.items():
        df = load_scores(path)
        nodes, scores = top_k(df, k)
        impl_data[name] = (nodes, scores)

    print(f"\nGenerating plots (top-{k}) → {out_dir}/\n")

    # Generate all plots
    plot_score_comparison(ref_nodes, ref_scores, impl_data, k, out_dir)
    plot_score_error(ref_scores, impl_data, k, out_dir)
    plot_relative_error(ref_scores, impl_data, k, out_dir)
    plot_summary_heatmap(ref_nodes, ref_scores, impl_data, k, out_dir)
    plot_radar(ref_nodes, ref_scores, impl_data, k, out_dir)
    plot_node_match_matrix(ref_nodes, impl_data, k, out_dir)
    plot_cumulative_overlap(ref_nodes, impl_data, out_dir)
    plot_summary_bars(ref_nodes, ref_scores, impl_data, k, out_dir)

    print(f"\n✅ All 8 plots saved to {out_dir}/\n")


if __name__ == "__main__":
    main()

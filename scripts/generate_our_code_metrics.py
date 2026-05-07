#!/usr/bin/env python3
"""Generate derived metrics from baseline/our_code step0_metrics.json files.

This script scans each dataset directory under baseline/our_code, reads the
existing step0_metrics.json file, and writes a consolidated summary with
additional derived metrics.

Derived metrics include:
- runtime_ms
- traversed_edges
- mteps (Million Traversed Edges Per Second)
- avg_iter_ms
- residual_per_node

The script does not require any code changes to the CUDA implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_traversed_edges(metrics: Dict[str, Any]) -> int:
    iterations = int(metrics.get("iterations", 0) or 0)
    nnz = int(metrics.get("nnz", 0) or 0)

    if iterations > 0 and nnz > 0:
        return nnz * iterations

    num_edges = int(metrics.get("num_edges", 0) or 0)
    return num_edges


def derive_row(dataset_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    runtime_seconds = float(metrics.get("runtime_seconds", 0.0) or 0.0)
    runtime_ms = runtime_seconds * 1000.0
    iterations = int(metrics.get("iterations", 0) or 0)
    num_nodes = int(metrics.get("num_nodes", 0) or 0)
    final_residual = float(metrics.get("final_residual", 0.0) or 0.0)
    traversed_edges = infer_traversed_edges(metrics)

    mteps = 0.0
    if runtime_seconds > 0.0:
        mteps = traversed_edges / runtime_seconds / 1e6

    avg_iter_ms = 0.0
    if iterations > 0:
        avg_iter_ms = runtime_ms / iterations

    residual_per_node = 0.0
    if num_nodes > 0:
        residual_per_node = final_residual / num_nodes

    row: Dict[str, Any] = {
        "dataset_key": metrics.get("dataset_key", dataset_dir.name),
        "dataset_dir": str(dataset_dir),
        "method": metrics.get("method", "our_code.power_iteration"),
        "graph_type": metrics.get("graph_type", ""),
        "num_nodes": num_nodes,
        "num_edges": int(metrics.get("num_edges", 0) or 0),
        "nnz": int(metrics.get("nnz", 0) or 0),
        "density": float(metrics.get("density", 0.0) or 0.0),
        "max_iter": int(metrics.get("max_iter", 0) or 0),
        "tol": float(metrics.get("tol", 0.0) or 0.0),
        "runtime_seconds": runtime_seconds,
        "runtime_ms": runtime_ms,
        "iterations": iterations,
        "avg_iter_ms": avg_iter_ms,
        "converged": bool(metrics.get("converged", False)),
        "final_residual": final_residual,
        "residual_per_node": residual_per_node,
        "top_node_id": int(metrics.get("top_node_id", -1) or -1),
        "top_score": float(metrics.get("top_score", 0.0) or 0.0),
        "traversed_edges": traversed_edges,
        "mteps": mteps,
    }

    # Preserve any extra fields already present in the source JSON so the
    # consolidated output can carry method-specific metadata forward.
    for key, value in metrics.items():
        if key not in row:
            row[key] = value

    return row


def iter_dataset_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            metrics_file = path / "step0_metrics.json"
            if metrics_file.exists():
                yield path


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("no metrics rows to write")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate derived metrics from baseline/our_code JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("baseline/our_code"),
        help="Directory containing per-dataset subfolders with step0_metrics.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("baseline/our_code/generated_metrics/derived_metrics.json"),
        help="Path to write the consolidated JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("baseline/our_code/generated_metrics/derived_metrics.csv"),
        help="Path to write the consolidated CSV summary.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {args.input_dir}")

    rows: List[Dict[str, Any]] = []
    for dataset_dir in iter_dataset_dirs(args.input_dir):
        metrics_file = dataset_dir / "step0_metrics.json"
        metrics = load_metrics(metrics_file)
        rows.append(derive_row(dataset_dir, metrics))

    rows.sort(key=lambda row: row["dataset_key"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=False)
        handle.write("\n")

    write_csv(rows, args.output_csv)

    print(f"Wrote {len(rows)} dataset summaries")
    print(f"JSON: {args.output_json}")
    print(f"CSV : {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
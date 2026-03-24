#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_edge_list(input_path: Path):
    adjacency = defaultdict(list)
    max_node = -1
    edge_count = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue

            try:
                src = int(parts[0])
                dst = int(parts[1])
            except ValueError:
                continue

            adjacency[src].append(dst)
            edge_count += 1
            max_node = max(max_node, src, dst)

    if max_node < 0:
        return [0], [], [], 0, 0

    num_nodes = max_node + 1

    indptr = [0] * (num_nodes + 1)
    indices = []
    data = []

    for node in range(num_nodes):
        neighbors = adjacency.get(node, [])
        neighbors.sort()
        indices.extend(neighbors)
        data.extend([1] * len(neighbors))
        indptr[node + 1] = len(indices)

    return indptr, indices, data, num_nodes, edge_count


def write_array(path: Path, values):
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(map(str, values)))
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an edge-list TXT graph file to CSR arrays."
    )
    parser.add_argument("input", type=Path, help="Path to input TXT edge-list file")
    parser.add_argument(
        "output_dir", type=Path, help="Directory where CSR files will be written"
    )

    args = parser.parse_args()

    indptr, indices, data, num_nodes, num_edges = parse_edge_list(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_array(args.output_dir / "indptr.txt", indptr)
    write_array(args.output_dir / "indices.txt", indices)
    write_array(args.output_dir / "data.txt", data)

    metadata = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "indptr_length": len(indptr),
        "indices_length": len(indices),
        "data_length": len(data),
        "input_file": str(args.input),
    }

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"CSR conversion complete. Output written to: {args.output_dir}")
    print(
        f"Nodes: {num_nodes}, Edges: {num_edges}, Non-zeros: {len(data)}"
    )


if __name__ == "__main__":
    main()

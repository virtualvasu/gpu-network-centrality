# Eigen Centrality Project — Big Picture

## Step 0 — CPU Baseline with NetworkX (Python)

### Objective
Establish a **reference baseline** for eigenvector centrality using NetworkX on the first dataset before moving to optimized/GPU implementations.

### Dataset (Step 0 Input)
- File: `dataset/amazon0302.txt/Amazon0302.txt`
- Graph type (from file header): **Directed graph**
- Nodes: **262111**
- Edges: **1234877**
- Edge format: `FromNodeId ToNodeId` (integer node IDs)

### Baseline Definition
- Library: `networkx`
- Graph object: `nx.DiGraph`
- Centrality method: `nx.eigenvector_centrality`
- Solver style: Power iteration
- Fixed baseline params:
	- `max_iter = 1000`
	- `tol = 1e-06`
	- `weight = None` (treat all edges as unweighted)

### Execution Spec
1. Load graph from `Amazon0302.txt` while ignoring comment lines starting with `#`.
2. Build a directed graph (`DiGraph`) from the edge list.
3. Run `nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight=None)`.
4. Record runtime (wall-clock) for centrality computation.
5. Save full centrality scores as `(node_id, score)`.
6. Save Top-20 nodes by score (descending).

### Deliverables for Step 0
- `baseline/networkx/amazon0302_eigenvector_scores.csv`
	- Columns: `node_id,score`
- `baseline/networkx/amazon0302_top20.csv`
	- Columns: `rank,node_id,score`
- `baseline/networkx/step0_metrics.json`
	- Fields:
		- `dataset`
		- `num_nodes`
		- `num_edges`
		- `method`
		- `max_iter`
		- `tol`
		- `runtime_seconds`
		- `converged` (true/false)

### Success Criteria
- Baseline completes and produces all three output files.
- Centrality scores are reproducible with the same parameters.
- This baseline is treated as the **reference** for accuracy and performance comparison in later steps.

---

_Future steps (Step 1, Step 2, ...) will be appended below this section._

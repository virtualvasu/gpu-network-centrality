# Eigen Centrality Workflow

This repository processes each dataset in the same order so CPU and GPU results stay comparable.

## Dataset Processing Steps

1. Get the dataset text file in `.txt` format. The graph may be directed or undirected, depending on the source dataset.
2. Convert the `.txt` file to CSR format for CPU evaluation.
3. Run the CPU baseline notebook:
   `/home/netweb/vasu/ugq/gpu-network-centrality/step0_networkx_eigen_baseline.ipynb`
4. Convert the graph to `csr.bin` using:
   `gpu-network-centrality/txt_to_csr.py`
5. Run the GPU evaluation using:
   `gpu-network-centrality/raw_code/mergepath.cu`
6. Save the eigenvector scores and run metrics for each dataset as CSV and JSON files.

## Output Files

For each dataset, keep:

- eigenvector scores in a CSV file
- summary metrics in a JSON file
- any top-k or ranking output used for comparison

## Notes

- Use the same dataset name across the text file, CSR files, and output files.
- Keep CPU and GPU runs aligned on the same graph input so the results are comparable.

#!/usr/bin/env python3
"""
Analyze CSR file format and directionality.
"""
import struct
import sys

def analyze_csr(path):
    """Read binary CSR and print structure info."""
    with open(path, "rb") as f:
        # Try reading as standard binary CSR (int32 n, int32 nnz)
        header = f.read(8)
        if len(header) < 8:
            print(f"File too small: {len(header)} bytes")
            return
        
        n, nnz = struct.unpack("<ii", header)
        print(f"Parsed header: n={n}, nnz={nnz}")
        
        # Sanity check
        if n < 0 or n > 1e9 or nnz < 0 or nnz > 1e11:
            print(f"Values seem invalid (n={n}, nnz={nnz}). Trying big-endian...")
            n, nnz = struct.unpack(">ii", header)
            print(f"  Big-endian: n={n}, nnz={nnz}")
        
        if n > 0 and n < 1e7 and nnz > 0 and nnz < 1e9:
            print(f"  ✓ Looks valid")
            
            # Read row_ptr to check sizes
            row_ptr_size = (n + 1) * 4
            row_ptr_data = f.read(row_ptr_size)
            if len(row_ptr_data) == row_ptr_size:
                print(f"  read row_ptr: {row_ptr_size} bytes ✓")
                row_ptr = struct.unpack(f"<{n+1}i", row_ptr_data)
                print(f"  row_ptr[0]={row_ptr[0]}, row_ptr[n]={row_ptr[n]}")
                
                # Read col_idx
                col_idx_size = nnz * 4
                col_idx_data = f.read(col_idx_size)
                if len(col_idx_data) == col_idx_size:
                    print(f"  read col_idx: {col_idx_size} bytes ✓")
                    
                    # Read vals
                    vals_size = nnz * 4
                    vals_data = f.read(vals_size)
                    if len(vals_data) == vals_size:
                        print(f"  read vals: {vals_size} bytes ✓")
                        print("\n✓ File appears to be valid binary CSR format")
                        
                        # Check directionality: if for every (u,v) with u!=v, (v,u) exists, it's undirected
                        col_idx = struct.unpack(f"<{nnz}i", col_idx_data)
                        vals = struct.unpack(f"<{nnz}f", vals_data)
                        
                        # Build a dict of edges for quick lookup
                        edges = set()
                        for i in range(n):
                            for j in range(row_ptr[i], row_ptr[i+1]):
                                u, v = i, col_idx[j]
                                edges.add((u, v))
                        
                        # Check if symmetric
                        is_undirected = True
                        num_asymmetric = 0
                        for u, v in list(edges):
                            if u != v and (v, u) not in edges:
                                is_undirected = False
                                num_asymmetric += 1
                                if num_asymmetric <= 5:
                                    print(f"  Asymmetric edge found: ({u}, {v}) exists but ({v}, {u}) does not")
                        
                        if is_undirected:
                            print(f"\n✓ Graph appears UNDIRECTED (symmetric)")
                        else:
                            print(f"\n✗ Graph appears DIRECTED (has {num_asymmetric} asymmetric edges)")
                        
                        return n, nnz, is_undirected
                    else:
                        print(f"  ERROR reading vals: expected {vals_size}, got {len(vals_data)}")
                else:
                    print(f"  ERROR reading col_idx: expected {col_idx_size}, got {len(col_idx_data)}")
            else:
                print(f"  ERROR reading row_ptr: expected {row_ptr_size}, got {len(row_ptr_data)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_csr(sys.argv[1])
    else:
        analyze_csr("dataset/big_graphs/uk-2002_output.csr")

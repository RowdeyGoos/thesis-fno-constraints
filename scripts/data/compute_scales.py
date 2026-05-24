#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
Compute normalization scales from training data.

This script computes median-based normalization scales from an HDF5 training dataset.
These scales are used to normalize input and output fields during training.

Usage:
    python compute_scales.py --datapath /path/to/data --filename train_k1_5_32k.h5 [--nx 128] [--ny 128] [--lx 1.0] [--ly 1.0]
    
Example:
    python compute_scales.py --datapath ./data/poisson_full --filename poissons_train_k1_5_32768.h5
"""

import os
import argparse
import numpy as np
import h5py


def compute_scales(datapath, filename, nx=128, ny=128, lx=1.0, ly=1.0):
    """
    Compute normalization scales from training dataset.
    
    Parameters
    ----------
    datapath : str
        Path to directory containing the HDF5 file
    filename : str
        Name of the HDF5 training file
    nx : int
        Grid resolution in x-direction (default: 128)
    ny : int
        Grid resolution in y-direction (default: 128)
    lx : float
        Domain length in x-direction (default: 1.0)
    ly : float
        Domain length in y-direction (default: 1.0)
        
    Returns
    -------
    scale : list
        List containing [source_scale, *tensor_scales, sol_scale, lx, ly]
        where:
        - source_scale: median norm of source functions
        - tensor_scales: median absolute values for each tensor component
        - sol_scale: median of solution max values
        - lx, ly: domain dimensions
    """
    
    filepath = os.path.join(datapath, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    print(f"Loading dataset from: {filepath}")
    
    with h5py.File(filepath, "r") as f:
        print(f"Available keys: {list(f.keys())}")
        
        x_train = f['fields'][:]
        x_tensor = f['tensor'][:]
        
        print(f"Fields shape: {x_train.shape}")
        print(f"Tensor shape: {x_tensor.shape}")
    
    # Compute scales
    source_norm = []
    sol_max = []
    tensor_max = []
    
    num_samples = x_train.shape[0]
    num_tensor_components = x_tensor.shape[1]
    
    print(f"\nProcessing {num_samples} samples...")
    
    for i in range(num_samples):
        # Source function norm (normalized by grid spacing)
        sn = np.linalg.norm(x_train[i, 0]) * lx / nx * ly / ny
        source_norm.append(sn)
        
        # Solution field maximum
        sol_max.append(np.max(np.abs(x_train[i, 1])))
        
        # Tensor components
        tensor_max.append([np.abs(x_tensor[i, t_idx]) for t_idx in range(num_tensor_components)])
    
    # Convert to numpy arrays for easier processing
    tensor_max = np.array(tensor_max)
    
    # Compute medians
    source_scale = np.median(source_norm)
    sol_scale = np.median(sol_max)
    tensor_scale = [np.median(tensor_max[:, j]) for j in range(num_tensor_components)]
    
    # Build scale list
    scale = [source_scale] + tensor_scale + [sol_scale] + [lx, ly]
    
    print("\nComputed scales:")
    print(f"  Source scale: {source_scale:.6f}")
    print(f"  Tensor scales ({num_tensor_components} components): {[f'{s:.6f}' for s in tensor_scale]}")
    print(f"  Solution scale: {sol_scale:.6f}")
    print(f"  Domain: Lx={lx}, Ly={ly}")
    print(f"  Grid: nx={nx}, ny={ny}")
    
    return scale


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization scales from HDF5 training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Poisson dataset
  python compute_scales.py --datapath ./data/poisson_full --filename poissons_train_k1_5_32768.h5
  
  # Advection-Diffusion dataset
  python compute_scales.py --datapath ./data/advdiff_full --filename advdiff_train_adr0.2_1.0_32768.h5
  
  # Helmholtz dataset
  python compute_scales.py --datapath ./data/helmholtz_full --filename helmholtz_train_o1_10_32768.h5
        """
    )
    
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to directory containing the HDF5 training file"
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Name of the HDF5 training file (e.g., poissons_train_k1_5_32768.h5)"
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=128,
        help="Grid resolution in x-direction (default: 128)"
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=128,
        help="Grid resolution in y-direction (default: 128)"
    )
    parser.add_argument(
        "--lx",
        type=float,
        default=1.0,
        help="Domain length in x-direction (default: 1.0)"
    )
    parser.add_argument(
        "--ly",
        type=float,
        default=1.0,
        help="Domain length in y-direction (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for scales (default: auto-generated from input filename)"
    )
    
    args = parser.parse_args()
    
    # Compute scales
    scale = compute_scales(args.datapath, args.filename, args.nx, args.ny, args.lx, args.ly)
    
    # Determine output filename
    if args.output is None:
        # Auto-generate from input filename by replacing .h5 with _scales.npy
        base_name = args.filename.replace('.h5', '_scales.npy')
        output_path = os.path.join(args.datapath, base_name)
    else:
        output_path = os.path.join(args.datapath, args.output)
    
    # Save scales
    np.save(output_path, scale)
    print(f"\n✓ Scales saved to: {output_path}")
    print(f"  Scale array: {scale}")


if __name__ == "__main__":
    main()

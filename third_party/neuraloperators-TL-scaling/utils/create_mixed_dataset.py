#!/usr/bin/env python3
"""
Create a mixed dataset combining Poisson, Advection-Diffusion, and Helmholtz data.

This script combines three PDE datasets into a single mixed dataset for multi-task learning,
following the approach described in the paper (Figure 6a). Datasets are unified by zero-padding
the tensor coefficients to a common size (5 components).

As described in the paper:
"When pre-training a single model on this 'mixed' dataset, we simply use zero channels 
for those coefficients that do not exist when using examples from a specific operator."

Usage:
    python create_mixed_dataset.py \
        --poisson_path data/poisson/_train_k1_5_32k.h5 \
        --advdiff_path data/advdiff/_train_adr0p2_1_32k.h5 \
        --helmholtz_path data/helmholtz/_train_o1_10_32k.h5 \
        --output_path data/mixed/_train_mixed_32k.h5 \
        --samples_per_system 10922

The mixed dataset maintains standard HDF5 format:
- fields: (n, 2, nx, ny) - source and solution
- tensor: (n, 5) - unified coefficient vector with zero-padding
  
Tensor layout: [coef0, coef1, coef2, coef3, coef4]
- Poisson:    [k11, k12, k22,  0,  0]      # diffusion only
- AdvDiff:    [k11, k12, k22, vx, vy]      # diffusion + advection
- Helmholtz:  [k,  omega,  0,  0,  0]      # constant diffusion + wavenumber

When loaded by PDESolns, each tensor component is expanded to a spatial channel,
resulting in: 1 (source) + 5 (tensor) = 6 input channels for the model.
"""

import argparse
import h5py
import numpy as np
import os
from tqdm import tqdm


def load_dataset_info(filepath):
    """Get dataset dimensions and sample count."""
    with h5py.File(filepath, 'r') as f:
        fields = f['fields'][:]
        tensor = f['tensor'][:] if 'tensor' in f.keys() else None
        print(f"Loaded {filepath}")
        print(f"  Fields shape: {fields.shape}")
        print(f"  Tensor shape: {tensor.shape if tensor is not None else 'None'}")
    return fields, tensor


def create_mixed_dataset(poisson_path, advdiff_path, helmholtz_path,
                         output_path, samples_per_system=10922):
    """
    Create a mixed dataset from three PDE systems.
    
    Parameters
    ----------
    poisson_path : str
        Path to Poisson dataset HDF5 file
    advdiff_path : str
        Path to Advection-Diffusion dataset HDF5 file
    helmholtz_path : str
        Path to Helmholtz dataset HDF5 file
    output_path : str
        Path to save mixed dataset HDF5 file
    samples_per_system : int
        Number of samples to use from each system (default: 10922 for ~33k total)
    """
    
    print("="*60)
    print("Creating Mixed Dataset for Multi-Task Learning")
    print("="*60)
    
    # Load datasets
    print("\nLoading Poisson dataset...")
    poisson_fields, poisson_tensor = load_dataset_info(poisson_path)
    
    print("\nLoading Advection-Diffusion dataset...")
    advdiff_fields, advdiff_tensor = load_dataset_info(advdiff_path)
    
    print("\nLoading Helmholtz dataset...")
    helmholtz_fields, helmholtz_tensor = load_dataset_info(helmholtz_path)
    
    # Verify consistent spatial dimensions
    nx, ny = poisson_fields.shape[2], poisson_fields.shape[3]
    assert advdiff_fields.shape[2:] == (nx, ny), "Spatial dimensions must match"
    assert helmholtz_fields.shape[2:] == (nx, ny), "Spatial dimensions must match"
    
    # Determine total samples
    n_samples_per_system = min(
        samples_per_system,
        poisson_fields.shape[0],
        advdiff_fields.shape[0],
        helmholtz_fields.shape[0]
    )
    total_samples = n_samples_per_system * 3
    
    print(f"\nDataset Configuration:")
    print(f"  Samples per system: {n_samples_per_system}")
    print(f"  Total samples: {total_samples}")
    print(f"  Spatial resolution: {nx} x {ny}")
    print(f"  Common input channels: 6 (max across all systems)")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize mixed dataset arrays in standard format
    # fields: (n_samples, 2, nx, ny) - source and solution only (standard format)
    # tensor: (n_samples, 5) - unified coefficient vector with zero-padding
    #   Layout: [k11_or_k, k12_or_omega, k22, vx, vy]
    #   Poisson:    [k11, k12, k22,  0,  0]  # diffusion only
    #   AdvDiff:    [k11, k12, k22, vx, vy]  # diffusion + advection
    #   Helmholtz:  [k,   omega, 0,  0,  0]  # constant diffusion + wavenumber
    mixed_fields = np.zeros((total_samples, 2, nx, ny), dtype=np.float32)
    mixed_tensor = np.zeros((total_samples, 5), dtype=np.float32)
    mixed_labels = np.zeros(total_samples, dtype=np.int32)  # 0=Poisson, 1=AdvDiff, 2=Helmholtz
    
    print("\nProcessing datasets...")
    
    # Process Poisson: tensor [k11, k12, k22] -> [k11, k12, k22, 0, 0]
    print("  Processing Poisson samples...")
    for i in tqdm(range(n_samples_per_system), desc="Poisson"):
        idx = i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = poisson_fields[i, 0]  # source
        mixed_fields[idx, 1] = poisson_fields[i, 1]  # solution
        # Tensor: diffusion coefficients, zero-pad for advection
        mixed_tensor[idx, :3] = poisson_tensor[i, :3]  # k11, k12, k22
        mixed_tensor[idx, 3:5] = 0  # no advection
        # Label
        mixed_labels[idx] = 0
    
    # Process Advection-Diffusion: tensor [k11, k12, k22, vx, vy] -> [k11, k12, k22, vx, vy]
    print("  Processing Advection-Diffusion samples...")
    for i in tqdm(range(n_samples_per_system), desc="AdvDiff"):
        idx = n_samples_per_system + i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = advdiff_fields[i, 0]  # source
        mixed_fields[idx, 1] = advdiff_fields[i, 1]  # solution
        # Tensor: diffusion coefficients + advection velocities (already 5 components)
        mixed_tensor[idx, :5] = advdiff_tensor[i, :5]  # k11, k12, k22, vx, vy
        # Label
        mixed_labels[idx] = 1
    
    # Process Helmholtz: tensor [k_constant, omega] -> [k_constant, omega, 0, 0, 0]
    print("  Processing Helmholtz samples...")
    for i in tqdm(range(n_samples_per_system), desc="Helmholtz"):
        idx = 2 * n_samples_per_system + i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = helmholtz_fields[i, 0]  # source
        mixed_fields[idx, 1] = helmholtz_fields[i, 1]  # solution
        # Tensor: constant diffusion + wavenumber, zero-pad rest
        mixed_tensor[idx, 0] = helmholtz_tensor[i, 0]  # k_constant
        mixed_tensor[idx, 1] = helmholtz_tensor[i, 1]  # omega
        mixed_tensor[idx, 2:5] = 0  # no other coefficients
        # Label
        mixed_labels[idx] = 2
    
    # Shuffle the mixed dataset
    print("\nShuffling mixed dataset...")
    shuffle_idx = np.random.permutation(total_samples)
    mixed_fields = mixed_fields[shuffle_idx]
    mixed_tensor = mixed_tensor[shuffle_idx]
    mixed_labels = mixed_labels[shuffle_idx]
    
    # Save mixed dataset
    print(f"\nSaving mixed dataset to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('fields', data=mixed_fields, dtype=np.float32)
        f.create_dataset('tensor', data=mixed_tensor, dtype=np.float32)
        f.create_dataset('labels', data=mixed_labels, dtype=np.int32)
        
        # Add metadata
        f.attrs['description'] = 'Mixed dataset: Poisson + Advection-Diffusion + Helmholtz'
        f.attrs['n_poisson'] = n_samples_per_system
        f.attrs['n_advdiff'] = n_samples_per_system
        f.attrs['n_helmholtz'] = n_samples_per_system
        f.attrs['total_samples'] = total_samples
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
    
    print("\n✓ Mixed dataset created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Total samples: {total_samples}")
    print(f"  Distribution: Poisson={n_samples_per_system}, AdvDiff={n_samples_per_system}, Helmholtz={n_samples_per_system}")
    
    # Compute and save scales
    scales_path = output_path.replace('.h5', '_scales.npy')
    print(f"\nComputing normalization scales...")
    compute_mixed_scales(output_path, scales_path)
    
    return output_path


def compute_mixed_scales(data_path, output_path):
    """Compute normalization scales for mixed dataset."""
    with h5py.File(data_path, 'r') as f:
        fields = f['fields'][:]
        tensor = f['tensor'][:]
        nx, ny = fields.shape[2], fields.shape[3]
        lx, ly = 1.0, 1.0  # Domain size
    
    source_norm = []
    sol_max = []
    tensor_max = []
    
    for i in range(fields.shape[0]):
        # Source norm
        sn = np.linalg.norm(fields[i, 0]) * lx / nx * ly / ny
        source_norm.append(sn)
        
        # Solution max
        sol_max.append(np.max(np.abs(fields[i, 1])))  # Fixed: index 1, not 6
        
        # Tensor components (5 total)
        tensor_max.append([np.abs(tensor[i, j]) for j in range(5)])
    
    tensor_max = np.array(tensor_max)
    
    # Compute medians
    source_scale = np.median(source_norm)
    sol_scale = np.median(sol_max)
    tensor_scale = [np.median(tensor_max[:, j]) for j in range(5)]  # Fixed: 5 components
    
    # Build scale array: [source, t0, t1, t2, t3, t4, solution, lx, ly]
    scale = [source_scale] + tensor_scale + [sol_scale] + [lx, ly]
    
    # Save scales
    np.save(output_path, scale)
    
    print(f"✓ Scales saved to {output_path}")
    print(f"  Source scale: {source_scale:.6f}")
    print(f"  Tensor scales: {[f'{s:.6f}' for s in tensor_scale]}")
    print(f"  Solution scale: {sol_scale:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed dataset for multi-task PDE learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--poisson_path',
        type=str,
        required=True,
        help='Path to Poisson dataset HDF5 file'
    )
    parser.add_argument(
        '--advdiff_path',
        type=str,
        required=True,
        help='Path to Advection-Diffusion dataset HDF5 file'
    )
    parser.add_argument(
        '--helmholtz_path',
        type=str,
        required=True,
        help='Path to Helmholtz dataset HDF5 file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save mixed dataset HDF5 file'
    )
    parser.add_argument(
        '--samples_per_system',
        type=int,
        default=10922,
        help='Number of samples to use from each system (default: 10922 for ~33k total)'
    )
    
    args = parser.parse_args()
    
    create_mixed_dataset(
        args.poisson_path,
        args.advdiff_path,
        args.helmholtz_path,
        args.output_path,
        args.samples_per_system
    )


if __name__ == '__main__':
    main()

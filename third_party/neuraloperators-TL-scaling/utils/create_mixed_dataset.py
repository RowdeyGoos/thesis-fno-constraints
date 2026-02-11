#!/usr/bin/env python3
"""
Create a mixed dataset combining Poisson, Advection-Diffusion, and Helmholtz data.

This script combines three PDE datasets into a single mixed dataset for multi-task learning,
following the approach described in the paper (Figure 6a). Datasets are unified by zero-padding
the tensor coefficients to a common size (6 components).

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
- tensor: (n, 6) - unified coefficient vector with zero-padding
  
Tensor layout: [k11, k12, k22, vx, vy, omega]
- Poisson:    [k11, k12, k22,  0,  0,     0]  # diffusion only
- AdvDiff:    [k11, k12, k22, vx, vy,     0]  # diffusion + advection
- Helmholtz:  [k,   0,   k,   0,  0, omega]  # identity diffusion + wavenumber

This ensures consistent semantics across PDEs with NO channel overlap:
- Channels [0,1,2]: Always diffusion tensor (k11, k12, k22)
- Channels [3,4]: Always advection velocities (vx, vy)
- Channel 5: Always wavenumber (omega)

When loaded by PDESolns, each tensor component is expanded to a spatial channel,
resulting in: 1 (source) + 6 (tensor) = 7 input channels for the model.
"""

import argparse
import h5py
import numpy as np
import os


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
    print(f"  Input channels: 7 (1 source + 6 tensor components)")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize mixed dataset arrays in standard format
    # fields: (n_samples, 2, nx, ny) - source and solution only (standard format)
    # tensor: (n_samples, 6) - unified coefficient vector with zero-padding
    #   Layout: [k11, k12, k22, vx, vy, omega]
    #   Poisson:    [k11, k12, k22,  0,  0,     0]  # diffusion only
    #   AdvDiff:    [k11, k12, k22, vx, vy,     0]  # diffusion + advection
    #   Helmholtz:  [k,   0,   k,   0,  0, omega]  # identity diffusion + wavenumber
    mixed_fields = np.zeros((total_samples, 2, nx, ny), dtype=np.float32)
    mixed_tensor = np.zeros((total_samples, 6), dtype=np.float32)
    mixed_labels = np.zeros(total_samples, dtype=np.int32)  # 0=Poisson, 1=AdvDiff, 2=Helmholtz
    
    print("\nProcessing datasets...")
    
    # Process Poisson: tensor [k11, k12, k22] -> [k11, k12, k22, 0, 0, 0]
    print("  Processing Poisson samples...")
    for i in range(n_samples_per_system):
        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_samples_per_system}")
        idx = i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = poisson_fields[i, 0]  # source
        mixed_fields[idx, 1] = poisson_fields[i, 1]  # solution
        # Tensor: diffusion coefficients, zero-pad for advection and wavenumber
        mixed_tensor[idx, :3] = poisson_tensor[i, :3]  # k11, k12, k22
        mixed_tensor[idx, 3:6] = 0  # no advection, no wavenumber
        # Label
        mixed_labels[idx] = 0
    print(f"    Completed: {n_samples_per_system}/{n_samples_per_system}")
    
    # Process Advection-Diffusion: tensor [k11, k12, k22, vx, vy] -> [k11, k12, k22, vx, vy, 0]
    print("  Processing Advection-Diffusion samples...")
    for i in range(n_samples_per_system):
        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_samples_per_system}")
        idx = n_samples_per_system + i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = advdiff_fields[i, 0]  # source
        mixed_fields[idx, 1] = advdiff_fields[i, 1]  # solution
        # Tensor: diffusion coefficients + advection velocities, zero-pad wavenumber
        mixed_tensor[idx, :5] = advdiff_tensor[i, :5]  # k11, k12, k22, vx, vy
        mixed_tensor[idx, 5] = 0  # no wavenumber
        # Label
        mixed_labels[idx] = 1
    print(f"    Completed: {n_samples_per_system}/{n_samples_per_system}")
    
    # Process Helmholtz: tensor [k_constant, omega] -> [k, 0, k, 0, 0, omega]
    # Identity diffusion tensor: k11=k, k12=0, k22=k (consistent with Poisson/AdvDiff diffusion channels)
    # Wavenumber in dedicated channel 5, no advection in channels 3-4
    print("  Processing Helmholtz samples...")
    for i in range(n_samples_per_system):
        if i % 1000 == 0:
            print(f"    Progress: {i}/{n_samples_per_system}")
        idx = 2 * n_samples_per_system + i
        # Copy fields (source and solution)
        mixed_fields[idx, 0] = helmholtz_fields[i, 0]  # source
        mixed_fields[idx, 1] = helmholtz_fields[i, 1]  # solution
        # Tensor: identity diffusion tensor + wavenumber in dedicated channel
        mixed_tensor[idx, 0] = helmholtz_tensor[i, 0]  # k11 = k_constant
        mixed_tensor[idx, 1] = 0                        # k12 = 0 (identity tensor)
        mixed_tensor[idx, 2] = helmholtz_tensor[i, 0]  # k22 = k_constant (identity tensor)
        mixed_tensor[idx, 3] = 0                        # vx = 0 (no advection)
        mixed_tensor[idx, 4] = 0                        # vy = 0 (no advection)
        mixed_tensor[idx, 5] = helmholtz_tensor[i, 1]  # omega (wavenumber in dedicated channel)
        # Label
        mixed_labels[idx] = 2
    print(f"    Completed: {n_samples_per_system}/{n_samples_per_system}")
    
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


def compute_mixed_scales(data_path, output_path, eps=1e-12, lx=1.0, ly=1.0):
    """
    Compute scales.npy in the exact format expected by the provided PDESolns loader.

    Output layout (1D array):
      [source_ref, t0, t1, t2, t3, t4, t5, sol_ref, lx, ly]

    Where tensor layout is assumed:
      [k11, k12, k22, vx, vy, omega]

    Fixes:
    - Avoids structural zeros in mixed dataset forcing median(t3/t4/t5)=0.
    - Uses labels to compute medians only where coefficient "exists":
        label 0: Poisson
        label 1: AdvDiff
        label 2: Helmholtz
    - Matches loader's definition of f_norm = ||f|| * measure, where measure=(lx/nx)*(ly/ny).
    """
    with h5py.File(data_path, "r") as f:
        fields = f["fields"][:]   # (N, 2, nx, ny)
        tensor = f["tensor"][:]   # (N, 6)
        if "labels" not in f:
            raise ValueError("Mixed dataset must include 'labels' (0=Poisson, 1=AdvDiff, 2=Helmholtz).")
        labels = f["labels"][:]

    N, _, nx, ny = fields.shape
    measure = (lx / nx) * (ly / ny)

    # Source reference: median of ||f|| * measure (to match loader exactly)
    f = fields[:, 0]  # (N, nx, ny)
    f_norm = np.linalg.norm(f.reshape(N, -1), axis=1) * measure
    source_ref = float(np.median(f_norm))
    source_ref = max(source_ref, eps)

    # Solution reference (not used by loader in __getitem__, but included for consistency)
    u = fields[:, 1]
    sol_max = np.max(np.abs(u.reshape(N, -1)), axis=1)
    sol_ref = float(np.median(sol_max))
    sol_ref = max(sol_ref, eps)

    # Coefficient refs (medians of absolute values, masked to avoid structural zeros)
    # Layout: [k11, k12, k22, vx, vy, omega]
    coeff_ref = np.zeros(6, dtype=np.float64)

    # diffusion exists for all in your unified representation
    diff_mask = np.ones(N, dtype=bool)
    # advection exists only for AdvDiff
    adv_mask = (labels == 1)
    # omega exists only for Helmholtz
    omg_mask = (labels == 2)

    masks = {
        0: diff_mask,  # k11
        1: diff_mask,  # k12
        2: diff_mask,  # k22
        3: adv_mask,   # vx
        4: adv_mask,   # vy
        5: omg_mask,   # omega
    }

    for j in range(6):
        m = masks[j]
        vals = np.abs(tensor[m, j])

        # If mask is empty (shouldn't happen), fallback to non-zero values
        if vals.size == 0:
            vals = np.abs(tensor[:, j])
            vals = vals[vals > eps]

        # If still empty (all zeros), set to 1 to avoid divide-by-zero
        if vals.size == 0:
            coeff_ref[j] = 1.0
        else:
            coeff_ref[j] = float(max(np.median(vals), eps))

    # Assemble array in the format the loader expects:
    # [source, t0..t5, sol, lx, ly]
    scales = np.array(
        [source_ref, *coeff_ref.tolist(), sol_ref, float(lx), float(ly)],
        dtype=np.float32
    )

    np.save(output_path, scales)

    print(f"✓ Scales saved to {output_path}")
    print(f"  source_ref (median ||f||*measure): {scales[0]:.6g}")
    print(f"  coeff_ref: {[f'{v:.6g}' for v in scales[1:7]]}  (k11,k12,k22,vx,vy,omega)")
    print(f"  sol_ref: {scales[7]:.6g}")
    print(f"  domain: lx={scales[8]:.6g}, ly={scales[9]:.6g}")


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

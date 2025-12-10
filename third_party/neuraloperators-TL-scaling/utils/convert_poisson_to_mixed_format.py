#!/usr/bin/env python3
"""
Convert Poisson dataset to mixed-compatible format with 5-component tensors.

This script modifies Poisson datasets (which have 3-component tensors: [k11, k12, k22])
to have 5-component tensors with zero-padding: [k11, k12, k22, 0, 0].

This is necessary when fine-tuning a mixed-pretrained model (which expects 5 tensor
components) on a single-domain Poisson task.

Usage:
    python utils/convert_poisson_to_mixed_format.py \
        --input_path data/poisson/_train_k1_2.5_32k.h5 \
        --output_path data/poisson/_train_k1_2.5_32k_mixed_format.h5

    # Or convert in place (overwrites original):
    python utils/convert_poisson_to_mixed_format.py \
        --input_path data/poisson/_train_k1_2.5_32k.h5 \
        --in_place
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def convert_poisson_to_mixed_format(input_path, output_path, in_place=False):
    """
    Convert Poisson dataset from 3-component to 5-component tensor format.
    
    Args:
        input_path: Path to input HDF5 file with 3-component tensors
        output_path: Path to output HDF5 file with 5-component tensors
        in_place: If True, overwrites the input file directly
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input data
    print(f"Reading input file: {input_path}")
    with h5py.File(input_path, 'r') as f:
        fields = f['fields'][:]  # Shape: (n, 2, nx, ny)
        tensor = f['tensor'][:]  # Shape: (n, 3) for Poisson
        
        print(f"  Fields shape: {fields.shape}")
        print(f"  Tensor shape: {tensor.shape}")
        
        # Verify it's a 3-component tensor (Poisson format)
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3-component tensor (Poisson format), got shape {tensor.shape}. "
                f"This dataset may already be in mixed format or is not a Poisson dataset."
            )
    
    # Create 5-component tensor with zero-padding
    n_samples = tensor.shape[0]
    tensor_5 = np.zeros((n_samples, 5), dtype=tensor.dtype)
    tensor_5[:, :3] = tensor  # Copy [k11, k12, k22] to first 3 components
    # Components 3 and 4 remain zero
    
    print(f"\nConverted tensor shape: {tensor_5.shape}")
    print(f"  Original: [k11, k12, k22]")
    print(f"  New:      [k11, k12, k22, 0, 0]")
    
    # Determine output path
    if in_place:
        # Create backup first
        backup_path = input_path.with_suffix('.h5.backup')
        print(f"\nCreating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
        final_output_path = input_path
    else:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output file
    print(f"\nWriting output file: {final_output_path}")
    with h5py.File(final_output_path, 'w') as f:
        f.create_dataset('fields', data=fields, compression='gzip')
        f.create_dataset('tensor', data=tensor_5, compression='gzip')
    
    print(f"\n✓ Conversion complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {final_output_path}")
    if in_place:
        print(f"  Backup: {backup_path}")
    
    # Verify the output
    with h5py.File(final_output_path, 'r') as f:
        verify_tensor = f['tensor'][:]
        print(f"\nVerification:")
        print(f"  Output tensor shape: {verify_tensor.shape}")
        print(f"  First sample (original): [{tensor[0, 0]:.4f}, {tensor[0, 1]:.4f}, {tensor[0, 2]:.4f}]")
        print(f"  First sample (new):      [{verify_tensor[0, 0]:.4f}, {verify_tensor[0, 1]:.4f}, "
              f"{verify_tensor[0, 2]:.4f}, {verify_tensor[0, 3]:.4f}, {verify_tensor[0, 4]:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Poisson dataset to mixed-compatible format with 5-component tensors.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to new file
  python utils/convert_poisson_to_mixed_format.py \\
      --input_path data/poisson/_train_k1_2.5_32k.h5 \\
      --output_path data/poisson/_train_k1_2.5_32k_mixed_format.h5
  
  # Convert in place (overwrites original, creates backup)
  python utils/convert_poisson_to_mixed_format.py \\
      --input_path data/poisson/_train_k1_2.5_32k.h5 \\
      --in_place
  
  # Batch convert all k1_2.5 datasets
  for split in train val test; do
      python utils/convert_poisson_to_mixed_format.py \\
          --input_path data/poisson/_${split}_k1_2.5_32k.h5 \\
          --in_place
  done
"""
    )
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input Poisson HDF5 file with 3-component tensors'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to output HDF5 file with 5-component tensors (required unless --in_place)'
    )
    parser.add_argument(
        '--in_place',
        action='store_true',
        help='Overwrite input file directly (creates .backup file first)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.in_place and not args.output_path:
        parser.error("Either --output_path or --in_place must be specified")
    
    if args.in_place and args.output_path:
        parser.error("Cannot specify both --output_path and --in_place")
    
    try:
        convert_poisson_to_mixed_format(
            args.input_path,
            args.output_path,
            args.in_place
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

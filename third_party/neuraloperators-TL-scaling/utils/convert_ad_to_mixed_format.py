#!/usr/bin/env python3
"""
Convert Advection-Diffusion dataset to mixed-compatible format with 6-component tensors.

This script converts AdvDiff datasets (which have 5-component tensors: [k11, k12, k22, vx, vy])
to have 6-component tensors with zero-padding: [k11, k12, k22, vx, vy, 0].

This is necessary when fine-tuning a mixed-pretrained model (which expects 6 tensor
components including the omega wavenumber channel) on a single-domain AdvDiff task.
If a dataset also contains boundary condition maps under the optional `bc` key, they
are preserved verbatim.

Usage:
    # Check if conversion is needed (AdvDiff should already be 5 components)
    python utils/convert_ad_to_mixed_format.py \
        --input_path data/advdiff/_train_adr0p2_1_32k.h5 \
        --check_only

    # Convert to 6-component format
    python utils/convert_ad_to_mixed_format.py \
        --input_path data/advdiff/_train_adr0p2_1_32k.h5 \
        --output_path data/advdiff/_train_adr0p2_1_32k_mixed.h5

    # Or convert in place (overwrites original):
    python utils/convert_ad_to_mixed_format.py \
        --input_path data/advdiff/_train_adr0p2_1_32k.h5 \
        --in_place
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def convert_ad_to_mixed_format(input_path, output_path, in_place=False, check_only=False):
    """
    Convert AdvDiff dataset from 5-component to 6-component tensor format.
    
    Args:
        input_path: Path to input HDF5 file with 5-component tensors
        output_path: Path to output HDF5 file with 6-component tensors
        in_place: If True, overwrites the input file directly
        check_only: If True, only check format and exit
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input data
    print(f"Reading input file: {input_path}")
    with h5py.File(input_path, 'r') as f:
        fields = f['fields'][:]  # Shape: (n, 2, nx, ny)
        tensor = f['tensor'][:]  # Shape: (n, 5) for AdvDiff
        bc = f['bc'][:] if 'bc' in f else None
        file_attrs = dict(f.attrs)
        
        # Check compression
        fields_comp = f['fields'].compression
        tensor_comp = f['tensor'].compression
        bc_comp = f['bc'].compression if 'bc' in f else None
        
        print(f"  Fields shape: {fields.shape}")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  BC shape: {bc.shape if bc is not None else 'None'}")
        print(f"  Fields compression: {fields_comp if fields_comp else 'None (uncompressed)'}")
        print(f"  Tensor compression: {tensor_comp if tensor_comp else 'None (uncompressed)'}")
        if bc is not None:
            print(f"  BC compression: {bc_comp if bc_comp else 'None (uncompressed)'}")
        
        # Verify it's a 5-component tensor (AdvDiff format)
        if tensor.shape[1] == 5:
            print("\n✓ Dataset has 5-component tensors (AdvDiff format)")
            print("  Tensor layout: [k11, k12, k22, vx, vy]")
            print("  Will convert to 6-component format: [k11, k12, k22, vx, vy, 0]")
            
            if check_only:
                if fields_comp is None and tensor_comp is None:
                    print("\n✓ Dataset is uncompressed.")
                else:
                    print(f"\n⚠️  Dataset uses compression: {fields_comp or tensor_comp}")
                    print("    This will cause slowdown with multiple data workers on NFS!")
                print("\nConversion needed to add 6th component (omega = 0) for mixed model compatibility.")
                return
        elif tensor.shape[1] == 6:
            print("\n✓ Dataset already has 6-component tensors (mixed-compatible format)")
            print("  Tensor layout: [k11, k12, k22, vx, vy, omega]")
            
            if fields_comp is None and tensor_comp is None:
                print("\n✓ Dataset is already uncompressed and ready for fast loading!")
                if check_only:
                    print("\nNo conversion needed. Dataset is perfect as-is.")
                    return
                else:
                    print("\nℹ️  Dataset doesn't need conversion, but will copy if requested.")
            else:
                print(f"\n⚠️  Dataset uses compression: {fields_comp or tensor_comp}")
                print("    This will cause slowdown with multiple data workers on NFS!")
                if check_only:
                    print("\n✓ Conversion recommended to remove compression.")
                    return
                else:
                    print("    Will create uncompressed copy...")
        elif tensor.shape[1] == 3:
            raise ValueError(
                f"Dataset has 3-component tensors (Poisson format). "
                f"Use convert_poisson_to_mixed_format.py instead."
            )
        elif tensor.shape[1] == 2:
            raise ValueError(
                f"Dataset has 2-component tensors (Helmholtz format). "
                f"Use convert_helmholtz_to_mixed_format.py instead."
            )
        else:
            raise ValueError(
                f"Unexpected tensor shape {tensor.shape}. Expected 5 components for AdvDiff."
            )
    
    if check_only:
        return
    
    # Convert to 6-component format if needed
    if tensor.shape[1] == 5:
        # Create 6-component tensor with zero-padding for omega
        n_samples = tensor.shape[0]
        tensor_6 = np.zeros((n_samples, 6), dtype=tensor.dtype)
        tensor_6[:, :5] = tensor  # Copy [k11, k12, k22, vx, vy] to first 5 components
        tensor_6[:, 5] = 0  # omega = 0 (no wavenumber for AdvDiff)
        
        print(f"\nConverted tensor shape: {tensor_6.shape}")
        print(f"  Original: [k11, k12, k22, vx, vy]")
        print(f"  New:      [k11, k12, k22, vx, vy, 0]")
    else:
        # Already 6 components, just copy (maybe to remove compression)
        tensor_6 = tensor
        print(f"\nTensor shape: {tensor_6.shape} (already 6 components)")
        print(f"  Layout: [k11, k12, k22, vx, vy, omega]")
    
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
    
    # Write output file WITHOUT compression
    print(f"\nWriting output file: {final_output_path}")
    with h5py.File(final_output_path, 'w') as f:
        # NO compression - must match the format of mixed dataset for fast loading
        # Compression causes severe slowdown with multiple data workers on NFS
        f.create_dataset('fields', data=fields, dtype=np.float32)
        f.create_dataset('tensor', data=tensor_6, dtype=np.float32)
        if bc is not None:
            f.create_dataset('bc', data=bc, dtype=np.float32)
        for key, value in file_attrs.items():
            f.attrs[key] = value
        if bc is not None:
            f.attrs['has_bc'] = True
    
    print(f"\n✓ Conversion complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {final_output_path}")
    if in_place:
        print(f"  Backup: {backup_path}")
    
    # Verify the output
    with h5py.File(final_output_path, 'r') as f:
        verify_fields_comp = f['fields'].compression
        verify_tensor_comp = f['tensor'].compression
        verify_bc_comp = f['bc'].compression if 'bc' in f else None
        verify_tensor = f['tensor'][:]
        verify_bc = f['bc'][:] if 'bc' in f else None
        
        print(f"\nVerification:")
        print(f"  Output tensor shape: {verify_tensor.shape}")
        print(f"  Output bc shape: {verify_bc.shape if verify_bc is not None else 'None'}")
        print(f"  Fields compression: {verify_fields_comp if verify_fields_comp else 'None (uncompressed) ✓'}")
        print(f"  Tensor compression: {verify_tensor_comp if verify_tensor_comp else 'None (uncompressed) ✓'}")
        if verify_bc is not None:
            print(f"  BC compression: {verify_bc_comp if verify_bc_comp else 'None (uncompressed) ✓'}")
        if verify_tensor.shape[1] == 6:
            print(f"  First sample: k11={verify_tensor[0, 0]:.4f}, k12={verify_tensor[0, 1]:.4f}, "
                  f"k22={verify_tensor[0, 2]:.4f}, vx={verify_tensor[0, 3]:.4f}, vy={verify_tensor[0, 4]:.4f}, omega={verify_tensor[0, 5]:.4f}")
        else:
            print(f"  First sample: k11={verify_tensor[0, 0]:.4f}, k12={verify_tensor[0, 1]:.4f}, "
                  f"k22={verify_tensor[0, 2]:.4f}, vx={verify_tensor[0, 3]:.4f}, vy={verify_tensor[0, 4]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AdvDiff dataset to mixed-compatible format with 6-component tensors.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: AdvDiff datasets typically have 5-component tensors [k11, k12, k22, vx, vy].
This script converts them to 6-component format [k11, k12, k22, vx, vy, 0] with
zero-padding for the omega (wavenumber) channel to match the mixed model format.
If the dataset has a `bc` key, it is copied through unchanged.

Examples:
  # Check if conversion is needed
  python utils/convert_ad_to_mixed_format.py \\
      --input_path data/advdiff/_train_adr0p2_1_32k.h5 \\
      --check_only
  
  # Convert to new file (removes compression)
  python utils/convert_ad_to_mixed_format.py \\
      --input_path data/advdiff/_train_adr0p2_1_32k.h5 \\
      --output_path data/advdiff/_train_adr0p2_1_32k_mixed.h5
  
  # Convert in place (overwrites original, creates backup)
  python utils/convert_ad_to_mixed_format.py \\
      --input_path data/advdiff/_train_adr0p2_1_32k.h5 \\
      --in_place
  
  # Batch convert all AdvDiff datasets
  for split in train val test; do
      python utils/convert_ad_to_mixed_format.py \\
          --input_path data/advdiff/_${split}_adr0p2_1_32k.h5 \\
          --in_place
  done
"""
    )
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input AdvDiff HDF5 file; optional bc maps are preserved'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to output HDF5 file (required unless --in_place or --check_only)'
    )
    parser.add_argument(
        '--in_place',
        action='store_true',
        help='Overwrite input file directly (creates .backup file first)'
    )
    parser.add_argument(
        '--check_only',
        action='store_true',
        help='Only check format and compression, do not convert'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.check_only and not args.in_place and not args.output_path:
        parser.error("Either --output_path, --in_place, or --check_only must be specified")
    
    if args.in_place and args.output_path:
        parser.error("Cannot specify both --output_path and --in_place")
    
    try:
        convert_ad_to_mixed_format(
            args.input_path,
            args.output_path,
            args.in_place,
            args.check_only
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

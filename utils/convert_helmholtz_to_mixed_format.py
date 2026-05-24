#!/usr/bin/env python3
"""
Convert Helmholtz dataset to mixed-compatible format with 6-component tensors.

This script converts Helmholtz datasets (which have 2-component tensors: [k, omega])
to have 6-component tensors using the mixed dataset convention:

    [k11, k12, k22, vx, vy, omega] = [k, 0, k, 0, 0, omega]

This is necessary when fine-tuning a mixed-pretrained model (which expects 6 tensor
components) on a single-domain Helmholtz task. If a dataset also contains boundary
condition maps under the optional `bc` key, they are preserved verbatim.
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def convert_helmholtz_to_mixed_format(input_path, output_path, in_place=False):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading input file: {input_path}")
    with h5py.File(input_path, 'r') as f:
        fields = f['fields'][:]  # (n, 2, nx, ny)
        tensor = f['tensor'][:]  # (n, 2) for Helmholtz: [k, omega]
        bc = f['bc'][:] if 'bc' in f else None
        file_attrs = dict(f.attrs)

        print(f"  Fields shape: {fields.shape}")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  BC shape:     {bc.shape if bc is not None else 'None'}")

        if tensor.shape[1] != 2:
            raise ValueError(
                f"Expected 2-component tensor (Helmholtz format), got shape {tensor.shape}."
            )

    n_samples = tensor.shape[0]
    tensor_6 = np.zeros((n_samples, 6), dtype=tensor.dtype)

    # Mixed convention: [k11, k12, k22, vx, vy, omega]
    # Helmholtz:        [k,   0,   k,   0,  0, omega]
    tensor_6[:, 0] = tensor[:, 0]  # k11 = k
    tensor_6[:, 1] = 0             # k12 = 0
    tensor_6[:, 2] = tensor[:, 0]  # k22 = k
    tensor_6[:, 3] = 0             # vx  = 0
    tensor_6[:, 4] = 0             # vy  = 0
    tensor_6[:, 5] = tensor[:, 1]  # omega

    print(f"\nConverted tensor shape: {tensor_6.shape}")
    print("  Original: [k, omega]")
    print("  New:      [k, 0, k, 0, 0, omega]")

    if in_place:
        backup_path = input_path.with_suffix('.h5.backup')
        print(f"\nCreating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
        final_output_path = input_path
    else:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting output file: {final_output_path}")
    with h5py.File(final_output_path, 'w') as f:
        # Keep uncompressed for performant multi-worker loading on NFS
        f.create_dataset('fields', data=fields, dtype=np.float32)
        f.create_dataset('tensor', data=tensor_6, dtype=np.float32)
        if bc is not None:
            f.create_dataset('bc', data=bc, dtype=np.float32)
        for key, value in file_attrs.items():
            f.attrs[key] = value
        if bc is not None:
            f.attrs['has_bc'] = True

    print("\n✓ Conversion complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {final_output_path}")
    if in_place:
        print(f"  Backup: {backup_path}")

    with h5py.File(final_output_path, 'r') as f:
        verify_tensor = f['tensor'][:]
        verify_bc = f['bc'][:] if 'bc' in f else None
        print("\nVerification:")
        print(f"  Output tensor shape: {verify_tensor.shape}")
        print(f"  Output bc shape:     {verify_bc.shape if verify_bc is not None else 'None'}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Helmholtz dataset to mixed-compatible 6-component tensor format.'
    )
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input Helmholtz HDF5 file with [k, omega] tensors; optional bc maps are preserved')
    parser.add_argument('--output_path', type=str,
                        help='Path to output HDF5 file (required unless --in_place)')
    parser.add_argument('--in_place', action='store_true',
                        help='Overwrite input file directly (creates .backup file)')

    args = parser.parse_args()

    if not args.in_place and not args.output_path:
        parser.error("Either --output_path or --in_place must be specified")
    if args.in_place and args.output_path:
        parser.error("Cannot specify both --output_path and --in_place")

    try:
        convert_helmholtz_to_mixed_format(args.input_path, args.output_path, args.in_place)
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

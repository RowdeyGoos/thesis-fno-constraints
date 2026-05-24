#!/usr/bin/env python3
import argparse
import pathlib
import sys

import h5py
import numpy as np


def _is_binary_mask(mask: np.ndarray, tol: float = 1.0e-6) -> bool:
    return np.all((np.abs(mask - 0.0) <= tol) | (np.abs(mask - 1.0) <= tol))


def _check_single_file(path: pathlib.Path, boundary_tol: float) -> tuple[bool, dict]:
    with h5py.File(path, "r") as f:
        if 'fields' not in f:
            return False, {"error": "missing 'fields'"}
        if 'bc' not in f:
            return False, {"error": "missing 'bc'"}

        fields = f['fields'][:]
        bc = f['bc'][:]

        if fields.ndim != 4 or fields.shape[1] < 2:
            return False, {"error": f"fields must have shape (N,>=2,nx,ny), got {fields.shape}"}

        expected_bc_shape = (fields.shape[0], 2, fields.shape[2], fields.shape[3])
        if bc.shape != expected_bc_shape:
            return False, {"error": f"bc must have shape {expected_bc_shape}, got {bc.shape}"}

        if np.any(~np.isfinite(fields)):
            return False, {"error": "fields contains NaN/Inf"}
        if np.any(~np.isfinite(bc)):
            return False, {"error": "bc contains NaN/Inf"}
        if 'tensor' in f and np.any(~np.isfinite(f['tensor'][:])):
            return False, {"error": "tensor contains NaN/Inf"}

        g = bc[:, 0]
        m = bc[:, 1]
        u = fields[:, 1]

        if not _is_binary_mask(m):
            return False, {"error": "bc mask is not binary"}

        # g should be zero away from boundary mask.
        interior_leak = np.max(np.abs(g * (1.0 - m)))
        if interior_leak > boundary_tol:
            return False, {"error": f"bc value map leaks into interior (max={interior_leak:.3e})"}

        # Boundary condition match error on boundary.
        denom = np.sum(m) + 1.0e-12
        bc_err = np.sum(np.abs((u - g) * m)) / denom

        return True, {
            "n": int(fields.shape[0]),
            "nx": int(fields.shape[2]),
            "ny": int(fields.shape[3]),
            "bc_mae": float(bc_err),
            "max_bc_interior_leak": float(interior_leak),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="BC dataset sanity checker")
    parser.add_argument(
        "--input",
        required=True,
        help="HDF5 file path or directory containing .h5 files",
    )
    parser.add_argument(
        "--glob",
        default="*.h5",
        help="Glob pattern used when --input is a directory (default: *.h5)",
    )
    parser.add_argument(
        "--boundary_tol",
        type=float,
        default=1.0e-5,
        help="Tolerance for boundary-only checks",
    )
    parser.add_argument(
        "--max_bc_mae",
        type=float,
        default=1.0e-4,
        help="Maximum allowed MAE on boundary between solution and g",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob(args.glob))
    else:
        files = [input_path]

    if not files:
        print(f"No files found for input={input_path} glob={args.glob}")
        return 2

    any_fail = False
    for path in files:
        ok, info = _check_single_file(path, boundary_tol=args.boundary_tol)
        if not ok:
            any_fail = True
            print(f"[FAIL] {path}: {info['error']}")
            continue

        bc_mae = info["bc_mae"]
        if bc_mae > args.max_bc_mae:
            any_fail = True
            print(
                f"[FAIL] {path}: bc_mae={bc_mae:.3e} exceeds max_bc_mae={args.max_bc_mae:.3e}"
            )
            continue

        print(
            f"[OK] {path}: n={info['n']} shape=({info['nx']},{info['ny']}) "
            f"bc_mae={info['bc_mae']:.3e} leak={info['max_bc_interior_leak']:.3e}"
        )

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())

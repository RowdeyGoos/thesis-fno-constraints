#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import pathlib
import sys

import h5py
import numpy as np


def infer_dataset_kind(path: pathlib.Path, tensor_dim: int, has_labels: bool) -> str:
    text = str(path).lower()
    if has_labels:
        return 'mixed'
    if tensor_dim == 3:
        return 'poisson'
    if tensor_dim == 5:
        return 'advdiff'
    if tensor_dim == 2:
        return 'helmholtz'
    if tensor_dim == 6:
        if 'poisson' in text:
            return 'poisson_mixed_compatible'
        if 'advection-diffusion' in text or 'advdiff' in text or '/ad/' in text:
            return 'advdiff_mixed_compatible'
        if 'helmholtz' in text:
            return 'helmholtz_mixed_compatible'
        return 'mixed_compatible'
    return 'unknown'


def expected_tensor_dim(kind: str) -> int | None:
    mapping = {
        'poisson': 3,
        'advdiff': 5,
        'helmholtz': 2,
        'mixed': 6,
        'poisson_mixed_compatible': 6,
        'advdiff_mixed_compatible': 6,
        'helmholtz_mixed_compatible': 6,
        'mixed_compatible': 6,
    }
    return mapping.get(kind)


def iter_batches(n_samples: int, batch_size: int):
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)


def _is_binary_mask(mask: np.ndarray, tol: float = 1.0e-6) -> bool:
    return np.all((np.abs(mask) <= tol) | (np.abs(mask - 1.0) <= tol))


def _all_finite(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))


def _maybe_load_scales(path: pathlib.Path):
    if not path.exists():
        return None, 'missing scales file'
    try:
        arr = np.load(path)
    except Exception as exc:
        return None, f'failed to load scales: {exc}'
    return arr, None


def check_scales(scales: np.ndarray, tensor_dim: int) -> tuple[bool, str | None]:
    expected_len = tensor_dim + 4
    if scales.ndim != 1:
        return False, f'scales must be 1D, got shape {scales.shape}'
    if len(scales) != expected_len:
        return False, f'scales length must be {expected_len}, got {len(scales)}'
    if not np.all(np.isfinite(scales)):
        return False, 'scales contains NaN/Inf'
    if scales[0] <= 0:
        return False, 'source scale must be > 0'
    if scales[-3] <= 0:
        return False, 'solution scale must be > 0'
    if scales[-2] <= 0 or scales[-1] <= 0:
        return False, 'domain sizes lx/ly must be > 0'
    return True, None


def validate_tensor_semantics(kind: str, tensor_batch: np.ndarray, tol: float = 1.0e-6) -> tuple[bool, str | None]:
    if tensor_batch.size == 0:
        return True, None

    if kind == 'poisson_mixed_compatible':
        if np.max(np.abs(tensor_batch[:, 3:6])) > tol:
            return False, 'poisson mixed-compatible tensor should have zero channels 3:6'
    elif kind == 'advdiff_mixed_compatible':
        if np.max(np.abs(tensor_batch[:, 5])) > tol:
            return False, 'advdiff mixed-compatible tensor should have zero omega channel'
    elif kind == 'helmholtz_mixed_compatible':
        if np.max(np.abs(tensor_batch[:, 1])) > tol:
            return False, 'helmholtz mixed-compatible tensor should have k12 == 0'
        if np.max(np.abs(tensor_batch[:, 3:5])) > tol:
            return False, 'helmholtz mixed-compatible tensor should have zero advection channels'
        if np.max(np.abs(tensor_batch[:, 0] - tensor_batch[:, 2])) > tol:
            return False, 'helmholtz mixed-compatible tensor should satisfy k11 == k22'
    return True, None


def validate_mixed_labels(labels: np.ndarray, tensor_batch: np.ndarray, tol: float = 1.0e-6) -> tuple[bool, str | None]:
    valid = np.isin(labels, [0, 1, 2])
    if not np.all(valid):
        bad = np.unique(labels[~valid])
        return False, f'mixed labels must be in {{0,1,2}}, got invalid labels {bad.tolist()}'

    poisson_mask = labels == 0
    if np.any(poisson_mask):
        if np.max(np.abs(tensor_batch[poisson_mask, 3:6])) > tol:
            return False, 'mixed poisson rows should have zero advection and omega channels'

    advdiff_mask = labels == 1
    if np.any(advdiff_mask):
        if np.max(np.abs(tensor_batch[advdiff_mask, 5])) > tol:
            return False, 'mixed advdiff rows should have zero omega channel'

    helm_mask = labels == 2
    if np.any(helm_mask):
        helm = tensor_batch[helm_mask]
        if np.max(np.abs(helm[:, 1])) > tol:
            return False, 'mixed helmholtz rows should have k12 == 0'
        if np.max(np.abs(helm[:, 3:5])) > tol:
            return False, 'mixed helmholtz rows should have zero advection channels'
        if np.max(np.abs(helm[:, 0] - helm[:, 2])) > tol:
            return False, 'mixed helmholtz rows should satisfy k11 == k22'

    return True, None


def summarize_range(arr: np.ndarray) -> tuple[float, float]:
    return float(np.min(arr)), float(np.max(arr))


def _check_single_file(
    path: pathlib.Path,
    batch_size: int,
    boundary_tol: float,
    require_scales: bool,
    scales_override: pathlib.Path | None,
) -> tuple[bool, dict]:
    with h5py.File(path, 'r') as f:
        if 'fields' not in f:
            return False, {'error': "missing 'fields'"}
        if 'tensor' not in f:
            return False, {'error': "missing 'tensor'"}

        fields_ds = f['fields']
        tensor_ds = f['tensor']
        other_ds = f['other'] if 'other' in f else None
        bc_ds = f['bc'] if 'bc' in f else None
        labels_ds = f['labels'] if 'labels' in f else None

        if fields_ds.ndim != 4 or fields_ds.shape[1] != 2:
            return False, {'error': f"fields must have shape (N,2,nx,ny), got {fields_ds.shape}"}
        if tensor_ds.ndim != 2:
            return False, {'error': f"tensor must have shape (N,tensor_dim), got {tensor_ds.shape}"}
        if tensor_ds.shape[0] != fields_ds.shape[0]:
            return False, {'error': 'tensor and fields sample counts differ'}
        if other_ds is not None and other_ds.shape[0] != fields_ds.shape[0]:
            return False, {'error': 'other and fields sample counts differ'}
        if labels_ds is not None and labels_ds.shape[0] != fields_ds.shape[0]:
            return False, {'error': 'labels and fields sample counts differ'}

        expected_bc_shape = (fields_ds.shape[0], 2, fields_ds.shape[2], fields_ds.shape[3])
        if bc_ds is not None and bc_ds.shape != expected_bc_shape:
            return False, {'error': f"bc must have shape {expected_bc_shape}, got {bc_ds.shape}"}

        n_samples = int(fields_ds.shape[0])
        nx = int(fields_ds.shape[2])
        ny = int(fields_ds.shape[3])
        tensor_dim = int(tensor_ds.shape[1])
        kind = infer_dataset_kind(path, tensor_dim, labels_ds is not None)
        expected_dim = expected_tensor_dim(kind)
        if expected_dim is not None and tensor_dim != expected_dim:
            return False, {'error': f'inferred kind={kind} expects tensor_dim={expected_dim}, got {tensor_dim}'}

        source_min = np.inf
        source_max = -np.inf
        solution_min = np.inf
        solution_max = -np.inf
        tensor_abs_max = np.zeros(tensor_dim, dtype=np.float64)
        other_abs_max = None
        label_counts = {0: 0, 1: 0, 2: 0}
        max_bc_interior_leak = 0.0
        bc_mae_numer = 0.0
        bc_mae_denom = 0.0

        for start, end in iter_batches(n_samples, batch_size):
            fields = fields_ds[start:end]
            tensor = tensor_ds[start:end]

            if not _all_finite(fields):
                return False, {'error': f'fields contains NaN/Inf in batch {start}:{end}'}
            if not _all_finite(tensor):
                return False, {'error': f'tensor contains NaN/Inf in batch {start}:{end}'}

            source_min = min(source_min, float(np.min(fields[:, 0])))
            source_max = max(source_max, float(np.max(fields[:, 0])))
            solution_min = min(solution_min, float(np.min(fields[:, 1])))
            solution_max = max(solution_max, float(np.max(fields[:, 1])))
            tensor_abs_max = np.maximum(tensor_abs_max, np.max(np.abs(tensor), axis=0))

            ok, err = validate_tensor_semantics(kind, tensor)
            if not ok:
                return False, {'error': f'{err} (batch {start}:{end})'}

            if other_ds is not None:
                other = other_ds[start:end]
                if not _all_finite(other):
                    return False, {'error': f'other contains NaN/Inf in batch {start}:{end}'}
                batch_abs_max = np.max(np.abs(other), axis=0)
                if other_abs_max is None:
                    other_abs_max = batch_abs_max.astype(np.float64)
                else:
                    other_abs_max = np.maximum(other_abs_max, batch_abs_max)

            if labels_ds is not None:
                labels = labels_ds[start:end]
                if not _all_finite(labels):
                    return False, {'error': f'labels contains NaN/Inf in batch {start}:{end}'}
                ok, err = validate_mixed_labels(labels.astype(np.int64), tensor)
                if not ok:
                    return False, {'error': f'{err} (batch {start}:{end})'}
                unique, counts = np.unique(labels.astype(np.int64), return_counts=True)
                for label, count in zip(unique.tolist(), counts.tolist()):
                    if label in label_counts:
                        label_counts[label] += count

            if bc_ds is not None:
                bc = bc_ds[start:end]
                if not _all_finite(bc):
                    return False, {'error': f'bc contains NaN/Inf in batch {start}:{end}'}
                g = bc[:, 0]
                m = bc[:, 1]
                u = fields[:, 1]
                if not _is_binary_mask(m):
                    return False, {'error': f'bc mask is not binary in batch {start}:{end}'}
                interior_leak = float(np.max(np.abs(g * (1.0 - m))))
                max_bc_interior_leak = max(max_bc_interior_leak, interior_leak)
                if interior_leak > boundary_tol:
                    return False, {'error': f'bc value map leaks into interior (max={interior_leak:.3e})'}
                bc_mae_numer += float(np.sum(np.abs((u - g) * m)))
                bc_mae_denom += float(np.sum(m))

        scales_path = scales_override or path.with_name(path.stem + '_scales.npy')
        scales = None
        if require_scales or scales_path.exists():
            scales, scales_error = _maybe_load_scales(scales_path)
            if scales_error is not None:
                return False, {'error': f'{scales_path}: {scales_error}'}
            if scales is None:
                return False, {'error': f'{scales_path}: missing scales contents'}
            ok, err = check_scales(scales, tensor_dim)
            if not ok:
                return False, {'error': f'{scales_path}: {err}'}

        info = {
            'kind': kind,
            'n': n_samples,
            'nx': nx,
            'ny': ny,
            'tensor_dim': tensor_dim,
            'has_other': other_ds is not None,
            'has_bc': bc_ds is not None,
            'has_labels': labels_ds is not None,
            'source_range': (source_min, source_max),
            'solution_range': (solution_min, solution_max),
            'tensor_abs_max': tensor_abs_max.tolist(),
            'scales_path': str(scales_path) if scales is not None else None,
        }
        if other_abs_max is not None:
            info['other_abs_max'] = other_abs_max.tolist()
        if labels_ds is not None:
            info['label_counts'] = label_counts
        if bc_ds is not None:
            denom = bc_mae_denom + 1.0e-12
            info['bc_mae'] = bc_mae_numer / denom
            info['max_bc_interior_leak'] = max_bc_interior_leak

        return True, info


def main() -> int:
    parser = argparse.ArgumentParser(description='General HDF5 dataset sanity checker')
    parser.add_argument('--input', required=True, help='HDF5 file path or directory containing .h5 files')
    parser.add_argument('--glob', default='*.h5', help='Glob pattern used when --input is a directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used when scanning HDF5 datasets')
    parser.add_argument('--boundary_tol', type=float, default=1.0e-5, help='Tolerance for BC interior leakage')
    parser.add_argument('--require_scales', action='store_true', help='Fail if companion *_scales.npy is missing')
    parser.add_argument('--scales', type=str, default=None, help='Explicit scales .npy to validate against a single file')
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if args.batch_size <= 0:
        print('--batch_size must be >= 1')
        return 2

    if input_path.is_dir():
        files = sorted(input_path.glob(args.glob))
    else:
        files = [input_path]

    if not files:
        print(f'No files found for input={input_path} glob={args.glob}')
        return 2

    if args.scales and len(files) != 1:
        print('--scales may only be used when validating a single HDF5 file')
        return 2

    any_fail = False
    scales_override = pathlib.Path(args.scales) if args.scales else None

    for path in files:
        ok, info = _check_single_file(
            path=path,
            batch_size=args.batch_size,
            boundary_tol=args.boundary_tol,
            require_scales=args.require_scales,
            scales_override=scales_override,
        )
        if not ok:
            any_fail = True
            print(f"[FAIL] {path}: {info['error']}")
            continue

        src_lo, src_hi = info['source_range']
        sol_lo, sol_hi = info['solution_range']
        summary = (
            f"[OK] {path}: kind={info['kind']} n={info['n']} "
            f"shape=({info['nx']},{info['ny']}) tensor_dim={info['tensor_dim']} "
            f"source=[{src_lo:.3e},{src_hi:.3e}] sol=[{sol_lo:.3e},{sol_hi:.3e}]"
        )
        if info['scales_path'] is not None:
            summary += f" scales={pathlib.Path(info['scales_path']).name}"
        print(summary)

        if info['has_other']:
            print(f"      other_abs_max={info['other_abs_max']}")
        print(f"      tensor_abs_max={info['tensor_abs_max']}")
        if info['has_labels']:
            print(f"      label_counts={info['label_counts']}")
        if info['has_bc']:
            print(
                f"      bc_mae={info['bc_mae']:.3e} "
                f"leak={info['max_bc_interior_leak']:.3e}"
            )

    return 1 if any_fail else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from collections import Counter
from typing import Callable, Dict, Iterable, List, Tuple

import h5py
import numpy as np


def infer_dataset_kind(path: pathlib.Path, tensor_dim: int, has_labels: bool) -> str:
    text = str(path).lower()
    if has_labels:
        return "mixed"
    if tensor_dim == 3:
        return "poisson"
    if tensor_dim == 5:
        return "advdiff"
    if tensor_dim == 2:
        return "helmholtz"
    if tensor_dim == 6:
        if "poisson" in text:
            return "poisson_mixed_compatible"
        if "advection-diffusion" in text or "advdiff" in text or "/ad/" in text:
            return "advdiff_mixed_compatible"
        if "helmholtz" in text:
            return "helmholtz_mixed_compatible"
        return "mixed_compatible"
    return "unknown"


def iter_batches(n_samples: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_samples, batch_size):
        yield start, min(start + batch_size, n_samples)


def resolve_primary_scalar(kind: str) -> Tuple[str | None, Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None]:
    if kind == "helmholtz":
        return "omega", lambda tensor, other: tensor[:, 1]
    if kind == "helmholtz_mixed_compatible":
        return "omega", lambda tensor, other: tensor[:, 5]
    if kind == "advdiff":
        return "adv_ratio", lambda tensor, other: other[:, 0]
    if kind == "advdiff_mixed_compatible":
        return "adv_ratio", lambda tensor, other: other[:, 0]
    if kind == "poisson":
        return "k11", lambda tensor, other: tensor[:, 0]
    if kind == "poisson_mixed_compatible":
        return "k11", lambda tensor, other: tensor[:, 0]
    return None, None


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def format_summary(name: str, stats: Dict[str, float]) -> str:
    parts = [
        f"{name}: min={stats['min']:.6g}",
        f"max={stats['max']:.6g}",
        f"mean={stats['mean']:.6g}",
        f"std={stats['std']:.6g}",
    ]
    if "median" in stats:
        parts.append(f"median={stats['median']:.6g}")
    return " ".join(parts)


def maybe_print_discrete_hist(label: str, values: np.ndarray, max_unique: int) -> None:
    unique = np.unique(values)
    if len(unique) == 0 or len(unique) > max_unique:
        return
    counts = Counter(np.asarray(values).tolist())
    ordered = ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))
    print(f"      {label} counts: {ordered}")


def format_overlap(intersection: int, union: int) -> str:
    jaccard = intersection / max(union, 1)
    return f"{intersection:>5}/{jaccard:0.3f}"


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.min = np.inf
        self.max = -np.inf
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.min = min(self.min, float(np.min(arr)))
        self.max = max(self.max, float(np.max(arr)))
        self.sum += float(np.sum(arr))
        self.sum_sq += float(np.sum(np.square(arr)))

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return {
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": float(np.sqrt(var)),
        }


def sample_fingerprint(
    source: np.ndarray,
    tensor: np.ndarray,
    other: np.ndarray | None,
    bc: np.ndarray | None,
    labels: np.ndarray | None,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(source).view(np.uint8))
    digest.update(np.ascontiguousarray(tensor).view(np.uint8))
    if other is not None:
        digest.update(np.ascontiguousarray(other).view(np.uint8))
    if bc is not None:
        digest.update(np.ascontiguousarray(bc).view(np.uint8))
    if labels is not None:
        digest.update(np.ascontiguousarray(labels).view(np.uint8))
    return digest.hexdigest()


def analyze_duplicates(
    fields_ds,
    tensor_ds,
    other_ds,
    bc_ds,
    labels_ds,
    batch_size: int,
) -> Tuple[int, int]:
    counts: Counter[str] = Counter()
    n_samples = fields_ds.shape[0]

    for start, end in iter_batches(n_samples, batch_size):
        fields = fields_ds[start:end]
        tensor = tensor_ds[start:end]
        other = other_ds[start:end] if other_ds is not None else None
        bc = bc_ds[start:end] if bc_ds is not None else None
        labels = labels_ds[start:end] if labels_ds is not None else None

        for idx in range(end - start):
            fp = sample_fingerprint(
                source=fields[idx, 0],
                tensor=tensor[idx],
                other=None if other is None else other[idx],
                bc=None if bc is None else bc[idx],
                labels=None if labels is None else labels[idx],
            )
            counts[fp] += 1

    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    duplicate_samples = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_groups, duplicate_samples


def analyze_file(path: pathlib.Path, batch_size: int, max_unique: int, skip_duplicates: bool) -> int:
    with h5py.File(path, "r") as f:
        if "fields" not in f or "tensor" not in f:
            print(f"[FAIL] {path}: requires at least 'fields' and 'tensor'")
            return 1

        fields_ds = f["fields"]
        tensor_ds = f["tensor"]
        other_ds = f["other"] if "other" in f else None
        bc_ds = f["bc"] if "bc" in f else None
        labels_ds = f["labels"] if "labels" in f else None

        kind = infer_dataset_kind(path, tensor_ds.shape[1], labels_ds is not None)
        n_samples = int(fields_ds.shape[0])
        nx = int(fields_ds.shape[2])
        ny = int(fields_ds.shape[3])

        print(f"{path}")
        print(f"  kind={kind} n={n_samples} shape=({nx},{ny}) tensor_dim={tensor_ds.shape[1]}")

        scalar_name, scalar_getter = resolve_primary_scalar(kind)
        source_stats = RunningStats()
        solution_stats = RunningStats()
        tensor_stats = [RunningStats() for _ in range(tensor_ds.shape[1])]
        other_stats = [RunningStats() for _ in range(other_ds.shape[1])] if other_ds is not None else []
        label_counts: Counter[int] = Counter()
        scalar_chunks: List[np.ndarray] = []

        for start, end in iter_batches(n_samples, batch_size):
            fields = fields_ds[start:end]
            tensor = tensor_ds[start:end]
            other = other_ds[start:end] if other_ds is not None else None
            labels = labels_ds[start:end] if labels_ds is not None else None

            source_stats.update(fields[:, 0])
            solution_stats.update(fields[:, 1])

            for tidx in range(tensor.shape[1]):
                tensor_stats[tidx].update(tensor[:, tidx])

            if other is not None:
                for oidx in range(other.shape[1]):
                    other_stats[oidx].update(other[:, oidx])

            if labels is not None:
                label_counts.update(labels.astype(np.int64).tolist())

            if (
                scalar_name is not None
                and scalar_getter is not None
                and not (kind in {"advdiff", "advdiff_mixed_compatible"} and other is None)
            ):
                scalar_chunks.append(np.asarray(scalar_getter(tensor, other), dtype=np.float64))

        print(f"    {format_summary('source', source_stats.summary())}")
        print(f"    {format_summary('solution', solution_stats.summary())}")

        for tidx, stats in enumerate(tensor_stats):
            print(f"    {format_summary(f'tensor[{tidx}]', stats.summary())}")

        if scalar_chunks:
            scalar_values = np.concatenate(scalar_chunks, axis=0)
            print(f"    {format_summary(scalar_name, summarize_array(scalar_values))}")
            maybe_print_discrete_hist(scalar_name, scalar_values, max_unique=max_unique)

        for oidx, stats in enumerate(other_stats):
            print(f"    {format_summary(f'other[{oidx}]', stats.summary())}")

        if label_counts:
            ordered = ", ".join(f"{key}:{label_counts[key]}" for key in sorted(label_counts))
            print(f"    label counts: {ordered}")

        if not skip_duplicates:
            duplicate_groups, duplicate_samples = analyze_duplicates(
                fields_ds=fields_ds,
                tensor_ds=tensor_ds,
                other_ds=other_ds,
                bc_ds=bc_ds,
                labels_ds=labels_ds,
                batch_size=batch_size,
            )
            print(
                "    duplicate summary: "
                f"duplicate_groups={duplicate_groups} duplicate_samples={duplicate_samples}"
            )

    return 0


def choose_subset_indices(n_samples: int, subsample: int, seed: int) -> np.ndarray:
    if subsample <= 1:
        return np.arange(n_samples, dtype=np.int64)
    target_n = int(n_samples / subsample)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=target_n, replace=False))


def compare_subsets(
    path: pathlib.Path,
    seeds: List[int],
    subsample: int,
    max_unique: int,
) -> int:
    with h5py.File(path, "r") as f:
        if "fields" not in f or "tensor" not in f:
            print(f"[FAIL] {path}: requires at least 'fields' and 'tensor'")
            return 1

        tensor = f["tensor"][:]
        other = f["other"][:] if "other" in f else None
        labels = f["labels"][:] if "labels" in f else None
        kind = infer_dataset_kind(path, tensor.shape[1], labels is not None)
        n_samples = int(f["fields"].shape[0])

    scalar_name, scalar_getter = resolve_primary_scalar(kind)
    scalar_values = None
    if (
        scalar_name is not None
        and scalar_getter is not None
        and not (kind in {"advdiff", "advdiff_mixed_compatible"} and other is None)
    ):
        scalar_values = np.asarray(scalar_getter(tensor, other))

    subsets: Dict[int, np.ndarray] = {seed: choose_subset_indices(n_samples, subsample, seed) for seed in seeds}

    print(f"{path}")
    print(f"  subset comparison: n={n_samples} subsample={subsample} subset_size={len(next(iter(subsets.values())))}")

    for seed in seeds:
        indices = subsets[seed]
        print(f"    seed={seed}: first_indices={indices[:10].tolist()}")
        if scalar_values is not None:
            subset_scalar = scalar_values[indices]
            print(f"      {format_summary(scalar_name, summarize_array(subset_scalar))}")
            maybe_print_discrete_hist(scalar_name, subset_scalar, max_unique=max_unique)

    print("    overlap matrix: cells are intersection_count / jaccard")
    header = "      seed".ljust(12) + "".join(f"{seed:>14}" for seed in seeds)
    print(header)
    for seed_a in seeds:
        row = f"{seed_a}".ljust(12)
        set_a = set(subsets[seed_a].tolist())
        for seed_b in seeds:
            set_b = set(subsets[seed_b].tolist())
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            row += f"{format_overlap(inter, union):>14}"
        print(f"      {row.rstrip()}")

    return 0


def resolve_inputs(input_path: pathlib.Path, glob_pattern: str) -> List[pathlib.Path]:
    if input_path.is_dir():
        return sorted(input_path.glob(glob_pattern))
    return [input_path]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dataset parameter distributions, exact duplicate samples, "
            "and seeded random-subset overlap."
        )
    )
    parser.add_argument("--input", required=True, help="HDF5 file path or directory")
    parser.add_argument("--glob", default="*.h5", help="Glob pattern when --input is a directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for duplicate hashing")
    parser.add_argument(
        "--mode",
        choices=["summary", "subset", "both"],
        default="both",
        help="Whether to print dataset summaries, subset overlap analysis, or both",
    )
    parser.add_argument(
        "--subset_file",
        type=str,
        default=None,
        help="Single HDF5 file to use for subset analysis. Defaults to each matched file.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=8,
        help="Subsample factor used by training (e.g. 8 => 4K subset from 32K train set)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seed values used for subset comparison",
    )
    parser.add_argument(
        "--max_unique",
        type=int,
        default=32,
        help="Print exact discrete counts when a scalar has at most this many unique values",
    )
    parser.add_argument(
        "--skip_duplicates",
        action="store_true",
        help="Skip exact duplicate hashing if you only want distribution summaries",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if args.batch_size <= 0:
        print("--batch_size must be >= 1")
        return 2
    if args.subsample <= 0:
        print("--subsample must be >= 1")
        return 2

    files = resolve_inputs(input_path, args.glob)
    if not files:
        print(f"No files found for input={input_path} glob={args.glob}")
        return 2

    status = 0

    if args.mode in ["summary", "both"]:
        for path in files:
            status |= analyze_file(
                path=path,
                batch_size=args.batch_size,
                max_unique=args.max_unique,
                skip_duplicates=args.skip_duplicates,
            )

    if args.mode in ["subset", "both"]:
        if args.subset_file is not None:
            subset_targets = [pathlib.Path(args.subset_file)]
        else:
            subset_targets = [path for path in files if "_train" in path.name]
            if not subset_targets:
                subset_targets = files
        for path in subset_targets:
            status |= compare_subsets(
                path=path,
                seeds=args.seeds,
                subsample=args.subsample,
                max_unique=args.max_unique,
            )

    return status


if __name__ == "__main__":
    sys.exit(main())

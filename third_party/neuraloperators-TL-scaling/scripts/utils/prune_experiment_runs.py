#!/usr/bin/env python3
"""Prune old run directories under an experiments/expts tree.

This script keeps only the newest N run directories inside each setup folder:

    experiments/expts/<setup>/<run_dir>

Safety features:
- Dry-run by default. Pass --apply to actually delete anything.
- Refuses to operate unless the target directory is named "expts", unless
  --allow-non-expts-root is provided.
- Only deletes direct child directories of each setup directory.
- Never deletes files, symlinks, or the setup directories themselves.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class RunEntry:
    path: Path
    mtime_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the latest N run directories inside each setup folder "
            "under experiments/expts."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="experiments/expts",
        help="Path to the expts directory. Default: experiments/expts",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of newest runs to keep per setup. Default: 3",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete old run directories. Default is dry-run.",
    )
    parser.add_argument(
        "--allow-non-expts-root",
        action="store_true",
        help='Allow operating on a directory whose basename is not "expts".',
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print kept directories as well as pruned ones.",
    )
    return parser.parse_args()


def validate_root(root_dir: Path, allow_non_expts_root: bool) -> Path:
    root = root_dir.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Error: root directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Error: root path is not a directory: {root}")
    if root.is_symlink():
        raise SystemExit(f"Error: refusing to operate on symlink root: {root}")
    if root.name != "expts" and not allow_non_expts_root:
        raise SystemExit(
            "Error: refusing to operate because target directory basename is "
            f'"{root.name}" instead of "expts". '
            "Pass --allow-non-expts-root if this is intentional."
        )
    return root


def iter_real_dirs(parent: Path) -> Iterable[Path]:
    for entry in sorted(parent.iterdir(), key=lambda p: p.name):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            yield entry


def collect_run_dirs(setup_dir: Path) -> List[RunEntry]:
    run_entries: List[RunEntry] = []
    for run_dir in iter_real_dirs(setup_dir):
        stat = run_dir.stat()
        run_entries.append(RunEntry(path=run_dir, mtime_ns=stat.st_mtime_ns))
    run_entries.sort(key=lambda item: (item.mtime_ns, item.path.name), reverse=True)
    return run_entries


def prune_setup(
    setup_dir: Path,
    keep: int,
    apply: bool,
    verbose: bool,
) -> tuple[int, int]:
    runs = collect_run_dirs(setup_dir)
    if len(runs) <= keep:
        if verbose:
            print(f"[skip] {setup_dir.name}: {len(runs)} run(s), nothing to prune")
        return 0, len(runs)

    kept = runs[:keep]
    pruned = runs[keep:]

    print(
        f"[setup] {setup_dir.name}: keeping {len(kept)} newest run(s), "
        f"{len(pruned)} older run(s) {'to delete' if apply else 'would be deleted'}"
    )

    if verbose:
        for entry in kept:
            print(f"  [keep] {entry.path.name}")

    for entry in pruned:
        print(f"  [{'delete' if apply else 'dry-run'}] {entry.path.name}")
        if apply:
            shutil.rmtree(entry.path)

    return len(pruned), len(kept)


def main() -> int:
    args = parse_args()
    if args.keep < 0:
        raise SystemExit("Error: --keep must be non-negative.")

    root = validate_root(Path(args.root_dir), args.allow_non_expts_root)
    setup_dirs = list(iter_real_dirs(root))

    if not setup_dirs:
        print(f"No setup directories found under {root}")
        return 0

    print(f"Scanning {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Keep per setup: {args.keep}")

    total_pruned = 0
    total_kept = 0
    setups_with_runs = 0

    for setup_dir in setup_dirs:
        pruned, kept = prune_setup(
            setup_dir=setup_dir,
            keep=args.keep,
            apply=args.apply,
            verbose=args.verbose,
        )
        total_pruned += pruned
        total_kept += kept
        if pruned or kept:
            setups_with_runs += 1

    print("")
    print("Summary:")
    print(f"  Setups scanned: {len(setup_dirs)}")
    print(f"  Setups with run directories: {setups_with_runs}")
    print(f"  Runs retained: {total_kept}")
    print(
        f"  Runs {'deleted' if args.apply else 'marked for deletion'}: {total_pruned}"
    )

    if not args.apply:
        print("")
        print("Dry-run only. Re-run with --apply to actually delete old runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

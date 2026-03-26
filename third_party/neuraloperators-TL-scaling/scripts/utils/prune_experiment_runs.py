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
- Only deletes top-level .out/.err files when they can be matched
  conservatively to a pruned run's Slurm job id and optional task id.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


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
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory containing top-level .out/.err files to prune alongside "
            "runs. Defaults to the parent experiments/ directory when root_dir "
            "is experiments/expts."
        ),
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


def iter_real_files(parent: Path) -> Iterable[Path]:
    for entry in sorted(parent.iterdir(), key=lambda p: p.name):
        if entry.is_symlink():
            continue
        if entry.is_file():
            yield entry


def collect_run_dirs(setup_dir: Path) -> List[RunEntry]:
    run_entries: List[RunEntry] = []
    for run_dir in iter_real_dirs(setup_dir):
        stat = run_dir.stat()
        run_entries.append(RunEntry(path=run_dir, mtime_ns=stat.st_mtime_ns))
    run_entries.sort(key=lambda item: (item.mtime_ns, item.path.name), reverse=True)
    return run_entries


def resolve_log_dir(root: Path, requested_log_dir: Optional[str]) -> Optional[Path]:
    if requested_log_dir is not None:
        log_dir = Path(requested_log_dir).expanduser().resolve()
        if not log_dir.exists():
            raise SystemExit(f"Error: log directory does not exist: {log_dir}")
        if not log_dir.is_dir():
            raise SystemExit(f"Error: log path is not a directory: {log_dir}")
        if log_dir.is_symlink():
            raise SystemExit(f"Error: refusing to operate on symlink log dir: {log_dir}")
        return log_dir

    parent = root.parent
    if parent.name == "experiments" and parent.exists() and parent.is_dir() and not parent.is_symlink():
        return parent
    return None


def parse_run_job_suffix(run_name: str) -> Optional[Tuple[str, Optional[str]]]:
    parts = run_name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[-2], parts[-1]
    if parts[-1].isdigit():
        return parts[-1], None
    return None


def collect_matching_logs(log_dir: Optional[Path], run_name: str) -> List[Path]:
    if log_dir is None:
        return []

    parsed = parse_run_job_suffix(run_name)
    if parsed is None:
        return []

    jobid, taskid = parsed
    suffixes: Set[str] = set()
    if taskid is None:
        suffixes.add(f"-{jobid}")
    else:
        suffixes.add(f"-{jobid}-{taskid}")
        suffixes.add(f"-{jobid}_{taskid}")
        if taskid == "0":
            # Some single-run Slurm jobs write logs as %x-%j.{out,err} while the
            # run directory still ends in "-<jobid>-0".
            suffixes.add(f"-{jobid}")

    matches: List[Path] = []
    for log_file in iter_real_files(log_dir):
        if log_file.suffix not in {".out", ".err"}:
            continue
        if any(log_file.stem.endswith(suffix) for suffix in suffixes):
            matches.append(log_file)
    return matches


def prune_setup(
    setup_dir: Path,
    log_dir: Optional[Path],
    keep: int,
    apply: bool,
    verbose: bool,
) -> tuple[int, int, int]:
    runs = collect_run_dirs(setup_dir)
    if len(runs) <= keep:
        if verbose:
            print(f"[skip] {setup_dir.name}: {len(runs)} run(s), nothing to prune")
        return 0, len(runs), 0

    kept = runs[:keep]
    pruned = runs[keep:]
    seen_logs: Set[Path] = set()
    logs_to_prune: List[Path] = []

    print(
        f"[setup] {setup_dir.name}: keeping {len(kept)} newest run(s), "
        f"{len(pruned)} older run(s) {'to delete' if apply else 'would be deleted'}"
    )

    if verbose:
        for entry in kept:
            print(f"  [keep] {entry.path.name}")

    for entry in pruned:
        print(f"  [{'delete' if apply else 'dry-run'}] {entry.path.name}")
        for log_file in collect_matching_logs(log_dir, entry.path.name):
            if log_file in seen_logs:
                continue
            seen_logs.add(log_file)
            logs_to_prune.append(log_file)
            print(f"  [{'delete-log' if apply else 'dry-run-log'}] {log_file.name}")
        if apply:
            shutil.rmtree(entry.path)

    if apply:
        for log_file in logs_to_prune:
            log_file.unlink()

    return len(pruned), len(kept), len(logs_to_prune)


def main() -> int:
    args = parse_args()
    if args.keep < 0:
        raise SystemExit("Error: --keep must be non-negative.")

    root = validate_root(Path(args.root_dir), args.allow_non_expts_root)
    log_dir = resolve_log_dir(root, args.log_dir)
    setup_dirs = list(iter_real_dirs(root))

    if not setup_dirs:
        print(f"No setup directories found under {root}")
        return 0

    print(f"Scanning {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Keep per setup: {args.keep}")
    print(f"Log directory: {log_dir if log_dir is not None else 'disabled'}")

    total_pruned = 0
    total_kept = 0
    total_logs_pruned = 0
    setups_with_runs = 0

    for setup_dir in setup_dirs:
        pruned, kept, logs_pruned = prune_setup(
            setup_dir=setup_dir,
            log_dir=log_dir,
            keep=args.keep,
            apply=args.apply,
            verbose=args.verbose,
        )
        total_pruned += pruned
        total_kept += kept
        total_logs_pruned += logs_pruned
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
    print(
        f"  Log files {'deleted' if args.apply else 'marked for deletion'}: "
        f"{total_logs_pruned}"
    )

    if not args.apply:
        print("")
        print("Dry-run only. Re-run with --apply to actually delete old runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

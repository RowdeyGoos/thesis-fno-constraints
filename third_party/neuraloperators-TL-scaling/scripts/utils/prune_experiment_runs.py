#!/usr/bin/env python3
"""Prune old run directories under an experiments/expts tree.

This script keeps only the newest N *successful* run directories inside each
setup folder:

    experiments/expts/<setup>/<run_dir>

It also prunes confirmed failed runs, even if they are recent.

Status detection is conservative:
- Prefer a `COMPLETED` / `FAILED` sentinel file inside the run directory.
- Otherwise, use `sacct` when available.
- Otherwise, fall back to matched top-level Slurm `.out` / `.err` files.
- Runs that are still active or cannot be classified confidently are kept.

Safety features:
- Dry-run by default. Pass --apply to actually delete anything.
- Refuses to operate unless the target directory is named "expts", unless
  --allow-non-expts-root is provided.
- Only deletes direct child directories of each setup directory.
- Never deletes files, symlinks, or the setup directories themselves.
- Only deletes top-level `.out` / `.err` files when they can be matched
  conservatively to a pruned run's Slurm job id and optional task id.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"
ACTIVE_STATUS = "active"
UNKNOWN_STATUS = "unknown"

ACTIVE_SLURM_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "STAGE_OUT",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_HOLD",
    "REQUEUE_FED",
    "SIGNALING",
}

FAILED_SLURM_STATE_PREFIXES = (
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
)

SUCCESS_LOG_MARKERS = (
    "completed successfully.",
    "info:root:done",
    "wandb: waiting for w&b process to finish... (success).",
)

FAILURE_LOG_MARKERS = (
    "failed with exit code",
    "traceback (most recent call last):",
    "slurmstepd: error",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "oom-kill",
    "out of memory",
    "memoryerror",
    "std::bad_alloc",
)


@dataclass(frozen=True)
class RunEntry:
    path: Path
    mtime_ns: int
    jobid: Optional[str]
    taskid: Optional[str]


@dataclass(frozen=True)
class ClassifiedRun:
    entry: RunEntry
    status: str
    status_source: str
    status_detail: str
    matched_logs: Tuple[Path, ...]


@dataclass(frozen=True)
class SetupPruneStats:
    successful_kept: int
    active_kept: int
    unknown_kept: int
    old_success_pruned: int
    failed_pruned: int
    logs_pruned: int
    had_runs: bool
    had_actions: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the latest N successful run directories inside each setup "
            "folder under experiments/expts, while pruning confirmed failed runs."
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
        help="Number of newest successful runs to keep per setup. Default: 3",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete old/failed run directories. Default is dry-run.",
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
    parser.add_argument(
        "--status-source",
        choices=("auto", "sacct", "logs"),
        default="auto",
        help=(
            "How to classify runs: auto (prefer sacct, fallback to logs), "
            "sacct, or logs. Default: auto"
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


def parse_run_job_suffix(run_name: str) -> Tuple[Optional[str], Optional[str]]:
    parts = run_name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[-2], parts[-1]
    if parts[-1].isdigit():
        return parts[-1], None
    return None, None


def collect_run_dirs(setup_dir: Path) -> List[RunEntry]:
    run_entries: List[RunEntry] = []
    for run_dir in iter_real_dirs(setup_dir):
        stat = run_dir.stat()
        jobid, taskid = parse_run_job_suffix(run_dir.name)
        run_entries.append(
            RunEntry(
                path=run_dir,
                mtime_ns=stat.st_mtime_ns,
                jobid=jobid,
                taskid=taskid,
            )
        )
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


def collect_matching_logs(log_dir: Optional[Path], run_entry: RunEntry) -> List[Path]:
    if log_dir is None or run_entry.jobid is None:
        return []

    suffixes: Set[str] = set()
    if run_entry.taskid is None:
        suffixes.add(f"-{run_entry.jobid}")
    else:
        suffixes.add(f"-{run_entry.jobid}-{run_entry.taskid}")
        suffixes.add(f"-{run_entry.jobid}_{run_entry.taskid}")
        if run_entry.taskid == "0":
            # Some single-run jobs produce %x-%j.{out,err} while the run
            # directory itself ends in "-<jobid>-0".
            suffixes.add(f"-{run_entry.jobid}")

    matches: List[Path] = []
    for log_file in iter_real_files(log_dir):
        if log_file.suffix not in {".out", ".err"}:
            continue
        if any(log_file.stem.endswith(suffix) for suffix in suffixes):
            matches.append(log_file)
    return matches


def read_text_tail(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="ignore")


def line_looks_like_done(line: str) -> bool:
    stripped = line.strip()
    if stripped == "DONE":
        return True
    upper = stripped.upper()
    return (
        upper.endswith(":DONE")
        or upper.endswith(": DONE")
        or upper.endswith(" - INFO - DONE")
    )


def classify_logs(log_files: Sequence[Path]) -> Tuple[str, str, str]:
    if not log_files:
        return UNKNOWN_STATUS, "logs", "no matched log files"

    for log_file in log_files:
        text = read_text_tail(log_file)
        text_lower = text.lower()
        last_success_pos = -1
        last_success_detail = ""
        last_failure_pos = -1
        last_failure_detail = ""

        for marker in FAILURE_LOG_MARKERS:
            pos = text_lower.rfind(marker)
            if pos > last_failure_pos:
                last_failure_pos = pos
                last_failure_detail = f"{log_file.name}: matched '{marker}'"

        for marker in SUCCESS_LOG_MARKERS:
            pos = text_lower.rfind(marker)
            if pos > last_success_pos:
                last_success_pos = pos
                last_success_detail = f"{log_file.name}: matched '{marker}'"

        for idx, line in enumerate(text.splitlines()):
            if line_looks_like_done(line):
                line_pos = idx
                if line_pos > last_success_pos:
                    last_success_pos = line_pos
                    last_success_detail = f"{log_file.name}: matched DONE line"

        if last_success_pos >= 0 and last_success_pos >= last_failure_pos:
            return SUCCESS_STATUS, "logs", last_success_detail
        if last_failure_pos >= 0:
            return FAILED_STATUS, "logs", last_failure_detail

    return UNKNOWN_STATUS, "logs", "matched logs exist but no final success/failure marker found"


def normalize_slurm_state(state: str) -> str:
    if not state:
        return ""
    return state.strip().upper().split()[0]


def classify_slurm_state(state: str, exit_code: str) -> str:
    normalized_state = normalize_slurm_state(state)
    normalized_exit = (exit_code or "").strip()

    if normalized_state.startswith("COMPLETED") and normalized_exit.startswith("0:0"):
        return SUCCESS_STATUS
    if normalized_state in ACTIVE_SLURM_STATES:
        return ACTIVE_STATUS
    if any(normalized_state.startswith(prefix) for prefix in FAILED_SLURM_STATE_PREFIXES):
        return FAILED_STATUS
    if normalized_exit and not normalized_exit.startswith("0:0"):
        return FAILED_STATUS
    return UNKNOWN_STATUS


class SacctResolver:
    def __init__(self, mode: str):
        self.mode = mode
        self.command = shutil.which("sacct")
        self.enabled = mode in {"auto", "sacct"} and self.command is not None
        self.query_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}
        self.disabled_reason: Optional[str] = None

        if mode == "sacct" and self.command is None:
            raise SystemExit("Error: --status-source=sacct requested but `sacct` is not available.")

    def summary(self) -> str:
        if self.enabled and self.command is not None:
            return "sacct"
        if self.mode == "sacct":
            return "sacct-unavailable"
        if self.disabled_reason:
            return f"logs-fallback ({self.disabled_reason})"
        return "logs-fallback"

    def _load_job(self, jobid: str) -> Dict[str, Tuple[str, str]]:
        if jobid in self.query_cache:
            return self.query_cache[jobid]

        if not self.enabled or self.command is None:
            self.query_cache[jobid] = {}
            return self.query_cache[jobid]

        result = subprocess.run(
            [
                self.command,
                "-n",
                "-P",
                "-j",
                jobid,
                "--format",
                "JobIDRaw,State,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            message = (result.stderr or result.stdout or f"return code {result.returncode}").strip()
            if self.mode == "sacct":
                raise SystemExit(f"Error: sacct query failed for job {jobid}: {message}")
            self.enabled = False
            self.disabled_reason = f"sacct query failed: {message}"
            self.query_cache[jobid] = {}
            return self.query_cache[jobid]

        records: Dict[str, Tuple[str, str]] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            jobid_raw, state, exit_code = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not jobid_raw:
                continue
            records[jobid_raw] = (state, exit_code)

        self.query_cache[jobid] = records
        return records

    def classify(self, run_entry: RunEntry) -> Optional[Tuple[str, str, str]]:
        if not self.enabled or run_entry.jobid is None:
            return None

        records = self._load_job(run_entry.jobid)
        if not records:
            return None

        candidate_keys: List[str] = []
        if run_entry.taskid is not None:
            candidate_keys.append(f"{run_entry.jobid}_{run_entry.taskid}")
            candidate_keys.append(f"{run_entry.jobid}.{run_entry.taskid}")
            if run_entry.taskid == "0":
                candidate_keys.append(run_entry.jobid)
        else:
            candidate_keys.append(run_entry.jobid)

        for key in candidate_keys:
            if key in records:
                state, exit_code = records[key]
                status = classify_slurm_state(state, exit_code)
                return status, "sacct", f"{key}: state={state} exit={exit_code}"

        return None


def classify_run(
    run_entry: RunEntry,
    log_dir: Optional[Path],
    sacct_resolver: SacctResolver,
) -> ClassifiedRun:
    matched_logs = tuple(collect_matching_logs(log_dir, run_entry))

    completed_sentinel = run_entry.path / "COMPLETED"
    failed_sentinel = run_entry.path / "FAILED"
    if completed_sentinel.is_file():
        return ClassifiedRun(
            entry=run_entry,
            status=SUCCESS_STATUS,
            status_source="sentinel",
            status_detail="COMPLETED sentinel present",
            matched_logs=matched_logs,
        )
    if failed_sentinel.is_file():
        return ClassifiedRun(
            entry=run_entry,
            status=FAILED_STATUS,
            status_source="sentinel",
            status_detail="FAILED sentinel present",
            matched_logs=matched_logs,
        )

    sacct_status = sacct_resolver.classify(run_entry)
    if sacct_status is not None:
        status, source, detail = sacct_status
        if status != UNKNOWN_STATUS:
            return ClassifiedRun(
                entry=run_entry,
                status=status,
                status_source=source,
                status_detail=detail,
                matched_logs=matched_logs,
            )

    log_status, log_source, log_detail = classify_logs(matched_logs)
    return ClassifiedRun(
        entry=run_entry,
        status=log_status,
        status_source=log_source,
        status_detail=log_detail,
        matched_logs=matched_logs,
    )


def format_run_status(run: ClassifiedRun) -> str:
    return f"{run.status} via {run.status_source} ({run.status_detail})"


def prune_setup(
    setup_dir: Path,
    log_dir: Optional[Path],
    sacct_resolver: SacctResolver,
    keep: int,
    apply: bool,
    verbose: bool,
) -> SetupPruneStats:
    run_entries = collect_run_dirs(setup_dir)
    if not run_entries:
        if verbose:
            print(f"[skip] {setup_dir.name}: no run directories found")
        return SetupPruneStats(0, 0, 0, 0, 0, 0, False, False)

    classified_runs = [
        classify_run(run_entry=entry, log_dir=log_dir, sacct_resolver=sacct_resolver)
        for entry in run_entries
    ]

    successful_runs = [run for run in classified_runs if run.status == SUCCESS_STATUS]
    failed_runs = [run for run in classified_runs if run.status == FAILED_STATUS]
    active_runs = [run for run in classified_runs if run.status == ACTIVE_STATUS]
    unknown_runs = [run for run in classified_runs if run.status == UNKNOWN_STATUS]

    successful_runs.sort(
        key=lambda run: (run.entry.mtime_ns, run.entry.path.name),
        reverse=True,
    )
    kept_successful_runs = successful_runs[:keep]
    pruned_old_successful_runs = successful_runs[keep:]
    pruned_failed_runs = list(failed_runs)

    had_actions = bool(pruned_old_successful_runs or pruned_failed_runs)
    if not had_actions and not verbose:
        return SetupPruneStats(
            successful_kept=len(kept_successful_runs),
            active_kept=len(active_runs),
            unknown_kept=len(unknown_runs),
            old_success_pruned=0,
            failed_pruned=0,
            logs_pruned=0,
            had_runs=True,
            had_actions=False,
        )

    print(
        f"[setup] {setup_dir.name}: success={len(successful_runs)} "
        f"active={len(active_runs)} unknown={len(unknown_runs)} failed={len(failed_runs)}; "
        f"keeping {len(kept_successful_runs)} successful run(s), "
        f"{len(pruned_old_successful_runs)} old successful run(s) "
        f"and {len(pruned_failed_runs)} failed run(s) "
        f"{'to delete' if apply else 'would be deleted'}"
    )

    if verbose:
        for run in kept_successful_runs:
            print(f"  [keep-success] {run.entry.path.name} :: {format_run_status(run)}")
        for run in active_runs:
            print(f"  [keep-active] {run.entry.path.name} :: {format_run_status(run)}")
        for run in unknown_runs:
            print(f"  [keep-unknown] {run.entry.path.name} :: {format_run_status(run)}")

    logs_to_prune: List[Path] = []
    seen_logs: Set[Path] = set()

    for run in pruned_old_successful_runs:
        print(f"  [{'delete-old-success' if apply else 'dry-run-old-success'}] {run.entry.path.name}")
        if verbose:
            print(f"    {format_run_status(run)}")
        for log_file in run.matched_logs:
            if log_file in seen_logs:
                continue
            seen_logs.add(log_file)
            logs_to_prune.append(log_file)
            print(f"  [{'delete-log' if apply else 'dry-run-log'}] {log_file.name}")

    for run in pruned_failed_runs:
        print(f"  [{'delete-failed' if apply else 'dry-run-failed'}] {run.entry.path.name}")
        if verbose:
            print(f"    {format_run_status(run)}")
        for log_file in run.matched_logs:
            if log_file in seen_logs:
                continue
            seen_logs.add(log_file)
            logs_to_prune.append(log_file)
            print(f"  [{'delete-log' if apply else 'dry-run-log'}] {log_file.name}")

    if apply:
        for run in pruned_old_successful_runs:
            shutil.rmtree(run.entry.path)
        for run in pruned_failed_runs:
            shutil.rmtree(run.entry.path)
        for log_file in logs_to_prune:
            log_file.unlink()

    return SetupPruneStats(
        successful_kept=len(kept_successful_runs),
        active_kept=len(active_runs),
        unknown_kept=len(unknown_runs),
        old_success_pruned=len(pruned_old_successful_runs),
        failed_pruned=len(pruned_failed_runs),
        logs_pruned=len(logs_to_prune),
        had_runs=True,
        had_actions=had_actions,
    )


def main() -> int:
    args = parse_args()
    if args.keep < 0:
        raise SystemExit("Error: --keep must be non-negative.")

    root = validate_root(Path(args.root_dir), args.allow_non_expts_root)
    log_dir = resolve_log_dir(root, args.log_dir)
    setup_dirs = list(iter_real_dirs(root))
    sacct_resolver = SacctResolver(args.status_source)

    if not setup_dirs:
        print(f"No setup directories found under {root}")
        return 0

    print(f"Scanning {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Keep successful runs per setup: {args.keep}")
    print(f"Status detection: {sacct_resolver.summary()}")
    print(f"Log directory: {log_dir if log_dir is not None else 'disabled'}")

    total_successful_kept = 0
    total_active_kept = 0
    total_unknown_kept = 0
    total_old_success_pruned = 0
    total_failed_pruned = 0
    total_logs_pruned = 0
    setups_with_runs = 0
    setups_with_actions = 0

    for setup_dir in setup_dirs:
        stats = prune_setup(
            setup_dir=setup_dir,
            log_dir=log_dir,
            sacct_resolver=sacct_resolver,
            keep=args.keep,
            apply=args.apply,
            verbose=args.verbose,
        )
        total_successful_kept += stats.successful_kept
        total_active_kept += stats.active_kept
        total_unknown_kept += stats.unknown_kept
        total_old_success_pruned += stats.old_success_pruned
        total_failed_pruned += stats.failed_pruned
        total_logs_pruned += stats.logs_pruned
        if stats.had_runs:
            setups_with_runs += 1
        if stats.had_actions:
            setups_with_actions += 1

    print("")
    print("Summary:")
    print(f"  Setups scanned: {len(setup_dirs)}")
    print(f"  Setups with run directories: {setups_with_runs}")
    print(f"  Setups with pruning actions: {setups_with_actions}")
    print(f"  Successful runs retained: {total_successful_kept}")
    print(f"  Active runs retained: {total_active_kept}")
    print(f"  Unknown-status runs retained: {total_unknown_kept}")
    print(
        f"  Old successful runs "
        f"{'deleted' if args.apply else 'marked for deletion'}: {total_old_success_pruned}"
    )
    print(
        f"  Failed runs "
        f"{'deleted' if args.apply else 'marked for deletion'}: {total_failed_pruned}"
    )
    print(
        f"  Log files "
        f"{'deleted' if args.apply else 'marked for deletion'}: {total_logs_pruned}"
    )

    if not args.apply:
        print("")
        print("Dry-run only. Re-run with --apply to actually delete pruned runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

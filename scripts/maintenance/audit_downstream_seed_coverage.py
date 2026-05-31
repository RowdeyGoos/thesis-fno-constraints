#!/usr/bin/env python3
"""Audit downstream experiment seed coverage under experiments/expts.

This script compares the expected downstream setup names from the operator
config files against the run directories currently present under
experiments/expts/<setup>/.

It reports which setups have all expected seeds, which are incomplete, and
which do not have any run directory yet. The default config set includes the
baseline operator YAMLs plus the mixed-constraint YAMLs, so constrained IID and
OOD setups are audited together. Report rows are tagged as ``IID`` or ``OOD``.

Seed detection rules:
- Prefer explicit `seedN` markers in the run directory name.
- Otherwise, if the run name ends with `-<jobid>-<taskid>`, infer the seed as
  `taskid % 3`, matching the downstream Slurm array convention used here.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_CONFIG_PATHS = (
    "config/operators_poisson.yaml",
    "config/operators_ad.yaml",
    "config/operators_helmholtz.yaml",
    "config/operators_poisson_mixed_constraints.yaml",
    "config/operators_ad_mixed_constraints.yaml",
    "config/operators_helmholtz_mixed_constraints.yaml",
)
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+):")
EXPLICIT_SEED_RE = re.compile(r"(?:^|-)seed(?P<seed>\d+)(?:-|$)")
POISSON_LOWER_BOUND_RE = re.compile(r"^poisson-k(?P<lower>\d+(?:p\d+)?)(?:_|-)")
ADVDIFF_LOWER_BOUND_RE = re.compile(r"^ad-adr(?P<lower>\d+(?:p\d+)?)(?:_|-)")
HELMHOLTZ_LOWER_BOUND_RE = re.compile(r"^helm-o(?P<lower>\d+(?:p\d+)?)(?:_|-)")


@dataclass(frozen=True)
class RunSeedInfo:
    run_name: str
    seed: Optional[int]
    source: str


@dataclass
class SetupCoverage:
    setup_name: str
    expected_seeds: Set[int]
    runs: List[RunSeedInfo]

    @property
    def present_seeds(self) -> Set[int]:
        return {run.seed for run in self.runs if run.seed is not None}

    @property
    def missing_seeds(self) -> Set[int]:
        return self.expected_seeds - self.present_seeds

    @property
    def explicit_seed_runs(self) -> int:
        return sum(1 for run in self.runs if run.source == "explicit")

    @property
    def inferred_seed_runs(self) -> int:
        return sum(1 for run in self.runs if run.source == "inferred_from_task_id")

    @property
    def unknown_seed_runs(self) -> int:
        return sum(1 for run in self.runs if run.seed is None)

    @property
    def is_complete(self) -> bool:
        return self.missing_seeds == set()

    @property
    def has_any_runs(self) -> bool:
        return bool(self.runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check which downstream finetune/scratch setups have all expected "
            "seeds present under experiments/expts, including constrained and "
            "OOD setups from the default operator YAMLs."
        )
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="experiments/expts",
        help="Path to experiments/expts. Default: experiments/expts",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIG_PATHS),
        help=(
            "Config files to scan for expected downstream setup names. "
            "Defaults to baseline plus mixed-constraint operator YAMLs."
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Expected seed values for each setup. Default: 0 1 2",
    )
    parser.add_argument(
        "--show-complete",
        action="store_true",
        help="Also print setups that already have all expected seeds.",
    )
    parser.add_argument(
        "--show-unexpected",
        action="store_true",
        help="Print downstream-like setup directories found in expts but not in configs.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path.resolve()
    alt = (repo_root() / path).resolve()
    if alt.exists():
        return alt
    return path.resolve()


def validate_root(root_dir: str) -> Path:
    root = resolve_path(root_dir)
    if not root.exists():
        raise SystemExit(f"Error: root directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Error: root path is not a directory: {root}")
    return root


def iter_real_dirs(parent: Path) -> Iterable[Path]:
    for entry in sorted(parent.iterdir(), key=lambda p: p.name):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            yield entry


def is_expected_downstream_key(key: str) -> bool:
    if "zeroshot" in key:
        return False
    return ("-finetune-" in key) or ("-scratch-" in key)


def collect_expected_setups(config_paths: Sequence[str]) -> Tuple[List[str], List[Path]]:
    expected: Set[str] = set()
    resolved_configs: List[Path] = []

    for config_path_str in config_paths:
        config_path = resolve_path(config_path_str)
        if not config_path.exists():
            raise SystemExit(f"Error: config file does not exist: {config_path}")
        if not config_path.is_file():
            raise SystemExit(f"Error: config path is not a file: {config_path}")
        resolved_configs.append(config_path)

        with config_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line or line[0].isspace():
                    continue
                match = TOP_LEVEL_KEY_RE.match(line)
                if not match:
                    continue
                key = match.group(1)
                if is_expected_downstream_key(key):
                    expected.add(key)

    return sorted(expected), resolved_configs


def parse_seed_from_run_name(run_name: str) -> Tuple[Optional[int], str]:
    explicit_match = EXPLICIT_SEED_RE.search(run_name)
    if explicit_match:
        return int(explicit_match.group("seed")), "explicit"

    parts = run_name.rsplit("-", 2)
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return int(parts[-1]) % 3, "inferred_from_task_id"

    return None, "unknown"


def collect_runs_for_setup(setup_dir: Path) -> List[RunSeedInfo]:
    runs: List[RunSeedInfo] = []
    for run_dir in iter_real_dirs(setup_dir):
        seed, source = parse_seed_from_run_name(run_dir.name)
        runs.append(RunSeedInfo(run_name=run_dir.name, seed=seed, source=source))
    return runs


def format_seed_set(seeds: Set[int]) -> str:
    if not seeds:
        return "-"
    return ",".join(str(seed) for seed in sorted(seeds))


def looks_like_downstream_setup(name: str) -> bool:
    return is_expected_downstream_key(name)


def parse_range_lower_bound(token: str) -> float:
    return float(token.replace("p", "."))


def is_ood_setup(name: str) -> bool:
    match = POISSON_LOWER_BOUND_RE.match(name)
    if match:
        return parse_range_lower_bound(match.group("lower")) >= 5.0

    match = ADVDIFF_LOWER_BOUND_RE.match(name)
    if match:
        return parse_range_lower_bound(match.group("lower")) >= 1.0

    match = HELMHOLTZ_LOWER_BOUND_RE.match(name)
    if match:
        return parse_range_lower_bound(match.group("lower")) >= 10.0

    return False


def setup_category(name: str) -> str:
    return "OOD" if is_ood_setup(name) else "IID"


def main() -> int:
    args = parse_args()
    root = validate_root(args.root_dir)
    expected_seeds = set(args.expected_seeds)
    expected_setups, resolved_configs = collect_expected_setups(args.configs)

    existing_setup_dirs = {path.name: path for path in iter_real_dirs(root)}
    coverage_rows: List[SetupCoverage] = []

    for setup_name in expected_setups:
        setup_dir = existing_setup_dirs.get(setup_name)
        runs = collect_runs_for_setup(setup_dir) if setup_dir is not None else []
        coverage_rows.append(
            SetupCoverage(
                setup_name=setup_name,
                expected_seeds=expected_seeds,
                runs=runs,
            )
        )

    complete = [row for row in coverage_rows if row.is_complete]
    incomplete = [row for row in coverage_rows if row.has_any_runs and not row.is_complete]
    missing_all = [row for row in coverage_rows if not row.has_any_runs]
    ood_rows = [row for row in coverage_rows if is_ood_setup(row.setup_name)]
    ood_complete = [row for row in ood_rows if row.is_complete]
    ood_incomplete = [row for row in ood_rows if row.has_any_runs and not row.is_complete]
    ood_missing_all = [row for row in ood_rows if not row.has_any_runs]

    print(f"Scanning run root: {root}")
    print("Configs:")
    for config_path in resolved_configs:
        print(f"  - {config_path}")
    print(f"Expected seeds: {format_seed_set(expected_seeds)}")
    print("")
    print("Summary:")
    print(f"  Expected downstream setups: {len(coverage_rows)}")
    print(f"  Complete (all seeds present): {len(complete)}")
    print(f"  Incomplete (some seeds missing): {len(incomplete)}")
    print(f"  Missing entirely (no run directory): {len(missing_all)}")
    print(f"  OOD setups tracked: {len(ood_rows)}")
    print(f"    OOD complete: {len(ood_complete)}")
    print(f"    OOD incomplete: {len(ood_incomplete)}")
    print(f"    OOD missing entirely: {len(ood_missing_all)}")

    if incomplete:
        print("")
        print("Incomplete setups:")
        for row in incomplete:
            print(
                f"  - {row.setup_name} [{setup_category(row.setup_name)}]: "
                f"present={format_seed_set(row.present_seeds)} "
                f"missing={format_seed_set(row.missing_seeds)} "
                f"runs={len(row.runs)} explicit={row.explicit_seed_runs} "
                f"inferred={row.inferred_seed_runs} unknown={row.unknown_seed_runs}"
            )

    if missing_all:
        print("")
        print("Missing setups:")
        for row in missing_all:
            print(
                f"  - {row.setup_name} [{setup_category(row.setup_name)}]: "
                f"present=- missing={format_seed_set(row.missing_seeds)}"
            )

    if args.show_complete and complete:
        print("")
        print("Complete setups:")
        for row in complete:
            print(
                f"  - {row.setup_name} [{setup_category(row.setup_name)}]: "
                f"present={format_seed_set(row.present_seeds)} "
                f"runs={len(row.runs)} explicit={row.explicit_seed_runs} "
                f"inferred={row.inferred_seed_runs} unknown={row.unknown_seed_runs}"
            )

    if args.show_unexpected:
        unexpected = [
            name
            for name in sorted(existing_setup_dirs)
            if looks_like_downstream_setup(name) and name not in set(expected_setups)
        ]
        if unexpected:
            print("")
            print("Unexpected downstream-like setup directories:")
            for name in unexpected:
                print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

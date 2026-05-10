#!/usr/bin/env python3
"""
Regenerate transfer-learning comparison plots from saved evaluation JSON files.

This script does not evaluate checkpoints. It only reads existing
``*_results.json`` files and calls ``utils/plot_transfer_learning_comparison.py``.

Examples:
    python scripts/utils/replot_transfer_learning_plots.py
    python scripts/utils/replot_transfer_learning_plots.py helmholtz
    python scripts/utils/replot_transfer_learning_plots.py constraints-poisson --strict
    python scripts/utils/replot_transfer_learning_plots.py --list
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTTER_PATH = REPO_ROOT / "utils" / "plot_transfer_learning_comparison.py"


SeriesSpec = Tuple[str, str, str]


@dataclass(frozen=True)
class PlotPreset:
    name: str
    description: str
    output_subdir: str
    output_name: str
    title: str
    series: Tuple[SeriesSpec, ...]


PRESETS: Dict[str, PlotPreset] = {
    "poisson": PlotPreset(
        name="poisson",
        description="Poisson k in [1,2.5] baseline transfer comparison",
        output_subdir="transfer_learning_k1_2.5",
        output_name="poisson_k1_2p5_transfer_learning_comparison",
        title="Transfer Learning: Poisson k∈[1,2.5]",
        series=(
            ("mixed", "mixed", "poisson_results.json"),
            ("k1_5", "k1_5", "poisson_results.json"),
            ("scratch", "scratch", "poisson_results.json"),
        ),
    ),
    "advdiff": PlotPreset(
        name="advdiff",
        description="Advection-Diffusion adr in [0.2,0.4] baseline transfer comparison",
        output_subdir="transfer_learning_advdiff_adr0.2_0.4",
        output_name="advdiff_adr0p2_0p4_transfer_learning_comparison",
        title="Transfer Learning: AdvDiff adr∈[0.2,0.4]",
        series=(
            ("mixed", "mixed", "advdiff_results.json"),
            ("k1_5", "advdiff", "advdiff_results.json"),
            ("scratch", "scratch", "advdiff_results.json"),
        ),
    ),
    "helmholtz": PlotPreset(
        name="helmholtz",
        description="Helmholtz omega in [1,5] baseline transfer comparison",
        output_subdir="transfer_learning_helmholtz_o1_5",
        output_name="helmholtz_o1_5_transfer_learning_comparison",
        title="Transfer Learning: Helmholtz o∈[1,5]",
        series=(
            ("mixed", "mixed", "helmholtz_results.json"),
            ("k1_5", "helmholtz", "helmholtz_results.json"),
            ("scratch", "scratch", "helmholtz_results.json"),
        ),
    ),
    "constraints-poisson": PlotPreset(
        name="constraints-poisson",
        description="Poisson transfer constraint comparison",
        output_subdir="transfer_learning_constraints_poisson_k1_2.5",
        output_name="poisson_k1_2p5_transfer_learning_constraints_comparison",
        title="Transfer Learning Constraints: Poisson k in [1,2.5]",
        series=(
            ("mixed", "mixed", "poisson_results.json"),
            ("mixed-zero-hard", "mixed-zero-hard", "poisson_results.json"),
            ("mixed-zero-soft", "mixed-zero-soft", "poisson_results.json"),
            ("mixed-penalty-pde", "mixed-penalty-pde", "poisson_results.json"),
        ),
    ),
    "constraints-advdiff": PlotPreset(
        name="constraints-advdiff",
        description="Advection-Diffusion transfer constraint comparison",
        output_subdir="transfer_learning_constraints_advdiff_adr0.2_0.4",
        output_name="advdiff_adr0p2_0p4_transfer_learning_constraints_comparison",
        title="Transfer Learning Constraints: AdvDiff adr in [0.2,0.4]",
        series=(
            ("mixed", "mixed", "advdiff_results.json"),
            ("mixed-zero-hard", "mixed-zero-hard", "advdiff_results.json"),
            ("mixed-zero-soft", "mixed-zero-soft", "advdiff_results.json"),
            ("mixed-penalty-pde", "mixed-penalty-pde", "advdiff_results.json"),
        ),
    ),
    "constraints-helmholtz": PlotPreset(
        name="constraints-helmholtz",
        description="Helmholtz transfer constraint comparison",
        output_subdir="transfer_learning_constraints_helmholtz_o1_5",
        output_name="helmholtz_o1_5_transfer_learning_constraints_comparison",
        title="Transfer Learning Constraints: Helmholtz omega in [1,5]",
        series=(
            ("mixed", "mixed", "helmholtz_results.json"),
            ("mixed-zero-hard", "mixed-zero-hard", "helmholtz_results.json"),
            ("mixed-zero-soft", "mixed-zero-soft", "helmholtz_results.json"),
            ("mixed-penalty-pde", "mixed-penalty-pde", "helmholtz_results.json"),
        ),
    ),
}


ALIASES = {
    "all": tuple(PRESETS),
    "baseline": ("poisson", "advdiff", "helmholtz"),
    "baselines": ("poisson", "advdiff", "helmholtz"),
    "constraints": ("constraints-poisson", "constraints-advdiff", "constraints-helmholtz"),
}


def load_plotter():
    """Load the plotting utility lazily so --list and --dry-run need no plotting deps."""
    spec = importlib.util.spec_from_file_location("transfer_plotter", PLOTTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotter from {PLOTTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_presets() -> None:
    print("Available plot presets:")
    for name in sorted(PRESETS):
        preset = PRESETS[name]
        print(f"  {name:<22} {preset.description}")
    print("\nAliases:")
    for alias in sorted(ALIASES):
        print(f"  {alias:<22} {', '.join(ALIASES[alias])}")


def resolve_requested_plots(requested: Sequence[str]) -> List[str]:
    if not requested:
        requested = ["all"]

    resolved: List[str] = []
    unknown: List[str] = []

    for name in requested:
        if name in ALIASES:
            resolved.extend(ALIASES[name])
        elif name in PRESETS:
            resolved.append(name)
        else:
            unknown.append(name)

    if unknown:
        valid = sorted(set(PRESETS) | set(ALIASES))
        raise ValueError(
            f"Unknown plot preset(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(valid)}"
        )

    deduped: List[str] = []
    for name in resolved:
        if name not in deduped:
            deduped.append(name)
    return deduped


def result_paths(preset: PlotPreset, results_root: Path) -> List[Path]:
    base_dir = results_root / preset.output_subdir
    return [base_dir / series_dir / filename for _, series_dir, filename in preset.series]


def build_series_specs(preset: PlotPreset, results_root: Path, plotter) -> List[Dict[str, str]]:
    base_dir = results_root / preset.output_subdir
    return [
        {
            "key": key,
            "label": plotter._series_label(key, {}),
            "path": str(base_dir / series_dir / filename),
        }
        for key, series_dir, filename in preset.series
    ]


def missing_result_paths(preset: PlotPreset, results_root: Path) -> List[Path]:
    return [path for path in result_paths(preset, results_root) if not path.exists()]


def print_plan(plot_names: Iterable[str], results_root: Path, output_root: Path) -> None:
    for name in plot_names:
        preset = PRESETS[name]
        output_dir = output_root / preset.output_subdir
        output_path = output_dir / f"{preset.output_name}.png"
        print(f"\n{name}:")
        print(f"  output: {output_path}")
        for path in result_paths(preset, results_root):
            status = "found" if path.exists() else "missing"
            print(f"  {status}: {path}")


def replot_preset(preset: PlotPreset, results_root: Path, output_root: Path, strict: bool) -> bool:
    missing = missing_result_paths(preset, results_root)
    if missing:
        print(f"\nSkipping {preset.name}: missing saved result JSON file(s)")
        for path in missing:
            print(f"  missing: {path}")
        if strict:
            raise FileNotFoundError(f"Missing inputs for {preset.name}")
        return False

    plotter = load_plotter()
    output_dir = output_root / preset.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{preset.output_name}.png"

    print(f"\nRegenerating {preset.name}: {output_path}")
    series_entries = plotter.load_series_entries(
        build_series_specs(preset, results_root, plotter)
    )
    plotter.print_summary_table(series_entries)
    plotter.plot_comparison(series_entries, str(output_path), preset.title)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate transfer-learning plots from saved result JSONs."
    )
    parser.add_argument(
        "plots",
        nargs="*",
        help="Plot preset(s) to regenerate. Use 'all', 'baseline', or 'constraints' for groups.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing saved result subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for regenerated plots. Defaults to --results-root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected preset is missing result JSONs instead of skipping it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inputs and outputs without importing plotting dependencies.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available presets and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_presets()
        return 0

    try:
        plot_names = resolve_requested_plots(args.plots)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    results_root = args.results_root
    output_root = args.output_root or results_root

    if args.dry_run:
        print_plan(plot_names, results_root, output_root)
        return 0

    generated = 0
    try:
        for name in plot_names:
            if replot_preset(PRESETS[name], results_root, output_root, args.strict):
                generated += 1
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"\nRegenerated {generated} plot preset(s).")
    if generated == 0 and not args.strict:
        print("No plots were generated because the selected result JSONs were missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

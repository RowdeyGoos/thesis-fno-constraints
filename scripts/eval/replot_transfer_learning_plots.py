#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
Regenerate transfer-learning and OOD plots from saved evaluation JSON files.

This script does not evaluate checkpoints. It only reads existing
``*_results.json`` files and calls the relevant plotting functions.

Examples:
    python scripts/eval/replot_transfer_learning_plots.py
    python scripts/eval/replot_transfer_learning_plots.py helmholtz
    python scripts/eval/replot_transfer_learning_plots.py constraints-poisson --strict
    python scripts/eval/replot_transfer_learning_plots.py ood
    python scripts/eval/replot_transfer_learning_plots.py ood-constraints-poisson
    python scripts/eval/replot_transfer_learning_plots.py --list
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTTER_PATH = REPO_ROOT / "scripts" / "eval" / "plot_transfer_learning_comparison.py"
ENTRYPOINTS_DIR = REPO_ROOT / "scripts" / "entrypoints"
DEFAULT_RESULTS_ROOT = Path("results")
LEGACY_RESULTS_ROOT = Path("third_party/neuraloperators-TL-scaling/results")


SeriesSpec = Tuple[str, str, str]


@dataclass(frozen=True)
class PlotPreset:
    name: str
    description: str
    output_subdir: str
    output_name: str
    title: str
    series: Tuple[SeriesSpec, ...]


@dataclass(frozen=True)
class OodPlotPreset:
    name: str
    description: str
    experiment_type: str
    output_subdir: str
    output_stem: str
    results_filename: str
    module_name: str


TRANSFER_PRESETS: Dict[str, PlotPreset] = {
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


OOD_PRESETS: Dict[str, OodPlotPreset] = {
    "ood-poisson": OodPlotPreset(
        name="ood-poisson",
        description="Poisson OOD degradation comparison",
        experiment_type="poisson",
        output_subdir="ood_poisson",
        output_stem="poisson_ood_comparison",
        results_filename="poisson_ood_results.json",
        module_name="eval_ood_comparison",
    ),
    "ood-advdiff": OodPlotPreset(
        name="ood-advdiff",
        description="Advection-Diffusion OOD degradation comparison",
        experiment_type="advdiff",
        output_subdir="ood_advdiff",
        output_stem="advdiff_ood_comparison",
        results_filename="advdiff_ood_results.json",
        module_name="eval_ood_comparison",
    ),
    "ood-helmholtz": OodPlotPreset(
        name="ood-helmholtz",
        description="Helmholtz OOD degradation comparison",
        experiment_type="helmholtz",
        output_subdir="ood_helmholtz",
        output_stem="helmholtz_ood_comparison",
        results_filename="helmholtz_ood_results.json",
        module_name="eval_ood_comparison",
    ),
    "ood-constraints-poisson": OodPlotPreset(
        name="ood-constraints-poisson",
        description="Poisson OOD constraint comparison",
        experiment_type="poisson",
        output_subdir="ood_constraints_poisson",
        output_stem="poisson_ood_constraints_comparison",
        results_filename="poisson_ood_constraints_results.json",
        module_name="eval_ood_constraint_comparison",
    ),
    "ood-constraints-advdiff": OodPlotPreset(
        name="ood-constraints-advdiff",
        description="Advection-Diffusion OOD constraint comparison",
        experiment_type="advdiff",
        output_subdir="ood_constraints_advdiff",
        output_stem="advdiff_ood_constraints_comparison",
        results_filename="advdiff_ood_constraints_results.json",
        module_name="eval_ood_constraint_comparison",
    ),
    "ood-constraints-helmholtz": OodPlotPreset(
        name="ood-constraints-helmholtz",
        description="Helmholtz OOD constraint comparison",
        experiment_type="helmholtz",
        output_subdir="ood_constraints_helmholtz",
        output_stem="helmholtz_ood_constraints_comparison",
        results_filename="helmholtz_ood_constraints_results.json",
        module_name="eval_ood_constraint_comparison",
    ),
}


PRESETS: Dict[str, Union[PlotPreset, OodPlotPreset]] = {
    **TRANSFER_PRESETS,
    **OOD_PRESETS,
}


ALIASES = {
    "all": tuple(PRESETS),
    "transfer": tuple(TRANSFER_PRESETS),
    "baseline": ("poisson", "advdiff", "helmholtz"),
    "baselines": ("poisson", "advdiff", "helmholtz"),
    "constraints": ("constraints-poisson", "constraints-advdiff", "constraints-helmholtz"),
    "ood": tuple(OOD_PRESETS),
    "ood-baseline": ("ood-poisson", "ood-advdiff", "ood-helmholtz"),
    "ood-baselines": ("ood-poisson", "ood-advdiff", "ood-helmholtz"),
    "ood-constraints": (
        "ood-constraints-poisson",
        "ood-constraints-advdiff",
        "ood-constraints-helmholtz",
    ),
}


def ensure_repo_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_plotter():
    """Load the plotting utility lazily so --list and --dry-run need no plotting deps."""
    spec = importlib.util.spec_from_file_location("transfer_plotter", PLOTTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotter from {PLOTTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ood_plotter(module_name: str):
    """Load an OOD plotting module lazily so --list and --dry-run need no plotting deps."""
    entrypoints_dir = str(ENTRYPOINTS_DIR)
    if entrypoints_dir not in sys.path:
        sys.path.insert(0, entrypoints_dir)

    module_path = ENTRYPOINTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OOD plotter from {module_path}")
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


def is_ood_preset(preset: Union[PlotPreset, OodPlotPreset]) -> bool:
    return isinstance(preset, OodPlotPreset)


def result_paths(preset: PlotPreset, results_root: Path) -> List[Path]:
    base_dir = results_root / preset.output_subdir
    return [base_dir / series_dir / filename for _, series_dir, filename in preset.series]


def ood_result_paths(preset: OodPlotPreset, results_root: Path) -> List[Path]:
    return [results_root / preset.output_subdir / preset.results_filename]


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


def missing_result_paths(preset: Union[PlotPreset, OodPlotPreset], results_root: Path) -> List[Path]:
    if is_ood_preset(preset):
        return [path for path in ood_result_paths(preset, results_root) if not path.exists()]
    return [path for path in result_paths(preset, results_root) if not path.exists()]


def any_result_paths_exist(preset: Union[PlotPreset, OodPlotPreset], results_root: Path) -> bool:
    if is_ood_preset(preset):
        return any(path.exists() for path in ood_result_paths(preset, results_root))
    return any(path.exists() for path in result_paths(preset, results_root))


def resolve_results_root(
    preset: Union[PlotPreset, OodPlotPreset],
    candidate_roots: Sequence[Path],
) -> Path:
    """Choose the best results root for a preset, including the pre-flatten cluster path."""
    for root in candidate_roots:
        if not missing_result_paths(preset, root):
            return root

    for root in candidate_roots:
        if any_result_paths_exist(preset, root):
            return root

    return candidate_roots[0]


def print_plan(
    plot_names: Iterable[str],
    candidate_roots: Sequence[Path],
    output_root: Union[Path, None],
) -> None:
    for name in plot_names:
        preset = PRESETS[name]
        results_root = resolve_results_root(preset, candidate_roots)
        preset_output_root = output_root or results_root
        print(f"\n{name}:")
        print(f"  results root: {results_root}")
        output_dir = preset_output_root / preset.output_subdir
        if is_ood_preset(preset):
            output_paths = [
                output_dir / f"{preset.output_stem}.png",
                output_dir / f"{preset.output_stem}.pdf",
            ]
            paths = ood_result_paths(preset, results_root)
        else:
            output_paths = [
                output_dir / f"{preset.output_name}.png",
                output_dir / f"{preset.output_name}.pdf",
            ]
            paths = result_paths(preset, results_root)

        for output_path in output_paths:
            print(f"  output: {output_path}")
        for path in paths:
            status = "found" if path.exists() else "missing"
            print(f"  {status}: {path}")


def replot_transfer_preset(
    preset: PlotPreset,
    results_root: Path,
    output_root: Path,
    strict: bool,
) -> bool:
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


def replot_ood_preset(
    preset: OodPlotPreset,
    results_root: Path,
    output_root: Path,
    strict: bool,
) -> bool:
    missing = missing_result_paths(preset, results_root)
    if missing:
        print(f"\nSkipping {preset.name}: missing saved result JSON file(s)")
        for path in missing:
            print(f"  missing: {path}")
        if strict:
            raise FileNotFoundError(f"Missing inputs for {preset.name}")
        return False

    input_path = ood_result_paths(preset, results_root)[0]
    output_dir = output_root / preset.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        results = json.load(f)

    plotter = load_ood_plotter(preset.module_name)
    print(f"\nRegenerating {preset.name} from {input_path}")
    for suffix in ("png", "pdf"):
        output_path = output_dir / f"{preset.output_stem}.{suffix}"
        plotter.plot_ood_degradation(results, preset.experiment_type, str(output_path))
        print(f"  saved: {output_path}")

    if hasattr(plotter, "plot_budget_ood_degradation"):
        for output_path in plotter.plot_budget_ood_degradation(
            results,
            preset.experiment_type,
            output_dir,
        ):
            print(f"  saved: {output_path}")
    return True


def replot_preset(
    preset: Union[PlotPreset, OodPlotPreset],
    results_root: Path,
    output_root: Path,
    strict: bool,
) -> bool:
    if is_ood_preset(preset):
        return replot_ood_preset(preset, results_root, output_root, strict)
    return replot_transfer_preset(preset, results_root, output_root, strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate transfer-learning and OOD plots from saved result JSONs."
    )
    parser.add_argument(
        "plots",
        nargs="*",
        help="Plot preset(s) to regenerate. Use 'all', 'transfer', 'ood', or group aliases.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing saved result subdirectories. "
            "Defaults to results/, with fallback to "
            "third_party/neuraloperators-TL-scaling/results/."
        ),
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

    candidate_roots = (
        [args.results_root]
        if args.results_root is not None
        else [DEFAULT_RESULTS_ROOT, LEGACY_RESULTS_ROOT]
    )
    output_root = args.output_root

    if args.dry_run:
        print_plan(plot_names, candidate_roots, output_root)
        return 0

    generated = 0
    try:
        for name in plot_names:
            preset = PRESETS[name]
            results_root = resolve_results_root(preset, candidate_roots)
            preset_output_root = output_root or results_root
            if results_root == LEGACY_RESULTS_ROOT:
                print(f"\nUsing legacy results root for {name}: {results_root}")
            if replot_preset(preset, results_root, preset_output_root, args.strict):
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

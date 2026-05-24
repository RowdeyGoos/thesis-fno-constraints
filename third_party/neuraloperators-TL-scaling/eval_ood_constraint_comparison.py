#!/usr/bin/env python3
"""
Evaluate constrained OOD transfer experiments and compare degradation curves.

This script evaluates OOD fine-tuning runs across the default OOD bins for:
  - the mixed baseline
  - mixed zero-hard pretraining
  - mixed zero-soft pretraining
  - mixed PDE-penalty pretraining

Each variant is evaluated at the 256-sample and 4K-sample downstream budgets,
then plotted against the OOD dataset bins on the x-axis.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from eval_ood_comparison import (
    MEAN_BASELINE_ORDER,
    MEAN_BASELINE_SPECS,
    OOD_EXPERIMENTS,
    evaluate_mean_baseline_series,
    evaluate_series,
    format_ood_bin_tick_label,
    format_log_decade_yaxis,
)
from eval_transfer_learning import get_reported_metric, save_results_json


mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['figure.dpi'] = 150


SERIES_ORDER = [
    'mixed_256',
    'mixed_4k',
    'mixed_zero_hard_256',
    'mixed_zero_hard_4k',
    'mixed_zero_soft_256',
    'mixed_zero_soft_4k',
    'mixed_penalty_pde_256',
    'mixed_penalty_pde_4k',
]


BUDGET_SERIES_ORDER = {
    '256': [
        'mixed_256',
        'mixed_zero_hard_256',
        'mixed_zero_soft_256',
        'mixed_penalty_pde_256',
        'mean_256',
    ],
    '4k': [
        'mixed_4k',
        'mixed_zero_hard_4k',
        'mixed_zero_soft_4k',
        'mixed_penalty_pde_4k',
        'mean_4k',
    ],
}


BUDGET_OUTPUT_SUFFIXES = {
    '256': '256',
    '4k': '4k',
}


BUDGET_TITLES = {
    '256': '256 samples',
    '4k': '4K samples',
}


SERIES_SPECS = {
    'mixed_256': {
        'label': 'Mixed baseline (256)',
        'legend_label': 'Mixed pretrain 256',
        'color': '#0f766e',
        'marker': 'o',
        'linestyle': '--',
        'variant': 'mixed',
        'budget_key': '256',
        'yaml_kind': 'baseline',
    },
    'mixed_4k': {
        'label': 'Mixed baseline (4K)',
        'legend_label': 'Mixed pretrain 4K',
        'color': '#0f766e',
        'marker': 's',
        'linestyle': '-',
        'variant': 'mixed',
        'budget_key': '4k',
        'yaml_kind': 'baseline',
    },
    'mixed_zero_hard_256': {
        'label': 'Mixed + zero-hard (256)',
        'legend_label': 'Zero-mode hard 256',
        'color': '#1d4ed8',
        'marker': 'o',
        'linestyle': '--',
        'variant': 'mixed-zero-hard',
        'budget_key': '256',
        'yaml_kind': 'constraints',
    },
    'mixed_zero_hard_4k': {
        'label': 'Mixed + zero-hard (4K)',
        'legend_label': 'Zero-mode hard 4K',
        'color': '#1d4ed8',
        'marker': 's',
        'linestyle': '-',
        'variant': 'mixed-zero-hard',
        'budget_key': '4k',
        'yaml_kind': 'constraints',
    },
    'mixed_zero_soft_256': {
        'label': 'Mixed + zero-soft (256)',
        'legend_label': 'Zero-mode soft 256',
        'color': '#d97706',
        'marker': 'o',
        'linestyle': '--',
        'variant': 'mixed-zero-soft',
        'budget_key': '256',
        'yaml_kind': 'constraints',
    },
    'mixed_zero_soft_4k': {
        'label': 'Mixed + zero-soft (4K)',
        'legend_label': 'Zero-mode soft 4K',
        'color': '#d97706',
        'marker': 's',
        'linestyle': '-',
        'variant': 'mixed-zero-soft',
        'budget_key': '4k',
        'yaml_kind': 'constraints',
    },
    'mixed_penalty_pde_256': {
        'label': 'Mixed + PDE penalty (256)',
        'legend_label': 'PDE penalty 256',
        'color': '#7c3aed',
        'marker': 'o',
        'linestyle': '--',
        'variant': 'mixed-penalty-pde',
        'budget_key': '256',
        'yaml_kind': 'constraints',
    },
    'mixed_penalty_pde_4k': {
        'label': 'Mixed + PDE penalty (4K)',
        'legend_label': 'PDE penalty 4K',
        'color': '#7c3aed',
        'marker': 's',
        'linestyle': '-',
        'variant': 'mixed-penalty-pde',
        'budget_key': '4k',
        'yaml_kind': 'constraints',
    },
}


CONFIG_PREFIXES = {
    'poisson': 'poisson',
    'advdiff': 'ad',
    'helmholtz': 'helm',
}


CONSTRAINT_YAMLS = {
    'poisson': 'config/operators_poisson_mixed_constraints.yaml',
    'advdiff': 'config/operators_ad_mixed_constraints.yaml',
    'helmholtz': 'config/operators_helmholtz_mixed_constraints.yaml',
}


OUTPUT_METADATA = {
    'poisson': {
        'title': 'Poisson OOD Constraint Comparison',
        'output_dir': 'results/ood_constraints_poisson',
        'output_stem': 'poisson_ood_constraints_comparison',
        'results_filename': 'poisson_ood_constraints_results.json',
    },
    'advdiff': {
        'title': 'AdvDiff OOD Constraint Comparison',
        'output_dir': 'results/ood_constraints_advdiff',
        'output_stem': 'advdiff_ood_constraints_comparison',
        'results_filename': 'advdiff_ood_constraints_results.json',
    },
    'helmholtz': {
        'title': 'Helmholtz OOD Constraint Comparison',
        'output_dir': 'results/ood_constraints_helmholtz',
        'output_stem': 'helmholtz_ood_constraints_comparison',
        'results_filename': 'helmholtz_ood_constraints_results.json',
    },
}


def build_series_plan(experiment_type: str) -> Dict[str, Dict]:
    """Build YAML/config mappings for each plotted OOD constraint series."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    prefix = CONFIG_PREFIXES[experiment_type]
    plan = {}

    for series_key in SERIES_ORDER:
        series_spec = SERIES_SPECS[series_key]
        yaml_config = (
            experiment_spec['yaml_config']
            if series_spec['yaml_kind'] == 'baseline'
            else CONSTRAINT_YAMLS[experiment_type]
        )
        budget_key = series_spec['budget_key']
        variant = series_spec['variant']

        plan[series_key] = {
            'label': series_spec['label'],
            'yaml_config': yaml_config,
            'config_names': {
                bin_key: f'{prefix}-{bin_key}-finetune-{variant}-{budget_key}'
                for bin_key in experiment_spec['bin_keys']
            },
        }

    return plan


def _legend_label_without_budget(label: str, budget_key: str) -> str:
    suffix = ' 4K' if budget_key == '4k' else f' {budget_key}'
    if label.endswith(suffix):
        return label[:-len(suffix)]
    return label


def plot_ood_degradation(results: Dict, experiment_type: str, output_path: str, budget_key: str = None):
    """Plot constraint-aware OOD degradation curves against the OOD bins."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    output_spec = OUTPUT_METADATA[experiment_type]
    bin_keys = experiment_spec['bin_keys']
    x_positions = np.arange(len(bin_keys), dtype=float)

    series_order = BUDGET_SERIES_ORDER.get(budget_key, [*SERIES_ORDER, *MEAN_BASELINE_ORDER])
    fig, ax = plt.subplots(figsize=(18, 9))

    for series_key in series_order:
        series_result = results['series'].get(series_key)
        if not series_result:
            continue

        style = MEAN_BASELINE_SPECS.get(series_key, SERIES_SPECS.get(series_key))
        points = series_result.get('points', {})
        errors = []
        min_band = []
        max_band = []

        for bin_key in bin_keys:
            metrics = points.get(bin_key)
            if metrics is None:
                errors.append(np.nan)
                min_band.append(np.nan)
                max_band.append(np.nan)
                continue

            errors.append(get_reported_metric(metrics, 'test_error'))
            min_band.append(metrics.get('test_error_min', np.nan))
            max_band.append(metrics.get('test_error_max', np.nan))

        errors_arr = np.asarray(errors, dtype=float)
        valid_mask = np.isfinite(errors_arr)
        if not np.any(valid_mask):
            continue

        x_valid = x_positions[valid_mask]
        y_valid = errors_arr[valid_mask]

        ax.plot(
            x_valid,
            y_valid,
            color=style['color'],
            marker=style['marker'],
            markerfacecolor=style['color'],
            markeredgecolor=style['color'],
            markeredgewidth=1.5,
            markersize=7.5,
            linestyle=style['linestyle'],
            linewidth=2.1,
            label=_legend_label_without_budget(
                style.get('legend_label', style['label']),
                budget_key,
            ) if budget_key else style.get('legend_label', style['label']),
        )

        min_arr = np.asarray(min_band, dtype=float)
        max_arr = np.asarray(max_band, dtype=float)
        band_mask = np.isfinite(min_arr) & np.isfinite(max_arr)
        if np.any(band_mask):
            ax.fill_between(
                x_positions[band_mask],
                min_arr[band_mask],
                max_arr[band_mask],
                color=style['color'],
                alpha=0.08,
                linewidth=0,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [format_ood_bin_tick_label(label) for label in experiment_spec['bin_labels']],
        fontsize=24,
    )
    ax.set_xlabel(
        f"{experiment_spec['range_axis_label']} (increasing OOD distance)",
        fontsize=26,
        fontweight='bold',
    )
    ax.set_ylabel('Test error (relative L2)', fontsize=26, fontweight='bold')
    ax.set_yscale('log', base=10)
    format_log_decade_yaxis(ax)
    ax.tick_params(axis='y', labelsize=24)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=22,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderaxespad=0.0,
    )
    title = output_spec['title']
    if budget_key:
        title += f" ({BUDGET_TITLES[budget_key]})"
    ax.set_title(
        f"{title}\n{experiment_spec['subtitle']}",
        fontsize=26,
        fontweight='bold',
        pad=18,
    )

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info('OOD constraint comparison plot saved to: %s', output_path)
    plt.close()


def plot_budget_ood_degradation(results: Dict, experiment_type: str, output_dir: Path):
    """Save separate 256-sample and 4K-sample OOD constraint plots."""
    output_spec = OUTPUT_METADATA[experiment_type]
    saved_paths = []
    for budget_key, suffix in BUDGET_OUTPUT_SUFFIXES.items():
        for extension in ('png', 'pdf'):
            output_path = output_dir / f"{output_spec['output_stem']}_{suffix}.{extension}"
            plot_ood_degradation(results, experiment_type, str(output_path), budget_key=budget_key)
            saved_paths.append(output_path)
    return saved_paths


def print_results_table(results: Dict, experiment_type: str):
    """Print a compact table of test error by OOD bin for each constrained series."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    print('\n' + '=' * 120)
    print('OOD CONSTRAINT EVALUATION RESULTS SUMMARY')
    print('=' * 120)

    header = f"{'Series':<34} | " + " | ".join(f"{label:<16}" for label in experiment_spec['bin_labels'])
    print(header)
    print('-' * len(header))

    for series_key in [*SERIES_ORDER, *MEAN_BASELINE_ORDER]:
        series_result = results['series'].get(series_key)
        if not series_result:
            continue

        row = f"{series_result['label']:<34} | "
        row_values = []
        for bin_key in experiment_spec['bin_keys']:
            metrics = series_result['points'].get(bin_key)
            if metrics is None:
                row_values.append(f"{'N/A':<16}")
                continue
            error = get_reported_metric(metrics, 'test_error')
            if np.isnan(error):
                row_values.append(f"{'N/A':<16}")
            else:
                row_values.append(f"{error:<16.6f}")
        row += ' | '.join(row_values)
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Evaluate constrained OOD degradation experiments')
    parser.add_argument(
        '--experiment_type',
        type=str,
        required=True,
        choices=sorted(OOD_EXPERIMENTS.keys()),
        help='Experiment type to evaluate',
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='experiments',
        help='Base experiment directory containing expts/',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to write JSON and plots. Defaults to the experiment-specific OOD-constraints dir.',
    )
    parser.add_argument(
        '--aggregate_runs',
        action='store_true',
        help='Aggregate multiple checkpoints matching run_pattern.',
    )
    parser.add_argument(
        '--run_pattern',
        type=str,
        default='*',
        help='Pattern for matching run directories when aggregating.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device for evaluation',
    )
    args = parser.parse_args()

    experiment_spec = OOD_EXPERIMENTS[args.experiment_type]
    output_spec = OUTPUT_METADATA[args.experiment_type]
    output_dir = Path(args.output_dir or output_spec['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'metadata': {
            'experiment_type': args.experiment_type,
            'experiment_dir': args.experiment_dir,
            'aggregate_runs': args.aggregate_runs,
            'run_pattern': args.run_pattern,
            'bin_keys': experiment_spec['bin_keys'],
            'bin_labels': experiment_spec['bin_labels'],
            'constraint_yaml_config': CONSTRAINT_YAMLS[args.experiment_type],
            'baseline_yaml_config': experiment_spec['yaml_config'],
        },
        'series': {},
    }

    series_plan = build_series_plan(args.experiment_type)

    for series_key in SERIES_ORDER:
        plan = series_plan[series_key]
        points = evaluate_series(
            yaml_config=plan['yaml_config'],
            experiment_dir=args.experiment_dir,
            bin_keys=experiment_spec['bin_keys'],
            config_names=plan['config_names'],
            aggregate_runs=args.aggregate_runs,
            run_pattern=args.run_pattern,
            device=args.device,
        )
        results['series'][series_key] = {
            'label': plan['label'],
            'yaml_config': plan['yaml_config'],
            'points': points,
        }

    for baseline_key in MEAN_BASELINE_ORDER:
        budget_key = MEAN_BASELINE_SPECS[baseline_key]['budget_key']
        reference_series = f'mixed_{budget_key}'
        plan = series_plan[reference_series]
        points = evaluate_mean_baseline_series(
            yaml_config=plan['yaml_config'],
            bin_keys=experiment_spec['bin_keys'],
            config_names=plan['config_names'],
        )
        results['series'][baseline_key] = {
            'label': MEAN_BASELINE_SPECS[baseline_key]['label'],
            'yaml_config': plan['yaml_config'],
            'points': points,
        }

    results_path = output_dir / output_spec['results_filename']
    save_results_json(results, str(results_path))
    print_results_table(results, args.experiment_type)

    plot_paths = plot_budget_ood_degradation(results, args.experiment_type, output_dir)

    logging.info('')
    logging.info('=' * 80)
    logging.info('OOD constraint evaluation complete')
    logging.info('Results JSON: %s', results_path)
    for plot_path in plot_paths:
        logging.info('Plot: %s', plot_path)
    logging.info('=' * 80)


if __name__ == '__main__':
    main()

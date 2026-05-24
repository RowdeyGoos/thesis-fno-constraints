#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
Evaluate OOD transfer experiments and plot degradation across OOD datasets.

This script evaluates the OOD fine-tuning experiments for mixed-pretrained and
from-scratch models, then generates a single plot with OOD dataset bins on the
x-axis so the degradation trend is visible in one figure.
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from eval_transfer_learning import (
    aggregate_metrics,
    evaluate_model,
    find_checkpoint,
    find_checkpoints,
    get_reported_metric,
    save_results_json,
)
from utils.YParams import YParams


mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['figure.dpi'] = 150


SERIES_SPECS = {
    'mixed_256': {
        'label': 'Mixed pre-trained (256 samples)',
        'legend_label': 'Mixed pretrain 256',
        'color': '#0f766e',
        'marker': 'o',
        'linestyle': '-',
        'template_key': 'mixed',
        'budget_key': '256',
        'budget_label': '256',
    },
    'mixed_4k': {
        'label': 'Mixed pre-trained (4K samples)',
        'legend_label': 'Mixed pretrain 4K',
        'color': '#115e59',
        'marker': 's',
        'linestyle': '-',
        'template_key': 'mixed',
        'budget_key': '4k',
        'budget_label': '4K',
    },
    'scratch_256': {
        'label': 'Scratch (256 samples)',
        'legend_label': 'Scratch 256',
        'color': '#dc2626',
        'marker': 'o',
        'linestyle': '--',
        'template_key': 'scratch',
        'budget_key': '256',
        'budget_label': '256',
    },
    'scratch_4k': {
        'label': 'Scratch (4K samples)',
        'legend_label': 'Scratch 4K',
        'color': '#991b1b',
        'marker': 's',
        'linestyle': '--',
        'template_key': 'scratch',
        'budget_key': '4k',
        'budget_label': '4K',
    },
}


MEAN_BASELINE_ORDER = ['mean_4k']


MEAN_BASELINE_SPECS = {
    'mean_256': {
        'label': 'Mean baseline',
        'legend_label': 'Mean baseline',
        'color': '#6b7280',
        'marker': 'D',
        'linestyle': ':',
        'budget_key': '256',
    },
    'mean_4k': {
        'label': 'Mean baseline',
        'legend_label': 'Mean baseline',
        'color': '#111827',
        'marker': 'D',
        'linestyle': ':',
        'budget_key': '4k',
    },
}


def format_log_decade_yaxis(ax):
    """Show log-scale y-axis labels only at powers of ten."""
    y_values = []
    for line in ax.lines:
        y_values.extend(
            float(y)
            for y in line.get_ydata()
            if np.isfinite(y) and y > 0
        )

    if not y_values:
        return

    y_min = 10 ** math.floor(math.log10(min(y_values)))
    y_max = max(y_values) * 10 ** 0.08

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def format_ood_bin_tick_label(label: str) -> str:
    """Use compact bin labels because the x-axis title names the parameter."""
    if ' in ' in label:
        return label.split(' in ', 1)[1]
    return label


OOD_EXPERIMENTS = {
    'poisson': {
        'yaml_config': 'config/operators_poisson.yaml',
        'title': 'Poisson OOD Comparison',
        'subtitle': 'Source/pretrain range: k in [1, 5]',
        'range_axis_label': 'Poisson coefficient range',
        'bin_keys': ['k5_7p5', 'k7p5_10', 'k10_12p5', 'k12p5_15'],
        'bin_labels': ['k in [5, 7.5]', 'k in [7.5, 10]', 'k in [10, 12.5]', 'k in [12.5, 15]'],
        'config_templates': {
            'mixed': 'poisson-{bin_key}-finetune-mixed-{budget_key}',
            'scratch': 'poisson-{bin_key}-scratch-{budget_key}',
        },
        'output_dir': 'results/ood_poisson',
        'output_stem': 'poisson_ood_comparison',
        'results_filename': 'poisson_ood_results.json',
    },
    'advdiff': {
        'yaml_config': 'config/operators_ad.yaml',
        'title': 'AdvDiff OOD Comparison',
        'subtitle': 'Source/pretrain range: adr in [0.2, 1.0]',
        'range_axis_label': 'AdvDiff adr range',
        'bin_keys': ['adr1_1p2', 'adr1p2_1p4', 'adr1p4_1p6', 'adr1p6_1p8'],
        'bin_labels': ['adr in [1.0, 1.2]', 'adr in [1.2, 1.4]', 'adr in [1.4, 1.6]', 'adr in [1.6, 1.8]'],
        'config_templates': {
            'mixed': 'ad-{bin_key}-finetune-mixed-{budget_key}',
            'scratch': 'ad-{bin_key}-scratch-{budget_key}',
        },
        'output_dir': 'results/ood_advdiff',
        'output_stem': 'advdiff_ood_comparison',
        'results_filename': 'advdiff_ood_results.json',
    },
    'helmholtz': {
        'yaml_config': 'config/operators_helmholtz.yaml',
        'title': 'Helmholtz OOD Comparison',
        'subtitle': 'Source/pretrain range: omega in [1, 10]',
        'range_axis_label': 'Helmholtz omega range',
        'bin_keys': ['o10_15', 'o15_20', 'o20_25', 'o25_30'],
        'bin_labels': ['omega in [10, 15]', 'omega in [15, 20]', 'omega in [20, 25]', 'omega in [25, 30]'],
        'config_templates': {
            'mixed': 'helm-{bin_key}-finetune-mixed-{budget_key}',
            'scratch': 'helm-{bin_key}-scratch-{budget_key}',
        },
        'output_dir': 'results/ood_helmholtz',
        'output_stem': 'helmholtz_ood_comparison',
        'results_filename': 'helmholtz_ood_results.json',
    },
}


def _resolve_data_path(path: str) -> str:
    """Resolve config data paths the same way the eval scripts are run: from cwd."""
    return path if os.path.isabs(path) else os.path.abspath(path)


def _training_indices(params, n_samples: int) -> np.ndarray:
    """Match PDESolns train-subset indexing for mean-baseline fitting."""
    subsample = int(getattr(params, 'subsample', 1))
    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")

    target_n = int(n_samples / subsample)
    if target_n <= 0:
        return np.asarray([], dtype=np.int64)

    random_subset = str(getattr(params, 'random_train_subset', False)).lower() in {'1', 'true', 'yes', 'y', 'on'}
    if random_subset:
        subset_seed = getattr(params, 'subset_seed', None)
        if subset_seed is None:
            subset_seed = getattr(params, 'seed', 0)
        rng = np.random.default_rng(int(subset_seed))
        return np.sort(rng.choice(n_samples, size=target_n, replace=False)).astype(np.int64)

    return (np.arange(target_n, dtype=np.int64) * subsample).astype(np.int64)


def _read_target_batch(fields, indices: np.ndarray) -> np.ndarray:
    """Read target channels from the HDF5 fields dataset."""
    target_start = fields.shape[1] - 1
    return fields[indices, target_start:target_start + 1].astype(np.float64)


def _compute_train_target_mean(train_path: str, indices: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Compute the spatial mean target field over the selected downstream train subset."""
    if indices.size == 0:
        raise ValueError(f"No training samples selected for mean baseline from {train_path}")

    with h5py.File(train_path, 'r') as f:
        fields = f['fields']
        target_sum = None
        n_seen = 0

        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start:start + batch_size]
            targets = _read_target_batch(fields, batch_indices)
            batch_sum = np.sum(targets, axis=0)
            target_sum = batch_sum if target_sum is None else target_sum + batch_sum
            n_seen += targets.shape[0]

    return target_sum / float(n_seen)


def evaluate_mean_baseline(yaml_config: str, config_name: str) -> Dict[str, float]:
    """Evaluate a constant predictor equal to the downstream train-target mean field."""
    logging.info('')
    logging.info('=' * 80)
    logging.info('Evaluating mean baseline for config %s', config_name)
    logging.info('=' * 80)

    try:
        params = YParams(os.path.abspath(yaml_config), config_name)
        train_path = _resolve_data_path(params.train_path)
        test_path = _resolve_data_path(params.test_path)

        with h5py.File(train_path, 'r') as f_train:
            train_n = int(f_train['fields'].shape[0])
        train_indices = _training_indices(params, train_n)
        mean_target = _compute_train_target_mean(train_path, train_indices)

        rel_errors = []
        squared_errors = []
        with h5py.File(test_path, 'r') as f_test:
            fields = f_test['fields']
            n_test = int(fields.shape[0])
            all_indices = np.arange(n_test, dtype=np.int64)

            for start in range(0, n_test, 512):
                batch_indices = all_indices[start:start + 512]
                targets = _read_target_batch(fields, batch_indices)
                diff = mean_target[None, ...] - targets
                numerator = np.sum(diff ** 2, axis=(-1, -2))
                denominator = np.sum(targets ** 2, axis=(-1, -2))
                rel_errors.append(np.sqrt(numerator / denominator))
                squared_errors.append(np.mean(diff ** 2, axis=(-1, -2)))

        rel_errors = np.concatenate(rel_errors, axis=0)
        squared_errors = np.concatenate(squared_errors, axis=0)
        metrics = {
            'test_error': float(np.mean(rel_errors)),
            'test_loss': float(np.mean(squared_errors)),
            'n_train_mean_samples': int(train_indices.size),
            'train_path': train_path,
            'test_path': test_path,
            'config_name': config_name,
            'baseline': 'train_target_mean',
        }
        logging.info(
            'Mean baseline results: test_error=%.6f, n_train_mean_samples=%d',
            metrics['test_error'],
            metrics['n_train_mean_samples'],
        )
        return metrics
    except Exception as exc:
        logging.error('Error evaluating mean baseline for %s: %s', config_name, exc)
        return {
            'test_error': np.nan,
            'test_loss': np.nan,
            'n_train_mean_samples': 0,
            'config_name': config_name,
            'baseline': 'train_target_mean',
        }


def evaluate_mean_baseline_series(
    yaml_config: str,
    bin_keys: List[str],
    config_names: Dict[str, str],
) -> Dict[str, Dict]:
    """Evaluate the mean baseline across all OOD bins for one sample budget."""
    return {
        bin_key: evaluate_mean_baseline(yaml_config, config_names[bin_key])
        for bin_key in bin_keys
    }


def build_series_plan(experiment_type: str) -> Dict[str, Dict]:
    """Build config names for each series and OOD bin."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    plan = {}

    for series_key, series_spec in SERIES_SPECS.items():
        template = experiment_spec['config_templates'][series_spec['template_key']]
        plan[series_key] = {
            'label': series_spec['label'],
            'config_names': {
                bin_key: template.format(
                    bin_key=bin_key,
                    budget_key=series_spec['budget_key'],
                )
                for bin_key in experiment_spec['bin_keys']
            },
        }

    return plan


def evaluate_series(
    yaml_config: str,
    experiment_dir: str,
    bin_keys: List[str],
    config_names: Dict[str, str],
    aggregate_runs: bool,
    run_pattern: str,
    device: str,
) -> Dict[str, Dict]:
    """Evaluate one plotted series across all OOD bins."""
    points = {}

    for bin_key in bin_keys:
        config_name = config_names[bin_key]
        logging.info('')
        logging.info('=' * 80)
        logging.info('Evaluating OOD bin %s with config %s', bin_key, config_name)
        logging.info('=' * 80)

        if aggregate_runs:
            checkpoints = find_checkpoints(experiment_dir, config_name, run_pattern)
            if not checkpoints:
                logging.warning('Skipping %s: checkpoints not found', config_name)
                continue

            trial_metrics = [
                evaluate_model(yaml_config, config_name, checkpoint, device)
                for checkpoint in checkpoints
            ]
            metrics = aggregate_metrics(trial_metrics)
        else:
            checkpoint = find_checkpoint(experiment_dir, config_name, run_pattern)
            if checkpoint is None:
                logging.warning('Skipping %s: checkpoint not found', config_name)
                continue
            metrics = evaluate_model(yaml_config, config_name, checkpoint, device)

        metrics['config_name'] = config_name
        points[bin_key] = metrics

    return points


def plot_ood_degradation(results: Dict, experiment_type: str, output_path: str):
    """Plot OOD degradation curves with OOD dataset bins on the x-axis."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    bin_keys = experiment_spec['bin_keys']
    x_positions = np.arange(len(bin_keys), dtype=float)

    fig, ax = plt.subplots(figsize=(18, 9))

    for series_key in ['mixed_256', 'mixed_4k', 'scratch_256', 'scratch_4k', *MEAN_BASELINE_ORDER]:
        series_result = results['series'].get(series_key)
        if not series_result:
            continue

        series_style = MEAN_BASELINE_SPECS.get(series_key, SERIES_SPECS.get(series_key))
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
        facecolor = 'none' if series_key.startswith('scratch') else series_style['color']

        ax.plot(
            x_valid,
            y_valid,
            color=series_style['color'],
            marker=series_style['marker'],
            markerfacecolor=facecolor,
            markeredgecolor=series_style['color'],
            markeredgewidth=2,
            markersize=8,
            linestyle=series_style['linestyle'],
            linewidth=2.3,
            label=series_style.get('legend_label', series_style['label']),
        )

        min_arr = np.asarray(min_band, dtype=float)
        max_arr = np.asarray(max_band, dtype=float)
        band_mask = np.isfinite(min_arr) & np.isfinite(max_arr)
        if np.any(band_mask):
            ax.fill_between(
                x_positions[band_mask],
                min_arr[band_mask],
                max_arr[band_mask],
                color=series_style['color'],
                alpha=0.12,
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
    ax.legend(loc='best', fontsize=22, frameon=True, fancybox=True, shadow=True)
    ax.set_title(
        f"{experiment_spec['title']}\n{experiment_spec['subtitle']}",
        fontsize=26,
        fontweight='bold',
        pad=18,
    )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info('OOD degradation plot saved to: %s', output_path)
    plt.close()


def print_results_table(results: Dict, experiment_type: str):
    """Print a compact table of test error by OOD bin."""
    experiment_spec = OOD_EXPERIMENTS[experiment_type]
    print('\n' + '=' * 100)
    print('OOD EVALUATION RESULTS SUMMARY')
    print('=' * 100)

    header = f"{'Series':<34} | " + " | ".join(f"{label:<16}" for label in experiment_spec['bin_labels'])
    print(header)
    print('-' * len(header))

    for series_key in ['mixed_256', 'mixed_4k', 'scratch_256', 'scratch_4k', *MEAN_BASELINE_ORDER]:
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
    parser = argparse.ArgumentParser(description='Evaluate OOD degradation experiments')
    parser.add_argument(
        '--experiment_type',
        type=str,
        required=True,
        choices=sorted(OOD_EXPERIMENTS.keys()),
        help='Experiment type to evaluate',
    )
    parser.add_argument(
        '--yaml_config',
        type=str,
        default=None,
        help='YAML config file. Defaults to the standard config for the experiment type.',
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
        help='Directory to write JSON and plots. Defaults to the experiment-specific OOD results dir.',
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
    yaml_config = args.yaml_config or experiment_spec['yaml_config']
    output_dir = Path(args.output_dir or experiment_spec['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'metadata': {
            'experiment_type': args.experiment_type,
            'yaml_config': yaml_config,
            'experiment_dir': args.experiment_dir,
            'aggregate_runs': args.aggregate_runs,
            'run_pattern': args.run_pattern,
            'bin_keys': experiment_spec['bin_keys'],
            'bin_labels': experiment_spec['bin_labels'],
        },
        'series': {},
    }

    series_plan = build_series_plan(args.experiment_type)

    for series_key in ['mixed_256', 'mixed_4k', 'scratch_256', 'scratch_4k']:
        plan = series_plan[series_key]
        points = evaluate_series(
            yaml_config=yaml_config,
            experiment_dir=args.experiment_dir,
            bin_keys=experiment_spec['bin_keys'],
            config_names=plan['config_names'],
            aggregate_runs=args.aggregate_runs,
            run_pattern=args.run_pattern,
            device=args.device,
        )
        results['series'][series_key] = {
            'label': plan['label'],
            'points': points,
        }

    for baseline_key in MEAN_BASELINE_ORDER:
        budget_key = MEAN_BASELINE_SPECS[baseline_key]['budget_key']
        reference_series = f'scratch_{budget_key}'
        plan = series_plan[reference_series]
        points = evaluate_mean_baseline_series(
            yaml_config=yaml_config,
            bin_keys=experiment_spec['bin_keys'],
            config_names=plan['config_names'],
        )
        results['series'][baseline_key] = {
            'label': MEAN_BASELINE_SPECS[baseline_key]['label'],
            'points': points,
        }

    results_path = output_dir / experiment_spec['results_filename']
    save_results_json(results, str(results_path))
    print_results_table(results, args.experiment_type)

    plot_path_png = output_dir / f"{experiment_spec['output_stem']}.png"
    plot_ood_degradation(results, args.experiment_type, str(plot_path_png))

    plot_path_pdf = output_dir / f"{experiment_spec['output_stem']}.pdf"
    plot_ood_degradation(results, args.experiment_type, str(plot_path_pdf))

    logging.info('')
    logging.info('=' * 80)
    logging.info('OOD evaluation complete')
    logging.info('Results JSON: %s', results_path)
    logging.info('Plot PNG: %s', plot_path_png)
    logging.info('Plot PDF: %s', plot_path_pdf)
    logging.info('=' * 80)


if __name__ == '__main__':
    main()

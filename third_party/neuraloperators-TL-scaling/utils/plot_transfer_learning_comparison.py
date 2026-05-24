#!/usr/bin/env python3
"""
Plot transfer-learning comparisons from separate evaluation result files.

The script supports two calling conventions:

1. Legacy three-way comparison:
   python utils/plot_transfer_learning_comparison.py \
       --mixed_results results/.../mixed/poisson_results.json \
       --k1_5_results results/.../k1_5/poisson_results.json \
       --scratch_results results/.../scratch/poisson_results.json

2. Generic multi-series comparison:
   python utils/plot_transfer_learning_comparison.py \
       --series mixed=results/.../mixed/poisson_results.json \
       --series mixed-zero-hard=results/.../mixed-zero-hard/poisson_results.json \
       --series mixed-zero-soft=results/.../mixed-zero-soft/poisson_results.json \
       --series mixed-penalty-pde=results/.../mixed-penalty-pde/poisson_results.json
"""

import argparse
import json
import math
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


SERIES_STYLES = {
    'mixed': {
        'color': '#0f766e',
        'marker': 'o',
        'label': 'Mixed pretrain',
    },
    'k1_5': {
        'color': '#2563eb',
        'marker': 's',
        'label': 'Equation-specific pretrain',
    },
    'scratch': {
        'color': '#dc2626',
        'marker': '^',
        'label': 'Scratch',
    },
    'mixed-zero-hard': {
        'color': '#1d4ed8',
        'marker': 's',
        'label': 'Zero-mode hard',
    },
    'mixed-zero-soft': {
        'color': '#d97706',
        'marker': 'D',
        'label': 'Zero-mode soft',
    },
    'mixed-penalty-pde': {
        'color': '#7c3aed',
        'marker': '^',
        'label': 'PDE penalty',
    },
    'mixed-bc-off': {
        'color': '#0f766e',
        'marker': 'o',
        'label': 'BC baseline',
    },
    'mixed-bc-soft': {
        'color': '#d97706',
        'marker': 'D',
        'label': 'BC soft',
    },
    'mixed-bc-hard': {
        'color': '#1d4ed8',
        'marker': 's',
        'label': 'BC hard',
    },
    'mixed-bc-hard-soft': {
        'color': '#7c3aed',
        'marker': '^',
        'label': 'BC hard+soft',
    },
}


def load_results(filepath):
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def slugify_filename(value):
    """Convert a plot title or label into a filesystem-friendly stem."""
    normalized = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    normalized = re.sub(r'(?<=\d)\.(?=\d)', 'p', normalized)
    normalized = normalized.lower()
    normalized = normalized.replace('&', ' and ')
    normalized = re.sub(r'[^a-z0-9]+', '_', normalized)
    normalized = normalized.strip('_')
    return normalized or 'transfer_learning_comparison'


def determine_output_stem(args):
    """Pick a descriptive output stem for the comparison plot."""
    if args.output_name:
        return slugify_filename(args.output_name)

    title_stem = slugify_filename(args.title)
    if title_stem != 'transfer_learning_comparison':
        return title_stem

    output_dir_name = Path(args.output_dir).name
    output_dir_stem = slugify_filename(output_dir_name)
    if output_dir_stem:
        return f'{output_dir_stem}_comparison'

    return 'transfer_learning_comparison'


def extract_metrics_by_size(results):
    """Extract test metrics organized by sample size."""
    metrics_by_size = {}
    
    # Handle different result structures
    if isinstance(results, dict):
        # Check if it's the nested structure {model_type: {size: metrics}}
        for model_type, size_dict in results.items():
            if isinstance(size_dict, dict):
                for size, metrics in size_dict.items():
                    size_int = int(size) if isinstance(size, str) else size
                    if isinstance(metrics, dict):
                        # Look for test_error (new format) or test_err (old format)
                        if 'test_error' in metrics:
                            metrics_by_size[size_int] = {
                                'test_error': metrics['test_error'],
                                'test_error_min': metrics.get('test_error_min'),
                                'test_error_max': metrics.get('test_error_max'),
                            }
                        elif 'test_err' in metrics:
                            metrics_by_size[size_int] = {
                                'test_error': metrics['test_err'],
                                'test_error_min': metrics.get('test_err_min'),
                                'test_error_max': metrics.get('test_err_max'),
                            }
                    elif isinstance(metrics, (int, float)):
                        metrics_by_size[size_int] = {
                            'test_error': metrics,
                            'test_error_min': None,
                            'test_error_max': None,
                        }
    
    return metrics_by_size


def _format_sample_tick(size):
    if size == 0:
        return '0-shot'
    if size >= 1024 and size % 1024 == 0:
        return f'{size // 1024}K'
    return str(size)


def _build_sample_x_map(sample_sizes):
    """Map sorted sample sizes to evenly spaced categorical x positions."""
    return {size: idx for idx, size in enumerate(sample_sizes)}


def _format_log_decade_yaxis(ax):
    """Show log-scale y-axis labels only at powers of ten."""
    y_values = []
    for line in ax.lines:
        y_values.extend(
            float(y)
            for y in line.get_ydata()
            if math.isfinite(y) and y > 0
        )

    if not y_values:
        return

    y_min = 10 ** math.floor(math.log10(min(y_values)))
    y_max = 10 ** math.ceil(math.log10(max(y_values)))
    if y_min == y_max:
        y_max *= 10

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _parse_kv_arg(spec, arg_name):
    if '=' not in spec:
        raise ValueError(f"{arg_name} must be in KEY=VALUE format: {spec}")
    key, value = spec.split('=', 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"{arg_name} must be in KEY=VALUE format: {spec}")
    return key, value


def _series_label(series_key, label_overrides):
    if series_key in label_overrides:
        return label_overrides[series_key]
    if series_key in SERIES_STYLES:
        return SERIES_STYLES[series_key]['label']
    return series_key.replace('-', ' ').replace('_', ' ').title()


def _series_style(series_key, series_index):
    fallback_colors = [
        '#0f766e',
        '#2563eb',
        '#d97706',
        '#7c3aed',
        '#dc2626',
        '#0891b2',
    ]
    fallback_markers = ['o', 's', 'D', '^', 'v', 'P']
    style = SERIES_STYLES.get(series_key, {})
    return {
        'color': style.get('color', fallback_colors[series_index % len(fallback_colors)]),
        'marker': style.get('marker', fallback_markers[series_index % len(fallback_markers)]),
        'label': style.get('label', _series_label(series_key, {})),
    }


def build_series_specs(args):
    label_overrides = {}
    for spec in args.series_label:
        key, label = _parse_kv_arg(spec, '--series_label')
        label_overrides[key] = label

    if args.series:
        return [
            {
                'key': key,
                'path': path,
                'label': _series_label(key, label_overrides),
            }
            for key, path in (_parse_kv_arg(spec, '--series') for spec in args.series)
        ]

    legacy_specs = [
        ('mixed', args.mixed_results),
        ('k1_5', args.k1_5_results),
        ('scratch', args.scratch_results),
    ]
    return [
        {
            'key': key,
            'path': path,
            'label': _series_label(key, label_overrides),
        }
        for key, path in legacy_specs
    ]


def load_series_entries(series_specs):
    series_entries = []

    print("\nLoading results...")
    for spec in series_specs:
        try:
            data = load_results(spec['path'])
            metrics = extract_metrics_by_size(data)
            series_entries.append({
                'key': spec['key'],
                'label': spec['label'],
                'path': spec['path'],
                'metrics': metrics,
            })
            print(f"  ✓ {spec['label']}: {spec['path']}")
        except Exception as e:
            series_entries.append({
                'key': spec['key'],
                'label': spec['label'],
                'path': spec['path'],
                'metrics': {},
            })
            print(f"  ✗ {spec['label']}: {e}")

    return series_entries


def plot_comparison(series_entries, output_path, title="Transfer Learning Comparison"):
    """Generate comparison plot for an arbitrary number of series."""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12

    # Combine all sample sizes
    all_sizes = sorted({
        size
        for entry in series_entries
        for size in entry['metrics'].keys()
    })

    if not all_sizes:
        print("⚠️  No data to plot!")
        return

    x_map = _build_sample_x_map(all_sizes)

    # Create figure
    fig_width = max(12, 10 + 1.2 * max(0, len(series_entries) - 3))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Plot each line
    for idx, entry in enumerate(series_entries):
        metrics_dict = entry['metrics']
        if metrics_dict:
            sizes = sorted(metrics_dict.keys())
            errors = [metrics_dict[s]['test_error'] for s in sizes]
            min_band = [metrics_dict[s].get('test_error_min') for s in sizes]
            max_band = [metrics_dict[s].get('test_error_max') for s in sizes]
            x_vals = [x_map[s] for s in sizes]

            style = _series_style(entry['key'], idx)
            ax.plot(x_vals, errors,
                   marker=style['marker'],
                   linestyle='-',
                   linewidth=2.5,
                   markersize=10,
                   label=entry['label'],
                   color=style['color'],
                   alpha=0.9)

            if any(v is not None for v in min_band) and any(v is not None for v in max_band):
                min_plot = [err if val is None else val for err, val in zip(errors, min_band)]
                max_plot = [err if val is None else val for err, val in zip(errors, max_band)]
                ax.fill_between(
                    x_vals,
                    min_plot,
                    max_plot,
                    color=style['color'],
                    alpha=0.15,
                    linewidth=0,
                )

    # Formatting
    ax.set_xlabel('Number of Downstream Training Samples', fontsize=26, fontweight='bold')
    ax.set_ylabel('Test Error (Relative L2)', fontsize=26, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    
    ax.set_yscale('log', base=10)
    _format_log_decade_yaxis(ax)
    xtick_positions = [x_map[s] for s in all_sizes]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([_format_sample_tick(n) for n in all_sizes], fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.margins(x=0.05)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    legend_cols = 1 if len(series_entries) <= 4 else 2
    ax.legend(loc='best', fontsize=22, frameon=True, shadow=True, fancybox=True, ncol=legend_cols)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    # Save PDF version
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Plot saved to: {pdf_path}")

    plt.close()


def print_summary_table(series_entries):
    """Print summary table of results for all provided series."""

    all_sizes = sorted({
        size
        for entry in series_entries
        for size in entry['metrics'].keys()
    })

    print("\n" + "="*80)
    print("TRANSFER LEARNING RESULTS SUMMARY")
    print("="*80)

    if not all_sizes:
        print("\nNo result points available.")
        return

    column_width = 20
    header = f"\n{'Samples':<12}" + "".join(f"{entry['label'][:column_width-1]:<{column_width}}" for entry in series_entries)
    print(header)
    print("-" * max(80, len(header)))

    for size in all_sizes:
        row = f"{size:<12}"
        for entry in series_entries:
            metrics = entry['metrics']
            if size in metrics:
                metric_str = f"{metrics[size]['test_error']:.6f}"
                row += f"{metric_str:<{column_width}}"
            else:
                row += f"{'N/A':<{column_width}}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Plot transfer learning comparison')

    parser.add_argument('--series', action='append', default=[],
                       help='Generic comparison series in KEY=PATH format. May be passed multiple times.')
    parser.add_argument('--series_label', action='append', default=[],
                       help='Optional legend label override in KEY=LABEL format. May be passed multiple times.')

    parser.add_argument('--mixed_results', type=str,
                       default='results/transfer_learning_k1_2.5/mixed/poisson_results.json',
                       help='Path to mixed-pretrained results JSON')
    parser.add_argument('--k1_5_results', type=str,
                       default='results/transfer_learning_k1_2.5/k1_5/poisson_results.json',
                       help='Path to k1_5-pretrained results JSON')
    parser.add_argument('--scratch_results', type=str,
                       default='results/transfer_learning_k1_2.5/scratch/poisson_results.json',
                       help='Path to from-scratch results JSON')
    parser.add_argument('--output_dir', type=str, default='results/transfer_learning_k1_2.5',
                       help='Output directory for plot')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Optional basename for the saved plot files, without extension')
    parser.add_argument('--title', type=str, default='Transfer Learning Comparison',
                       help='Plot title')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING COMPARISON PLOT")
    print("="*80)

    series_specs = build_series_specs(args)
    series_entries = load_series_entries(series_specs)

    print("\nExtracted data points:")
    for entry in series_entries:
        print(f"  {entry['label']}: {len(entry['metrics'])} sizes")

    # Print summary table
    print_summary_table(series_entries)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plot
    output_stem = determine_output_stem(args)
    output_path = output_dir / f'{output_stem}.png'
    plot_comparison(series_entries, str(output_path), args.title)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

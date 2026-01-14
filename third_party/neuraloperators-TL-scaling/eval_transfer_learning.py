#!/usr/bin/env python3
"""
Evaluate Transfer Learning Experiments and Generate Plots

This script evaluates multiple models (pretrained+finetuned, from-scratch, mixed-pretrained)
and generates plots similar to Figure 3a from the paper showing data efficiency curves.

Usage:
    # Evaluate all models and generate plot
    python eval_transfer_learning.py \
        --yaml_config config/operators_poisson.yaml \
        --experiment_type poisson \
        --output_dir results/transfer_learning

    # Evaluate only specific configurations
    python eval_transfer_learning.py \
        --yaml_config config/operators_poisson.yaml \
        --experiment_type poisson \
        --configs poisson-k5_10-finetune-16 poisson-k5_10-scratch-16 \
        --output_dir results/transfer_learning

    # Include mixed-pretrained models
    python eval_transfer_learning.py \
        --yaml_config config/operators_poisson.yaml \
        --experiment_type poisson \
        --include_mixed \
        --output_dir results/transfer_learning
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.distributed as dist
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.inferencer import Inferencer


# Plot styling
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['figure.dpi'] = 150


def parse_sample_size(config_name: str) -> Optional[int]:
    """Extract sample size from config name (e.g., 'finetune-16' -> 16)."""
    parts = config_name.split('-')
    for part in parts:
        if part.isdigit():
            return int(part)
        # Handle 1k, 4k notation
        if part.endswith('k') and part[:-1].isdigit():
            return int(part[:-1]) * 1024
    return None


def get_experiment_groups(experiment_type: str, include_mixed: bool = False) -> Dict[str, List[str]]:
    """
    Get experiment configuration groups for different scenarios.
    
    Args:
        experiment_type: Type of experiment ('poisson', 'advdiff', 'helmholtz')
        include_mixed: Whether to include mixed-pretrained experiments
    
    Returns:
        Dictionary mapping sample sizes to list of config names
    """
    if experiment_type == 'poisson':
        base_configs = {
            16: {
                'finetune': 'poisson-k5_10-finetune-16',
                'scratch': 'poisson-k5_10-scratch-16',
                'mixed': 'poisson-k5_10-finetune-mixed-16'
            },
            64: {
                'finetune': 'poisson-k5_10-finetune-64',
                'scratch': 'poisson-k5_10-scratch-64',
                'mixed': 'poisson-k5_10-finetune-mixed-64'
            },
            256: {
                'finetune': 'poisson-k5_10-finetune-256',
                'scratch': 'poisson-k5_10-scratch-256',
                'mixed': 'poisson-k5_10-finetune-mixed-256'
            },
            1024: {
                'finetune': 'poisson-k5_10-finetune-1k',
                'scratch': 'poisson-k5_10-scratch-1k',
                'mixed': 'poisson-k5_10-finetune-mixed-1k'
            },
            4096: {
                'finetune': 'poisson-k5_10-finetune-4k',
                'scratch': 'poisson-k5_10-scratch-4k',
                'mixed': 'poisson-k5_10-finetune-mixed-4k'
            },
            8192: {
                'finetune': 'poisson-k5_10-finetune-8k',
                'scratch': 'poisson-k5_10-scratch-8k',
                'mixed': 'poisson-k5_10-finetune-mixed-8k'
            },
        }
    elif experiment_type == 'advdiff':
        # Add advdiff configs here when ready
        base_configs = {}
    elif experiment_type == 'helmholtz':
        # Add helmholtz configs here when ready
        base_configs = {}
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Filter out mixed if not requested
    if not include_mixed:
        for size_configs in base_configs.values():
            size_configs.pop('mixed', None)
    
    return base_configs


def find_checkpoint(experiment_dir: str, config_name: str, run_pattern: str = "*") -> Optional[str]:
    """
    Find checkpoint file for a given experiment.
    
    Args:
        experiment_dir: Base experiments directory
        config_name: Configuration name
        run_pattern: Pattern to match run directories (e.g., "*", "finetune-16-*")
    
    Returns:
        Path to checkpoint file or None if not found
    """
    config_dir = Path(experiment_dir) / 'expts' / config_name
    
    if not config_dir.exists():
        logging.warning(f"Config directory not found: {config_dir}")
        return None
    
    # Find run directories matching pattern
    run_dirs = sorted(config_dir.glob(run_pattern))
    
    if not run_dirs:
        logging.warning(f"No run directories found in {config_dir} matching {run_pattern}")
        return None
    
    # Use the most recent run (or first if sorted differently)
    run_dir = run_dirs[-1]
    
    # Look for checkpoint
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        logging.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return None
    
    # Try best checkpoint first, then latest
    for ckpt_name in ['ckpt_best.tar', 'ckpt.tar']:
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            return str(ckpt_path)
    
    logging.warning(f"No checkpoint found in {ckpt_dir}")
    return None


def evaluate_model(yaml_config: str, config_name: str, checkpoint_path: str, 
                   device: str = 'cuda:0') -> Dict[str, float]:
    """
    Evaluate a single model and return metrics.
    
    Args:
        yaml_config: Path to YAML configuration file
        config_name: Configuration name
        checkpoint_path: Path to model checkpoint
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating: {config_name}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"{'='*60}")
    
    # Create temporary args
    class Args:
        def __init__(self):
            self.yaml_config = yaml_config
            self.config = config_name
            self.root_dir = './evaluation_tmp'
            self.run_num = 'eval'
            self.sweep = 'none'
            self.weights = checkpoint_path
    
    args = Args()
    
    try:
        # Load parameters
        params = YParams(os.path.abspath(args.yaml_config), args.config)
        params['weights'] = checkpoint_path
        
        # Create inferencer
        inferencer = Inferencer(params, args)
        
        # Launch inferencer (initializes model and runs test)
        inferencer.launch()
        
        # Extract metrics from logs
        test_error = inferencer.logs['test_err']
        test_loss = inferencer.logs['test_loss']
        
        # Handle tensor values
        if torch.is_tensor(test_error):
            test_error = test_error.item()
        if torch.is_tensor(test_loss):
            test_loss = test_loss.item()
        
        metrics = {
            'test_error': test_error,
            'test_loss': test_loss,
            'test_time': 0,  # Not returned by launch()
        }
        
        logging.info(f"Results: test_error={metrics['test_error']:.6f}, "
                    f"test_loss={metrics['test_loss']:.6f}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'test_error': np.nan, 'test_loss': np.nan, 'test_time': 0}


def plot_transfer_learning_curve(results: Dict[str, Dict[int, Dict[str, float]]],
                                 experiment_type: str,
                                 output_path: str,
                                 include_mixed: bool = False):
    """
    Create transfer learning data efficiency plot (Figure 3a style).
    
    Args:
        results: Nested dict {model_type: {sample_size: metrics}}
        experiment_type: Type of experiment for title
        output_path: Path to save plot
        include_mixed: Whether mixed results are included
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and markers
    colors = {
        'scratch': '#87CEEB',  # Light blue (open circles in paper)
        'finetune': '#4169E1',  # Royal blue (filled circles in paper)
        'mixed': '#FF6347',     # Tomato red (for mixed experiments)
    }
    
    markers = {
        'scratch': 'o',   # Open circle
        'finetune': 'o',  # Filled circle
        'mixed': 's',     # Square
    }
    
    labels = {
        'scratch': 'Training from Scratch',
        'finetune': 'TL from Pre-trained',
        'mixed': 'TL from Mixed Pre-trained',
    }
    
    linestyles = {
        'scratch': '--',
        'finetune': '-',
        'mixed': '-',
    }
    
    # Plot each model type
    for model_type in ['scratch', 'finetune', 'mixed']:
        if model_type not in results:
            continue
        
        sample_sizes = sorted(results[model_type].keys())
        errors = [results[model_type][size]['test_error'] for size in sample_sizes]
        
        # Filter out NaN values
        valid_points = [(s, e) for s, e in zip(sample_sizes, errors) if not np.isnan(e)]
        if not valid_points:
            continue
        
        sample_sizes_valid, errors_valid = zip(*valid_points)
        
        # Plot with appropriate style
        facecolors = 'none' if model_type == 'scratch' else colors[model_type]
        
        ax.plot(sample_sizes_valid, errors_valid,
                marker=markers[model_type],
                color=colors[model_type],
                markerfacecolor=facecolors,
                markeredgecolor=colors[model_type],
                markeredgewidth=2,
                markersize=8,
                linestyle=linestyles[model_type],
                linewidth=2,
                label=labels[model_type])
    
    # Formatting
    ax.set_xlabel('Number of downstream examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Testing error (relative $\ell_2$)', fontsize=14, fontweight='bold')
    ax.set_yscale('log', base=10)  # Base-10 log scale for better differentiation
    ax.set_xscale('log', base=2)
    
    # Set x-ticks to match paper
    xticks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    xtick_labels = ['8', '16', '32', '64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K']
    
    # Filter to only show ticks in data range
    data_samples = []
    for model_results in results.values():
        data_samples.extend(model_results.keys())
    
    if data_samples:
        min_sample = min(data_samples)
        max_sample = max(data_samples)
        
        valid_ticks = [(t, l) for t, l in zip(xticks, xtick_labels) 
                      if t >= min_sample/2 and t <= max_sample*2]
        
        if valid_ticks:
            xticks_filtered, xtick_labels_filtered = zip(*valid_ticks)
            ax.set_xticks(xticks_filtered)
            ax.set_xticklabels(xtick_labels_filtered)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add title with pre-training and downstream info
    if experiment_type == 'poisson':
        title = 'Poisson\'s Equation Transfer Learning\n'
        title += 'Pretrain: SYS-1(1,5)    Downstream: SYS-1(5,10)'
    elif experiment_type == 'advdiff':
        title = 'Advection-Diffusion Transfer Learning\n'
        title += 'Pretrain: SYS-2(0.2,1)    Downstream: SYS-2(...)'
    elif experiment_type == 'helmholtz':
        title = 'Helmholtz Transfer Learning\n'
        title += 'Pretrain: SYS-3(1,10)    Downstream: SYS-3(...)'
    else:
        title = f'{experiment_type.title()} Transfer Learning'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to: {output_path}")
    plt.close()


def plot_individual_comparison(results: Dict[str, Dict[int, Dict[str, float]]],
                               experiment_type: str,
                               output_path: str,
                               include_mixed: bool = False):
    """
    Create a 3-subplot figure comparing all three approaches side by side.
    
    Args:
        results: Nested dict {model_type: {sample_size: metrics}}
        experiment_type: Type of experiment for title
        output_path: Path to save plot
        include_mixed: Whether mixed results are included
    """
    # Determine how many subplots we need
    model_types = []
    if 'scratch' in results:
        model_types.append('scratch')
    if 'finetune' in results:
        model_types.append('finetune')
    if 'mixed' in results and include_mixed:
        model_types.append('mixed')
    
    if not model_types:
        logging.warning("No results to plot")
        return
    
    n_plots = len(model_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    
    # Make axes iterable if only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Define colors and markers
    colors = {
        'scratch': '#87CEEB',  # Light blue
        'finetune': '#4169E1',  # Royal blue
        'mixed': '#FF6347',     # Tomato red
    }
    
    titles = {
        'scratch': 'Training from Scratch',
        'finetune': 'TL from Pre-trained',
        'mixed': 'TL from Mixed Pre-trained',
    }
    
    # Set x-ticks
    xticks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    xtick_labels = ['8', '16', '32', '64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32K']
    
    # Get data range
    data_samples = []
    for model_results in results.values():
        data_samples.extend(model_results.keys())
    
    min_sample = min(data_samples) if data_samples else 16
    max_sample = max(data_samples) if data_samples else 4096
    
    valid_ticks = [(t, label) for t, label in zip(xticks, xtick_labels) 
                  if t >= min_sample/2 and t <= max_sample*2]
    
    if valid_ticks:
        xticks_filtered, xtick_labels_filtered = zip(*valid_ticks)
    else:
        xticks_filtered, xtick_labels_filtered = xticks, xtick_labels
    
    # Plot each model type in its own subplot
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in results:
            continue
        
        sample_sizes = sorted(results[model_type].keys())
        errors = [results[model_type][size]['test_error'] for size in sample_sizes]
        
        # Filter out NaN values
        valid_points = [(s, e) for s, e in zip(sample_sizes, errors) if not np.isnan(e)]
        if not valid_points:
            continue
        
        sample_sizes_valid, errors_valid = zip(*valid_points)
        
        # Plot with appropriate style
        facecolors = 'none' if model_type == 'scratch' else colors[model_type]
        linestyle = '--' if model_type == 'scratch' else '-'
        
        ax.plot(sample_sizes_valid, errors_valid,
                marker='o',
                color=colors[model_type],
                markerfacecolor=facecolors,
                markeredgecolor=colors[model_type],
                markeredgewidth=2,
                markersize=10,
                linestyle=linestyle,
                linewidth=2.5)
        
        # Formatting
        ax.set_xlabel('Number of downstream examples', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Testing error (relative $\ell_2$)', fontsize=12, fontweight='bold')
        ax.set_yscale('log', base=10)  # Base-10 log scale for better differentiation
        ax.set_xscale('log', base=2)
        ax.set_xticks(xticks_filtered)
        ax.set_xticklabels(xtick_labels_filtered)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_title(titles[model_type], fontsize=13, fontweight='bold', pad=10)
    
    # Add overall title
    if experiment_type == 'poisson':
        suptitle = 'Poisson\'s Equation Transfer Learning Comparison\n'
        suptitle += 'Pretrain: SYS-1(1,5)    Downstream: SYS-1(5,10)'
    elif experiment_type == 'advdiff':
        suptitle = 'Advection-Diffusion Transfer Learning Comparison\n'
        suptitle += 'Pretrain: SYS-2(0.2,1)    Downstream: SYS-2(...)'
    elif experiment_type == 'helmholtz':
        suptitle = 'Helmholtz Transfer Learning Comparison\n'
        suptitle += 'Pretrain: SYS-3(1,10)    Downstream: SYS-3(...)'
    else:
        suptitle = f'{experiment_type.title()} Transfer Learning Comparison'
    
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Individual comparison plot saved to: {output_path}")
    plt.close()


def save_results_json(results: Dict, output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results_clean = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


def print_results_table(results: Dict[str, Dict[int, Dict[str, float]]]):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Get all sample sizes
    all_sizes = set()
    for model_results in results.values():
        all_sizes.update(model_results.keys())
    
    sample_sizes = sorted(all_sizes)
    
    # Print header
    header = f"{'Model Type':<25} | " + " | ".join([f"{s:>8}" for s in sample_sizes])
    print(header)
    print("-" * len(header))
    
    # Print results for each model type
    for model_type in ['scratch', 'finetune', 'mixed']:
        if model_type not in results:
            continue
        
        row = f"{model_type.capitalize():<25} | "
        for size in sample_sizes:
            if size in results[model_type]:
                error = results[model_type][size]['test_error']
                if np.isnan(error):
                    row += f"{'N/A':>8} | "
                else:
                    row += f"{error:>8.6f} | "
            else:
                row += f"{'---':>8} | "
        
        print(row.rstrip(' |'))
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transfer learning experiments and generate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--yaml_config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--experiment_type',
        type=str,
        choices=['poisson', 'advdiff', 'helmholtz'],
        default='poisson',
        help='Type of experiment to evaluate'
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='experiments',
        help='Base directory containing experiment results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/transfer_learning',
        help='Directory to save evaluation results and plots'
    )
    
    parser.add_argument(
        '--include_mixed',
        action='store_true',
        help='Include mixed-pretrained models in evaluation'
    )
    
    parser.add_argument(
        '--configs',
        nargs='+',
        type=str,
        help='Specific config names to evaluate (overrides automatic detection)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run evaluation on'
    )
    
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip evaluation and only regenerate plots from existing results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f'{args.experiment_type}_results.json'
    
    # Load existing results if skipping evaluation
    if args.skip_evaluation and results_file.exists():
        logging.info(f"Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            results_raw = json.load(f)
        
        # Convert back to proper structure
        results = {}
        for model_type, size_dict in results_raw.items():
            results[model_type] = {int(k): v for k, v in size_dict.items()}
    
    else:
        # Get experiment configurations
        if args.configs:
            # Manual config specification
            configs_to_eval = args.configs
        else:
            # Automatic detection based on experiment type
            config_groups = get_experiment_groups(args.experiment_type, args.include_mixed)
            configs_to_eval = []
            for size, model_configs in config_groups.items():
                configs_to_eval.extend(model_configs.values())
        
        logging.info(f"Evaluating {len(configs_to_eval)} configurations")
        
        # Evaluate all models
        results = defaultdict(dict)
        
        for config_name in configs_to_eval:
            # Find checkpoint
            checkpoint = find_checkpoint(args.experiment_dir, config_name)
            
            if checkpoint is None:
                logging.warning(f"Skipping {config_name}: checkpoint not found")
                continue
            
            # Determine model type and sample size
            if 'scratch' in config_name:
                model_type = 'scratch'
            elif 'mixed' in config_name:
                model_type = 'mixed'
            else:
                model_type = 'finetune'
            
            sample_size = parse_sample_size(config_name)
            if sample_size is None:
                logging.warning(f"Could not parse sample size from {config_name}")
                continue
            
            # Evaluate
            metrics = evaluate_model(
                args.yaml_config,
                config_name,
                checkpoint,
                args.device
            )
            
            results[model_type][sample_size] = metrics
        
        # Convert defaultdict to regular dict for JSON serialization
        results = dict(results)
        
        # Save results
        save_results_json(results, results_file)
    
    # Print results table
    print_results_table(results)
    
    # Generate combined plot (all approaches on one graph)
    plot_path = output_dir / f'{args.experiment_type}_transfer_learning.png'
    plot_transfer_learning_curve(
        results,
        args.experiment_type,
        str(plot_path),
        args.include_mixed
    )
    
    # Also save as PDF
    plot_path_pdf = output_dir / f'{args.experiment_type}_transfer_learning.pdf'
    plot_transfer_learning_curve(
        results,
        args.experiment_type,
        str(plot_path_pdf),
        args.include_mixed
    )
    
    # Generate individual comparison plot (3 subplots side by side)
    comparison_path = output_dir / f'{args.experiment_type}_individual_comparison.png'
    plot_individual_comparison(
        results,
        args.experiment_type,
        str(comparison_path),
        args.include_mixed
    )
    
    # Also save comparison as PDF
    comparison_path_pdf = output_dir / f'{args.experiment_type}_individual_comparison.pdf'
    plot_individual_comparison(
        results,
        args.experiment_type,
        str(comparison_path_pdf),
        args.include_mixed
    )
    
    logging.info("\n" + "="*80)
    logging.info("EVALUATION COMPLETE!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("="*80)


if __name__ == '__main__':
    main()

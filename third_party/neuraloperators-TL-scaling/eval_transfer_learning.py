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
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from collections import defaultdict
from typing import Dict, List, Optional

from utils import logging_utils
from utils.YParams import YParams
from utils.inferencer import Inferencer


logging_utils.config_logger()


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
    if 'zeroshot' in config_name.lower():
        return 0

    parts = config_name.split('-')
    for part in parts:
        if part.isdigit():
            return int(part)
        # Handle 1k, 4k notation
        if part.endswith('k') and part[:-1].isdigit():
            return int(part[:-1]) * 1024
    return None


def format_sample_tick(size: int) -> str:
    """Format sample-size tick labels compactly to avoid overlap."""
    if size == 0:
        return '0-shot'
    if size >= 1024 and size % 1024 == 0:
        return f'{size // 1024}K'
    return str(size)


def sample_to_plot_x(size: int, min_positive_size: Optional[int]) -> float:
    """
    Map sample sizes to plotting coordinates with explicit log2 spacing.

    Positive sizes use log2(sample_size). Zero-shot gets a dedicated slot to the
    left of the smallest positive sample, avoiding symlog's nonlinear transition
    near zero that causes awkward spacing in these transfer-learning plots.
    """
    if size > 0:
        return float(np.log2(size))
    if min_positive_size is None:
        return 0.0
    return float(np.log2(min_positive_size) - 1.5)


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
            16384: {
                'finetune': 'poisson-k5_10-finetune-16k',
                'scratch': 'poisson-k5_10-scratch-16k',
                'mixed': 'poisson-k5_10-finetune-mixed-16k'
            },
            32768: {
                'finetune': 'poisson-k5_10-finetune-32k',
                'scratch': 'poisson-k5_10-scratch-32k',
                'mixed': 'poisson-k5_10-finetune-mixed-32k'
            },
        }
    elif experiment_type == 'advdiff':
        # Advection-Diffusion transfer learning experiments (see config/operators_ad.yaml)
        # Downstream: adr∈[0.2,0.4]
        # Pretrain:   adr∈[0.2,1]
        base_configs = {
            16: {
                'finetune': 'ad-adr0p2_0p4-finetune-16',
                'scratch': 'ad-adr0p2_0p4-scratch-16',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-16',
            },
            64: {
                'finetune': 'ad-adr0p2_0p4-finetune-64',
                'scratch': 'ad-adr0p2_0p4-scratch-64',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-64',
            },
            256: {
                'finetune': 'ad-adr0p2_0p4-finetune-256',
                'scratch': 'ad-adr0p2_0p4-scratch-256',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-256',
            },
            1024: {
                'finetune': 'ad-adr0p2_0p4-finetune-1k',
                'scratch': 'ad-adr0p2_0p4-scratch-1k',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-1k',
            },
            4096: {
                'finetune': 'ad-adr0p2_0p4-finetune-4k',
                'scratch': 'ad-adr0p2_0p4-scratch-4k',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-4k',
            },
            8192: {
                'finetune': 'ad-adr0p2_0p4-finetune-8k',
                'scratch': 'ad-adr0p2_0p4-scratch-8k',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-8k',
            },
            16384: {
                'finetune': 'ad-adr0p2_0p4-finetune-16k',
                'scratch': 'ad-adr0p2_0p4-scratch-16k',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-16k',
            },
            32768: {
                'finetune': 'ad-adr0p2_0p4-finetune-32k',
                'scratch': 'ad-adr0p2_0p4-scratch-32k',
                'mixed': 'ad-adr0p2_0p4-finetune-mixed-32k',
            },
        }
    elif experiment_type == 'helmholtz':
        # Helmholtz transfer learning experiments (see config/operators_helmholtz.yaml)
        # Downstream: o∈[1,5]
        # Pretrain:   o∈[1,10]
        base_configs = {
            16: {
                'finetune': 'helm-o1_5-finetune-16',
                'scratch': 'helm-o1_5-scratch-16',
                'mixed': 'helm-o1_5-finetune-mixed-16',
            },
            64: {
                'finetune': 'helm-o1_5-finetune-64',
                'scratch': 'helm-o1_5-scratch-64',
                'mixed': 'helm-o1_5-finetune-mixed-64',
            },
            256: {
                'finetune': 'helm-o1_5-finetune-256',
                'scratch': 'helm-o1_5-scratch-256',
                'mixed': 'helm-o1_5-finetune-mixed-256',
            },
            1024: {
                'finetune': 'helm-o1_5-finetune-1k',
                'scratch': 'helm-o1_5-scratch-1k',
                'mixed': 'helm-o1_5-finetune-mixed-1k',
            },
            4096: {
                'finetune': 'helm-o1_5-finetune-4k',
                'scratch': 'helm-o1_5-scratch-4k',
                'mixed': 'helm-o1_5-finetune-mixed-4k',
            },
            8192: {
                'finetune': 'helm-o1_5-finetune-8k',
                'scratch': 'helm-o1_5-scratch-8k',
                'mixed': 'helm-o1_5-finetune-mixed-8k',
            },
            16384: {
                'finetune': 'helm-o1_5-finetune-16k',
                'scratch': 'helm-o1_5-scratch-16k',
                'mixed': 'helm-o1_5-finetune-mixed-16k',
            },
            32768: {
                'finetune': 'helm-o1_5-finetune-32k',
                'scratch': 'helm-o1_5-scratch-32k',
                'mixed': 'helm-o1_5-finetune-mixed-32k',
            },
        }
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Filter out mixed if not requested
    if not include_mixed:
        for size_configs in base_configs.values():
            size_configs.pop('mixed', None)
    
    return base_configs


def _config_exists(yaml_config: str, config_name: str) -> bool:
    """Return True if a config exists in the YAML file."""
    try:
        YParams(os.path.abspath(yaml_config), config_name)
        return True
    except Exception:
        return False


def find_checkpoints(experiment_dir: str, config_name: str, run_pattern: str = "*") -> List[str]:
    """
    Find checkpoint files for a given experiment.
    
    Args:
        experiment_dir: Base experiments directory
        config_name: Configuration name
        run_pattern: Pattern to match run directories (e.g., "*", "finetune-16-*")
    
    Returns:
        Sorted list of checkpoint paths
    """
    config_dir = Path(experiment_dir) / 'expts' / config_name
    
    if not config_dir.exists():
        logging.warning(f"Config directory not found: {config_dir}")
        return []
    
    # Find run directories matching pattern
    run_dirs = sorted(config_dir.glob(run_pattern))
    
    if not run_dirs:
        logging.warning(f"No run directories found in {config_dir} matching {run_pattern}")
        return []

    checkpoints = []
    for run_dir in run_dirs:
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.exists():
            logging.warning(f"Checkpoint directory not found: {ckpt_dir}")
            continue
        for ckpt_name in ['ckpt_best.tar', 'ckpt.tar']:
            ckpt_path = ckpt_dir / ckpt_name
            if ckpt_path.exists():
                checkpoints.append(str(ckpt_path))
                break
        else:
            logging.warning(f"No checkpoint found in {ckpt_dir}")

    return checkpoints


def find_checkpoint(experiment_dir: str, config_name: str, run_pattern: str = "*") -> Optional[str]:
    """Return the latest matching checkpoint for backward-compatible single-run evaluation."""
    checkpoints = find_checkpoints(experiment_dir, config_name, run_pattern)
    if not checkpoints:
        return None
    return checkpoints[-1]


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate repeated-run metrics for mean/std and quartile reporting."""
    if not metrics_list:
        return {
            'test_error': np.nan,
            'test_loss': np.nan,
            'test_zero_mode_constraint_loss': np.nan,
            'test_pde_residual_norm': np.nan,
            'test_zero_mode_violation': np.nan,
            'n_trials': 0,
        }

    aggregated = {}
    metric_names = [
        'test_error',
        'test_loss',
        'test_zero_mode_constraint_loss',
        'test_pde_residual_norm',
        'test_zero_mode_violation',
    ]

    for metric_name in metric_names:
        values = np.array([m[metric_name] for m in metrics_list], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            aggregated[metric_name] = np.nan
            aggregated[f'{metric_name}_std'] = np.nan
            aggregated[f'{metric_name}_q1'] = np.nan
            aggregated[f'{metric_name}_q3'] = np.nan
            continue

        aggregated[metric_name] = float(np.mean(finite_values))
        aggregated[f'{metric_name}_std'] = float(np.std(finite_values))
        aggregated[f'{metric_name}_q1'] = float(np.quantile(finite_values, 0.25))
        aggregated[f'{metric_name}_q3'] = float(np.quantile(finite_values, 0.75))

    aggregated['n_trials'] = len(metrics_list)
    aggregated['trial_metrics'] = metrics_list
    return aggregated


def get_checkpoint_from_config(yaml_config: str, config_name: str) -> Optional[str]:
    """
    Read checkpoint path from a config's `weights` field.

    Used for zero-shot configs that are evaluated directly from pretraining
    checkpoints and therefore have no downstream run directory.
    """
    try:
        params = YParams(os.path.abspath(yaml_config), config_name)
    except Exception as exc:
        logging.warning(f"Could not load config {config_name}: {exc}")
        return None

    if 'weights' not in params:
        logging.warning(f"Config {config_name} has no `weights` field")
        return None

    checkpoint = params['weights']
    if not checkpoint:
        logging.warning(f"Config {config_name} has empty `weights`")
        return None

    return str(checkpoint)


def evaluate_model(yaml_config: str, config_name: str, checkpoint_path: Optional[str],
                   device: str = 'cuda:0') -> Dict[str, float]:
    """
    Evaluate a single model and return metrics.
    
    Args:
        yaml_config: Path to YAML configuration file
        config_name: Configuration name
        checkpoint_path: Path to model checkpoint, or None for random-init eval
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating: {config_name}")
    if checkpoint_path is None:
        logging.info("Checkpoint: <random initialization>")
    else:
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
        if checkpoint_path is not None:
            params['weights'] = checkpoint_path
        
        # Create inferencer
        inferencer = Inferencer(params, args)
        
        # Launch inferencer (initializes model and runs test)
        inferencer.launch()
        
        # Extract metrics from logs
        test_error = inferencer.logs['test_err']
        test_loss = inferencer.logs['test_loss']
        test_zero_mode_constraint_loss = inferencer.logs.get('test_zero_mode_constraint_loss', np.nan)
        test_pde_residual_norm = inferencer.logs.get('test_pde_residual_norm', np.nan)
        test_zero_mode_violation = inferencer.logs.get('test_zero_mode_violation', np.nan)
        
        # Handle tensor values
        if torch.is_tensor(test_error):
            test_error = test_error.item()
        if torch.is_tensor(test_loss):
            test_loss = test_loss.item()
        if torch.is_tensor(test_zero_mode_constraint_loss):
            test_zero_mode_constraint_loss = test_zero_mode_constraint_loss.item()
        if torch.is_tensor(test_pde_residual_norm):
            test_pde_residual_norm = test_pde_residual_norm.item()
        if torch.is_tensor(test_zero_mode_violation):
            test_zero_mode_violation = test_zero_mode_violation.item()
        
        metrics = {
            'test_error': test_error,
            'test_loss': test_loss,
            'test_zero_mode_constraint_loss': test_zero_mode_constraint_loss,
            'test_pde_residual_norm': test_pde_residual_norm,
            'test_zero_mode_violation': test_zero_mode_violation,
            'test_time': 0,  # Not returned by launch()
        }
        
        logging.info(f"Results: test_error={metrics['test_error']:.6f}, "
                    f"test_loss={metrics['test_loss']:.6f}, "
                    f"test_zero_mode_constraint_loss={metrics['test_zero_mode_constraint_loss']:.6f}, "
                    f"test_pde_residual_norm={metrics['test_pde_residual_norm']:.6f}, "
                    f"test_zero_mode_violation={metrics['test_zero_mode_violation']:.6f}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'test_error': np.nan,
            'test_loss': np.nan,
            'test_zero_mode_constraint_loss': np.nan,
            'test_pde_residual_norm': np.nan,
            'test_zero_mode_violation': np.nan,
            'test_time': 0
        }


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

    data_samples = []
    for model_results in results.values():
        data_samples.extend(model_results.keys())
    positive_data_samples = [s for s in data_samples if s > 0]
    min_positive_size = min(positive_data_samples) if positive_data_samples else None
    
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
        lower_quartiles = [results[model_type][size].get('test_error_q1', np.nan) for size in sample_sizes]
        upper_quartiles = [results[model_type][size].get('test_error_q3', np.nan) for size in sample_sizes]
        
        # Filter out NaN values
        valid_points = [
            (s, e, q1, q3)
            for s, e, q1, q3 in zip(sample_sizes, errors, lower_quartiles, upper_quartiles)
            if not np.isnan(e)
        ]
        if not valid_points:
            continue
        
        sample_sizes_valid, errors_valid, q1_valid, q3_valid = zip(*valid_points)
        
        # Plot with appropriate style
        facecolors = 'none' if model_type == 'scratch' else colors[model_type]
        
        x_vals = [sample_to_plot_x(s, min_positive_size) for s in sample_sizes_valid]

        ax.plot(x_vals, errors_valid,
                marker=markers[model_type],
                color=colors[model_type],
                markerfacecolor=facecolors,
                markeredgecolor=colors[model_type],
                markeredgewidth=2,
                markersize=8,
                linestyle=linestyles[model_type],
                linewidth=2,
                label=labels[model_type])

        if any(not np.isnan(v) for v in q1_valid) and any(not np.isnan(v) for v in q3_valid):
            ax.fill_between(
                x_vals,
                q1_valid,
                q3_valid,
                color=colors[model_type],
                alpha=0.15,
                linewidth=0,
            )
    
    # Formatting
    ax.set_xlabel('Number of downstream examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Testing error (relative $\ell_2$)', fontsize=14, fontweight='bold')
    ax.set_yscale('log', base=10)  # Base-10 log scale for better differentiation
    
    # Set x-ticks to match paper
    xticks = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    xtick_labels = [format_sample_tick(x) for x in xticks]
    
    # Filter to only show ticks in data range
    if data_samples:
        min_sample = min(data_samples)
        max_sample = max(data_samples)
        
        valid_ticks = []
        for t, label in zip(xticks, xtick_labels):
            if t == 0 and min_sample == 0:
                valid_ticks.append((t, label))
            elif t > 0 and t >= max(min_sample / 2, 1) and t <= max_sample * 2:
                valid_ticks.append((t, label))
        
        if valid_ticks:
            xticks_filtered, xtick_labels_filtered = zip(*valid_ticks)
            xtick_positions = [sample_to_plot_x(int(t), min_positive_size) for t in xticks_filtered]
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels_filtered)
            ax.margins(x=0.05)
    
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
    xticks = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    xtick_labels = [format_sample_tick(x) for x in xticks]
    
    # Get data range
    data_samples = []
    for model_results in results.values():
        data_samples.extend(model_results.keys())
    
    min_sample = min(data_samples) if data_samples else 16
    max_sample = max(data_samples) if data_samples else 4096
    positive_data_samples = [s for s in data_samples if s > 0]
    min_positive_size = min(positive_data_samples) if positive_data_samples else None
    
    valid_ticks = []
    for t, label in zip(xticks, xtick_labels):
        if t == 0 and min_sample == 0:
            valid_ticks.append((t, label))
        elif t > 0 and t >= max(min_sample / 2, 1) and t <= max_sample * 2:
            valid_ticks.append((t, label))
    
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
        
        x_vals = [sample_to_plot_x(s, min_positive_size) for s in sample_sizes_valid]

        ax.plot(x_vals, errors_valid,
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
        xtick_positions = [sample_to_plot_x(int(t), min_positive_size) for t in xticks_filtered]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels_filtered)
        ax.margins(x=0.05)
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
                std = results[model_type][size].get('test_error_std', np.nan)
                n_trials = int(results[model_type][size].get('n_trials', 1))
                if np.isnan(error):
                    row += f"{'N/A':>8} | "
                elif n_trials > 1 and np.isfinite(std):
                    row += f"{error:.4f}±{std:.2g} | "
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

    parser.add_argument(
        '--aggregate_runs',
        action='store_true',
        help='Evaluate all matching downstream runs per config and aggregate mean/std/quartiles'
    )

    parser.add_argument(
        '--run_pattern',
        type=str,
        default='*',
        help='Glob for downstream run directories, e.g. "*-seed*" when aggregating multi-seed runs'
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

            # Optional scratch zero-shot baseline (random initialization, no training).
            # Add it only when the config exists for the selected YAML/experiment.
            scratch_zeroshot_by_experiment = {
                'poisson': 'poisson-k5_10-scratch-zeroshot',
                'advdiff': 'ad-adr0p2_0p4-scratch-zeroshot',
                'helmholtz': 'helm-o1_5-scratch-zeroshot',
            }
            scratch_zeroshot_cfg = scratch_zeroshot_by_experiment.get(args.experiment_type)
            if scratch_zeroshot_cfg and scratch_zeroshot_cfg not in configs_to_eval:
                if _config_exists(args.yaml_config, scratch_zeroshot_cfg):
                    configs_to_eval = [scratch_zeroshot_cfg] + configs_to_eval
                    logging.info(f"Added scratch zero-shot config to automatic evaluation: {scratch_zeroshot_cfg}")
                else:
                    logging.info(
                        f"Scratch zero-shot config not found in {args.yaml_config}; skipping automatic addition: "
                        f"{scratch_zeroshot_cfg}"
                    )
        
        logging.info(f"Evaluating {len(configs_to_eval)} configurations")
        
        # Evaluate all models
        results = defaultdict(dict)
        
        for config_name in configs_to_eval:
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

            # Zero-shot configs evaluate directly from their configured pretraining
            # checkpoint instead of a downstream run directory.
            if sample_size == 0:
                if model_type == 'scratch':
                    checkpoint = None
                    logging.info(f"Using random-init zero-shot evaluation for {config_name}")
                else:
                    checkpoint = get_checkpoint_from_config(args.yaml_config, config_name)
                    if checkpoint is None:
                        logging.warning(f"Skipping {config_name}: zero-shot weights not found in config")
                        continue
                metrics = evaluate_model(
                    args.yaml_config,
                    config_name,
                    checkpoint,
                    args.device
                )
            else:
                if args.aggregate_runs:
                    checkpoints = find_checkpoints(args.experiment_dir, config_name, args.run_pattern)
                    if not checkpoints:
                        logging.warning(f"Skipping {config_name}: checkpoints not found")
                        continue

                    trial_metrics = []
                    for checkpoint in checkpoints:
                        trial_metrics.append(
                            evaluate_model(
                                args.yaml_config,
                                config_name,
                                checkpoint,
                                args.device
                            )
                        )
                    metrics = aggregate_metrics(trial_metrics)
                else:
                    checkpoint = find_checkpoint(args.experiment_dir, config_name, args.run_pattern)
                    if checkpoint is None:
                        logging.warning(f"Skipping {config_name}: checkpoint not found")
                        continue

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

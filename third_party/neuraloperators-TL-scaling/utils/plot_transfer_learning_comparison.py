#!/usr/bin/env python3
"""
Plot combined transfer learning comparison from separate evaluation results.

This script takes the results from three separate evaluations (mixed, k1_5, scratch)
and creates a single comparison plot.

Usage:
    # With default paths (poisson_results.json)
    python utils/plot_transfer_learning_comparison.py
    
    # With custom paths
    python utils/plot_transfer_learning_comparison.py \
        --mixed_results results/transfer_learning_k1_2.5/mixed/poisson_results.json \
        --k1_5_results results/transfer_learning_k1_2.5/k1_5/poisson_results.json \
        --scratch_results results/transfer_learning_k1_2.5/scratch/poisson_results.json \
        --output_dir results/transfer_learning_k1_2.5
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(filepath):
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_errors_by_size(results):
    """Extract test errors organized by sample size"""
    errors = {}
    
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
                            errors[size_int] = metrics['test_error']
                        elif 'test_err' in metrics:
                            errors[size_int] = metrics['test_err']
                    elif isinstance(metrics, (int, float)):
                        errors[size_int] = metrics
    
    return errors


def plot_comparison(mixed_errors, k1_5_errors, scratch_errors, output_path, title="Transfer Learning Comparison"):
    """Generate comparison plot"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 12
    
    # Combine all sample sizes
    all_sizes = sorted(set(
        list(mixed_errors.keys()) + 
        list(k1_5_errors.keys()) + 
        list(scratch_errors.keys())
    ))
    
    if not all_sizes:
        print("⚠️  No data to plot!")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each line
    markers = {'mixed': 'o', 'k1_5': 's', 'scratch': '^'}
    colors = {'mixed': '#2ecc71', 'k1_5': '#3498db', 'scratch': '#e74c3c'}
    # Keep legend labels experiment-agnostic; the meaning of the middle curve
    # depends on what results you feed in (Poisson-pretrained, AdvDiff-pretrained, etc.).
    labels = {
        'mixed': 'Fine-tuned (Mixed-Domain Pretraining)',
        'k1_5': 'Fine-tuned (Single-Domain Pretraining)',
        'scratch': 'Trained from Scratch'
    }
    
    for errors_dict, key in [(mixed_errors, 'mixed'), (k1_5_errors, 'k1_5'), (scratch_errors, 'scratch')]:
        if errors_dict:
            sizes = sorted(errors_dict.keys())
            errors = [errors_dict[s] for s in sizes]
            
            ax.plot(sizes, errors, 
                   marker=markers[key],
                   linestyle='-',
                   linewidth=2.5,
                   markersize=10,
                   label=labels[key],
                   color=colors[key],
                   alpha=0.9)
    
    # Formatting
    ax.set_xlabel('Number of Downstream Training Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Error (Relative L2)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Use symlog on x-axis so zero-shot (0 samples) can be plotted.
    ax.set_xscale('symlog', linthresh=1)
    ax.set_yscale('log', base=10)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels(['0-shot' if n == 0 else str(n) for n in all_sizes])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True, fancybox=True)
    
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


def print_summary_table(mixed_errors, k1_5_errors, scratch_errors):
    """Print summary table of results"""
    
    all_sizes = sorted(set(
        list(mixed_errors.keys()) + 
        list(k1_5_errors.keys()) + 
        list(scratch_errors.keys())
    ))
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Samples':<12} {'Mixed':<18} {'k1_5':<18} {'Scratch':<18}")
    print("-" * 80)
    
    for size in all_sizes:
        mixed_str = f"{mixed_errors[size]:.6f}" if size in mixed_errors else "N/A"
        k1_5_str = f"{k1_5_errors[size]:.6f}" if size in k1_5_errors else "N/A"
        scratch_str = f"{scratch_errors[size]:.6f}" if size in scratch_errors else "N/A"
        
        print(f"{size:<12} {mixed_str:<18} {k1_5_str:<18} {scratch_str:<18}")
    
    # Calculate improvements
    print("\n" + "="*80)
    print("TRANSFER LEARNING BENEFITS (% improvement over scratch)")
    print("="*80)
    
    for size in all_sizes:
        scratch_err = scratch_errors.get(size)
        if scratch_err:
            print(f"\n{size} samples:")
            
            # Mixed improvement
            mixed_err = mixed_errors.get(size)
            if mixed_err:
                improvement = ((scratch_err - mixed_err) / scratch_err) * 100
                print(f"  Mixed pretraining:   {improvement:+6.2f}%")
            
            # k1_5 improvement
            k1_5_err = k1_5_errors.get(size)
            if k1_5_err:
                improvement = ((scratch_err - k1_5_err) / scratch_err) * 100
                print(f"  k1_5 pretraining:    {improvement:+6.2f}%")
            
            # Mixed vs k1_5
            if mixed_err and k1_5_err:
                if mixed_err < k1_5_err:
                    improvement = ((k1_5_err - mixed_err) / k1_5_err) * 100
                    winner = "mixed"
                else:
                    improvement = ((mixed_err - k1_5_err) / mixed_err) * 100
                    winner = "k1_5"
                print(f"  Mixed vs k1_5:       {improvement:+6.2f}% ({winner} better)")


def main():
    parser = argparse.ArgumentParser(description='Plot transfer learning comparison')
    
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
    parser.add_argument('--title', type=str, default='Transfer Learning Comparison',
                       help='Plot title')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TRANSFER LEARNING COMPARISON PLOT")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    try:
        mixed_data = load_results(args.mixed_results)
        print(f"  ✓ Mixed: {args.mixed_results}")
    except Exception as e:
        print(f"  ✗ Mixed: {e}")
        mixed_data = {}
    
    try:
        k1_5_data = load_results(args.k1_5_results)
        print(f"  ✓ k1_5: {args.k1_5_results}")
    except Exception as e:
        print(f"  ✗ k1_5: {e}")
        k1_5_data = {}
    
    try:
        scratch_data = load_results(args.scratch_results)
        print(f"  ✓ Scratch: {args.scratch_results}")
    except Exception as e:
        print(f"  ✗ Scratch: {e}")
        scratch_data = {}
    
    # Extract errors by sample size
    mixed_errors = extract_errors_by_size(mixed_data)
    k1_5_errors = extract_errors_by_size(k1_5_data)
    scratch_errors = extract_errors_by_size(scratch_data)
    
    print("\nExtracted data points:")
    print(f"  Mixed: {len(mixed_errors)} sizes")
    print(f"  k1_5: {len(k1_5_errors)} sizes")
    print(f"  Scratch: {len(scratch_errors)} sizes")
    
    # Print summary table
    print_summary_table(mixed_errors, k1_5_errors, scratch_errors)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plot
    output_path = output_dir / 'transfer_learning_comparison.png'
    plot_comparison(mixed_errors, k1_5_errors, scratch_errors, str(output_path), args.title)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

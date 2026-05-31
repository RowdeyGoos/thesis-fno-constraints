# Transfer Learning Evaluation Guide

## Overview

This guide explains how to evaluate your transfer learning experiments and generate comparison plots showing test error vs number of training samples.

## What Gets Evaluated

You have trained models using three different approaches:

1. **Mixed-Pretrained Fine-tuning**: Models pretrained on mixed-domain data (Poisson + AdvDiff + Helmholtz), then fine-tuned on Poisson k∈[1,2.5]
2. **k1_5-Pretrained Fine-tuning**: Models pretrained on Poisson k∈[1,5], then fine-tuned on Poisson k∈[1,2.5]  
3. **From-Scratch**: Models trained directly on Poisson k∈[1,2.5] without pretraining

Each approach was tested at 5 data sizes: 16, 64, 256, 1k, 4k samples.

## Quick Start

### Option 1: SLURM Batch Evaluation (Recommended)

Run all evaluations and generate plots in a single SLURM job:

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh
```

This will:
1. Evaluate all mixed-pretrained models
2. Evaluate all k1_5-pretrained models
3. Evaluate all from-scratch models
4. Generate a combined comparison plot

**Time:** ~2-4 hours total

### Option 2: Manual Evaluation

Evaluate each approach separately:

```bash
# 1. Mixed-pretrained
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_mixed.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-mixed-16 \
             poisson-k1_2.5-finetune-mixed-64 \
             poisson-k1_2.5-finetune-mixed-256 \
             poisson-k1_2.5-finetune-mixed-1k \
             poisson-k1_2.5-finetune-mixed-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/mixed

# 2. k1_5-pretrained
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-16 \
             poisson-k1_2.5-finetune-64 \
             poisson-k1_2.5-finetune-256 \
             poisson-k1_2.5-finetune-1k \
             poisson-k1_2.5-finetune-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/k1_5

# 3. From-scratch
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-scratch-16 \
             poisson-k1_2.5-scratch-64 \
             poisson-k1_2.5-scratch-256 \
             poisson-k1_2.5-scratch-1k \
             poisson-k1_2.5-scratch-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/scratch

# 4. Generate combined plot
python scripts/eval/plot_transfer_learning_comparison.py \
    --mixed_results results/transfer_learning_k1_2.5/mixed/results.json \
    --k1_5_results results/transfer_learning_k1_2.5/k1_5/results.json \
    --scratch_results results/transfer_learning_k1_2.5/scratch/results.json \
    --output_dir results/transfer_learning_k1_2.5 \
    --title "Transfer Learning: Poisson k∈[1,2.5]"
```

### Option 3: Bash Script (Local)

```bash
bash scripts/eval/eval_all_transfer_learning.sh
```

## Understanding the Evaluation Process

### What `scripts/entrypoints/eval_transfer_learning.py` Does

For each model configuration:
1. **Finds checkpoint**: Locates `ckpt_best.tar` in the experiment directory
2. **Loads model**: Restores model weights from checkpoint
3. **Runs inference**: Evaluates on test set using `scripts/entrypoints/eval.py`
4. **Computes metrics**: Calculates test error (relative L2 norm)
5. **Saves results**: Stores metrics in JSON format

### What `plot_transfer_learning_comparison.py` Does

1. **Loads results**: Reads JSON files from all three approaches
2. **Extracts errors**: Organizes test errors by sample size
3. **Generates plot**: Creates comparison plot with 3 lines
4. **Calculates improvements**: Shows % improvement of fine-tuning over scratch
5. **Saves outputs**: PNG and PDF versions of the plot

## Output Structure

After evaluation, you'll have:

```
results/transfer_learning_k1_2.5/
├── mixed/
│   ├── results.json                      # Mixed-pretrained results
│   ├── mixed_k1_2.5_transfer_learning.png
│   └── mixed_k1_2.5_transfer_learning.pdf
├── k1_5/
│   ├── results.json                      # k1_5-pretrained results
│   ├── k1_2.5_finetune_transfer_learning.png
│   └── k1_2.5_finetune_transfer_learning.pdf
├── scratch/
│   ├── results.json                      # From-scratch results
│   ├── k1_2.5_scratch_transfer_learning.png
│   └── k1_2.5_scratch_transfer_learning.pdf
├── transfer_learning_comparison.png      # Combined comparison plot ⭐
├── transfer_learning_comparison.pdf      # PDF version
└── eval_logs/                            # Detailed evaluation logs
```

The **main result** is `transfer_learning_comparison.png` which shows all three approaches on one plot.

## Understanding the Plot

The comparison plot shows:

- **X-axis**: Number of downstream training samples (log scale)
- **Y-axis**: Test error (relative L2 norm, lower is better)
- **Three lines**:
  - 🟢 **Green circles**: Mixed-pretrained fine-tuning
  - 🔵 **Blue squares**: k1_5-pretrained fine-tuning
  - 🔴 **Red triangles**: From-scratch training

### Expected Patterns

1. **All lines decrease** as training data increases (more data = better performance)
2. **Fine-tuning below scratch**: Transfer learning should help, especially at low data
3. **Lines converge**: As data increases, the benefit of pretraining decreases
4. **Mixed vs k1_5**: May vary depending on domain similarity

## Analyzing Results

### Summary Table

The script prints a summary table:

```
Samples      Mixed             k1_5              Scratch          
--------------------------------------------------------------------------------
16           0.123456          0.134567          0.156789
64           0.098765          0.104321          0.123456
...
```

### Transfer Learning Benefits

The script calculates improvement percentages:

```
16 samples:
  Mixed pretraining:   +20.35% improvement over scratch
  k1_5 pretraining:    +14.12% improvement over scratch
  Mixed vs k1_5:       +5.45% (mixed better)
```

**Interpretation:**
- **Positive %**: Transfer learning helps
- **Higher % at low data**: Confirms data efficiency benefit
- **Smaller % at high data**: Diminishing returns with more data

## Troubleshooting

### Checkpoint Not Found

```
⚠️  No checkpoint found for config: poisson-k1_2.5-finetune-mixed-16
```

**Solution:** Check if training completed successfully:
```bash
ls experiments/expts/poisson-k1_2.5-finetune-mixed-16/*/checkpoints/
```

### Evaluation Failed

```
⚠️  Evaluation failed for config: ...
```

**Possible causes:**
1. Checkpoint corrupted
2. Config mismatch
3. Dataset path wrong
4. GPU memory issue

**Debug:**
```bash
# Run evaluation manually for that specific config
python scripts/entrypoints/eval.py \
    --yaml_config config/operators_mixed.yaml \
    --config poisson-k1_2.5-finetune-mixed-16 \
    --weights experiments/expts/.../checkpoints/ckpt_best.tar \
    --root_dir debug_eval
```

### Missing Data Points

If some sample sizes are missing from the plot:

1. Check if training jobs completed
2. Verify checkpoint exists
3. Re-run evaluation for that specific size
4. Use `--force_reeval` flag

### Plot Looks Wrong

Common issues:
- **Lines crossing unexpectedly**: May indicate training issue
- **Large jumps**: Check if using correct checkpoints
- **Scratch better than fine-tuning**: Unusual, verify pretrained model was loaded

## Advanced Usage

### Evaluate Specific Configs Only

```bash
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_mixed.yaml \
    --configs poisson-k1_2.5-finetune-mixed-16 poisson-k1_2.5-finetune-mixed-64 \
    --output_dir results/mixed_subset
```

### Use Existing Results (Skip Re-evaluation)

```bash
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_mixed.yaml \
    --experiment_type mixed_k1_2.5 \
    --results_file results/transfer_learning_k1_2.5/mixed/results.json
```

### Custom Plot Title

```bash
python scripts/eval/plot_transfer_learning_comparison.py \
    --mixed_results results/transfer_learning_k1_2.5/mixed/results.json \
    --k1_5_results results/transfer_learning_k1_2.5/k1_5/results.json \
    --scratch_results results/transfer_learning_k1_2.5/scratch/results.json \
    --output_dir results \
    --title "Data Efficiency: Transfer Learning on SYS-1"
```

## Integration with Paper

This evaluation generates **Figure 3a-style plots** for your thesis:

1. **Include in paper**: Use the PDF version for publication quality
2. **Caption**: "Transfer learning data efficiency on Poisson k∈[1,2.5] downstream task"
3. **Discussion points**:
   - Quantify improvement at each data size
   - Compare mixed-domain vs in-domain pretraining
   - Discuss when pretraining is most beneficial

## Next Steps

After evaluation:

1. ✅ Verify results make sense (fine-tuning should generally beat scratch)
2. ✅ Calculate statistical significance if you have multiple runs
3. ✅ Compare with literature baselines
4. ✅ Analyze failure cases (which sample sizes don't show improvement?)
5. ✅ Include plots in your thesis/paper

## Additional Plots

You can create additional visualizations:

```python
# In a Jupyter notebook
import json
import matplotlib.pyplot as plt

# Load results
with open('results/transfer_learning_k1_2.5/mixed/results.json') as f:
    mixed = json.load(f)

# Custom plot
# ... your analysis code ...
```

See `notebooks/02_baseline_results.ipynb` for analysis templates.

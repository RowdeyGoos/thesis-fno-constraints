# Transfer Learning Evaluation - Quick Reference

## Files Created

### 1. Evaluation Scripts
- **`scripts/eval_all_transfer_learning.sh`** - Bash script to run all evaluations
- **`utils/plot_transfer_learning_comparison.py`** - Create combined comparison plot
- **`scripts/slurm/submit_eval_transfer_learning.sh`** - SLURM job submission

### 2. Documentation
- **`EVALUATION_GUIDE.md`** - Comprehensive evaluation guide

### 3. Existing Script (Already in Repo)
- **`eval_transfer_learning.py`** - Main evaluation script (already exists!)
- **`eval.py`** - Core evaluation using Inferencer class

## Quick Start

### SLURM (Recommended)

```bash
sbatch scripts/slurm/submit_eval_transfer_learning.sh
```

This evaluates all models and creates the comparison plot automatically.

### Local Execution

```bash
bash scripts/eval_all_transfer_learning.sh
```

## What You Get

A comparison plot with 3 lines showing test error vs number of training samples:

1. 🟢 **Mixed-pretrained** fine-tuning (green circles)
2. 🔵 **k1_5-pretrained** fine-tuning (blue squares)  
3. 🔴 **From-scratch** training (red triangles)

**Output:** `results/transfer_learning_k1_2.5/transfer_learning_comparison.png`

## How It Works

### Step 1: Evaluate Each Approach

The script `eval_transfer_learning.py` (which already exists in your repo):
1. Finds checkpoints for each config
2. Runs `eval.py` on test sets
3. Collects test errors
4. Saves results to JSON

### Step 2: Generate Combined Plot

The script `plot_transfer_learning_comparison.py`:
1. Loads results from all three JSON files
2. Creates a single plot comparing all approaches
3. Prints summary table and improvement percentages

## Verifying `eval.py` Works

Yes, `eval.py` is the correct tool for evaluation! It:

- ✅ Takes a checkpoint path via `--weights`
- ✅ Loads model from checkpoint
- ✅ Runs inference on test set
- ✅ Computes test error (relative L2)
- ✅ Saves results to `logs_best.txt`

The existing `eval_transfer_learning.py` is a wrapper that:
- Automatically finds checkpoints
- Runs `eval.py` for multiple configs
- Aggregates results into comparison plots

## Expected Results

### Data Efficiency Pattern

```
Low data (16-256 samples):
  → Large gap between fine-tuning and scratch
  → Transfer learning provides big benefit

High data (1k-4k samples):
  → Smaller gap
  → Diminishing returns from pretraining
```

### Mixed vs k1_5 Comparison

- **Mixed**: Broader pretraining (Poisson + AdvDiff + Helmholtz)
- **k1_5**: In-domain pretraining (Poisson k∈[1,5] only)

Which is better depends on:
- Domain overlap: k∈[1,2.5] is subset of k∈[1,5] (favors k1_5)
- Diversity: Mixed has broader features (may help generalization)

## Directory Structure

```
results/transfer_learning_k1_2.5/
├── mixed/
│   ├── results.json
│   └── *.png/pdf plots
├── k1_5/
│   ├── results.json  
│   └── *.png/pdf plots
├── scratch/
│   ├── results.json
│   └── *.png/pdf plots
└── transfer_learning_comparison.png  ⭐ Main result
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Checkpoint not found | Check training completed: `ls experiments/expts/<config>/*/checkpoints/` |
| Evaluation fails | Run manually with `eval.py` to see error |
| Missing data points | Re-run training for that sample size |
| Plot empty | Check JSON files have data |

## Next Steps

1. Submit evaluation job: `sbatch scripts/slurm/submit_eval_transfer_learning.sh`
2. Wait for completion (~2-4 hours)
3. Check results: `ls results/transfer_learning_k1_2.5/`
4. View plot: `transfer_learning_comparison.png`
5. Include in thesis/paper

## Manual Evaluation (Alternative)

If you prefer more control:

```bash
# Evaluate one approach at a time
python eval_transfer_learning.py \
    --yaml_config config/operators_mixed.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-mixed-16 \
             poisson-k1_2.5-finetune-mixed-64 \
             poisson-k1_2.5-finetune-mixed-256 \
             poisson-k1_2.5-finetune-mixed-1k \
             poisson-k1_2.5-finetune-mixed-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/mixed

python eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-16 \
             poisson-k1_2.5-finetune-64 \
             poisson-k1_2.5-finetune-256 \
             poisson-k1_2.5-finetune-1k \
             poisson-k1_2.5-finetune-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/k1_5

python eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-scratch-16 \
             poisson-k1_2.5-scratch-64 \
             poisson-k1_2.5-scratch-256 \
             poisson-k1_2.5-scratch-1k \
             poisson-k1_2.5-scratch-4k \
    --experiment_dir experiments \
    --output_dir results/transfer_learning_k1_2.5/scratch

# Combine into one plot
python utils/plot_transfer_learning_comparison.py \
    --mixed_results results/transfer_learning_k1_2.5/mixed/results.json \
    --k1_5_results results/transfer_learning_k1_2.5/k1_5/results.json \
    --scratch_results results/transfer_learning_k1_2.5/scratch/results.json \
    --output_dir results/transfer_learning_k1_2.5
```

## Summary

✅ **eval.py works** for evaluation (uses Inferencer class)
✅ **eval_transfer_learning.py exists** (wrapper for batch evaluation)
✅ **New scripts created** for combined plotting and SLURM submission
✅ **Ready to use** - just submit the SLURM job!

The comparison plot will show exactly what you need: 3 lines (mixed, k1_5, scratch) with test error vs number of training samples.

# Transfer Learning Evaluation and Plotting

This directory contains tools for evaluating transfer learning experiments and generating publication-quality plots similar to Figure 3a from the paper.

## Overview

The evaluation system can:
- ✅ Evaluate multiple models (pretrained+finetuned, from-scratch, mixed-pretrained)
- ✅ Generate data efficiency curves showing performance vs. number of samples
- ✅ Compare different pretraining strategies
- ✅ Export results in JSON format for further analysis
- ✅ Create plots in both PNG and PDF formats

## Files

- **`scripts/entrypoints/eval_transfer_learning.py`**: Main evaluation and plotting script
- **`scripts/eval/evaluate_transfer_learning.sh`**: Quick helper script for common use cases
- **`EVAL_TRANSFER_LEARNING.md`**: This documentation file

## Quick Start

### Basic Usage

Evaluate all Poisson transfer learning experiments:

```bash
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --output_dir results/transfer_learning
```

### Include Mixed-Pretrained Models

```bash
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --include_mixed \
    --output_dir results/transfer_learning
```

### Using the Helper Script

```bash
# Basic evaluation
bash scripts/eval/evaluate_transfer_learning.sh

# Include mixed models
bash scripts/eval/evaluate_transfer_learning.sh --mixed

# Regenerate plots without re-evaluating
bash scripts/eval/evaluate_transfer_learning.sh --skip-eval

# Different experiment type
bash scripts/eval/evaluate_transfer_learning.sh --type advdiff
```

## Command Line Arguments

### Required Arguments

- `--yaml_config`: Path to YAML configuration file (e.g., `config/operators_poisson.yaml`)
- `--experiment_type`: Type of experiment (`poisson`, `advdiff`, or `helmholtz`)

### Optional Arguments

- `--experiment_dir`: Base directory containing experiment results (default: `experiments`)
- `--output_dir`: Directory to save results and plots (default: `results/transfer_learning`)
- `--include_mixed`: Include mixed-pretrained models in evaluation
- `--configs`: Specify exact config names to evaluate (overrides automatic detection)
- `--device`: Device to run evaluation on (default: `cuda:0`)
- `--skip_evaluation`: Skip evaluation and only regenerate plots from existing results

## Output Files

After running, you'll find the following in `output_dir`:

1. **`{experiment_type}_results.json`**: Raw evaluation results in JSON format
2. **`{experiment_type}_transfer_learning.png`**: High-resolution plot (300 DPI)
3. **`{experiment_type}_transfer_learning.pdf`**: Vector PDF plot for publications

### Results JSON Structure

```json
{
  "scratch": {
    "16": {
      "test_error": 0.123456,
      "test_loss": 0.234567,
      "test_time": 1.23
    },
    ...
  },
  "finetune": {
    "16": {
      "test_error": 0.012345,
      "test_loss": 0.023456,
      "test_time": 1.23
    },
    ...
  },
  "mixed": {
    ...
  }
}
```

## Expected Experiment Structure

The script expects experiments to be organized as follows:

```
experiments/
├── expts/
│   ├── poisson-k5_10-finetune-16/
│   │   └── finetune-16-JOBID-0/
│   │       └── checkpoints/
│   │           └── ckpt_best.tar
│   ├── poisson-k5_10-scratch-16/
│   │   └── scratch-16-JOBID-5/
│   │       └── checkpoints/
│   │           └── ckpt_best.tar
│   ├── poisson-k5_10-finetune-mixed-16/
│   │   └── finetune-mixed-16-JOBID-0/
│   │       └── checkpoints/
│   │           └── ckpt_best.tar
│   ...
```

## Configuration Naming Convention

The script automatically detects experiments based on config names:

### Poisson Experiments
- **Fine-tuning**: `poisson-k5_10-finetune-{16,64,256,1k,4k}`
- **From scratch**: `poisson-k5_10-scratch-{16,64,256,1k,4k}`
- **Mixed fine-tuning**: `poisson-k5_10-finetune-mixed-{16,64,256,1k,4k}`

### Sample Size Notation
- `16`, `64`, `256`: Exact numbers
- `1k`: 1024 samples
- `4k`: 4096 samples

## Evaluating Specific Configurations

If you want to evaluate only specific experiments:

```bash
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k5_10-finetune-16 \
             poisson-k5_10-finetune-64 \
             poisson-k5_10-scratch-16 \
             poisson-k5_10-scratch-64 \
    --output_dir results/custom_eval
```

## Plot Customization

The plot follows the paper's style:

- **From Scratch**: Open blue circles (○) with dashed line
- **Fine-tuned**: Filled blue circles (●) with solid line
- **Mixed Fine-tuned**: Filled red squares (■) with solid line
- **Y-axis**: Log scale (relative L2 error)
- **X-axis**: Log₂ scale with custom labels (16, 64, 256, 1K, 4K, etc.)

To customize the plot appearance, edit the `plot_transfer_learning_curve()` function in `scripts/entrypoints/eval_transfer_learning.py`.

## Adding New Experiment Types

To add support for new PDEs (e.g., Helmholtz, Advection-Diffusion):

1. Edit `get_experiment_groups()` in `scripts/entrypoints/eval_transfer_learning.py`
2. Add the config name patterns for your experiments
3. Update the title generation in `plot_transfer_learning_curve()`

Example for Advection-Diffusion:

```python
elif experiment_type == 'advdiff':
    base_configs = {
        16: {
            'finetune': 'ad-downstream-finetune-16',
            'scratch': 'ad-downstream-scratch-16',
        },
        64: {
            'finetune': 'ad-downstream-finetune-64',
            'scratch': 'ad-downstream-scratch-64',
        },
        # ... more configs
    }
```

## Troubleshooting

### "Checkpoint not found" warnings

**Cause**: Experiment hasn't completed or checkpoint is in a different location

**Solution**:
1. Check that experiments have finished running
2. Verify checkpoint files exist in `experiments/expts/{config_name}/*/checkpoints/`
3. Use `--configs` to manually specify which configs to evaluate

### "Could not parse sample size" warnings

**Cause**: Config name doesn't follow expected pattern

**Solution**:
1. Ensure config names include sample size (e.g., `-16`, `-64`, `-1k`)
2. Or manually specify configs with `--configs`

### Import errors

**Cause**: Script needs to be run from the repository root

**Solution**:
```bash
cd /path/to/thesis-fno-constraints
python scripts/entrypoints/eval_transfer_learning.py ...
```

### CUDA out of memory

**Cause**: Evaluation uses GPU memory

**Solution**:
1. Use smaller batch sizes in config files
2. Evaluate on CPU: `--device cpu`
3. Evaluate configs one at a time

## Advanced Usage

### Batch Evaluation of Multiple Experiments

Create a script to evaluate different experiment types:

```bash
#!/bin/bash

for exp_type in poisson advdiff helmholtz; do
    echo "Evaluating $exp_type..."
    python scripts/entrypoints/eval_transfer_learning.py \
        --yaml_config config/operators_${exp_type}.yaml \
        --experiment_type $exp_type \
        --include_mixed \
        --output_dir results/${exp_type}_transfer_learning
done
```

### Comparing Different Pretraining Strategies

To compare single-domain vs. mixed-domain pretraining:

```bash
# Generate plot with both
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --include_mixed \
    --output_dir results/pretraining_comparison
```

The plot will show three curves:
1. Training from scratch (baseline)
2. Transfer from single-domain pretraining
3. Transfer from mixed-domain pretraining

### Regenerating Plots

If you want to modify the plot style without re-evaluating:

```bash
# First evaluation (saves results.json)
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --output_dir results/transfer_learning

# Modify plot_transfer_learning_curve() in scripts/entrypoints/eval_transfer_learning.py

# Regenerate plots from saved results
python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --output_dir results/transfer_learning \
    --skip_evaluation
```

## Integration with Weights & Biases

If you want to export results to W&B:

```python
import wandb
import json

# Load results
with open('results/transfer_learning/poisson_results.json', 'r') as f:
    results = json.load(f)

# Log to W&B
wandb.init(project="neuraloperators", name="transfer_learning_eval")

for model_type, size_results in results.items():
    for size, metrics in size_results.items():
        wandb.log({
            f"{model_type}/samples_{size}/test_error": metrics['test_error'],
            f"{model_type}/samples_{size}/test_loss": metrics['test_loss'],
        })

wandb.finish()
```

## Citation

If you use this evaluation framework, please cite the original paper:

```bibtex
@article{subramanian2023towards,
  title={Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior},
  author={Subramanian, Shashank and others},
  journal={arXiv preprint arXiv:2306.00258},
  year={2023}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example commands in this README
3. Check logs in `evaluation_tmp/` directory for detailed error messages

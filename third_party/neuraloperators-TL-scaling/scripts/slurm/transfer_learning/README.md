# Transfer Learning SLURM Scripts

This directory contains organized SLURM job submission scripts for transfer learning experiments, grouped by sample size to optimize time allocation.

## Directory Structure

```
transfer_learning/
├── README.md                          # This file
├── submit_k1_2.5_small.sh            # Poisson k∈[1,2.5]: 16, 64 samples (30 min)
├── submit_k1_2.5_medium.sh           # Poisson k∈[1,2.5]: 256, 1k samples (1 hr)
├── submit_k1_2.5_large.sh            # Poisson k∈[1,2.5]: 4k, 8k, 16k, 32k samples (2 hr)
├── submit_mixed_small.sh             # Mixed fine-tuning: 16, 64 samples (30 min)
├── submit_mixed_medium.sh            # Mixed fine-tuning: 256, 1k samples (1 hr)
└── submit_mixed_large.sh             # Mixed fine-tuning: 4k, 8k, 16k, 32k samples (2 hr)
```

## Scripts Overview

### Poisson k∈[1,2.5] Transfer Learning

These scripts compare fine-tuning (with k1_5 pre-trained weights) vs. training from scratch on the Poisson k∈[1,2.5] downstream task.

#### `submit_k1_2.5_small.sh`
- **Time allocation**: 30 minutes
- **Array size**: 4 jobs (0-3)
- **Experiments**:
  - Fine-tuning: 16, 64 samples (tasks 0-1)
  - From-scratch: 16, 64 samples (tasks 2-3)

#### `submit_k1_2.5_medium.sh`
- **Time allocation**: 1 hour
- **Array size**: 4 jobs (0-3)
- **Experiments**:
  - Fine-tuning: 256, 1k samples (tasks 0-1)
  - From-scratch: 256, 1k samples (tasks 2-3)

#### `submit_k1_2.5_large.sh`
- **Time allocation**: 2 hours
- **Array size**: 8 jobs (0-7)
- **Experiments**:
  - Fine-tuning: 4k, 8k, 16k, 32k samples (tasks 0-3)
  - From-scratch: 4k, 8k, 16k, 32k samples (tasks 4-7)

### Mixed Dataset Fine-Tuning

These scripts fine-tune the mixed-pretrained model (trained on Poisson + AdvDiff + Helmholtz) on the Poisson k∈[1,2.5] downstream task.

#### `submit_mixed_small.sh`
- **Time allocation**: 30 minutes
- **Array size**: 2 jobs (0-1)
- **Experiments**: 16, 64 samples

#### `submit_mixed_medium.sh`
- **Time allocation**: 1 hour
- **Array size**: 2 jobs (0-1)
- **Experiments**: 256, 1k samples

#### `submit_mixed_large.sh`
- **Time allocation**: 2 hours
- **Array size**: 4 jobs (0-3)
- **Experiments**: 4k, 8k, 16k, 32k samples

## Prerequisites

### For Poisson k∈[1,2.5] scripts:
1. Pre-trained model checkpoint from `poisson-scale-k1_5` pretraining
2. Generated Poisson k∈[1,2.5] data (3-component tensor format)
3. Computed scales for k∈[1,2.5] data
4. Update `PRETRAIN_CHECKPOINT` variable in scripts with actual job ID

### For Mixed fine-tuning scripts:
1. Completed mixed dataset pretraining
2. Updated checkpoint paths in `config/operators_mixed.yaml`
3. Poisson k∈[1,2.5] data with compatible tensor format

## Usage

### Submit all Poisson k∈[1,2.5] experiments:
```bash
# Small samples (16, 64)
sbatch scripts/slurm/transfer_learning/submit_k1_2.5_small.sh

# Medium samples (256, 1k)
sbatch scripts/slurm/transfer_learning/submit_k1_2.5_medium.sh

# Large samples (4k, 8k, 16k, 32k)
sbatch scripts/slurm/transfer_learning/submit_k1_2.5_large.sh
```

### Submit all Mixed fine-tuning experiments:
```bash
# Small samples (16, 64)
sbatch scripts/slurm/transfer_learning/submit_mixed_small.sh

# Medium samples (256, 1k)
sbatch scripts/slurm/transfer_learning/submit_mixed_medium.sh

# Large samples (4k, 8k, 16k, 32k)
sbatch scripts/slurm/transfer_learning/submit_mixed_large.sh
```

### Submit specific sample sizes only:
```bash
# Only small and medium samples for both Poisson and Mixed
sbatch scripts/slurm/transfer_learning/submit_k1_2.5_small.sh
sbatch scripts/slurm/transfer_learning/submit_k1_2.5_medium.sh
sbatch scripts/slurm/transfer_learning/submit_mixed_small.sh
sbatch scripts/slurm/transfer_learning/submit_mixed_medium.sh
```

## Configuration

### Time Allocations
- **30 minutes**: Sufficient for 16-64 sample training (typically converges quickly)
- **1 hour**: Appropriate for 256-1k sample training
- **2 hours**: Required for 4k-32k sample training (more epochs needed)

### Resource Allocation
All scripts use:
- **Partition**: `insy,general`
- **GPU**: 1x A40 (40GB VRAM)
- **CPUs**: 10 cores
- **Memory**: 32GB RAM
- **Nodes**: 1

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View specific job logs:
```bash
# Poisson k∈[1,2.5] logs
tail -f experiments/neuralop-k1_2.5-small-<JOBID>-<TASKID>.out
tail -f experiments/neuralop-k1_2.5-medium-<JOBID>-<TASKID>.out
tail -f experiments/neuralop-k1_2.5-large-<JOBID>-<TASKID>.out

# Mixed fine-tuning logs
tail -f experiments/neuralop-mixed-small-<JOBID>_<TASKID>.out
tail -f experiments/neuralop-mixed-medium-<JOBID>_<TASKID>.out
tail -f experiments/neuralop-mixed-large-<JOBID>_<TASKID>.out
```

## Output Locations

Results are saved to:
```
experiments/expts/<config_name>/
├── checkpoints/
│   └── ckpt_best.tar
├── logs/
└── figures/
```

Where `<config_name>` is one of:
- `poisson-k1_2.5-finetune-{16,64,256,1k,4k,8k,16k,32k}`
- `poisson-k1_2.5-scratch-{16,64,256,1k,4k,8k,16k,32k}`
- `poisson-k1_2.5-finetune-mixed-{16,64,256,1k,4k,8k,16k,32k}`

## Notes

- Scripts automatically create necessary directories (`experiments/`, `wandb/`)
- Pre-trained checkpoint paths must be updated before running fine-tuning experiments
- Mixed fine-tuning requires `in_dim: 6` to match the mixed-pretrained model
- From-scratch experiments provide baselines for comparison with fine-tuning
- All experiments log to Weights & Biases (W&B) when `log_to_wandb: true`

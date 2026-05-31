# Transfer Learning Setup - Summary

This document summarizes the transfer learning experimental setup created for comparing pre-trained vs from-scratch models.

## 🎯 Experiment Goal

Compare the data efficiency of:
1. **Fine-tuned models**: Pre-trained on Poisson k∈[1,5], fine-tuned on Poisson k∈[5,10]
2. **From-scratch models**: Trained directly on Poisson k∈[5,10] without pre-training

Test with varying amounts of downstream data: **16, 64, 256, 1k, and 4k samples**

## 📁 Files Created

### 1. Configuration File Updates
**File**: `config/operators_poisson.yaml`

**Added 11 new configurations**:

#### Base Configuration
- `poisson-k5_10-base`: Base settings for k5_10 domain

#### Fine-tuning Configurations (with pre-trained weights)
- `poisson-k5_10-finetune-16`: Fine-tune with 16 samples
- `poisson-k5_10-finetune-64`: Fine-tune with 64 samples
- `poisson-k5_10-finetune-256`: Fine-tune with 256 samples
- `poisson-k5_10-finetune-1k`: Fine-tune with 1k samples
- `poisson-k5_10-finetune-4k`: Fine-tune with 4k samples

#### From-Scratch Configurations (no pre-training)
- `poisson-k5_10-scratch-16`: Train from scratch with 16 samples
- `poisson-k5_10-scratch-64`: Train from scratch with 64 samples
- `poisson-k5_10-scratch-256`: Train from scratch with 256 samples
- `poisson-k5_10-scratch-1k`: Train from scratch with 1k samples
- `poisson-k5_10-scratch-4k`: Train from scratch with 4k samples

### 2. SLURM Array Script
**File**: `scripts/slurm/finetune/poisson/submit_finetune_poisson_k5_10_array.sh`

**Features**:
- Single GPU per task (10 tasks total)
- Array tasks 0-4: Fine-tuning experiments
- Array tasks 5-9: From-scratch experiments
- Time limit: 24 hours
- Partition: insy,general
- Automatic checkpoint verification

### 3. Helper Script
**File**: `scripts/maintenance/update_checkpoint_path.sh`

**Purpose**: Update checkpoint paths in configs and scripts after pretraining completes

**Usage**:
```bash
bash scripts/maintenance/update_checkpoint_path.sh <PRETRAIN_JOBID>
```

### 4. Documentation Updates
**File**: `REPRODUCTION_GUIDE.md`

**Added comprehensive transfer learning section**:
- Step-by-step workflow
- Expected results table
- Troubleshooting guide
- Alternative manual approach

**File**: `scripts/slurm/README.md`

**Created complete SLURM scripts documentation**:
- All available scripts
- Usage examples
- Quick start workflow
- Monitoring commands
- Common issues and solutions

## 🔧 Configuration Details

### Sample Size Implementation
Uses the `subsample` parameter to control training data size:

| Samples | Subsample Value | Calculation |
|---------|----------------|-------------|
| 16      | 2048          | 32768 / 2048 = 16 |
| 64      | 512           | 32768 / 512 = 64 |
| 256     | 128           | 32768 / 128 = 256 |
| 1k      | 32            | 32768 / 32 = 1024 |
| 4k      | 8             | 32768 / 8 = 4096 |

### Training Parameters

#### Fine-tuning (with pre-training)
- Max epochs: 200 (reduced from 500)
- Learning rate: 1E-3
- Loads checkpoint via `weights` parameter

#### From-scratch (no pre-training)
- Max epochs: 500 (full training)
- Learning rate: 1E-3
- No checkpoint loading

### Batch Sizes (adjusted for small datasets)
- 16 samples: batch_size = 16
- 64 samples: batch_size = 32
- 256 samples: batch_size = 64
- 1k samples: batch_size = 128
- 4k samples: batch_size = 128

## 🚀 Complete Workflow

### Step 1: Generate Data
```bash
# Source domain (k1_5) - for pretraining
python scripts/data/gen_data_poisson.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/poisson --e1 1 --e2 5
python scripts/data/compute_scales.py --datapath data/poisson --filename _train_k1_5_32k.h5

# Target domain (k5_10) - for transfer learning
python scripts/data/gen_data_poisson.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/poisson --e1 5 --e2 10
python scripts/data/compute_scales.py --datapath data/poisson --filename _train_k5_10_32k.h5
```

### Step 2: Pretrain
```bash
sbatch scripts/slurm/pretrain/submit_pretrain_array_single_gpu.sh
# Note the job ID (e.g., 12345)
```

### Step 3: Update Checkpoint Path
```bash
bash scripts/maintenance/update_checkpoint_path.sh 12345
```

### Step 4: Run Transfer Learning Experiments
```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k5_10_array.sh
```

### Step 5: Analyze Results
Check experiment outputs in:
```
experiments/expts/
├── poisson-k5_10-finetune-16/
├── poisson-k5_10-finetune-64/
├── poisson-k5_10-finetune-256/
├── poisson-k5_10-finetune-1k/
├── poisson-k5_10-finetune-4k/
├── poisson-k5_10-scratch-16/
├── poisson-k5_10-scratch-64/
├── poisson-k5_10-scratch-256/
├── poisson-k5_10-scratch-1k/
└── poisson-k5_10-scratch-4k/
```

## 📊 Expected Results

Based on the paper, you should observe:

1. **Large gap with few samples**: Fine-tuned models significantly outperform from-scratch models when data is limited (16-256 samples)

2. **Gap narrows with more data**: The advantage of pre-training decreases as more downstream data becomes available

3. **Convergence speed**: Fine-tuned models converge faster (200 epochs) compared to from-scratch models (500 epochs)

### Approximate L2 Error Expectations

| Samples | From-Scratch | Fine-tuned | Improvement |
|---------|-------------|------------|-------------|
| 16      | 0.15-0.20   | 0.05-0.08  | ~60-70%     |
| 64      | 0.10-0.15   | 0.04-0.06  | ~50-60%     |
| 256     | 0.08-0.12   | 0.03-0.05  | ~40-50%     |
| 1k      | 0.05-0.08   | 0.03-0.04  | ~30-40%     |
| 4k      | 0.04-0.06   | 0.02-0.03  | ~20-30%     |

## 🔍 Monitoring

### Check Job Status
```bash
squeue -u $USER | grep poisson-k5_10-transfer
```

### View Logs
```bash
# Specific task
cat experiments/neuralop-poisson-k5_10-transfer-<JOBID>-<TASKID>.out

# All tasks
ls -lh experiments/neuralop-poisson-k5_10-transfer-*
```

### Check W&B Dashboard
If W&B is configured, view real-time training metrics at:
```
https://wandb.ai/rowdey_goos-tu-delft/neuraloperators
```

## 📈 Analysis Tips

1. **Compare validation curves**: Plot validation loss over epochs for fine-tuned vs from-scratch

2. **Test set performance**: Compare final test L2 error for each sample size

3. **Data efficiency plot**: Create plot showing L2 error vs number of training samples for both approaches

4. **Training time**: Compare convergence speed (epochs to reach certain error threshold)

5. **Statistical significance**: Run multiple seeds if needed for confidence intervals

## 🎓 Paper Reference

This experimental setup is based on:
> Subramanian et al., "Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior of Neural Operators", 2023

Section on transfer learning demonstrates that pre-training on diverse PDE problems enables efficient adaptation to new domains with limited data.

## ✅ Verification Checklist

- [ ] Data generated for k1_5 (source domain)
- [ ] Data generated for k5_10 (target domain)
- [ ] Scales computed for both domains
- [ ] Pretraining completed successfully
- [ ] Checkpoint path updated in configs
- [ ] Transfer learning array job submitted
- [ ] Jobs completed successfully
- [ ] Results logged to W&B or local storage
- [ ] Performance gap observed between fine-tuned and from-scratch
- [ ] Results match paper's trends

## 🐛 Troubleshooting

### Checkpoint not found
- Check pretraining actually completed
- Verify job ID in `update_checkpoint_path.sh`
- Manually check path: `ls experiments/expts/poisson-scale-k1_5/*/checkpoints/`

### Jobs failing with small datasets
- Batch size might be too large for 16 samples
- Config already adjusted, but verify in logs
- Check data loading is working correctly

### Poor performance on small datasets
- Expected for from-scratch models
- Fine-tuned models should perform much better
- If both perform poorly, check data quality

### W&B not logging
- Check W&B credentials: `wandb login`
- Verify `log_to_wandb: !!bool True` in configs
- Check network connectivity from compute nodes

---

**Created**: December 10, 2025  
**For**: Thesis FNO Constraints Project  
**Contact**: See main README for support

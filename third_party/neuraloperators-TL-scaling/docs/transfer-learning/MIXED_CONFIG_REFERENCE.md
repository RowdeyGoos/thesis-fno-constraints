# Mixed Dataset Configuration Reference

This document describes the new separated configuration structure for mixed dataset experiments.

## File Structure

All mixed dataset configurations are now in:
```
config/operators_mixed.yaml
```

Single-domain configurations remain in:
```
config/operators_poisson.yaml
config/operators_helmholtz.yaml  
config/operators_ad.yaml
```

## Configuration Hierarchy

### Mixed Dataset Pretraining

**Config name**: `mixed-scale-all`
- **Base**: `mixed-pretrain-base` (defined in operators_mixed.yaml)
- **Training data**: Combines Poisson k∈[1,5] + AdvDiff αdr∈[0.2,1] + Helmholtz ω∈[1,10]
- **Model**: FNO with 6 input channels (1 source + 5 tensor)
- **Dataset**: `data/mixed/_train_mixed_32k.h5`

### Downstream Fine-Tuning

All fine-tuning configs are in `operators_mixed.yaml`:
- `poisson-k1_2.5-finetune-mixed-16` - 16 samples
- `poisson-k1_2.5-finetune-mixed-64` - 64 samples
- `poisson-k1_2.5-finetune-mixed-256` - 256 samples
- `poisson-k1_2.5-finetune-mixed-1k` - 1024 samples
- `poisson-k1_2.5-finetune-mixed-4k` - 4096 samples

Each inherits from `poisson-k1_2.5-base` and:
- Uses 6 input channels (to match mixed pretrained weights)
- Loads checkpoint from mixed pretraining
- Trains on Poisson k∈[1,2.5] downstream task (SYS-1 in paper)

## Workflow

### 1. Create Mixed Dataset

```bash
# Training set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_train_k1_5_32k.h5 \
    --advdiff_path data/advdiff/_train_adr0.2_1_32k.h5 \
    --helmholtz_path data/helmholtz/_train_o1_10_32k.h5 \
    --output_path data/mixed/_train_mixed_32k.h5

# Validation set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_val_k1_5_4k.h5 \
    --advdiff_path data/advdiff/_val_adr0.2_1_4k.h5 \
    --helmholtz_path data/helmholtz/_val_o1_10_4k.h5 \
    --output_path data/mixed/_val_mixed_4k.h5

# Test set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_test_k1_5_4k.h5 \
    --advdiff_path data/advdiff/_test_adr0.2_1_4k.h5 \
    --helmholtz_path data/helmholtz/_test_o1_10_4k.h5 \
    --output_path data/mixed/_test_mixed_4k.h5
```

### 2. Pretrain on Mixed Dataset

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

This uses `config/operators_mixed.yaml` with config name `mixed-scale-all`.

### 3. Update Checkpoint Paths

After pretraining completes (note the job ID):

```bash
bash scripts/utils/update_mixed_checkpoint_path.sh <job_id>
```

This updates all fine-tuning configs in `operators_mixed.yaml` to point to the correct checkpoint.

### 4. Fine-Tune on Downstream Tasks

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

This submits 5 jobs (array 0-4) for different data sizes, all using `config/operators_mixed.yaml`.

### 5. Evaluate Results

```bash
python eval_transfer_learning.py --include_mixed
```

This will:
- Auto-detect all experiments (scratch, pretrained, mixed-pretrained)
- Generate Figure 3a-style plots comparing all approaches
- Output results to JSON and PNG/PDF files

## Key Differences from Single-Domain Configs

### Input Channels
- **Single-domain** (e.g., Poisson k∈[1,2.5]): `in_dim: 4` (1 source + 3 tensor)
- **Mixed-domain**: `in_dim: 6` (1 source + 5 tensor)

### Zero-Padding Strategy
When fine-tuning mixed-pretrained model on Poisson:
- Keep `in_dim: 6` to match pretrained weights
- Poisson data is automatically zero-padded to 5 tensor components
- Extra channels receive zeros but model has learned to handle them

### Tensor Layout
```
Poisson:    [k11, k12, k22,  0,  0]
AdvDiff:    [k11, k12, k22, vx, vy]  
Helmholtz:  [k,  omega,  0,  0,  0]
```

## Configuration Details

### Mixed Pretraining Config

```yaml
mixed-scale-all:
  <<: *mixed_pretrain_base
  train_path:    'data/mixed/_train_mixed_32k.h5'
  val_path:      'data/mixed/_val_mixed_4k.h5'
  test_path:     'data/mixed/_test_mixed_4k.h5'
  scales_path:   'data/mixed/train_mixed_scales.npy'
  batch_size: 128
  valid_batch_size: 128
  log_to_wandb: true
  mode_cut: 32
  embed_cut: 64
  fc_cut: 2
  max_epochs: 50
  in_dim: 6
```

### Fine-Tuning Config Example

```yaml
poisson-k1_2.5-finetune-mixed-256:
  <<: *poisson_k1_2p5_base
  weights: 'experiments/expts/mixed-scale-all/pretrain-mixed-JOBID-0/checkpoints/ckpt_best.tar'
  subsample: 128   # 32768 / 128 = 256 samples
  batch_size: 64
  valid_batch_size: 128
  max_epochs: 50
  in_dim: 6  # Must match mixed model (6 input channels)
```

## Scripts Updated

The following scripts now use `config/operators_mixed.yaml`:

1. **`scripts/slurm/pretrain/submit_pretrain_mixed.sh`**
   - Line 57: `CONFIG_FILE="config/operators_mixed.yaml"`
   
2. **`scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh`**
   - Line 59: `CONFIG_FILE="config/operators_mixed.yaml"`

3. **`scripts/utils/update_mixed_checkpoint_path.sh`** (new file)
   - Updates checkpoint paths in `operators_mixed.yaml`

## Related Documentation

- `MIXED_DATASET_EXPLANATION.md` - Mixed dataset format and implementation
- `EVAL_TRANSFER_LEARNING.md` - Evaluation and plotting system
- `TRANSFER_LEARNING_QUICK_START.md` - Complete transfer learning workflow

## Verification

To verify the configuration is correct:

```bash
# Check that mixed configs exist
grep -n "mixed-scale-all:" config/operators_mixed.yaml

# Check that scripts reference correct file
grep "CONFIG_FILE=" scripts/slurm/pretrain/submit_pretrain_mixed.sh
grep "CONFIG_FILE=" scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh

# Verify no mixed configs remain in Poisson file
grep -i "mixed" config/operators_poisson.yaml
```

The last command should return no results (mixed configs have been moved).

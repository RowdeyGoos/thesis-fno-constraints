# Converting Poisson Datasets for Mixed-Pretrained Model Fine-Tuning

## Problem

When fine-tuning a mixed-pretrained model on a single-domain Poisson task, there's a tensor dimension mismatch:

- **Standard Poisson datasets**: 3-component tensors `[k11, k12, k22]`
- **Mixed-pretrained model**: Expects 6 input channels = 1 (source) + 5 (tensor)

## Solution

Convert Poisson datasets to have 5-component tensors with zero-padding: `[k11, k12, k22, 0, 0]`

This maintains compatibility with the mixed-pretrained model while preserving all Poisson data.

## Quick Start

### Option 1: Batch Convert All Splits (Recommended)

```bash
# Convert train, val, and test datasets at once
bash scripts/utils/convert_k1_2.5_to_mixed_format.sh
```

This will:
- Convert all three splits in place
- Create `.backup` files of originals
- Verify the conversion

### Option 2: Convert Individual Files

```bash
# Convert a single dataset
python utils/convert_poisson_to_mixed_format.py \
    --input_path data/poisson/_train_k1_2.5_32k.h5 \
    --in_place
```

### Option 3: Convert to New File (Keep Original)

```bash
# Create a new file instead of modifying in place
python utils/convert_poisson_to_mixed_format.py \
    --input_path data/poisson/_train_k1_2.5_32k.h5 \
    --output_path data/poisson/_train_k1_2.5_32k_mixed_format.h5
```

## What the Conversion Does

### Before Conversion
```
HDF5 Structure:
  fields: (n, 2, 128, 128)  - Source field and solution
  tensor: (n, 3)            - [k11, k12, k22]
```

### After Conversion
```
HDF5 Structure:
  fields: (n, 2, 128, 128)  - Unchanged
  tensor: (n, 5)            - [k11, k12, k22, 0, 0]
```

The last two components are zero-padded to match the mixed dataset format.

## Complete Workflow

### 1. Generate Poisson k∈[1,2.5] Dataset

```bash
# On the cluster or locally
python utils/gen_data_poisson.py \
    --train_samples 32768 \
    --val_samples 4096 \
    --test_samples 4096 \
    --e1 1.0 \
    --e2 2.5 \
    --save_name k1_2.5 \
    --nx 128 \
    --ny 128
```

### 2. Convert to Mixed-Compatible Format

```bash
# Batch convert all splits
bash scripts/utils/convert_k1_2.5_to_mixed_format.sh
```

### 3. Update Config Checkpoint Paths

```bash
# After mixed pretraining completes
bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
```

### 4. Fine-Tune on Converted Datasets

```bash
# Submit fine-tuning array job
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

The config already points to the correct paths:
- `data/poisson/_train_k1_2.5_32k.h5`
- `data/poisson/_val_k1_2.5_4k.h5`
- `data/poisson/_test_k1_2.5_4k.h5`

## Verification

After conversion, the script shows verification output:

```
Verification:
  Output tensor shape: (32768, 5)
  First sample (original): [k11_value, k12_value, k22_value]
  First sample (new):      [k11_value, k12_value, k22_value, 0.0000, 0.0000]
```

## Backup and Recovery

### Automatic Backup
When using `--in_place`, a backup is automatically created:
```
data/poisson/_train_k1_2.5_32k.h5.backup
```

### Restore from Backup
```bash
# If you need to restore the original
cp data/poisson/_train_k1_2.5_32k.h5.backup data/poisson/_train_k1_2.5_32k.h5
```

## Technical Details

### Why This is Needed

The mixed-pretrained model has learned to handle 6 input channels:
- 1 source field
- 5 tensor components (expanded spatially by PDESolns)

When loading Poisson data for fine-tuning:
- Original format: 3 tensor components → 4 total channels (1 + 3)
- Mixed format: 5 tensor components → 6 total channels (1 + 5)

By zero-padding Poisson tensors to 5 components, we match the mixed model's architecture without retraining.

### Model Compatibility

The fine-tuning configs in `operators_mixed.yaml` specify:
```yaml
in_dim: 6  # Must match mixed model (6 input channels)
```

This ensures the model architecture matches between pretraining and fine-tuning.

### Zero-Padding Layout

The tensor layout across all three PDE systems is:
```
Poisson:    [k11, k12, k22,  0,  0]  ← zero-padded
AdvDiff:    [k11, k12, k22, vx, vy]
Helmholtz:  [k,  omega,  0,  0,  0]
```

## Troubleshooting

### "Expected 3-component tensor" Error
The dataset is already in 5-component format or not a Poisson dataset. Check:
```bash
python -c "import h5py; f=h5py.File('data/poisson/_train_k1_2.5_32k.h5','r'); print(f['tensor'].shape)"
```

Should show `(n, 3)` before conversion, `(n, 5)` after.

### File Not Found
Generate the dataset first:
```bash
python utils/gen_data_poisson.py --e1 1.0 --e2 2.5 --save_name k1_2.5 ...
```

### Different Number of Samples
Update the script if you generated datasets with different sizes:
- Training: default 32768 (32k)
- Validation: default 4096 (4k)
- Test: default 4096 (4k)

## Related Documentation

- `MIXED_CONFIG_REFERENCE.md` - Mixed dataset configuration guide
- `MIXED_DATASET_EXPLANATION.md` - Mixed dataset format details
- `EVAL_TRANSFER_LEARNING.md` - Evaluation and comparison guide

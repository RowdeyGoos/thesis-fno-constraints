# Transfer Learning Example - Reproducing Figure 3a

This guide shows **exactly** how to reproduce Figure 3a transfer learning results from the paper using the correct `weights` parameter approach.

## 📊 What is Figure 3a?

Figure 3a from the paper shows:
- **Plot**: Test error vs. amount of downstream training data
- **Two curves compared**:
  - 🔵 **Pre-trained**: Model initialized with source domain weights, then fine-tuned
  - 🟠 **From scratch**: Model randomly initialized, trained only on target data
- **Demonstrates**: Pre-trained models need less target domain data to achieve same performance

## ⚡ Quick Reference

**Correct way to load pre-trained weights:**
```yaml
# In your YAML config file
poisson-finetune:
  <<: *base_config
  weights: './path/to/pretrained/ckpt_best.tar'  # ✅ THIS IS THE KEY!
```

**NOT** via command-line: ~~`--pretrained_weights`~~ ❌ (this doesn't exist!)

---

## Overview

**Figure 3a** from "Towards Foundation Models for Scientific Machine Learning" (Subramanian et al., 2023) shows:
- **Transfer learning performance**: Comparison of pre-trained models vs. from-scratch models on downstream tasks
- **Data efficiency**: How much target domain training data is needed to achieve good performance
- **X-axis**: Amount of downstream training data (e.g., 12.5%, 25%, 50%, 100%)
- **Y-axis**: Test error on target domain
- **Two curves**: 
  - Pre-trained model (initialized with weights from source domain, then fine-tuned)
  - From-scratch model (randomly initialized, trained only on target domain)
- **Key finding**: Pre-trained models achieve better performance with significantly less target domain data

## Complete Workflow

### Step 1: Pre-train on Source Domain (Poisson k∈[1,5])

```bash
# Generate source domain data (if not already done)
mkdir -p data/poisson_k1_5

python utils/gen_data_poisson.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/poisson_k1_5 \
    --e1 1 \
    --e2 5

python compute_scales.py \
    data/poisson_k1_5/poissons_train_k1_5_32768.h5 \
    data/poisson_k1_5/poissons_train_k1_5_32768_scales.npy

# Pre-train model
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k1_5 \
    --run_num pretrain_k1_5 \
    --root_dir ./results

# Wait for training to complete (~10 hours on 4 GPUs)
# Best checkpoint will be saved at:
# ./results/expts/poisson-scale-k1_5/pretrain_k1_5/checkpoints/ckpt_best.tar
```

### Step 2: Generate Target Domain Data (Poisson k∈[5,10])

```bash
mkdir -p data/poisson_k5_10

python utils/gen_data_poisson.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/poisson_k5_10 \
    --e1 5 \
    --e2 10

python compute_scales.py \
    data/poisson_k5_10/poissons_train_k5_10_32768.h5 \
    data/poisson_k5_10/poissons_train_k5_10_32768_scales.npy
```

### Step 3: Create Config for Transfer Learning

Add to `config/operators_poisson.yaml`:

```yaml
# Target domain: k∈[5,10]
poisson-scale-k5_10: &poisson_scale_k5_10
  <<: *poisson
  train_path:    'data/poisson_k5_10/poissons_train_k5_10_32768.h5'
  val_path:      'data/poisson_k5_10/poissons_val_k5_10_4096.h5'
  test_path:     'data/poisson_k5_10/poissons_test_k5_10_4096.h5'
  scales_path:   'data/poisson_k5_10/poissons_train_k5_10_32768_scales.npy'
  batch_size: 128
  valid_batch_size: 128
  log_to_wandb: !!bool True
  mode_cut: 32
  embed_cut: 64
  fc_cut: 2

# Fine-tuning with pre-trained weights
poisson-scale-k5_10-pretrained: &poisson_k5_10_pt
  <<: *poisson_scale_k5_10
  weights: './results/expts/poisson-scale-k1_5/pretrain_k1_5/checkpoints/ckpt_best.tar'
  max_epochs: 200  # Can use fewer epochs for fine-tuning
  lr: 1E-4  # Lower learning rate for fine-tuning (optional)

# Data scaling experiments on target domain
poisson-scale-k5_10-pretrained-sub2: &poisson_k5_10_pt_sub2
  <<: *poisson_k5_10_pt
  subsample: 2  # 50% of data

poisson-scale-k5_10-pretrained-sub4: &poisson_k5_10_pt_sub4
  <<: *poisson_k5_10_pt
  subsample: 4  # 25% of data

poisson-scale-k5_10-pretrained-sub8: &poisson_k5_10_pt_sub8
  <<: *poisson_k5_10_pt
  subsample: 8  # 12.5% of data

# From-scratch baselines (no pre-training)
poisson-scale-k5_10-fromscratch-sub2:
  <<: *poisson_scale_k5_10
  subsample: 2

poisson-scale-k5_10-fromscratch-sub4:
  <<: *poisson_scale_k5_10
  subsample: 4

poisson-scale-k5_10-fromscratch-sub8:
  <<: *poisson_scale_k5_10
  subsample: 8
```

### Step 4: Run Transfer Learning Experiments

#### Baseline: Train from scratch on target domain

```bash
# Full data (100%)
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10 \
    --run_num fromscratch_100pct \
    --root_dir ./results

# 50% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-fromscratch-sub2 \
    --run_num fromscratch_50pct \
    --root_dir ./results

# 25% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-fromscratch-sub4 \
    --run_num fromscratch_25pct \
    --root_dir ./results

# 12.5% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-fromscratch-sub8 \
    --run_num fromscratch_12pct \
    --root_dir ./results
```

#### Transfer Learning: Fine-tune with pre-trained weights

```bash
# Full data (100%)
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-pretrained \
    --run_num pretrained_100pct \
    --root_dir ./results

# 50% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-pretrained-sub2 \
    --run_num pretrained_50pct \
    --root_dir ./results

# 25% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-pretrained-sub4 \
    --run_num pretrained_25pct \
    --root_dir ./results

# 12.5% data
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-pretrained-sub8 \
    --run_num pretrained_12pct \
    --root_dir ./results
```

**Note**: The trainer automatically detects the `weights` parameter in the config and loads the checkpoint before training begins. You'll see this message in the logs:
```
Loading IC weights ./results/expts/poisson-scale-k1_5/pretrain_k1_5/checkpoints/ckpt_best.tar
```

### Step 5: Evaluate Results

```bash
# Evaluate from-scratch model
python eval.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-fromscratch-sub4 \
    --run_num fromscratch_25pct \
    --root_dir ./results \
    --weights ./results/expts/poisson-scale-k5_10-fromscratch-sub4/fromscratch_25pct/checkpoints/ckpt_best.tar

# Evaluate pre-trained model
python eval.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k5_10-pretrained-sub4 \
    --run_num pretrained_25pct \
    --root_dir ./results \
    --weights ./results/expts/poisson-scale-k5_10-pretrained-sub4/pretrained_25pct/checkpoints/ckpt_best.tar
```

### Step 6: Compare Results

Check the logs for final test errors:

```bash
# From-scratch
cat ./results/expts/poisson-scale-k5_10-fromscratch-sub4/fromscratch_25pct/logs_best.txt

# Pre-trained
cat ./results/expts/poisson-scale-k5_10-pretrained-sub4/pretrained_25pct/logs_best.txt
```

Expected results:
- **Pre-trained model**: Lower test error, faster convergence
- **From-scratch model**: Higher test error, slower convergence

## Key Points

1. ✅ **Use `weights` parameter in YAML config**, not a command-line argument
2. ✅ **Path to weights** must point to a valid checkpoint file (`.tar`)
3. ✅ **`subsample` parameter** is the correct way to reduce training data
4. ✅ **The trainer checks** `if hasattr(self.params, 'weights')` and loads automatically
5. ✅ **Normalization scales** should be computed from the target domain data

## Alternative: Cross-PDE Transfer

You can also try transfer learning across different PDE types:

```yaml
# Pre-train on Poisson, fine-tune on Advection-Diffusion
advdiff-from-poisson: &ad_from_poisson
  <<: *ad_scale_adr0p2_1  # Advection-Diffusion base config
  weights: './results/expts/poisson-scale-k1_5/pretrain_k1_5/checkpoints/ckpt_best.tar'
  max_epochs: 200
  lr: 1E-4
```

## Troubleshooting

**Error: "weights file not found"**
- Verify the checkpoint path exists: `ls -lh ./results/expts/poisson-scale-k1_5/pretrain_k1_5/checkpoints/`
- Make sure pre-training completed successfully

**Error: "size mismatch" when loading weights**
- Ensure target domain config has same model architecture as source
- Check `mode_cut`, `embed_cut`, `fc_cut` match between configs

**Training doesn't improve with pre-trained weights**
- Try lower learning rate (e.g., `lr: 1E-4` instead of `1E-3`)
- Check that weights loaded successfully in logs
- Verify target domain data is correct

**Weights load but training starts from scratch**
- Make sure `self.params.resuming = False` is set (this is done automatically when weights parameter exists)
- Check logs for "Loading IC weights" message

## Summary

To reproduce **Figure 3a** transfer learning results:

### What Figure 3a Shows
A plot comparing **test error vs. training data size** for two scenarios:
1. **Pre-trained** (blue line): Model pre-trained on source domain (e.g., Poisson k∈[1,5]), then fine-tuned on target domain (e.g., Poisson k∈[5,10])
2. **From scratch** (orange line): Model trained only on target domain data with random initialization

### Experimental Setup
- **Source domain (pre-training)**: Poisson k∈[1,5] with 32k samples, train for 500 epochs
- **Target domain (evaluation)**: Poisson k∈[5,10] with varying amounts of data (12.5%, 25%, 50%, 100% of 32k)
- **Comparison**: For each data amount, train one model from scratch and one with pre-trained weights
- **Metric**: Test L2 error on target domain test set

### Key Implementation Details
1. **`weights` parameter in config file** is how you specify pre-trained weights (NOT command-line argument!)
2. **`subsample` parameter** controls the fraction of training data used (subsample=2 → 50%, subsample=4 → 25%, etc.)
3. **Normalization scales** must be computed separately for source and target domains
4. **Same model architecture** must be used for pre-training and fine-tuning
5. **Learning rate** can optionally be reduced for fine-tuning (e.g., 1E-4 instead of 1E-3)

### Steps to Reproduce
1. Pre-train on source domain (k∈[1,5])
2. Generate target domain data (k∈[5,10])
3. Add configs with `weights` parameter pointing to pre-trained checkpoint
4. Run experiments with different `subsample` values (1, 2, 4, 8 for 100%, 50%, 25%, 12.5%)
5. Compare test errors: pre-trained vs. from-scratch for each data amount

The key insight: **The `weights` parameter in the YAML config file** is how the codebase implements transfer learning, not through command-line arguments!

# Mixed Dataset Configuration Reference

This document describes the mixed-dataset configuration structure.

## File Structure

Mixed-dataset configs are in:

```text
config/operators_mixed.yaml
```

Single-domain configs remain in:

```text
config/operators_poisson.yaml
config/operators_helmholtz.yaml
config/operators_ad.yaml
```

## Configuration Hierarchy

### Mixed Pretraining

Config: `mixed-scale-all`

- Base: `mixed-pretrain-base` in `operators_mixed.yaml`
- Data: Poisson `k in [1,5]` + AdvDiff `adr in [0.2,1]` + Helmholtz `omega in [1,10]`
- Model input: `in_dim: 7` = `1 source + 6 tensor channels`
- Dataset: `data/mixed/_train_mixed_32k.h5`

### Downstream Fine-Tuning From Mixed Pretraining

Configs in `operators_mixed.yaml`:

- `poisson-k1_2.5-finetune-mixed-16`
- `poisson-k1_2.5-finetune-mixed-64`
- `poisson-k1_2.5-finetune-mixed-256`
- `poisson-k1_2.5-finetune-mixed-1k`
- `poisson-k1_2.5-finetune-mixed-4k`
- `poisson-k1_2.5-finetune-mixed-8k`
- `poisson-k1_2.5-finetune-mixed-16k`
- `poisson-k1_2.5-finetune-mixed-32k`

Each uses mixed-compatible downstream data and keeps `in_dim: 7` to match the pretrained checkpoint.

## Mixed Tensor Layout (Canonical)

All mixed-format datasets use 6 tensor components:

```text
[k11, k12, k22, vx, vy, omega]
```

Per PDE mapping:

```text
Poisson:    [k11, k12, k22,  0,  0,     0]
AdvDiff:    [k11, k12, k22, vx, vy,     0]
Helmholtz:  [k,   0,   k,   0,  0, omega]
```

## Workflow

### 1) Create mixed pretraining datasets

```bash
python utils/create_mixed_dataset.py   --poisson_path data/poisson/_train_k1_5_32k.h5   --advdiff_path data/advdiff/_train_adr0.2_1_32k.h5   --helmholtz_path data/helmholtz/_train_o1_10_32k.h5   --output_path data/mixed/_train_mixed_32k.h5

python utils/create_mixed_dataset.py   --poisson_path data/poisson/_val_k1_5_4k.h5   --advdiff_path data/advdiff/_val_adr0.2_1_4k.h5   --helmholtz_path data/helmholtz/_val_o1_10_4k.h5   --output_path data/mixed/_val_mixed_4k.h5

python utils/create_mixed_dataset.py   --poisson_path data/poisson/_test_k1_5_4k.h5   --advdiff_path data/advdiff/_test_adr0.2_1_4k.h5   --helmholtz_path data/helmholtz/_test_o1_10_4k.h5   --output_path data/mixed/_test_mixed_4k.h5
```

### 2) Pretrain

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

### 3) Update checkpoint path in configs

```bash
bash scripts/utils/update_mixed_checkpoint_path.sh <job_id>
```

### 4) Fine-tune

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

### 5) Evaluate

```bash
python eval_transfer_learning.py   --yaml_config config/operators_mixed.yaml   --experiment_type poisson   --include_mixed
```

## Config Snippets (Current)

Mixed pretraining (from `operators_mixed.yaml`):

```yaml
mixed-scale-all:
  <<: *mixed_pretrain_base
  train_path: 'data/mixed/_train_mixed_32k.h5'
  val_path: 'data/mixed/_val_mixed_4k.h5'
  test_path: 'data/mixed/_test_mixed_4k.h5'
  scales_path: 'data/mixed/_train_mixed_32k_scales.npy'
  in_dim: 7
  max_epochs: 500
```

Mixed fine-tuning example:

```yaml
poisson-k1_2.5-finetune-mixed-256:
  <<: *poisson_k1_2p5_base
  weights: 'experiments/expts/mixed-scale-all/pretrain-mixed-JOBID-0/checkpoints/ckpt_best.tar'
  subsample: 128
  batch_size: 64
  valid_batch_size: 128
  in_dim: 7
```

## Verification

```bash
grep -n "mixed-scale-all:" config/operators_mixed.yaml
grep -n "in_dim:" config/operators_mixed.yaml | head
```

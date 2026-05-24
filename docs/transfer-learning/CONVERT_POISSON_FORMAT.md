# Converting Poisson Datasets For Mixed-Pretrained Fine-Tuning

## Problem

Mixed-pretrained checkpoints expect mixed-format inputs:

- mixed model input: `in_dim = 7`
- composition: `1 source + 6 tensor channels`

Standard Poisson datasets use only 3 tensor coefficients (`k11, k12, k22`), so they must be converted for mixed-finetune configs.

## Solution

Convert Poisson tensors from 3 to 6 components with zero padding:

```text
[k11, k12, k22] -> [k11, k12, k22, 0, 0, 0]
```

## Quick Start

Batch convert all `k in [1.0, 2.5]` Poisson splits:

```bash
bash scripts/utils/convert_k1_2.5_to_mixed_format.sh
```

Or convert one file:

```bash
python utils/convert_poisson_to_mixed_format.py   --input_path data/poisson/_train_k1.0_2.5_32k.h5   --in_place
```

## Before vs After

Before:

- `fields`: `(n, 2, 128, 128)`
- `tensor`: `(n, 3)`

After:

- `fields`: `(n, 2, 128, 128)`
- `tensor`: `(n, 6)`

## Workflow

1. Generate Poisson downstream data (`k in [1.0, 2.5]`)
2. Convert with `convert_poisson_to_mixed_format.py`
3. Update mixed checkpoint path:

```bash
bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
```

4. Run mixed fine-tuning:

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

## Verification

Expected converter output:

```text
Output tensor shape: (N, 6)
First sample (new): [k11, k12, k22, 0, 0, 0]
```

## Notes

- `--in_place` creates `.backup` files automatically.
- Mixed layout convention remains:
  - Poisson: `[k11, k12, k22, 0, 0, 0]`
  - AdvDiff: `[k11, k12, k22, vx, vy, 0]`
  - Helmholtz: `[k, 0, k, 0, 0, omega]`

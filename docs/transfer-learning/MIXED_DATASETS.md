# Mixed Datasets

This guide describes the canonical mixed-format datasets used for multi-PDE
pretraining and mixed-pretrained downstream finetuning.

## Canonical Format

Mixed datasets keep the standard HDF5 fields:

- `fields`: `(N, 2, nx, ny)`
  - `fields[:,0]`: source field
  - `fields[:,1]`: solution field
- `tensor`: `(N, 6)`
- `labels`: `(N,)`, with `0=Poisson`, `1=AdvDiff`, `2=Helmholtz`

Tensor channel order is fixed:

```text
[k11, k12, k22, vx, vy, omega]
```

Per-system mapping:

```text
Poisson:    [k11, k12, k22,  0,  0,     0]
AdvDiff:    [k11, k12, k22, vx, vy,     0]
Helmholtz:  [k,   0,   k,   0,  0, omega]
```

This avoids coefficient-channel overlap and keeps `in_dim: 7` for mixed
models: one source channel plus six tensor channels expanded over the spatial
grid.

## Create Mixed Pretraining Datasets

Use `scripts/data/create_mixed_dataset.py` to combine Poisson,
Advection-Diffusion, and Helmholtz splits:

```bash
python scripts/data/create_mixed_dataset.py \
  --poisson_path data/poisson/_train_k1_5_32k.h5 \
  --advdiff_path data/advdiff/_train_adr0.2_1_32k.h5 \
  --helmholtz_path data/helmholtz/_train_o1_10_32k.h5 \
  --output_path data/mixed/_train_mixed_32k.h5
```

Repeat for validation and test splits. The script also computes the mixed
normalization scales next to the output file.

## Mixed Configs

Mixed configs live in `config/operators_mixed.yaml`.

Main pretraining config:

```yaml
mixed-scale-all:
  train_path: data/mixed/_train_mixed_32k.h5
  val_path: data/mixed/_val_mixed_4k.h5
  test_path: data/mixed/_test_mixed_4k.h5
  scales_path: data/mixed/_train_mixed_32k_scales.npy
  in_dim: 7
```

Run mixed pretraining with:

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

## Convert Downstream Data For Mixed Finetuning

Mixed-pretrained checkpoints require downstream data in the same six-component
tensor format. For Poisson data, convert:

```text
[k11, k12, k22] -> [k11, k12, k22, 0, 0, 0]
```

Batch convert the `k in [1.0,2.5]` Poisson splits:

```bash
bash scripts/data/convert_k1_2.5_to_mixed_format.sh
```

Or convert a single file in place:

```bash
python scripts/data/convert_poisson_to_mixed_format.py \
  --input_path data/poisson/_train_k1.0_2.5_32k.h5 \
  --in_place
```

The converter creates `.backup` files for in-place conversions.

## Checkpoint Handoff And Finetuning

After mixed pretraining finishes:

```bash
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

For constrained mixed checkpoints, pass the selected pretraining config and run
prefix:

```bash
PRETRAIN_CONFIG_NAME=mixed-scale-all-constraints-penalty-hard \
PRETRAIN_RUN_PREFIX=pretrain-mixed-penalty-hard \
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
```

## Verification

Inspect one mixed dataset:

```python
import h5py
import numpy as np

with h5py.File("data/mixed/_train_mixed_32k.h5", "r") as f:
    fields = f["fields"][:]
    tensor = f["tensor"][:]
    labels = f["labels"][:]

print(fields.shape)
print(tensor.shape)
print(labels.shape)

for label, name in [(0, "Poisson"), (1, "AdvDiff"), (2, "Helmholtz")]:
    idx = np.where(labels == label)[0][0]
    print(name, tensor[idx])
```

Expected tensor shape is `(N, 6)`.

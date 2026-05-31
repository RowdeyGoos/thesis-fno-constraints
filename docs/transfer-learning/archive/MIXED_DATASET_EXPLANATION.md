# Mixed Dataset Implementation - Technical Details

## Overview

This document explains the mixed-dataset format used for multi-task pretraining across Poisson, Advection-Diffusion, and Helmholtz.

## Canonical Mixed Format

Mixed datasets keep standard HDF5 keys:

- `fields`: `(n_samples, 2, nx, ny)`
  - index 0: source field
  - index 1: solution field
- `tensor`: `(n_samples, 6)`
- `labels`: `(n_samples,)` where `0=Poisson`, `1=AdvDiff`, `2=Helmholtz`

Tensor channel order is fixed:

```text
[k11, k12, k22, vx, vy, omega]
```

Per PDE mapping:

```text
Poisson:    [k11, k12, k22,  0,  0,     0]
AdvDiff:    [k11, k12, k22, vx, vy,     0]
Helmholtz:  [k,   0,   k,   0,  0, omega]
```

## How Data Loader Expands Inputs

`utils/data_utils.py` (`PDESolns`) expands each tensor component into a spatial channel.

So mixed input dimensionality is:

- `1` source channel
- `6` tensor channels (expanded)
- total `in_dim = 7`

## Why this format matters

This layout avoids coefficient channel overlap between PDE families and keeps semantics fixed per channel.

## Configuration usage

Mixed pretraining config lives in:

```text
config/operators_mixed.yaml
```

Typical mixed settings:

```yaml
system: 'mixed'
in_dim: 7
train_path: 'data/mixed/_train_mixed_32k.h5'
scales_path: 'data/mixed/_train_mixed_32k_scales.npy'
```

Downstream mixed-finetune configs also keep `in_dim: 7` and use converted downstream datasets in mixed tensor format.

## Implementation files

1. `scripts/data/create_mixed_dataset.py`
2. `config/operators_mixed.yaml`
3. `scripts/slurm/pretrain/submit_pretrain_mixed.sh`
4. `scripts/slurm/finetune/*/*mixed*.sh`

## Verification snippet

```python
import h5py
import numpy as np

with h5py.File('data/mixed/_train_mixed_32k.h5', 'r') as f:
    fields = f['fields'][:]
    tensor = f['tensor'][:]
    labels = f['labels'][:]

print(fields.shape)  # (n, 2, 128, 128)
print(tensor.shape)  # (n, 6)
print(labels.shape)  # (n,)

poisson_idx = np.where(labels == 0)[0][0]
advdiff_idx = np.where(labels == 1)[0][0]
helm_idx = np.where(labels == 2)[0][0]

print('Poisson tensor:', tensor[poisson_idx])
print('AdvDiff tensor:', tensor[advdiff_idx])
print('Helmholtz tensor:', tensor[helm_idx])
```

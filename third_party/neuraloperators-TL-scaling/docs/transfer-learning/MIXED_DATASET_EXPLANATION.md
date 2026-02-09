# Mixed Dataset Implementation - Technical Details

## Overview
This document explains the mixed dataset implementation for multi-task learning across three PDE systems (Poisson, Advection-Diffusion, and Helmholtz), as described in Section 4.2 and Figure 6a of the paper.

## Paper Quote
> "When pre-training a single model on this 'mixed' dataset, we simply use zero channels for those coefficients that do not exist when using examples from a specific operator. For example, the Helmholtz equation has a diffusion tensor input (identity matrix) with an additional input for the wavenumber but no advection (zero channel), while the Poisson's equation only has a diffusion tensor input and hence we append zero channels to signify no wavenumbers and advection; similarly for Advection-Diffusion."

## Data Format Understanding

### Individual PDE Datasets
Each PDE dataset uses the standard HDF5 format:
- **fields**: shape `(n_samples, 2, nx, ny)`
  - `[0, :, :]` = source function
  - `[1, :, :]` = solution
- **tensor**: shape `(n_samples, n_coeffs)` - PDE-specific coefficients

### Tensor Coefficients by System
1. **Poisson** (`n_coeffs = 3`):
   ```
   tensor[i, 0] = k11  (diffusion tensor component)
   tensor[i, 1] = k12  (diffusion tensor component)
   tensor[i, 2] = k22  (diffusion tensor component)
   ```

2. **Advection-Diffusion** (`n_coeffs = 5`):
   ```
   tensor[i, 0] = k11  (diffusion tensor component)
   tensor[i, 1] = k12  (diffusion tensor component)
   tensor[i, 2] = k22  (diffusion tensor component)
   tensor[i, 3] = vx   (advection velocity x)
   tensor[i, 4] = vy   (advection velocity y)
   ```

3. **Helmholtz** (`n_coeffs = 2`):
   ```
   tensor[i, 0] = k_constant  (constant diffusion coefficient)
   tensor[i, 1] = omega       (wavenumber)
   ```

### How PDESolns Loads Data
The `PDESolns` class in `utils/data_utils.py` performs the following:
1. Loads `fields[:, 0]` as source (1 spatial channel)
2. Loads `tensor[:]` as coefficient values
3. **Expands each tensor component to a full spatial channel** by broadcasting
4. Concatenates: `input = [source] + [expanded_tensors]`

For example, Poisson with 3 tensor components:
- Original: 1 source channel + 3 scalar coefficients
- After expansion: 1 source + 3 spatial channels = **4 input channels**

## Mixed Dataset Implementation

### Zero-Padding Strategy
To unify all three systems, we pad tensor coefficients to size 5 (the maximum):

```python
# Unified tensor layout: [coef0, coef1, coef2, coef3, coef4]

Poisson (3 → 5):
    [k11, k12, k22, 0, 0]
    # Zeros for missing advection

AdvDiff (5 → 5):  
    [k11, k12, k22, vx, vy]
    # No change, already 5 components

Helmholtz (2 → 5):
    [k_constant, omega, 0, 0, 0]
    # Zeros for missing diffusion tensor components and advection
```

### Result After PDESolns Expansion
When the mixed dataset is loaded:
- Source: 1 spatial channel
- Tensor (5 components expanded): 5 spatial channels
- **Total input**: 1 + 5 = **6 channels**

The model sees:
```
Channel 0: source function
Channel 1: coefficient 0 (k11 for Poisson/AdvDiff, k_constant for Helmholtz)
Channel 2: coefficient 1 (k12 for Poisson/AdvDiff, omega for Helmholtz, 0 elsewhere)
Channel 3: coefficient 2 (k22 for Poisson/AdvDiff, 0 for Helmholtz)
Channel 4: coefficient 3 (0 for Poisson/Helmholtz, vx for AdvDiff)
Channel 5: coefficient 4 (0 for Poisson/Helmholtz, vy for AdvDiff)
```

The zero-valued channels effectively act as an implicit operator selector, allowing the model to learn which PDE system each example belongs to.

## Configuration
In `config/operators_poisson.yaml`, the mixed dataset configuration uses:
```yaml
mixed-scale-all:
  in_dim: 6  # 1 source + 5 tensor components (after expansion)
  out_dim: 1
  system: 'mixed'
  train_path: 'data/mixed/_train_mixed_32k.h5'
  scales_path: 'data/mixed/train_mixed_scales.npy'
```

## Transfer Learning from Mixed Pretraining
When fine-tuning the mixed-pretrained model on a single PDE (e.g., Poisson k∈[5,10]):
- The model maintains `in_dim: 6` to match the pretrained weights
- The downstream Poisson data (normally 4 channels) needs to be loaded in a way compatible with the 6-channel model
- This is handled by the config: `in_dim: 6` override in fine-tuning configs

## Implementation Files
1. **`utils/create_mixed_dataset.py`**: Creates the mixed HDF5 file with zero-padding
2. **`config/operators_poisson.yaml`**: Contains `mixed-scale-all` and fine-tuning configs
3. **`scripts/slurm/pretrain/submit_pretrain_mixed.sh`**: Runs mixed pretraining
4. **`scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh`**: Runs fine-tuning experiments

## Verification
To verify the mixed dataset was created correctly:
```python
import h5py
import numpy as np

with h5py.File('data/mixed/_train_mixed_32k.h5', 'r') as f:
    fields = f['fields'][:]
    tensor = f['tensor'][:]
    labels = f['labels'][:]
    
    print(f"Fields shape: {fields.shape}")  # Should be (n, 2, 128, 128)
    print(f"Tensor shape: {tensor.shape}")  # Should be (n, 5)
    print(f"Labels shape: {labels.shape}")  # Should be (n,)
    
    # Check a few examples
    poisson_idx = np.where(labels == 0)[0][0]
    advdiff_idx = np.where(labels == 1)[0][0]
    helmholtz_idx = np.where(labels == 2)[0][0]
    
    print(f"\nPoisson tensor (should have 3 nonzero, 2 zero): {tensor[poisson_idx]}")
    print(f"AdvDiff tensor (should have 5 nonzero): {tensor[advdiff_idx]}")
    print(f"Helmholtz tensor (should have 2 nonzero, 3 zero): {tensor[helmholtz_idx]}")
```

## Expected Results (from Figure 6a)
When comparing mixed-domain vs single-domain pretraining on downstream Poisson k∈[5,10]:
- Both pretraining approaches should significantly outperform training from scratch
- Mixed-domain pretraining should achieve comparable or better performance than single-domain
- The benefit is most pronounced in low-data regimes (16-256 samples)

This demonstrates that a single model can learn useful representations across multiple PDE systems through simple zero-padding, without requiring complex operator selection mechanisms.

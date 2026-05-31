# Block A Constraints: Zero-Mode + PDE Residual

This document describes the Block A implementation in this repository.

## Summary

Implemented modules:

1. Zero-mode enforcement (`off | hard | soft`) with shared masking semantics
2. Soft PDE residual loss (multi-operator residual in `utils/loss_utils.py`)
3. Optional augmented Lagrangian objective for PDE residual

Both modules are config-driven and default to off in all operator configs.

## Zero-mode enforcement

Config keys:

```yaml
constraint_zero_mode_enable: false
constraint_zero_mode_enforcement: 'off'   # off | hard | soft
constraint_zero_mode_weight: 0.0          # soft mode only
constraint_zero_mode_warmup_fraction: 0.0 # soft mode only
constraint_zero_mode_mode: 'gauge_aware'   # all | gauge_aware
constraint_zero_mode_omega_tol: 1.0e-8
```

Behavior:

- Enforcement mode:
  - `off`: no zero-mode constraint.
  - `hard`: project model outputs to zero mean in `models/fno.py`.
  - `soft`: add a zero-mode penalty term in `utils/loss_utils.py`.
- Mask mode:
  - `all`: constrain every sample.
  - `gauge_aware`: only constrain samples with `|omega| <= tol`.
  - This keeps Poisson/AdvDiff constrained.
  - It avoids forcing Helmholtz samples (with non-zero reaction term) to zero mean.

Soft penalty form:

- Per-sample DC component: `dc_i = mean(u_i)` (over spatial dims)
- Penalty: `L_zero = w_t * mean(dc_i^2)` over selected samples/channels
- Warmup: `w_t` ramps linearly to `constraint_zero_mode_weight` over
  `constraint_zero_mode_warmup_fraction * max_epochs`

Backward compatibility:

- If `constraint_zero_mode_enforcement` is omitted, legacy
  `constraint_zero_mode_enable: true` still maps to hard enforcement.

## PDE residual loss

Config keys:

```yaml
constraint_pde_enable: false
constraint_pde_weight: 0.0
constraint_pde_warmup_fraction: 0.1
constraint_pde_eps: 1.0e-8
constraint_pde_relative_norm: true
constraint_diffusion_tensor_order: 'k11_k22_k12'  # or k11_k12_k22
```

Residuals are computed with spectral derivatives:

- Poisson: `R = div(K grad u) + f`
- AdvDiff: `R = div(K grad u) - v·grad(u) + f`
- Helmholtz: `R = div(K grad u) + omega*u + f`

Mixed batches are routed per-sample from coefficient channels.

Loss scalar:

- Relative norm metric: `r = ||R||_2 / (||f||_2 + eps)`
- Penalty form: `L_pde = lambda_t * mean(r^2)`
- Warmup: `lambda_t` ramps linearly to `constraint_pde_weight` over
  `constraint_pde_warmup_fraction * max_epochs`.

## Augmented Lagrangian option

Enable with:

```yaml
constraint_pde_method: 'augmented_lagrangian'  # penalty | augmented_lagrangian
constraint_pde_al_rho: 1.0
constraint_pde_al_lambda0: 0.0
constraint_pde_al_dual_clip: 1.0e6
```

Form used:

- Constraint value per batch: `c = mean(r)`
- Objective: `L_pde = lambda_t * (mu * c + 0.5 * rho * c^2)`
- Dual update once per epoch: `mu <- clip(max(0, mu + rho * c_epoch_avg))`

## Logged metrics

Training:

- `pde_loss`
- `zero_mode_constraint_loss`
- `pde_residual_norm`
- `zero_mode_violation`
- `pde_al_lambda`

Validation:

- `val_zero_mode_constraint_loss`
- `val_pde_residual_norm`
- `val_zero_mode_violation`

Inference / transfer eval:

- `test_zero_mode_constraint_loss`
- `test_pde_residual_norm`
- `test_zero_mode_violation`

## Checkpoint selection

Training supports:

```yaml
checkpoint_selection_metric: 'val_loss'  # val_loss | val_err
```

- Default: `val_loss` (backward compatible).
- For accuracy-first studies with soft penalties, `val_err` is often preferred so
  `ckpt_best.tar` reflects predictive performance rather than weighted constraint loss.

## Example toggles (A0-A5)

```yaml
# A0 baseline
constraint_zero_mode_enforcement: 'off'
constraint_pde_enable: false

# A1 zero-mode hard only
constraint_zero_mode_enforcement: 'hard'
constraint_pde_enable: false

# A2 zero-mode soft only
constraint_zero_mode_enforcement: 'soft'
constraint_zero_mode_weight: 0.1
constraint_zero_mode_mode: 'gauge_aware'
constraint_pde_enable: false

# A3 PDE only
constraint_zero_mode_enforcement: 'off'
constraint_pde_enable: true
constraint_pde_weight: 0.1
constraint_pde_method: 'penalty'

# A4 combined (hard + PDE)
constraint_zero_mode_enforcement: 'hard'
constraint_pde_enable: true
constraint_pde_weight: 0.1

# A5 combined (soft + PDE)
constraint_zero_mode_enforcement: 'soft'
constraint_zero_mode_weight: 0.1
constraint_pde_enable: true
constraint_pde_weight: 0.1
```

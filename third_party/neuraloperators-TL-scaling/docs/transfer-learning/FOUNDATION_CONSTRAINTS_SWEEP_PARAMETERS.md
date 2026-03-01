# Foundation Constraints Sweep Parameters

This document lists the exact hyperparameter search spaces used by the mixed-pretraining constraint sweeps.

It is intended as a report reference for:

- what is swept
- what is fixed
- which sweep file corresponds to each experiment stage

---

## 1) Common sweep settings (all files)

All six sweep files in `config/sweep_constraints_pretrain_*.yaml` share:

- `method: bayes`
- `metric.name: best_val_err` with `goal: minimize`
- `run_cap: 12`
- `program: train.py`

Common fixed training/runtime params in every sweep:

- `max_epochs: 150`
- `checkpoint_selection_metric: val_err`
- `plot_figs: false`
- `save_checkpoint: true`
- `log_to_wandb: true`
- `constraint_pde_enable: true`
- `constraint_pde_relative_norm: true`
- `constraint_zero_mode_mode: gauge_aware`
- `constraint_zero_mode_omega_tol: 1.0e-8`

---

## 2) Sweep matrix by experiment stage

| Stage | Sweep file | PDE method | Zero-mode mode | Swept parameters |
|---|---|---|---|---|
| A (PDE-only control) | `config/sweep_constraints_pretrain_penalty_pde_only.yaml` | `penalty` | `off` | `constraint_pde_weight`, `constraint_pde_warmup_fraction` |
| A (PDE-only control) | `config/sweep_constraints_pretrain_al_pde_only.yaml` | `augmented_lagrangian` | `off` | `constraint_pde_weight`, `constraint_pde_warmup_fraction`, `constraint_pde_al_rho` |
| B (hard zero-mode) | `config/sweep_constraints_pretrain_penalty_hard.yaml` | `penalty` | `hard` | `constraint_pde_weight`, `constraint_pde_warmup_fraction` |
| B (hard zero-mode) | `config/sweep_constraints_pretrain_al_hard.yaml` | `augmented_lagrangian` | `hard` | `constraint_pde_weight`, `constraint_pde_warmup_fraction`, `constraint_pde_al_rho` |
| C (soft zero-mode) | `config/sweep_constraints_pretrain_penalty_soft.yaml` | `penalty` | `soft` | `constraint_zero_mode_weight`, `constraint_zero_mode_warmup_fraction`, `constraint_pde_weight`, `constraint_pde_warmup_fraction` |
| C (soft zero-mode) | `config/sweep_constraints_pretrain_al_soft.yaml` | `augmented_lagrangian` | `soft` | `constraint_zero_mode_weight`, `constraint_zero_mode_warmup_fraction`, `constraint_pde_weight`, `constraint_pde_warmup_fraction`, `constraint_pde_al_rho` |

---

## 3) Exact search spaces

### 3.1 Shared PDE search ranges

Used in all sweep files:

- `constraint_pde_weight`
  - distribution: `log_uniform_values`
  - range: `[0.01, 0.3]`
- `constraint_pde_warmup_fraction`
  - categorical values: `[0.0, 0.1]`

### 3.2 AL-specific PDE search range

Used only when `constraint_pde_method: augmented_lagrangian`:

- `constraint_pde_al_rho`
  - categorical values: `[0.5, 1.0, 2.0]`

Fixed AL params in AL sweep files:

- `constraint_pde_al_lambda0: 0.0`
- `constraint_pde_al_dual_clip: 1.0e6`

### 3.3 Soft zero-mode search range

Used only when `constraint_zero_mode_enforcement: soft`:

- `constraint_zero_mode_weight`
  - distribution: `log_uniform_values`
  - range: `[0.001, 0.3]`
- `constraint_zero_mode_warmup_fraction`
  - categorical values: `[0.0, 0.1]`

---

## 4) Fixed constraint settings by zero-mode regime

### PDE-only (`off`)

- `constraint_zero_mode_enable: false`
- `constraint_zero_mode_enforcement: off`
- `constraint_zero_mode_weight: 0.0`
- `constraint_zero_mode_warmup_fraction: 0.0`

### Hard zero-mode (`hard`)

- `constraint_zero_mode_enable: true`
- `constraint_zero_mode_enforcement: hard`
- `constraint_zero_mode_weight: 0.0`
- `constraint_zero_mode_warmup_fraction: 0.0`

### Soft zero-mode (`soft`)

- `constraint_zero_mode_enable: false`
- `constraint_zero_mode_enforcement: soft`
- `constraint_zero_mode_weight`: swept in `[0.001, 0.3]` (log-uniform)
- `constraint_zero_mode_warmup_fraction`: swept in `{0.0, 0.1}`

---

## 5) Effective sweep dimensionality

Because `method: bayes`, these are not full Cartesian grids; this is the number of tuned dimensions per sweep:

- penalty + PDE-only: 2D
- AL + PDE-only: 3D
- penalty + hard: 2D
- AL + hard: 3D
- penalty + soft: 4D
- AL + soft: 5D

With `run_cap: 12`, each sweep evaluates up to 12 Bayesian trials in its respective search space.


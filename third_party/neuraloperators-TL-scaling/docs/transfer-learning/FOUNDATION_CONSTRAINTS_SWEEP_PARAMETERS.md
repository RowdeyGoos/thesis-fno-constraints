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

### What each parameter does

This section explains each sweep key in plain language.

Sweep metadata keys:

- `name`: W&B sweep name shown in the dashboard.
- `entity`: W&B account or team that owns the sweep.
- `project`: W&B project where runs are stored.
- `program`: training entrypoint that W&B agent executes (`train.py`).
- `method`: search strategy (`bayes` here).
- `metric.name`: optimization target tracked by W&B (`best_val_err`).
- `metric.goal`: whether the target should be minimized or maximized.
- `run_cap`: maximum number of trials the sweep is allowed to launch.
- `parameters`: dictionary of values/ranges that define the search space.

General training/control parameters:

- `max_epochs`: number of training epochs per trial.
- `checkpoint_selection_metric`: metric used for choosing `ckpt_best.tar`.
- `plot_figs`: enables/disables figure generation during training.
- `save_checkpoint`: enables checkpoint and `logs_best.txt` writing.
- `log_to_wandb`: enables W&B scalar logging.

Zero-mode parameters:

- `constraint_zero_mode_enable`: legacy boolean switch (kept for compatibility).
- `constraint_zero_mode_enforcement`:
  - `off`: no zero-mode constraint.
  - `hard`: enforce zero mean by projection in model forward.
  - `soft`: enforce zero mean via extra loss penalty.
- `constraint_zero_mode_mode`:
  - `all`: apply zero-mode constraint to all samples.
  - `gauge_aware`: apply only where `|omega| <= constraint_zero_mode_omega_tol`.
- `constraint_zero_mode_omega_tol`: tolerance used by gauge-aware masking.
- `constraint_zero_mode_weight`: coefficient of soft zero-mode penalty.
- `constraint_zero_mode_warmup_fraction`: linear warmup fraction for soft zero-mode weight.

PDE constraint parameters:

- `constraint_pde_enable`: enables PDE residual loss term.
- `constraint_pde_method`:
  - `penalty`: weighted residual penalty.
  - `augmented_lagrangian`: penalty + dual variable update.
- `constraint_pde_weight`: base coefficient for PDE residual term.
- `constraint_pde_warmup_fraction`: linear warmup fraction for PDE weight.
- `constraint_pde_relative_norm`: uses residual normalization by source norm.
- `constraint_pde_al_rho`: AL quadratic penalty / dual step scale.
- `constraint_pde_al_lambda0`: initial AL dual variable value.
- `constraint_pde_al_dual_clip`: upper clip on AL dual variable magnitude.

Distribution/value specifiers in sweep YAML:

- `value`: fixed constant for all trials.
- `values`: discrete candidate set sampled by the sweep.
- `distribution: log_uniform_values`: sample in log-space between `min` and `max`.

---

## 2) Sweep matrix by experiment stage

| Stage | Sweep file | PDE method | Zero-mode mode | Swept parameters |
|---|---|---|---|---|
| Optional Z (zero-only) | `config/sweep_constraints_pretrain_zero_soft_only.yaml` | `off` | `soft` | `constraint_zero_mode_weight`, `constraint_zero_mode_warmup_fraction` |
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

In the optional zero-only sweep (`sweep_constraints_pretrain_zero_soft_only.yaml`),
these are the only tuned constraint parameters; PDE is fully disabled.

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

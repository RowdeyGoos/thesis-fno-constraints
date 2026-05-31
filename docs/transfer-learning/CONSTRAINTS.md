# Constraint-Aware Mixed Pretraining

This guide covers zero-mode constraints, PDE residual objectives, augmented
Lagrangian sweeps, and the strict foundation-model protocol used for
constraint-aware mixed pretraining.

## Implementation Reference

Constraint logic is implemented in `utils/loss_utils.py` and wired through the
trainer, inferencer, and configs. All constraint defaults are off unless a
specific experiment config enables them.

### Zero-mode enforcement

Config keys:

```yaml
constraint_zero_mode_enforcement: "off"   # off | hard | soft
constraint_zero_mode_weight: 0.0
constraint_zero_mode_warmup_fraction: 0.0
constraint_zero_mode_mode: "gauge_aware"  # all | gauge_aware
constraint_zero_mode_omega_tol: 1.0e-8
```

Modes:

- `off`: no zero-mode constraint.
- `hard`: project outputs to zero mean.
- `soft`: add a zero-mode penalty term to the loss.

`gauge_aware` applies the constraint only where `|omega| <= tol`, so
Poisson/AdvDiff samples are constrained while Helmholtz reaction samples are
not forced to zero mean.

### PDE residual loss

Config keys:

```yaml
constraint_pde_enable: false
constraint_pde_method: "penalty"          # penalty | augmented_lagrangian
constraint_pde_weight: 0.0
constraint_pde_warmup_fraction: 0.1
constraint_pde_relative_norm: true
constraint_pde_eps: 1.0e-8
constraint_diffusion_tensor_order: "k11_k12_k22"
```

Residuals are computed from the predicted solution and input coefficients:

- Poisson: `R = div(K grad u) + f`
- AdvDiff: `R = div(K grad u) - v dot grad(u) + f`
- Helmholtz: `R = div(K grad u) + omega*u + f`

The default penalty uses a relative residual norm normalized by the source
norm.

### Augmented Lagrangian

Enable with:

```yaml
constraint_pde_method: "augmented_lagrangian"
constraint_pde_al_rho: 1.0
constraint_pde_al_lambda0: 0.0
constraint_pde_al_dual_clip: 1.0e6
```

The dual variable is updated once per epoch from the epoch-average residual
constraint value.

### Logged metrics

Training, validation, and eval logs may include:

- `pde_loss`
- `pde_residual_norm`
- `pde_al_lambda`
- `zero_mode_constraint_loss`
- `zero_mode_violation`
- `test_pde_residual_norm`
- `test_zero_mode_violation`

For accuracy-first studies with soft penalties, use:

```yaml
checkpoint_selection_metric: "val_err"
```

## Strict Foundation-Model Protocol

The main constrained mixed-pretraining track compares:

1. PDE residual method with zero-mode off: `penalty` vs `augmented_lagrangian`.
2. PDE residual method with hard zero-mode: `penalty` vs
   `augmented_lagrangian`.
3. Zero-mode strategy using the winning PDE method: `hard` vs `soft`.

The foundation-pretraining track is `mixed-scale-all`. Candidate selection uses
mixed-pretraining validation metrics only, so downstream transfer results do
not leak into hyperparameter selection.

## Sweep Launch Workflow

Run a smoke test before long sweeps:

```bash
sbatch scripts/slurm/smoke/submit_smoke_train_eval_constraints.sh
bash scripts/experiments/submit_constraints_sanity_sweep.sh
```

Stage launch helpers:

```bash
bash scripts/experiments/submit_constraints_stage_a.sh
bash scripts/experiments/submit_constraints_stage_b.sh
bash scripts/experiments/submit_constraints_stage_c.sh penalty   # or: al
```

Single sweep launch:

```bash
bash scripts/experiments/submit_constraints_sweep.sh config/sweep_constraints_pretrain_al_hard.yaml
```

Rank local sweep outputs:

```bash
python scripts/experiments/select_constraints_candidate.py \
  --sweep_root experiments/sweeps/<SWEEP_ID> \
  --top_k 5 \
  --output_json results/constraints/<SWEEP_ID>_ranking.json
```

## Sweep Matrix

| Stage | Sweep file | PDE method | Zero-mode | Tuned parameters |
|---|---|---|---|---|
| optional | `config/sweep_constraints_pretrain_zero_soft_only.yaml` | off | soft | zero-mode weight, warmup |
| A | `config/sweep_constraints_pretrain_penalty_pde_only.yaml` | penalty | off | PDE weight, warmup |
| A | `config/sweep_constraints_pretrain_al_pde_only.yaml` | AL | off | PDE weight, warmup, rho |
| B | `config/sweep_constraints_pretrain_penalty_hard.yaml` | penalty | hard | PDE weight, warmup |
| B | `config/sweep_constraints_pretrain_al_hard.yaml` | AL | hard | PDE weight, warmup, rho |
| C | `config/sweep_constraints_pretrain_penalty_soft.yaml` | penalty | soft | zero-mode and PDE weights/warmups |
| C | `config/sweep_constraints_pretrain_al_soft.yaml` | AL | soft | zero-mode, PDE, rho |

Common sweep settings:

- `method: bayes`
- `metric.name: best_val_err`
- `run_cap: 12`
- `max_epochs: 150`
- `checkpoint_selection_metric: val_err`

## Final Presets

Locked final-run presets live in `config/operators_mixed.yaml`:

- `mixed-scale-all-constraints-penalty-pde-only`
- `mixed-scale-all-constraints-al-pde-only`
- `mixed-scale-all-constraints-penalty-hard`
- `mixed-scale-all-constraints-al-hard`
- `mixed-scale-all-constraints-penalty-soft`
- `mixed-scale-all-constraints-al-soft`

Run a selected preset:

```bash
sbatch --export=ALL,CONFIG_NAME=<CONFIG_NAME>,RUN_NAME=<RUN_NAME> \
  scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

Then push the selected checkpoint into downstream mixed-transfer configs:

```bash
PRETRAIN_CONFIG_NAME=<CONFIG_NAME> \
PRETRAIN_RUN_PREFIX=<RUN_NAME> \
bash scripts/maintenance/update_mixed_checkpoint_path.sh <pretrain_job_id>
```

## Decision Rules

Candidate selection during sweeps:

1. lowest `best_val_err`,
2. exclude NaN/divergent runs,
3. tie-break by lower `val_pde_residual_norm`.

Final reporting:

1. best mean low-data transfer error over Poisson, AdvDiff, and Helmholtz,
2. if tied, lower mean `test_pde_residual_norm`,
3. if still tied, prefer simpler settings: hard over soft, penalty over AL.

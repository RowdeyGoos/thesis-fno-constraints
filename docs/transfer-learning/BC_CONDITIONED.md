# Boundary-Conditioned Mixed Pretraining

This guide covers boundary-conditioned (BC) data, mixed BC pretraining, BC
constraint modes, smoke tests, and the soft/hard comparison workflow.

## Data Contract

BC-enabled HDF5 files use:

- `fields`: `(N, 2, nx, ny)`
  - `fields[:,0]`: source
  - `fields[:,1]`: solution
- `tensor`: system-specific coefficients
- `bc`: `(N, 2, nx, ny)`
  - `bc[:,0]`: boundary value map `g(x,y)`
  - `bc[:,1]`: boundary mask `m(x,y)`, with `1` on the boundary and `0` in
    the interior

Mixed BC datasets additionally store:

- `labels`: `0=Poisson`, `1=AdvDiff`, `2=Helmholtz`

BC mixed model input uses `in_dim: 9`:

```text
0      source
1..6   tensor channels [k11, k12, k22, vx, vy, omega]
7      BC value map g
8      BC mask m
```

## Data Generation And Assembly

Generate per-PDE BC datasets:

```bash
bash scripts/workflows/run_gen_data_bc.sh
```

Build mixed BC train/val/test datasets:

```bash
bash scripts/workflows/run_build_mixed_bc.sh
```

The mixed builder requires `bc` in all inputs when called with `--require_bc`
and preserves alignment across `fields`, `tensor`, `labels`, and `bc`.

Sanity-check generated data:

```bash
python scripts/data/check_bc_dataset_sanity.py --input data/poisson --glob "*_bc.h5"
python scripts/data/check_bc_dataset_sanity.py --input data/advdiff --glob "*_bc.h5"
python scripts/data/check_bc_dataset_sanity.py --input data/helmholtz --glob "*_bc.h5"
```

## Training Semantics

BC configs live in `config/operators_mixed_bc.yaml`.

Key config knobs:

```yaml
use_bc_channels: true
bc_value_channel_idx: 7
bc_mask_channel_idx: 8
constraint_bc_enforcement: "off"   # off | soft | hard | hard+soft
constraint_bc_weight: 0.1
constraint_bc_warmup_fraction: 0.0
constraint_bc_loss_norm: "l2"       # l2 | l1
constraint_pde_discretization: "spectral"
```

Prediction flow:

1. `u_raw = model(inputs)`
2. `u_final = project_bc(u_raw, g, m)` for hard modes
3. data loss is computed on `u_final`
4. BC soft loss is computed on `u_raw` for `soft` and `hard+soft`
5. metrics are logged for both raw and final predictions

Hard projection:

```text
u_final = u_raw * (1 - m) + g * m
```

Mode behavior:

| Mode | Projection | Soft BC loss |
|---|---|---|
| `off` | no | no |
| `soft` | no | yes |
| `hard` | yes | no |
| `hard+soft` | yes | yes |

Important current limitation: `constraint_pde_discretization: fd` is parsed for
API compatibility, but the non-periodic FD residual loss path is not active in
the current implementation.

## Smoke Tests

Local smoke:

```bash
MODES="off soft hard hard+soft" bash scripts/workflows/run_local_smoke_train_eval_bc_constraints.sh
```

SLURM smoke:

```bash
sbatch scripts/slurm/smoke/submit_smoke_train_eval_bc_constraints.sh
```

Pass criteria:

- train logs include `bc_violation_raw`, `bc_violation_final`, and
  `val_err_interior`;
- eval logs include `test_bc_violation_raw`, `test_bc_violation_final`, and
  `test_err_interior`;
- hard modes have near-zero final BC violation.

## BC Experiment Workflow

Run fixed mode comparison:

```bash
bash scripts/experiments/submit_bc_constraints_mode_off.sh
bash scripts/experiments/submit_bc_constraints_mode_soft.sh
bash scripts/experiments/submit_bc_constraints_mode_hard.sh
bash scripts/experiments/submit_bc_constraints_mode_hard_soft.sh
```

Or run only soft-vs-hard:

```bash
bash scripts/experiments/submit_bc_constraints_soft_hard_compare.sh
```

Run a soft-mode hyperparameter sweep:

```bash
bash scripts/experiments/submit_bc_constraints_stage_soft.sh
```

Rank sweep candidates:

```bash
python scripts/experiments/select_constraints_candidate.py \
  --sweep_root experiments/sweeps/<SWEEP_ID> \
  --top_k 5 \
  --output_json results/constraints/<SWEEP_ID>_ranking.json
```

## Selection Rule

Choose the final BC model by:

1. lowest `test_err_interior`,
2. if close, lower `test_bc_violation_final`,
3. if still close, simpler enforcement: `hard` over `hard+soft`, fixed mode
   over heavily tuned soft mode.

## Checkpoint Compatibility

The trainer and inferencer support warm-starting legacy mixed checkpoints
(`in_dim: 7`) into BC mixed models (`in_dim: 9`). Existing input columns are
copied and new BC input columns are initialized to zero before strict loading.

## Known Limitations

- BC channels are affected by the global source normalization step when
  `scales_path` is used.
- No Neumann or Robin boundary-condition support is included.
- FD residual loss for non-periodic BC data is scaffolded but not active.

## Future Work: Seam Padding

FFT-based FNOs are implicitly periodic, which can create seam artifacts for
non-periodic BC data. A future experiment can move the artificial seam into a
padded halo:

1. pad physical input to `[N+2p, N+2p]`,
2. run the unchanged FFT-FNO,
3. crop back to the physical `[N, N]` domain,
4. compute data and BC losses only on the cropped domain.

Suggested first defaults for a future implementation:

- `p = 8` for `N=64`,
- `p = 16` for `N=128`,
- `pad_mode = replicate`,
- compare `p=0` against BC modes `off`, `soft`, `hard`, and `hard+soft`.

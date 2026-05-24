# BC Constraints Experiment Walkthrough

This document is an execution checklist for running BC-conditioned mixed pretraining end-to-end.

It answers:

1. What to run
2. In which order
3. What to check in outputs before moving on

Use this together with:

- `docs/transfer-learning/BC_CONDITIONED_IMPLEMENTATION_GUIDE.md`
- `config/operators_mixed_bc.yaml`

---

## 1) Target comparisons

You should evaluate these questions in order:

1. BC enforcement mode with fixed hyperparameters: `off` vs `soft` vs `hard` vs `hard+soft`
2. Soft-mode hyperparameters: tune `constraint_bc_weight` (and optionally warmup/loss norm)
3. Final BC winner based on interior error vs boundary violation tradeoff

---

## 2) One-time preflight

Run from:

```bash
cd <repo-root>
```

Quick checks:

```bash
ls containers/neuraloperators.sif
ls config/operators_mixed_bc.yaml
ls config/sweep_constraints_pretrain_bc_soft.yaml
ls run_gen_data_bc.sh run_build_mixed_bc.sh
```

---

## 3) Stage 0: Smoke test (recommended)

Use this first to catch config/runtime issues before full generation and long pretraining runs.

Local smoke:

```bash
MODES="off soft hard hard+soft" bash scripts/utils/run_local_smoke_train_eval_bc_constraints.sh
```

SLURM smoke:

```bash
sbatch scripts/slurm/smoke/submit_smoke_train_eval_bc_constraints.sh
```

Pass criteria:

- Job exits successfully.
- Train log contains BC metrics: `bc_violation_raw`, `bc_violation_final`, `val_err_interior`
- Eval log contains BC metrics: `test_bc_violation_raw`, `test_bc_violation_final`, `test_err_interior`
- In hard modes, final BC violation is near zero.

---

## 4) Stage 1: Generate BC datasets per PDE system

Recommended (single command):

```bash
bash run_gen_data_bc.sh
```

This generates:

- Poisson BC datasets in `data_root/poisson`
- AdvDiff BC datasets in `data_root/advdiff`
- Helmholtz BC datasets in `data_root/helmholtz`

Important knobs in `run_gen_data_bc.sh`:

- split sizes: `ntrain`, `nval`, `ntest`
- grid/source controls: `n`, `ng`
- BC controls: `bc_modes`, `bc_amplitude`, `bc_width`
- I/O/performance: `h5_chunk_samples`, `progress_every`

Pass criteria:

- Expected files exist for each system:
  - train: `*_32k_bc.h5`
  - val/test: `*_4k_bc.h5`
- No generator runtime failures.

Optional sanity checks:

```bash
python3 scripts/utils/check_bc_dataset_sanity.py --input data/poisson --glob "*_bc.h5"
python3 scripts/utils/check_bc_dataset_sanity.py --input data/advdiff --glob "*_bc.h5"
python3 scripts/utils/check_bc_dataset_sanity.py --input data/helmholtz --glob "*_bc.h5"
```

---

## 5) Stage 2: Build mixed BC train/val/test datasets

Recommended (single command):

```bash
bash run_build_mixed_bc.sh
```

This builds:

- `data/mixed/_train_mixed_32k_bc.h5`
- `data/mixed/_val_mixed_4k_bc.h5`
- `data/mixed/_test_mixed_4k_bc.h5`
- `data/mixed/_train_mixed_32k_bc_scales.npy`

Pass criteria:

- All 4 files above exist.
- Mixed dataset sanity check passes if enabled in `run_build_mixed_bc.sh`.

---

## 6) Stage 3: Fixed-mode BC pretraining comparison

Run the four BC modes with separate submit scripts:

```bash
bash scripts/utils/submit_bc_constraints_mode_off.sh
bash scripts/utils/submit_bc_constraints_mode_soft.sh
bash scripts/utils/submit_bc_constraints_mode_hard.sh
bash scripts/utils/submit_bc_constraints_mode_hard_soft.sh
```

This submits:

- `mixed-bc-scale-all-off`
- `mixed-bc-scale-all-soft`
- `mixed-bc-scale-all-hard`
- `mixed-bc-scale-all-hard-soft`

Soft-vs-hard only:

```bash
bash scripts/utils/submit_bc_constraints_soft_hard_compare.sh
```

Single-run template:

```bash
sbatch --export=ALL,CONFIG_NAME=<CONFIG_NAME>,RUN_NAME=<RUN_NAME> \
  scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh
```

What to compare:

- `val_err`, `val_err_interior`
- `val_bc_violation_raw`, `val_bc_violation_final`
- same metrics on test (`test_*`)

Expected behavior:

- `hard`: very low `*_bc_violation_final`
- `soft`: lower raw boundary violation than `off`, but not exactly zero final violation
- `hard+soft`: hard boundary match with additional raw regularization

Decision output of Stage 3:

- `BC_MODE_WINNER_FIXED = off | soft | hard | hard+soft`

---

## 7) Stage 4: Soft-mode BC hyperparameter sweep

Run BC soft sweep:

```bash
bash scripts/utils/submit_bc_constraints_stage_soft.sh
```

Equivalent:

```bash
bash scripts/utils/submit_bc_constraints_sweep.sh config/sweep_constraints_pretrain_bc_soft.yaml
```

Default sweep targets:

- `constraint_bc_weight`
- `constraint_bc_warmup_fraction`
- `constraint_bc_loss_norm`

Fixed during sweep:

- `constraint_bc_enforcement=soft`
- PDE residual constraints off
- zero-mode constraints off

Rank candidates:

```bash
python scripts/utils/select_constraints_candidate.py \
  --sweep_root experiments/sweeps/<SWEEP_ID> \
  --top_k 5 \
  --output_json results/constraints/<SWEEP_ID>_ranking.json
```

What to look for:

- Primary: lower `val_err`
- Secondary: lower `val_err_interior`
- Constraint quality: reduced `val_bc_violation_raw` and `val_bc_violation_final`
- Stability: no NaN/divergence in top runs

Decision output of Stage 4:

- `BC_SOFT_WINNER = <best config from sweep>`

---

## 8) Stage 5: Final BC model selection and eval

Run final pretraining jobs for:

1. `BC_MODE_WINNER_FIXED` from Stage 3
2. `BC_SOFT_WINNER` from Stage 4 (if soft sweep was run)

Then run eval:

```bash
python3 eval.py \
  --yaml_config config/operators_mixed_bc.yaml \
  --config <CONFIG_NAME> \
  --run_num <EVAL_RUN_NAME> \
  --root_dir experiments \
  --weights <CHECKPOINT_PATH>
```

Primary decision signal:

- Best interior error without unacceptable boundary violations.

Recommended rule:

1. Minimize `test_err_interior`
2. If close, prefer lower `test_bc_violation_final`
3. If still close, prefer simpler enforcement (`hard` over `hard+soft`, fixed mode over heavily tuned mode)

---

## 9) Useful BC SLURM entry points

- Pretrain single BC config:
  - `sbatch scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh`
- BC smoke:
  - `sbatch scripts/slurm/smoke/submit_smoke_train_eval_bc_constraints.sh`
- Submit BC mode `off`:
  - `bash scripts/utils/submit_bc_constraints_mode_off.sh`
- Submit BC mode `soft`:
  - `bash scripts/utils/submit_bc_constraints_mode_soft.sh`
- Submit BC mode `hard`:
  - `bash scripts/utils/submit_bc_constraints_mode_hard.sh`
- Submit BC mode `hard+soft`:
  - `bash scripts/utils/submit_bc_constraints_mode_hard_soft.sh`
- Submit BC soft sweep:
  - `bash scripts/utils/submit_bc_constraints_stage_soft.sh`

---

## 10) Tracking template (copy into your notes)

```text
Stage 1 datasets generated: yes/no
Stage 2 mixed BC datasets built: yes/no
Stage 3 fixed-mode winner:
Stage 4 soft-sweep winner:
Final selected config:
Final pretrain job ID:
Final checkpoint path:

val_err:
val_err_interior:
val_bc_violation_raw:
val_bc_violation_final:
test_err:
test_err_interior:
test_bc_violation_raw:
test_bc_violation_final:

Overall recommendation:
```

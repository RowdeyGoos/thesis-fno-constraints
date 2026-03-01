# Foundation Constraints Experiment Walkthrough

This document is an execution checklist for running the constraints study end-to-end.

It answers:

1. What to run
2. In which order
3. What to check in the outputs before moving on

Use this together with:

- `docs/transfer-learning/FOUNDATION_CONSTRAINTS_EXPERIMENT_RUNBOOK.md`
- `docs/transfer-learning/BLOCK_A_CONSTRAINTS.md`

---

## 1) Target comparisons

You will evaluate three questions in this order:

1. PDE method with zero-mode off (control): `penalty` vs `augmented_lagrangian`
2. PDE method with hard zero-mode: `penalty` vs `augmented_lagrangian`
3. Zero-mode strategy with winning PDE method: `hard` vs `soft`

Then you run final 500-epoch mixed pretraining and downstream transfer evaluation.

---

## 2) One-time preflight

Run from:

```bash
cd third_party/neuraloperators-TL-scaling
```

Quick checks:

```bash
ls containers/neuraloperators.sif
ls config/sweep_constraints_pretrain_*.yaml
```

You should see:

- `sweep_constraints_pretrain_penalty_pde_only.yaml`
- `sweep_constraints_pretrain_al_pde_only.yaml`
- `sweep_constraints_pretrain_penalty_hard.yaml`
- `sweep_constraints_pretrain_al_hard.yaml`
- `sweep_constraints_pretrain_penalty_soft.yaml`
- `sweep_constraints_pretrain_al_soft.yaml`

---

## 3) Stage 0: Smoke test (recommended)

Use this to catch config/runtime issues before long sweeps.

```bash
sbatch scripts/slurm/smoke/submit_smoke_train_eval_constraints.sh
```

Pass criteria:

- Job exits successfully.
- Train log contains: `pde_residual_norm`, `pde_al_lambda`, `zero_mode_constraint_loss`
- Eval log contains: `test_pde_residual_norm`, `test_zero_mode_violation`, `test_zero_mode_constraint_loss`

---

## 4) Stage A: PDE method comparison with zero-mode OFF (control)

Run two sweeps.

### A1) Penalty + PDE-only

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_penalty_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "Penalty PDE-only sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_penalty_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

### A2) AL + PDE-only

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "AL PDE-only sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_al_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

### A3) Rank candidates

```bash
python scripts/utils/select_constraints_candidate.py \
  --sweep_root experiments/sweeps/<SWEEP_ID> \
  --top_k 5 \
  --output_json results/constraints/<SWEEP_ID>_ranking.json
```

What to look for:

- Primary: lower `val_err`
- Secondary: lower `val_pde_residual_norm`
- Stability: no NaN/divergence runs among top candidates

Decision output of Stage A:

- `PDE_WINNER_PDE_ONLY = penalty | augmented_lagrangian`

---

## 5) Stage B: PDE method comparison with hard zero-mode

Run two sweeps.

### B1) Penalty + hard

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_penalty_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "Penalty hard sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_penalty_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

### B2) AL + hard

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "AL hard sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

What to look for:

- Primary: lower `val_err`
- Secondary: lower `val_pde_residual_norm`
- For hard mode: top runs should have very low `val_zero_mode_violation`

Decision output of Stage B:

- `PDE_WINNER_HARD = penalty | augmented_lagrangian`

---

## 6) Stage C: Hard vs soft zero-mode (using Stage-B winner PDE)

Run one new soft sweep (hard baseline already exists from Stage B for same PDE method).

If `PDE_WINNER_HARD=penalty`:

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_penalty_soft.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "Penalty soft sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_penalty_soft.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

If `PDE_WINNER_HARD=augmented_lagrangian`:

```bash
CREATE_JOB=$(sbatch --parsable \
  --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_soft.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh)

SWEEP_ID=$(grep -ho 'SWEEP_ID=.*' experiments/neuralop-constraints-sweep-${CREATE_JOB}-*.out | tail -n1 | cut -d= -f2)

echo "AL soft sweep: $SWEEP_ID"

sbatch --array=0-3 \
  --export=ALL,MODE=agent,SWEEP_ID=${SWEEP_ID},SWEEP_YAML=config/sweep_constraints_pretrain_al_soft.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

What to look for:

- Primary: `val_err` improvement vs hard baseline for same PDE method
- Tradeoff: `val_zero_mode_violation` will usually be higher for soft than hard
- Accept soft if it gives better error without unacceptable constraint drift

Decision output of Stage C:

- `ZERO_MODE_WINNER = hard | soft`

---

## 7) Stage D: Final 500-epoch mixed pretraining runs

Run locked presets from `config/operators_mixed.yaml`.

Recommended minimum final runs:

1. Best PDE-only control (`penalty-pde-only` or `al-pde-only`)
2. Best constrained model from Stage C

Command template:

```bash
sbatch --export=ALL,CONFIG_NAME=<CONFIG_NAME>,RUN_NAME=<RUN_NAME> \
  scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

Preset names:

- `mixed-scale-all-constraints-penalty-pde-only`
- `mixed-scale-all-constraints-al-pde-only`
- `mixed-scale-all-constraints-penalty-hard`
- `mixed-scale-all-constraints-al-hard`
- `mixed-scale-all-constraints-penalty-soft`
- `mixed-scale-all-constraints-al-soft`

What to look for:

- `logs_best.txt` has finite `val_err`
- `pde_residual_norm` reduced vs unconstrained baseline
- hard mode: very low zero-mode violation
- soft mode: nonzero but controlled zero-mode violation with better/similar error

---

## 8) Stage E: Push winning mixed checkpoint to downstream configs

After final mixed pretraining, update mixed checkpoint paths in:

- `config/operators_poisson.yaml`
- `config/operators_ad.yaml`
- `config/operators_helmholtz.yaml`

Command:

```bash
PRETRAIN_CONFIG_NAME=<WINNING_PRETRAIN_CONFIG> \
PRETRAIN_RUN_PREFIX=<WINNING_RUN_PREFIX> \
bash scripts/utils/update_mixed_checkpoint_path.sh <PRETRAIN_JOBID>
```

Sanity check:

```bash
rg -n "weights:" config/operators_poisson.yaml config/operators_ad.yaml config/operators_helmholtz.yaml
```

Ensure no `JOBID` placeholders remain for mixed fine-tune configs.

---

## 9) Stage F: Run downstream mixed fine-tuning

Run mixed fine-tuning jobs to populate transfer curves.

### Poisson mixed fine-tuning

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh   # 16..4k
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_medium.sh  # 8k
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_large.sh   # 16k,32k
```

### AdvDiff mixed fine-tuning

```bash
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_small.sh
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_medium.sh
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_large.sh
```

### Helmholtz mixed fine-tuning

```bash
sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh
sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_medium.sh
sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_large.sh
```

---

## 10) Stage G: Batch transfer evaluation and plots

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh
sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh
sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh
```

Main result files:

- `results/transfer_learning_k1_2.5/mixed/poisson_results.json`
- `results/transfer_learning_advdiff_adr0.2_0.4/mixed/advdiff_results.json`
- `results/transfer_learning_helmholtz_o1_5/mixed/helmholtz_results.json`

What to look for:

- Primary: low-data transfer (`16, 64, 256, 1k`) should improve or stay neutral vs previous mixed baseline.
- Secondary: full-curve mean (`16` to `32k`) should not regress badly.
- Diagnostics:
  - `test_pde_residual_norm` lower is better.
  - `test_zero_mode_violation`:
    - hard: should be very small for constrained samples.
    - soft: higher than hard is expected; assess tradeoff against error.

---

## 11) Decision rule (recommended)

Pick the final winner by:

1. Best mean low-data transfer error over Poisson/AdvDiff/Helmholtz.
2. If nearly tied, prefer lower mean `test_pde_residual_norm`.
3. If still tied, prefer simpler setting:
   - hard over soft
   - penalty over AL

Keep Stage-A PDE-only winner as control in your report.

---

## 12) Tracking template (copy into your notes)

```text
Stage A winner (PDE-only):
Stage B winner (hard):
Stage C winner (hard vs soft):
Final pretrain config:
Final pretrain job ID:
Checkpoint path used for transfer:

Poisson low-data mean:
AdvDiff low-data mean:
Helmholtz low-data mean:
Overall recommendation:
```

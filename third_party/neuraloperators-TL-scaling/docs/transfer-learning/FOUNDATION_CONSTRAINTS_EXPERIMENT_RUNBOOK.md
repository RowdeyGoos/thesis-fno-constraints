# Foundation Constraints Experiment Runbook

This runbook defines a strict foundation-model protocol for comparing:

1. PDE residual objective: `penalty` vs `augmented_lagrangian`
2. Zero-mean enforcement: `hard` vs `soft`
3. PDE-only control: zero-mode fully `off`

The workflow avoids downstream leakage by tuning only on mixed pretraining validation metrics.

---

## 1) Study design (strict FM)

- Foundation pretraining track: `mixed-scale-all`
- Tuning metric: `best_val_err` (accuracy-first)
- Checkpoint selection: `checkpoint_selection_metric: val_err`
- Proxy sweep budget: `max_epochs=150`, `run_cap=12`
- Final confirmation budget: `max_epochs=500`, `2 seeds`

### Stage A (PDE method)

Control comparison with zero-mode disabled:

- `penalty + PDE-only`: `config/sweep_constraints_pretrain_penalty_pde_only.yaml`
- `AL + PDE-only`: `config/sweep_constraints_pretrain_al_pde_only.yaml`

### Stage B (PDE method under hard zero-mode)

Compare hard zero-mode with equal tuning budgets:

- `penalty + hard`: `config/sweep_constraints_pretrain_penalty_hard.yaml`
- `AL + hard`: `config/sweep_constraints_pretrain_al_hard.yaml`

### Stage C (Zero-mode)

Fix PDE method to Stage-B winner and compare:

- `<winner PDE> + hard`
- `<winner PDE> + soft`

Use:

- `config/sweep_constraints_pretrain_penalty_soft.yaml` (if penalty wins)
- `config/sweep_constraints_pretrain_al_soft.yaml` (if AL wins)

---

## 2) Sweep launch workflow

Use `scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh`.

Recommended helper scripts:

```bash
# One sweep (create + wait for SWEEP_ID + submit agents)
bash scripts/utils/submit_constraints_sweep.sh config/sweep_constraints_pretrain_al_hard.yaml

# Whole stages
bash scripts/utils/submit_constraints_stage_a.sh
bash scripts/utils/submit_constraints_stage_b.sh
bash scripts/utils/submit_constraints_stage_c.sh penalty   # or: al
```

### 2.1 Create sweep ID

```bash
sbatch --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

Read `SWEEP_ID=...` from the job output.

PDE-only control example:

```bash
sbatch --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

### 2.2 Launch sweep agents

```bash
sbatch --array=0-3 --export=ALL,MODE=agent,SWEEP_ID=<SWEEP_ID>,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

PDE-only control example:

```bash
sbatch --array=0-3 --export=ALL,MODE=agent,SWEEP_ID=<SWEEP_ID>,SWEEP_YAML=config/sweep_constraints_pretrain_al_pde_only.yaml \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

Notes:

- Each array task runs one W&B agent by default.
- Increase per-task agents via `AGENT_COUNT_PER_TASK`, e.g.:

```bash
sbatch --array=0-1 --export=ALL,MODE=agent,SWEEP_ID=<SWEEP_ID>,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml,AGENT_COUNT_PER_TASK=2 \
  scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
```

---

## 3) Candidate selection rules

Pick top candidates by:

1. Primary: lowest `best_val_err`
2. Exclude unstable runs (`NaN`, divergence)
3. Tie-breaker: lower `val_pde_residual_norm`

Then run final 500-epoch confirmation with 2 seeds.

You can rank local sweep outputs with:

```bash
python scripts/utils/select_constraints_candidate.py \
  --sweep_root experiments/sweeps/<SWEEP_ID> \
  --top_k 5 \
  --output_json results/constraints/<SWEEP_ID>_ranking.json
```

---

## 4) Final constrained pretraining presets

`config/operators_mixed.yaml` includes fixed presets:

- `mixed-scale-all-constraints-penalty-pde-only`
- `mixed-scale-all-constraints-al-pde-only`
- `mixed-scale-all-constraints-penalty-hard`
- `mixed-scale-all-constraints-al-hard`
- `mixed-scale-all-constraints-penalty-soft`
- `mixed-scale-all-constraints-al-soft`

Run via pretrain script overrides:

```bash
sbatch --export=ALL,CONFIG_NAME=mixed-scale-all-constraints-al-pde-only,RUN_NAME=pretrain-mixed-al-pde-only \
  scripts/slurm/pretrain/submit_pretrain_mixed.sh

sbatch --export=ALL,CONFIG_NAME=mixed-scale-all-constraints-al-hard,RUN_NAME=pretrain-mixed-al-hard \
  scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

These presets are intended as locked final runs after sweep tuning.

---

## 5) Push mixed checkpoint into transfer configs

After choosing a final mixed pretraining run, update all mixed-transfer weights in:

- `config/operators_poisson.yaml`
- `config/operators_ad.yaml`
- `config/operators_helmholtz.yaml`

Using:

```bash
bash scripts/utils/update_mixed_checkpoint_path.sh <JOBID>
```

For constrained pretraining names, override defaults:

```bash
PRETRAIN_CONFIG_NAME=mixed-scale-all-constraints-al-hard \
PRETRAIN_RUN_PREFIX=pretrain-mixed-al-hard \
bash scripts/utils/update_mixed_checkpoint_path.sh <JOBID>
```

---

## 6) Transfer evaluation (main report)

Keep downstream finetuning recipes fixed and evaluate transfer from constrained foundation checkpoints.

Run existing eval scripts:

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh
sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh
sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh
```

Primary winner metric for reporting:

- Mean low-data transfer error over sample sizes: `16, 64, 256, 1k`

Secondary diagnostics:

- `test_pde_residual_norm`
- `test_zero_mode_violation`
- full-curve mean over all sample sizes

---

## 7) Reproducibility checklist

- [ ] Sweep YAML used is archived with its `SWEEP_ID`
- [ ] Final run config names + job IDs recorded
- [ ] `update_mixed_checkpoint_path.sh` invocation logged
- [ ] Transfer eval result JSON files archived
- [ ] Comparison plots regenerated from archived JSONs

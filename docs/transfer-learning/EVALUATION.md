# Transfer Learning Evaluation

This guide explains how to evaluate downstream transfer-learning runs and
generate data-efficiency plots.

## Quick Start

Run the batch evaluation jobs:

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh
sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh
sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh
```

For constrained mixed checkpoint comparisons:

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer_constraints.sh
sbatch scripts/slurm/eval/submit_eval_advdiff_transfer_constraints.sh
sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer_constraints.sh
```

Local Poisson evaluation can be launched with:

```bash
bash scripts/eval/eval_all_transfer_learning.sh
```

## What Gets Evaluated

The main Poisson bundle compares:

- mixed-pretrained finetuning,
- `k1_5` single-domain pretrained finetuning,
- from-scratch downstream training.

Each approach is evaluated over the downstream sample sizes available for that
experiment family.

## Manual Evaluation

Evaluate each approach directly when debugging or when only a subset of configs
is ready:

```bash
python scripts/entrypoints/eval_transfer_learning.py \
  --yaml_config config/operators_mixed.yaml \
  --experiment_type poisson \
  --configs poisson-k1_2.5-finetune-mixed-16 \
           poisson-k1_2.5-finetune-mixed-64 \
           poisson-k1_2.5-finetune-mixed-256 \
           poisson-k1_2.5-finetune-mixed-1k \
           poisson-k1_2.5-finetune-mixed-4k \
  --experiment_dir experiments \
  --output_dir results/transfer_learning_k1_2.5/mixed
```

Then combine result files into one comparison plot:

```bash
python scripts/eval/plot_transfer_learning_comparison.py \
  --mixed_results results/transfer_learning_k1_2.5/mixed/results.json \
  --k1_5_results results/transfer_learning_k1_2.5/k1_5/results.json \
  --scratch_results results/transfer_learning_k1_2.5/scratch/results.json \
  --output_dir results/transfer_learning_k1_2.5 \
  --title "Transfer Learning: Poisson k in [1,2.5]"
```

## Evaluation Flow

`scripts/entrypoints/eval_transfer_learning.py`:

1. resolves the requested config names,
2. finds `ckpt_best.tar` for each completed run,
3. invokes the standard inferencer/evaluation path,
4. stores metrics as JSON,
5. writes PNG/PDF plots when plotting is enabled.

`scripts/eval/plot_transfer_learning_comparison.py` creates combined plots from
separate result JSON files.

## Output Layout

Typical Poisson output:

```text
results/transfer_learning_k1_2.5/
|-- mixed/
|   `-- results.json
|-- k1_5/
|   `-- results.json
|-- scratch/
|   `-- results.json
|-- transfer_learning_comparison.png
|-- transfer_learning_comparison.pdf
`-- eval_logs/
```

Primary metrics:

- `test_error`: relative L2 error.
- `test_loss`: evaluation loss.
- `test_pde_residual_norm`: PDE residual diagnostic when constraints are
  enabled.
- `test_zero_mode_violation`: zero-mode diagnostic when relevant.
- `test_bc_violation_raw`, `test_bc_violation_final`, `test_err_interior`:
  BC diagnostics for boundary-conditioned runs.

## Plot Interpretation

- X-axis: downstream training samples, usually on a log scale.
- Y-axis: relative L2 test error; lower is better.
- Transfer benefit is strongest when pretrained curves sit below scratch at
  low sample sizes.
- Mixed vs single-domain pretraining is task dependent: in-domain pretraining
  often helps interpolation tasks, while mixed pretraining tests broader
  cross-operator reuse.

## Troubleshooting

- Checkpoint not found: confirm training finished and inspect
  `experiments/expts/<config>/*/checkpoints/`.
- Missing data points: evaluate only finished configs with `--configs`, then
  rerun the full bundle after the remaining jobs complete.
- Evaluation fails: run `scripts/entrypoints/eval.py` manually for one config
  and checkpoint to expose the underlying error.
- CUDA out of memory: set `--device cpu`, reduce eval batch sizes in the
  config, or evaluate configs one at a time.
- Plot looks wrong: verify the result JSON files correspond to the same
  downstream task and checkpoint family.

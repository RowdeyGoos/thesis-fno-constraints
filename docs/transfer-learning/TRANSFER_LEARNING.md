# Transfer Learning Workflows

This guide is the active entry point for downstream transfer-learning
experiments. It covers the Poisson `k5_10` extrapolation task, the Poisson
`k1_2.5` interpolation task, and mixed-pretrained downstream runs.

## Experiment Families

The repository compares three downstream strategies:

- single-domain finetuning: initialize from a pretrained single-PDE checkpoint,
  then train on the downstream task.
- mixed-pretrained finetuning: initialize from `mixed-scale-all` or a
  constrained mixed checkpoint, then train on mixed-format downstream data.
- from-scratch training: train directly on the downstream task without
  pretrained weights.

Common downstream sample sizes are `16`, `64`, `256`, `1k`, `4k`, and, for
some mixed runs, `8k`, `16k`, and `32k`.

## Poisson Transfer Tasks

### `k5_10` extrapolation

- Source pretraining: Poisson `k in [1,5]`.
- Downstream target: Poisson `k in [5,10]`.
- Config family: `poisson-k5_10-*` in `config/operators_poisson.yaml`.
- Launcher:

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k5_10_array.sh
```

### `k1_2.5` interpolation

- Source pretraining: Poisson `k in [1,5]`.
- Downstream target: Poisson `k in [1,2.5]`.
- Config family: `poisson-k1_2.5-*` in `config/operators_poisson.yaml`.
- Launcher:

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_array.sh
```

### Mixed-pretrained `k1_2.5`

Mixed-pretrained checkpoints expect mixed-format input (`in_dim: 7`). Convert
or generate downstream data in mixed format before running these configs.

```bash
bash scripts/data/convert_k1_2.5_to_mixed_format.sh
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

Larger mixed data regimes use:

```bash
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_medium.sh
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_large.sh
```

## Checkpoint Handoff

Single-domain Poisson finetuning uses:

```bash
bash scripts/maintenance/update_checkpoint_path.sh <pretrain_job_id>
bash scripts/maintenance/update_k1_2.5_checkpoint_path.sh <pretrain_job_id>
```

Mixed-pretrained finetuning uses:

```bash
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
```

For constrained mixed checkpoints, pass the expected pretraining config and run
prefix:

```bash
PRETRAIN_CONFIG_NAME=mixed-scale-all-constraints-al-hard \
PRETRAIN_RUN_PREFIX=pretrain-mixed-al-hard \
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
```

Always verify checkpoint paths before launching long jobs:

```bash
rg -n "weights:" config/operators_poisson.yaml config/operators_ad.yaml config/operators_helmholtz.yaml
```

## Sample-Size Convention

Most downstream configs use `subsample` to select a fixed-size subset from a
32k training dataset:

| Samples | Subsample |
|---:|---:|
| 16 | 2048 |
| 64 | 512 |
| 256 | 128 |
| 1k | 32 |
| 4k | 8 |
| 8k | 4 |
| 16k | 2 |
| 32k | 1 |

Small data configs also reduce batch size so the batch is not larger than the
effective training set.

## Cross-PDE Downstream Bundles

Mixed-pretrained downstream runs are split by PDE family and data regime:

```bash
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_small.sh
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_medium.sh
sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_large.sh

sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh
sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_medium.sh
sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_large.sh
```

For the complete list of launchers, see
`scripts/slurm/finetune/README.md`.

## Troubleshooting

- Checkpoint missing: inspect `experiments/expts/<config>/*/checkpoints/` and
  rerun the relevant checkpoint update script.
- Data missing: verify the configured `train_path`, `val_path`, `test_path`,
  and `scales_path` in the YAML file.
- Mixed checkpoint load failure: confirm downstream data has six tensor
  components and the config uses `in_dim: 7`.
- Slow jobs: check GPU utilization with `nvidia-smi`, confirm `plot_figs` is
  disabled for small downstream sweeps, and check storage I/O.

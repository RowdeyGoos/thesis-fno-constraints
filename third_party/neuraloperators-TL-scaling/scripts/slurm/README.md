# SLURM Scripts

This directory is organized by workflow stage.

## Layout

- `scripts/slurm/pretrain/` - pretraining jobs (single, array, mixed)
- `scripts/slurm/finetune/` - downstream fine-tuning and transfer learning jobs
- `scripts/slurm/eval/` - batch evaluation and plotting jobs
- `scripts/slurm/smoke/` - quick runtime smoke tests (train + eval)
- `scripts/slurm/legacy/` - legacy/venv-based submission scripts

## Primary Entry Points

### Pretraining

- `sbatch scripts/slurm/pretrain/submit_pretrain_array_ddp.sh`
- `sbatch scripts/slurm/pretrain/submit_pretrain_array_single_gpu.sh`
- `sbatch scripts/slurm/pretrain/submit_pretrain_single_ddp.sh`
- `sbatch scripts/slurm/pretrain/submit_pretrain_single_gpu.sh`
- `sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh`

### Fine-Tuning

See `scripts/slurm/finetune/README.md`.

### Evaluation

- `sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh`
- `sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh`
- `sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh`

### Smoke test

- `sbatch scripts/slurm/smoke/submit_smoke_train_eval_constraints.sh`

### Legacy

- `sbatch scripts/slurm/legacy/submit_pretrain_venv.sh`

## Notes

- Paths under `scripts/slurm/` are canonical; old paths were removed.
- If you have local aliases/scripts, update them to the new locations above.

# Fine-Tuning SLURM Scripts

This folder contains transfer-learning / downstream fine-tuning scripts by PDE family.

Default downstream protocol:

- Every downstream training config is launched for seeds `0`, `1`, and `2`
- Seeded runs are launcher-driven; the YAML config names do not change
- Seeded run directories include `seed0`, `seed1`, or `seed2` in `run_num`
- Pretraining and zero-shot evaluation behavior are unchanged
- Downstream training scripts pass `--seed`, `--train_shuffle`, `--random_train_subset`, and `--subset_seed`

## Poisson

### Array Jobs

- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k5_10_array.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_array.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh`

### Split by Data Regime

- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_small.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_medium.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_large.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_small.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_medium.sh`
- `sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_large.sh`

## Advection-Diffusion

### Split by Data Regime

- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_small.sh`
- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_medium.sh`
- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_large.sh`
- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_small.sh`
- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_medium.sh`
- `sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_large.sh`

## Helmholtz

### Split by Data Regime

- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_small.sh`
- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_medium.sh`
- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_large.sh`
- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh`
- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_medium.sh`
- `sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_large.sh`

## Related Evaluation Jobs

- `sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh`
- `sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh`
- `sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh`

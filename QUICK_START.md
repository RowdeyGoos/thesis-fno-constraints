# Quick Start

This is the shortest path for verifying that the repository works before
launching full thesis experiments.

## Documentation Map

- [README.md](README.md) - thesis-facing repository overview.
- [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) - upstream-paper baseline
  reproduction guide.
- [docs/transfer-learning/README.md](docs/transfer-learning/README.md) -
  transfer learning, mixed datasets, constraints, BC conditioning, and
  evaluation guides.
- [DAIC_SETUP.md](DAIC_SETUP.md) - TU Delft DAIC container and SLURM setup.

## 1. Environment

```bash
cd <repo-root>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

The original neuraloperators stack targets PyTorch 1.12 with CUDA 11.3. On
DAIC, prefer the container workflow in [DAIC_SETUP.md](DAIC_SETUP.md).

## 2. Run The Local Smoke Test

```bash
chmod +x scripts/workflows/quick_test.sh
./scripts/workflows/quick_test.sh
```

This generates a small Poisson dataset, computes normalization scales, trains a
small FNO for a few epochs, and verifies that the loss decreases.

## 3. Constraint Smoke Tests

Use these before launching long constrained pretraining jobs:

```bash
bash scripts/workflows/run_local_smoke_train_eval_constraints.sh
MODES="off soft hard hard+soft" bash scripts/workflows/run_local_smoke_train_eval_bc_constraints.sh
```

On DAIC:

```bash
sbatch scripts/slurm/smoke/submit_smoke_train_eval_constraints.sh
sbatch scripts/slurm/smoke/submit_smoke_train_eval_bc_constraints.sh
```

## 4. Main Workflows

Baseline reproduction:

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_array_ddp.sh
```

Mixed pretraining:

```bash
sbatch scripts/slurm/pretrain/submit_pretrain_mixed.sh
```

Mixed transfer finetuning:

```bash
bash scripts/maintenance/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>
sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh
```

Transfer evaluation:

```bash
sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh
```

## Expected Baseline Scale

| PDE system | Typical test L2 error | Training time, 4 GPUs |
|---|---:|---:|
| Poisson `k in [1,5]` | about 0.02-0.03 | about 10 hours |
| Advection-Diffusion | about 0.03-0.05 | about 10 hours |
| Helmholtz `omega in [1,10]` | about 0.05-0.08 | about 10 hours |

For the full thesis workflow, continue with
[docs/transfer-learning/README.md](docs/transfer-learning/README.md).

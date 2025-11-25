# Scripts Directory

Executable scripts for training, evaluation, and cluster job submission.

## 📁 Directory Structure

```
scripts/
├── container/          # Container build and management
├── slurm/             # SLURM job submission scripts
├── setup/             # Environment setup scripts
├── utils/             # Utility scripts
└── train.py           # Main training script
```

## 📦 Container Scripts (`container/`)

Scripts for building and managing Docker/Apptainer containers.

### `build_container.sh`
Build Docker image locally and export as tar.

```bash
bash scripts/container/build_container.sh
```

### `build_apptainer.sh`
Build Apptainer image directly (requires Apptainer installed).

```bash
bash scripts/container/build_apptainer.sh
```

### `transfer_container.sh`
Transfer container to DAIC and convert to Apptainer format.

```bash
bash scripts/container/transfer_container.sh <netid>
```

## 🖥️ SLURM Job Submission (`slurm/`)

Scripts for submitting jobs to the DAIC cluster.

### Container-Based Jobs (Recommended)

**`submit_job_container.sh`** - Single GPU training job with container
```bash
sbatch scripts/slurm/submit_job_container.sh
```

**`submit_job_array_container.sh`** - Array job for multiple experiments
```bash
sbatch scripts/slurm/submit_job_array_container.sh
```

### Virtual Environment Jobs

**`submit_job.sh`** - Single GPU training job
```bash
sbatch scripts/slurm/submit_job.sh
```

**`submit_job_array.sh`** - Array job for multiple experiments
```bash
sbatch scripts/slurm/submit_job_array.sh
```

## ⚙️ Setup Scripts (`setup/`)

One-time environment setup scripts.

### `setup_environment.sh`
Create Python virtual environment with all dependencies.

```bash
bash scripts/setup/setup_environment.sh
```

## 🛠️ Utility Scripts (`utils/`)

Helper scripts for monitoring and interactive sessions.

### `check_job_status.sh`
Check status of submitted jobs and view recent logs.

```bash
bash scripts/utils/check_job_status.sh
```

### `interactive_gpu.sh`
Request an interactive GPU session for debugging.

```bash
bash scripts/utils/interactive_gpu.sh
```

## 🐍 Training Scripts (root)

### `train.py`
Main training script for FNO models.

```bash
python scripts/train.py \
    --config configs/training/default.yaml \
    --model configs/models/fno_baseline.yaml \
    --dataset configs/datasets/pdebench.yaml \
    --output experiments/runs/test
```

## 🚀 Quick Usage Examples

### Example 1: Build and Use Container

```bash
# Build container
bash scripts/container/build_container.sh

# Transfer to DAIC
bash scripts/container/transfer_container.sh mynetid

# On DAIC, submit job
sbatch scripts/slurm/submit_job_container.sh
```

### Example 2: Virtual Environment Workflow

```bash
# One-time setup
bash scripts/setup/setup_environment.sh

# Submit training job
sbatch scripts/slurm/submit_job.sh

# Check status
bash scripts/utils/check_job_status.sh
```

### Example 3: Interactive Development

```bash
# Request GPU session
bash scripts/utils/interactive_gpu.sh

# Inside session, run training
python scripts/train.py --config configs/training/default.yaml
```

### Example 4: Multiple Experiments

```bash
# Edit array job script to define experiments
# Then submit array job
sbatch scripts/slurm/submit_job_array_container.sh

# Monitor all jobs
watch -n 10 squeue -u $USER
```

## 📝 Script Categories

### By Purpose

| Purpose | Scripts |
|---------|---------|
| **Build Environment** | `container/build_*.sh`, `setup/setup_environment.sh` |
| **Submit Jobs** | `slurm/submit_job*.sh` |
| **Monitor Jobs** | `utils/check_job_status.sh` |
| **Interactive Work** | `utils/interactive_gpu.sh`, `train.py` |
| **Transfer Files** | `container/transfer_container.sh` |

### By Use Case

| Use Case | Recommended Scripts |
|----------|-------------------|
| **First-time setup** | `setup/setup_environment.sh` OR `container/build_container.sh` |
| **Development** | `utils/interactive_gpu.sh`, `slurm/submit_job.sh` |
| **Production runs** | `slurm/submit_job_container.sh` |
| **Multiple experiments** | `slurm/submit_job_array_container.sh` |
| **Debugging** | `utils/check_job_status.sh`, `utils/interactive_gpu.sh` |

## 🔗 See Also

- [CONTAINER_GUIDE.md](../CONTAINER_GUIDE.md) - Detailed container instructions
- [CLUSTER_USAGE.md](../CLUSTER_USAGE.md) - Complete cluster guide
- [CONTAINER_VS_VENV.md](../CONTAINER_VS_VENV.md) - Choosing between approaches
- [CLUSTER_QUICK_REF.md](../CLUSTER_QUICK_REF.md) - Quick reference


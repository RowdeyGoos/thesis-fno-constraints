# DAIC Cluster Setup for neuraloperators-TL-scaling

This repository contains scripts and configurations for running the neuraloperators transfer-learning project on TU Delft's DAIC cluster.

## 🔑 Key Differences from Main Project

| Aspect | Archived Root Scaffold | Active Neuraloperators Project |
|--------|---------------------|---------------------------|
| PyTorch Version | 2.1.0 | 1.12.0 |
| CUDA Version | 11.8 | 11.3 |
| cuDNN Version | 8.6 | 8.2 |
| Environment | `~/venv/thesis-fno` | `~/venv/neuraloperators` |
| Container | `thesis-fno.sif` | `neuraloperators.sif` |

## 📁 Directory Structure

```
./
├── scripts/
│   ├── container/
│   │   ├── build_container.sh         # Build Docker container
│   │   ├── build_apptainer.sh         # Build Apptainer directly
│   │   └── transfer_container.sh      # Transfer to DAIC
│   ├── slurm/
│   │   ├── submit_job_container.sh    # Single training job (container)
│   │   ├── submit_job_array_container.sh  # Array job (container)
│   │   └── submit_job.sh              # Training with venv
│   └── setup/
│       └── setup_environment.sh       # Setup virtual environment
├── Dockerfile                         # Docker container definition
├── apptainer.def                      # Apptainer/Singularity definition
├── .dockerignore                      # Docker build exclusions
├── Makefile                          # Build automation
└── DAIC_SETUP.md                     # This file
```

## 🚀 Quick Start

### Option 1: Container-Based (Recommended for DAIC)

1. **Build container locally:**
   ```bash
   cd <repo-root>
   make build-container
   # or: bash scripts/container/build_container.sh
   ```

2. **Transfer to DAIC:**
   ```bash
   make transfer-container NETID=your-netid
   # or: bash scripts/container/transfer_container.sh your-netid
   ```

3. **On DAIC - Convert to Apptainer:**
   ```bash
   ssh your-netid@login.daic.tudelft.nl
   cd ~/neuraloperators
   module load apptainer
   apptainer build neuraloperators.sif docker-archive://neuraloperators_latest.tar
   ```

4. **Navigate to project and submit job:**
   ```bash
   cd ~/thesis-fno-constraints
   sbatch scripts/slurm/pretrain/submit_pretrain_single_ddp.sh
   ```

### Option 2: Virtual Environment (for Development)

1. **On DAIC - Setup environment:**
   ```bash
   cd ~/thesis-fno-constraints
   bash scripts/setup/setup_environment.sh
   ```

2. **Submit job:**
   ```bash
   sbatch scripts/slurm/legacy/submit_pretrain_venv.sh
   ```

## 📝 Customizing Training Jobs

### Single Training Run

Edit `scripts/slurm/pretrain/submit_pretrain_single_ddp.sh` to modify:

```bash
# Configuration file
CONFIG_FILE="config/operators_poisson.yaml"
RUN_NAME="poisson-scale-k1_5"

# Job resources (in SBATCH headers)
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
```

### Multiple PDE Systems (Array Job)

The array job script runs three different PDE systems:
- Poisson's equation
- Advection-Diffusion
- Helmholtz equation

Submit with:
```bash
sbatch scripts/slurm/pretrain/submit_pretrain_array_ddp.sh
```

Customize the configurations in the script:
```bash
configs=(
    "config/operators_poisson.yaml:poisson-scale-k1_5"
    "config/operators_ad.yaml:ad-scale_adr0p2_1"
    "config/operators_helmholtz.yaml:helm-scale-o1_10"
)
```

## 🔧 Job Management

### Submit Jobs
```bash
# Single training run (container)
sbatch scripts/slurm/pretrain/submit_pretrain_single_ddp.sh

# Array job for multiple PDEs (container)
sbatch scripts/slurm/pretrain/submit_pretrain_array_ddp.sh

# Single run with venv (development)
sbatch scripts/slurm/legacy/submit_pretrain_venv.sh
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Watch job status (updates every 10 seconds)
watch -n 10 squeue -u $USER

# Check specific job
squeue -j <job-id>

# Cancel job
scancel <job-id>
```

### View Logs
```bash
# View output
tail -f experiments/neuralop-train-<job-id>.out

# View errors
tail -f experiments/neuralop-train-<job-id>.err

# For array jobs
tail -f experiments/neuralop-array-<array-id>-<task-id>.out
```

## 📊 Data Setup

Before running training, ensure your data is properly configured:

1. **Generate data** (if needed):
   ```bash
   # On DAIC or locally
   python utils/gen_data_poisson.py --help
   ```

2. **Generate normalization scales:**
   ```bash
   python generate_scales.py --data_path /path/to/data
   ```

3. **Update config files** (`config/operators_*.yaml`):
   ```yaml
   poisson-scale-k1_5:
     train_path: /path/to/train.h5
     val_path: /path/to/val.h5
     test_path: /path/to/test.h5
     scales_path: /path/to/scales.npy
   ```

## 🐳 Container Details

### What's Included
- PyTorch 1.12.0 with CUDA 11.3 support
- All dependencies from `requirements.txt`
- Project code (models, utils, config)
- GPU support via `--nv` flag

### Testing Container Locally
```bash
# Test PyTorch
docker run --rm neuraloperators:latest python -c "import torch; print(torch.__version__)"

# Test with GPU (if available)
docker run --gpus all --rm neuraloperators:latest python -c "import torch; print(torch.cuda.is_available())"

# Interactive shell
docker run -it --rm neuraloperators:latest /bin/bash
```

### Testing Container on DAIC
```bash
# Test basic import
apptainer exec neuraloperators.sif python -c "import torch; print(torch.__version__)"

# Test with GPU
apptainer exec --nv neuraloperators.sif python -c "import torch; print(torch.cuda.is_available())"

# Interactive shell
apptainer shell --nv neuraloperators.sif
```

## 🔄 Updating the Container

If you modify the code or dependencies:

1. **Rebuild locally:**
   ```bash
   make build-container
   ```

2. **Transfer to DAIC:**
   ```bash
   make transfer-container NETID=your-netid
   ```

3. **On DAIC - Rebuild Apptainer image:**
   ```bash
   cd ~/neuraloperators
   rm -f neuraloperators.sif  # Remove old version
   apptainer build neuraloperators.sif docker-archive://neuraloperators_latest.tar
   ```

## 🆘 Troubleshooting

### Container Build Issues

**Problem:** Docker build fails with "no space left on device"
```bash
# Clean up Docker
docker system prune -a
```

**Problem:** Permission denied when building Apptainer
```bash
# Use --fakeroot flag
apptainer build --fakeroot neuraloperators.sif docker-archive://neuraloperators_latest.tar
```

### Job Submission Issues

**Problem:** Job stays in pending state
```bash
# Check job details
scontrol show job <job-id>

# Check available resources
sinfo -p gpu
```

**Problem:** Container not found error
```bash
# Verify container exists
ls -lh ~/neuraloperators.sif

# Check path in SLURM script
grep CONTAINER_PATH scripts/slurm/pretrain/submit_pretrain_single_ddp.sh
```

### Module Loading Issues

**Problem:** Module not found
```bash
# List available modules
module avail

# Load alternative module names
module load singularity  # instead of apptainer
module load cuda/11.3.1  # if exact version not available
```

## 📚 Additional Resources

- [Original Project README](README.md)
- [Archived Local Testing Notes](archive/root-scaffold/docs/generated/README_LOCAL_TESTING.md)
- [DAIC User Guide](https://doc.dhpc.tudelft.nl/)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [SLURM Documentation](https://slurm.schedmd.com/)

## 🔗 Results Layout

The flattened project writes active experiment outputs under the repository root:

```bash
~/thesis-fno-constraints/experiments/
```

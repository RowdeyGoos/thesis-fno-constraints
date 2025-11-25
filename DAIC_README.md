# Using the DAIC Cluster - Complete Guide

This document provides an overview of all resources for using the TU Delft DAIC cluster with this project.

## 📚 Documentation Index

| Document | Description | When to Use |
|----------|-------------|-------------|
| **[CONTAINER_VS_VENV.md](CONTAINER_VS_VENV.md)** | 🆕 Container vs Virtual Env comparison | Deciding which approach to use |
| **[CONTAINER_GUIDE.md](CONTAINER_GUIDE.md)** | Complete container guide | Building and using containers (recommended approach) |
| **[CLUSTER_USAGE.md](CLUSTER_USAGE.md)** | Full cluster usage guide | Understanding all cluster features and options |
| **[CLUSTER_QUICK_REF.md](CLUSTER_QUICK_REF.md)** | Quick reference cheat sheet | Quick command lookup |
| **[scripts/README.md](scripts/README.md)** | Scripts documentation | Understanding available scripts |

## 🚀 Quick Start (3 Steps)

### Using Containers (Recommended) ✅

```bash
# Step 1: Build container on your local machine (requires Docker)
bash scripts/build_container.sh

# Step 2: Transfer to DAIC
bash scripts/transfer_container.sh <your-netid>

# Step 3: On DAIC, submit job
ssh <netid>@login.daic.tudelft.nl
cd thesis-fno-constraints
sbatch scripts/submit_job_container.sh
```

**Why containers?** 
- ✅ Maximum reproducibility
- ✅ Exact environment specification
- ✅ Easier dependency management
- ✅ Recommended by DAIC for production runs

**Best for:** Final experiments, paper results, reproducible research

### Using Virtual Environment (Alternative)

```bash
# Step 1: SSH to DAIC
ssh <netid>@login.daic.tudelft.nl

# Step 2: Clone and setup
git clone https://github.com/RowdeyGoos/thesis-fno-constraints.git
cd thesis-fno-constraints
bash scripts/setup_environment.sh

# Step 3: Submit job
sbatch scripts/submit_job.sh
```

**Why virtual environment?**
- ⚡ Faster iteration during development
- 🔧 Easier to modify dependencies
- 🐛 Better for debugging
- 💻 No local Docker required

**Best for:** Development, testing, quick iterations, debugging

## 📦 Available Scripts

### Container Management
- **`build_container.sh`** - Build Docker image
- **`build_apptainer.sh`** - Build Apptainer image
- **`transfer_container.sh`** - Transfer to DAIC

### Job Submission
- **`submit_job_container.sh`** - Single job (container)
- **`submit_job_array_container.sh`** - Array job (container)
- **`submit_job.sh`** - Single job (venv)
- **`submit_job_array.sh`** - Array job (venv)

### Utilities
- **`setup_environment.sh`** - Environment setup
- **`interactive_gpu.sh`** - Interactive GPU session
- **`check_job_status.sh`** - Monitor jobs

## 🎯 Common Tasks

### Submit a Training Job

```bash
# With container
sbatch scripts/submit_job_container.sh

# With virtual environment
sbatch scripts/submit_job.sh
```

### Check Job Status

```bash
squeue -u $USER
# or
bash scripts/check_job_status.sh
```

### View Output Logs

```bash
tail -f experiments/logs/fno-*.out
```

### Cancel a Job

```bash
scancel <JOB_ID>
```

### Interactive Session

```bash
# With container
srun --partition=gpu --gpus=1 --mem=32G --time=04:00:00 --pty bash
module load apptainer
apptainer shell --nv ~/thesis-fno.sif

# With virtual environment
bash scripts/interactive_gpu.sh
```

## 🔧 Configuration

### Adjusting Job Resources

Edit the SBATCH parameters in submit scripts:

```bash
#SBATCH --time=24:00:00        # Max runtime
#SBATCH --gpus=1               # Number of GPUs
#SBATCH --cpus-per-task=8      # CPUs
#SBATCH --mem=32G              # Memory
```

### Modifying Training Config

Edit configuration files in `configs/`:
- `configs/training/default.yaml` - Training parameters
- `configs/models/*.yaml` - Model architectures
- `configs/datasets/*.yaml` - Dataset settings

## 📊 Monitoring and Results

### View Training Progress

```bash
# Real-time log
tail -f experiments/logs/fno-<JOB_ID>.out

# All recent logs
ls -lt experiments/logs/
```

### Download Results

```bash
# From your local machine
scp -r <netid>@login.daic.tudelft.nl:~/thesis-fno-constraints/experiments/runs/ ./experiments/
```

### Check GPU Usage

```bash
# In an interactive session
nvidia-smi
```

## 🆘 Troubleshooting

### Container not found
```bash
# Check if it exists
ls -lh ~/thesis-fno.sif

# Rebuild if needed
bash scripts/build_container.sh
bash scripts/transfer_container.sh <netid>
```

### Module not found
```bash
module load python/3.10 cuda/11.8 cudnn/8.6
source ~/venv/thesis-fno/bin/activate
```

### Out of memory
```bash
# Increase in submit script
#SBATCH --mem=64G
```

### Job pending
```bash
# Check why
squeue -u $USER

# Check partition availability
sinfo -p gpu
```

## 📖 Additional Resources

- **DAIC Documentation**: https://daic.tudelft.nl/
- **SLURM Commands**: https://slurm.schedmd.com/
- **Apptainer Docs**: https://apptainer.org/docs/
- **Support Email**: hpc-support@tudelft.nl

## 🏗️ Project Structure

```
thesis-fno-constraints/
├── scripts/               # All executable scripts
│   ├── build_container.sh
│   ├── submit_job*.sh
│   └── train.py
├── configs/              # Configuration files
├── src/thesis_fno/       # Source code
├── experiments/          # Training outputs
│   ├── logs/            # Job logs
│   └── runs/            # Training results
├── data/                # Datasets
├── Dockerfile           # Docker image definition
├── apptainer.def        # Apptainer image definition
└── DAIC_README.md       # This file
```

## ✅ Workflow Checklist

### First Time Setup

- [ ] Clone repository on DAIC
- [ ] Choose approach (container or venv)
- [ ] Build/setup environment
- [ ] Test with small job
- [ ] Review configurations

### Regular Workflow

- [ ] Update code: `git pull`
- [ ] Modify configs if needed
- [ ] Submit job: `sbatch scripts/submit_job*.sh`
- [ ] Monitor: `squeue -u $USER`
- [ ] Check logs: `tail -f experiments/logs/*.out`
- [ ] Download results when complete

### Before Large Experiments

- [ ] Test on small dataset first
- [ ] Verify GPU availability: `sinfo -p gpu`
- [ ] Check storage quota: `quota -s`
- [ ] Backup important data
- [ ] Document experiment parameters

## 💡 Best Practices

1. **Use containers** for production runs (better reproducibility)
2. **Test locally first** before submitting to cluster
3. **Use array jobs** for hyperparameter sweeps
4. **Monitor resources** to optimize usage
5. **Save checkpoints** regularly to prevent data loss
6. **Document experiments** in `experiments/configs_used/`
7. **Use scratch storage** (`/scratch/$USER`) for large datasets
8. **Clean up** old results to manage quota

## 🎓 Learning Path

1. **Start here** → Read [CLUSTER_QUICK_REF.md](CLUSTER_QUICK_REF.md)
2. **Setup** → Follow Quick Start above
3. **Submit test job** → Use provided scripts
4. **Learn more** → Read [CLUSTER_USAGE.md](CLUSTER_USAGE.md)
5. **Use containers** → Read [CONTAINER_GUIDE.md](CONTAINER_GUIDE.md)
6. **Master it** → Experiment with different configurations

## 📞 Getting Help

1. **Check documentation** in this repository
2. **Check DAIC docs**: https://daic.tudelft.nl/
3. **Ask DAIC support**: hpc-support@tudelft.nl
4. **GitHub Issues**: For project-specific problems

---

**Ready to start?** Follow the Quick Start guide above! 🚀

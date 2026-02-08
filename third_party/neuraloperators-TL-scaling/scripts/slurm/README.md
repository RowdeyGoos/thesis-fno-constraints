# SLURM Scripts Overview

This directory contains SLURM job submission scripts for running experiments on the DAIC cluster.

## 📋 Available Scripts

### Pretraining Scripts

#### `submit_job_array_container.sh`
**Purpose**: Submit array job for pretraining all three PDE systems with DDP (2 GPUs per job)

**Usage**:
```bash
sbatch scripts/slurm/submit_job_array_container.sh
```

**Details**:
- Runs 3 array tasks (0-2): Poisson, Advection-Diffusion, Helmholtz
- Each task uses 2 A40 GPUs with Distributed Data Parallel training
- Partition: `insy,general`
- Time limit: 24 hours
- Memory: 32GB

**Configs used**:
- Task 0: `poisson-scale-k1_5`
- Task 1: `ad-scale-adr0p2_1`
- Task 2: `helm-scale-o1_10`

---

#### `submit_job_array_container_single_gpu.sh`
**Purpose**: Submit array job for pretraining all three PDE systems (1 GPU per job)

**Usage**:
```bash
sbatch scripts/slurm/submit_job_array_container_single_gpu.sh
```

**Details**:
- Runs 3 array tasks (0-2): Poisson, Advection-Diffusion, Helmholtz
- Each task uses 1 A40 GPU (no DDP)
- Partition: `insy,general`
- Time limit: 24 hours
- Memory: 32GB

**Use when**:
- GPU resources are limited
- Testing configurations
- DDP setup issues

---

#### `submit_job_container.sh`
**Purpose**: Submit a single pretraining job with DDP (2 GPUs)

**Usage**:
```bash
# Edit the script to set CONFIG_NAME and RUN_BASE, then:
sbatch scripts/slurm/submit_job_container.sh
```

**Details**:
- Uses 2 A40 GPUs with DDP
- Currently configured for Poisson k1_5
- Partition: `insy,general`
- Time limit: 2 hours (QOS: short)
- Memory: 32GB

**Use when**:
- Running a single pretraining experiment
- Testing DDP setup
- Need more control over single job

---

### Transfer Learning Scripts

#### `submit_transfer_learning_array.sh`
**Purpose**: Submit array job for transfer learning experiments (fine-tuning vs from-scratch comparison)

**Usage**:
```bash
# 1. First, update checkpoint path with your pretraining job ID:
bash scripts/utils/update_checkpoint_path.sh <PRETRAIN_JOBID>

# 2. Then submit the array job:
sbatch scripts/slurm/submit_transfer_learning_array.sh
```

**Details**:
- Runs 10 array tasks (0-9)
  - Tasks 0-4: Fine-tuning with pre-trained weights (16, 64, 256, 1k, 4k samples)
  - Tasks 5-9: Training from scratch (16, 64, 256, 1k, 4k samples)
- Each task uses 1 A40 GPU
- Partition: `insy,general`
- Time limit: 24 hours
- Memory: 32GB

**Configs used**:
- Task 0: `poisson-k5_10-finetune-16`
- Task 1: `poisson-k5_10-finetune-64`
- Task 2: `poisson-k5_10-finetune-256`
- Task 3: `poisson-k5_10-finetune-1k`
- Task 4: `poisson-k5_10-finetune-4k`
- Task 5: `poisson-k5_10-scratch-16`
- Task 6: `poisson-k5_10-scratch-64`
- Task 7: `poisson-k5_10-scratch-256`
- Task 8: `poisson-k5_10-scratch-1k`
- Task 9: `poisson-k5_10-scratch-4k`

**Prerequisites**:
1. Completed pretraining on Poisson k1_5
2. Generated data for Poisson k5_10 domain
3. Computed scales for k5_10 data
4. Updated checkpoint path using `update_checkpoint_path.sh`

---

## 🚀 Quick Start Workflow

### Full Reproduction Pipeline

```bash
# 1. Generate all datasets (run on login node or interactive session)
cd third_party/neuraloperators-TL-scaling

# Generate Poisson k1_5 (source domain)
mkdir -p data/poisson
python utils/gen_data_poisson.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/poisson --e1 1 --e2 5
python utils/compute_scales.py --datapath data/poisson --filename _train_k1_5_32k.h5

# Generate Poisson k5_10 (target domain)
python utils/gen_data_poisson.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/poisson --e1 5 --e2 10
python utils/compute_scales.py --datapath data/poisson --filename _train_k5_10_32k.h5

# Generate Advection-Diffusion
mkdir -p data/advdiff
python utils/gen_data_advdiff.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/advdiff --adr1 0.2 --adr2 1.0
python utils/compute_scales.py --datapath data/advdiff --filename _train_adr0p2_1_32k.h5

# Generate Helmholtz
mkdir -p data/helmholtz
python utils/gen_data_helmholtz.py --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse --datapath data/helmholtz --o1 1 --o2 10
python utils/compute_scales.py --datapath data/helmholtz --filename _train_o1_10_32k.h5

# 2. Submit pretraining array job
sbatch scripts/slurm/submit_job_array_container_single_gpu.sh

# 3. Wait for pretraining to complete, note the job ID (e.g., 12345)
# Check status: squeue -u $USER

# 4. Update checkpoint path for transfer learning
bash scripts/utils/update_checkpoint_path.sh 12345

# 5. Submit transfer learning experiments
sbatch scripts/slurm/submit_transfer_learning_array.sh

# 6. Monitor results
# View W&B dashboard or check experiment logs
ls experiments/expts/
```

---

## 📊 Understanding Array Jobs

Array jobs submit multiple independent tasks with a single command. Each task has:
- **Array Job ID**: Shared across all tasks (e.g., `12345`)
- **Array Task ID**: Unique per task (e.g., `0`, `1`, `2`, ...)

**Example**:
```bash
sbatch --array=0-2 my_script.sh
```
Creates 3 jobs:
- Job `12345_0` (task 0)
- Job `12345_1` (task 1)
- Job `12345_2` (task 2)

**Benefits**:
- Submit many jobs at once
- Efficient resource allocation
- Easy to track related experiments

---

## 🔍 Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific array job
squeue -j <JOBID>

# View output logs
cat experiments/<job-name>-<jobid>-<taskid>.out

# View error logs
cat experiments/<job-name>-<jobid>-<taskid>.err

# Cancel job
scancel <JOBID>

# Cancel specific array task
scancel <JOBID>_<TASKID>

# Cancel all your jobs
scancel -u $USER
```

---

## 🛠️ Customizing Scripts

### Changing GPU Type
```bash
#SBATCH --gres=gpu:a40:2    # Use A40 GPUs
#SBATCH --gres=gpu:v100:2   # Use V100 GPUs
#SBATCH --gres=gpu:2        # Use any available GPUs
```

### Changing Partition
```bash
#SBATCH --partition=insy,general  # Try insy first, fallback to general
#SBATCH --partition=gpu           # GPU partition only
```

### Changing Time Limit
```bash
#SBATCH --time=2:00:00    # 2 hours
#SBATCH --time=24:00:00   # 24 hours
#SBATCH --time=3-00:00:00 # 3 days
```

### Changing Memory
```bash
#SBATCH --mem=32G   # 32 GB
#SBATCH --mem=64G   # 64 GB
#SBATCH --mem=128G  # 128 GB
```

---

## 📁 Output Files

Jobs produce several output files:

### Log Files
```
experiments/
├── neuralop-pretrain-array-<jobid>-<taskid>.out  # Standard output
├── neuralop-pretrain-array-<jobid>-<taskid>.err  # Standard error
```

### Result Files
```
experiments/expts/
├── <config-name>/
│   ├── <run-name>/
│   │   ├── checkpoints/
│   │   │   ├── ckpt_best.tar     # Best model checkpoint
│   │   │   └── ckpt_latest.tar   # Latest checkpoint
│   │   ├── logs/
│   │   │   └── train.log         # Training logs
│   │   └── figures/              # Plots (if enabled)
```

---

## ⚠️ Common Issues

### Container Not Found
```
Error: Container not found at /path/to/container
```
**Solution**: Update `CONTAINER_PATH` in the script to your actual container location.

### Checkpoint Not Found (Transfer Learning)
```
WARNING: Pre-trained checkpoint not found
```
**Solution**: Run `bash scripts/utils/update_checkpoint_path.sh <JOBID>` with your actual pretraining job ID.

### Out of Memory
```
slurmstepd: error: Detected 1 oom-kill event(s)
```
**Solution**: 
- Reduce batch size in config
- Request more memory: `#SBATCH --mem=64G`
- Use single GPU instead of DDP

### GPU Not Available
```
srun: error: Unable to allocate resources
```
**Solution**:
- Check partition availability: `sinfo`
- Try different partition: `--partition=general`
- Try different GPU: `--gres=gpu:1` (any type)

---

## 📚 Additional Resources

- [DAIC Documentation](https://daic.tudelft.nl/)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [Project README](../../README.md)
- [Reproduction Guide](../../REPRODUCTION_GUIDE.md)

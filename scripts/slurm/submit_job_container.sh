#!/bin/bash
#SBATCH --job-name=fno-container
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

# Job using Apptainer container
# Submit from project root: sbatch scripts/slurm/submit_job_container.sh

echo "=========================================="
echo "Starting FNO training job (Container)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=~/thesis-fno.sif

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please build and transfer the container first:"
    echo "  1. Locally: bash scripts/container/build_container.sh"
    echo "  2. Transfer: bash scripts/container/transfer_container.sh"
    exit 1
fi

# Load Apptainer module if needed
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

# Set environment variables
export PYTHONUNBUFFERED=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Bind directories
# This makes host directories available inside the container
BIND_DIRS="--bind $PWD/data:/workspace/data"
BIND_DIRS="$BIND_DIRS --bind $PWD/experiments:/workspace/experiments"
BIND_DIRS="$BIND_DIRS --bind $PWD/models:/workspace/models"

# Run training with container
# --nv enables NVIDIA GPU support
apptainer exec --nv $BIND_DIRS $CONTAINER_PATH \
    python scripts/train.py \
    --config configs/training/default.yaml \
    --model configs/models/fno_baseline.yaml \
    --dataset configs/datasets/pdebench.yaml \
    --output experiments/runs/$SLURM_JOB_ID

echo "=========================================="
echo "Job completed"
echo "=========================================="

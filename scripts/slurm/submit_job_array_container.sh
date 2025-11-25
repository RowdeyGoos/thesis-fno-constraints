#!/bin/bash
#SBATCH --job-name=fno-array-container
#SBATCH --output=experiments/logs/%x-%A-%a.out
#SBATCH --error=experiments/logs/%x-%A-%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --array=0-4

# Array job using Apptainer container
# Submit from project root: sbatch scripts/slurm/submit_job_array_container.sh

echo "=========================================="
echo "Starting FNO array job (Container)"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
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

# Load Apptainer module
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

# Set environment variables
export PYTHONUNBUFFERED=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Define experiment configurations
configs=(
    "configs/models/fno_baseline.yaml"
    "configs/models/fno_divfree.yaml"
    "configs/models/fno_conservation.yaml"
    "configs/models/fno_multi_constraint.yaml"
    "configs/models/fno_large.yaml"
)

# Get config for this task
config=${configs[$SLURM_ARRAY_TASK_ID]}

echo "Running with config: $config"

# Bind directories
BIND_DIRS="--bind $PWD/data:/workspace/data"
BIND_DIRS="$BIND_DIRS --bind $PWD/experiments:/workspace/experiments"
BIND_DIRS="$BIND_DIRS --bind $PWD/models:/workspace/models"

# Run training
apptainer exec --nv $BIND_DIRS $CONTAINER_PATH \
    python scripts/train.py \
    --config configs/training/default.yaml \
    --model $config \
    --dataset configs/datasets/pdebench.yaml \
    --output experiments/runs/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="

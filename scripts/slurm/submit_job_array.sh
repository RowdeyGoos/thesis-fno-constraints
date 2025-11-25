#!/bin/bash
#SBATCH --job-name=fno-array
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

# Array job for hyperparameter sweep or multiple experiments (venv)
# Submit from project root: sbatch scripts/slurm/submit_job_array.sh

echo "=========================================="
echo "Starting FNO array job"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module purge
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
# If not set up: bash scripts/setup/setup_environment.sh
source ~/venv/thesis-fno/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

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

# Run training script
python scripts/train.py \
    --config configs/training/default.yaml \
    --model $config \
    --dataset configs/datasets/pdebench.yaml \
    --output experiments/runs/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="

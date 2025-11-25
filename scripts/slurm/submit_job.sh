#!/bin/bash
#SBATCH --job-name=fno-training
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

# Job using virtual environment (for development/debugging)
# Submit from project root: sbatch scripts/slurm/submit_job.sh

echo "=========================================="
echo "Starting FNO training job"
echo "Job ID: $SLURM_JOB_ID"
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

# Run training script
python scripts/train.py \
    --config configs/training/default.yaml \
    --model configs/models/fno_baseline.yaml \
    --dataset configs/datasets/pdebench.yaml \
    --output experiments/runs/$SLURM_JOB_ID

echo "=========================================="
echo "Job completed"
echo "=========================================="

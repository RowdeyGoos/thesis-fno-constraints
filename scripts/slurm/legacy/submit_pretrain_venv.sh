#!/bin/bash
#SBATCH --job-name=neuralop-train-venv
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=8G

# SLURM job using virtual environment (for development/debugging)
# Submit from project root: sbatch scripts/slurm/legacy/submit_pretrain_venv.sh

echo "=========================================="
echo "Starting neuraloperators-TL-scaling Training (venv)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module purge
module load python/3.10
module load cuda/11.3
module load cudnn/8.2

# Activate virtual environment
# If not set up: bash scripts/setup/setup_environment.sh
source ~/venv/neuraloperators/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Configuration file (modify as needed)
CONFIG_FILE="config/operators_poisson.yaml"
RUN_NAME="poisson-scale-k1_5"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo ""

# Run training
echo "Starting training..."
python train.py \
    --yaml_config "$CONFIG_FILE" \
    --config "$RUN_NAME" \
    --run_num "${RUN_NAME}-${SLURM_JOB_ID}"

echo ""
echo "=========================================="
echo "Training completed"
echo "Results saved to: experiments/${RUN_NAME}-${SLURM_JOB_ID}/"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=neuralop-train
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

# SLURM job for neuraloperators-TL-scaling training using Apptainer container
# Submit from project root: sbatch scripts/slurm/submit_job_container.sh

echo "=========================================="
echo "Starting neuraloperators-TL-scaling Training (Container)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=~/neuraloperators.sif

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please build and transfer the container first:"
    echo "  1. Locally: bash scripts/container/build_container.sh"
    echo "  2. Transfer: bash scripts/container/transfer_container.sh <netid>"
    exit 1
fi

# Load Apptainer module
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

# Set environment variables
export PYTHONUNBUFFERED=1

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

# Bind directories
BIND_DIRS="--bind $PWD/data:/workspace/data"
BIND_DIRS="$BIND_DIRS --bind $PWD/experiments:/workspace/experiments"
BIND_DIRS="$BIND_DIRS --bind $PWD/config:/workspace/config"

# Run training
echo "Starting training..."
apptainer exec --nv $BIND_DIRS $CONTAINER_PATH \
    python /workspace/train.py \
    --yaml_config "$CONFIG_FILE" \
    --config "$RUN_NAME" \
    --run_num "${RUN_NAME}-${SLURM_JOB_ID}"

echo ""
echo "=========================================="
echo "Training completed"
echo "Results saved to: experiments/${RUN_NAME}-${SLURM_JOB_ID}/"
echo "=========================================="

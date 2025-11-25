#!/bin/bash
#SBATCH --job-name=neuralop-array
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --array=0-2

# Array job for multiple PDE systems using Apptainer container
# Submit from project root: sbatch scripts/slurm/submit_job_array_container.sh

echo "=========================================="
echo "Starting neuraloperators-TL-scaling Array Job (Container)"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
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

# Define configurations for different PDEs
configs=(
    "config/operators_poisson.yaml:poisson-scale-k1_5"
    "config/operators_ad.yaml:ad-scale_adr0p2_1"
    "config/operators_helmholtz.yaml:helm-scale-o1_10"
)

# Get config for this task
IFS=':' read -r config_file run_name <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $config_file"
echo "Run name: $run_name"
echo ""

# Check if config exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found"
    exit 1
fi

# Bind directories
BIND_DIRS="--bind $PWD/data:/workspace/data"
BIND_DIRS="$BIND_DIRS --bind $PWD/experiments:/workspace/experiments"
BIND_DIRS="$BIND_DIRS --bind $PWD/config:/workspace/config"

# Run training
echo "Starting training..."
apptainer exec --nv $BIND_DIRS $CONTAINER_PATH \
    python /workspace/train.py \
    --yaml_config "$config_file" \
    --config "$run_name" \
    --run_num "${run_name}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="

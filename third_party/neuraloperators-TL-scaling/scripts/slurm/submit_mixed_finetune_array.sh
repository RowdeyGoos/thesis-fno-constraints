#!/bin/bash
#SBATCH --job-name=neuralop-mixed-finetune
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --array=0-4

# Mixed Dataset Fine-Tuning Array Job
# This script fine-tunes the mixed-pretrained model on Poisson k∈[5,10] domain
# with different numbers of downstream examples: 16, 64, 256, 1k, 4k
#
# Prerequisites:
#   1. Completed mixed dataset pretraining (submit_mixed_pretrain.sh)
#   2. Updated checkpoint paths using update_checkpoint_path.sh
#
# Usage:
#   # Update checkpoint path after pretraining
#   bash scripts/utils/update_checkpoint_path.sh <mixed_pretrain_job_id>
#   
#   # Submit fine-tuning jobs
#   sbatch scripts/slurm/submit_mixed_finetune_array.sh

echo "=========================================="
echo "Mixed Dataset Fine-Tuning (Task $SLURM_ARRAY_TASK_ID)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Load Apptainer module
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

# -------- W&B config --------
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp

cd "$SLURM_SUBMIT_DIR"

# Configuration file
CONFIG_FILE="config/operators_mixed.yaml"

# Array of configurations for mixed fine-tuning
declare -a configs=(
    "poisson-k5_10-finetune-mixed-16:finetune-mixed-16"
    "poisson-k5_10-finetune-mixed-64:finetune-mixed-64"
    "poisson-k5_10-finetune-mixed-256:finetune-mixed-256"
    "poisson-k5_10-finetune-mixed-1k:finetune-mixed-1k"
    "poisson-k5_10-finetune-mixed-4k:finetune-mixed-4k"
)

# Get current task configuration
IFS=':' read -r CONFIG_NAME RUN_NAME <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Run name: $RUN_NAME"
echo ""

# Verify checkpoint path is set
CHECKPOINT_LINE=$(grep -A 20 "^$CONFIG_NAME:" "$CONFIG_FILE" | grep "weights:" | head -n 1)
if [[ "$CHECKPOINT_LINE" == *"JOBID"* ]]; then
    echo "ERROR: Checkpoint path contains placeholder 'JOBID'!"
    echo "Please run: bash scripts/utils/update_checkpoint_path.sh <mixed_pretrain_job_id>"
    echo "Found line: $CHECKPOINT_LINE"
    exit 1
fi

echo "Checkpoint line: $CHECKPOINT_LINE"
echo ""

# Create directories
mkdir -p experiments

# Bind directories
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command
CMD="python /workspace/train.py \
    --yaml_config=/workspace/$CONFIG_FILE \
    --config=$CONFIG_NAME \
    --run_num=${RUN_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments"

echo "Running mixed fine-tuning (Task $SLURM_ARRAY_TASK_ID)..."
echo "Command: $CMD"
echo ""

# Run training
apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             mkdir -p wandb wandb/cache wandb/tmp tmp experiments && \
             export TMPDIR=/workspace/tmp && \
             '"$CMD"

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID ($CONFIG_NAME) completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID ($CONFIG_NAME) FAILED with exit code $status."
fi
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results: experiments/expts/$CONFIG_NAME/"
echo "=========================================="

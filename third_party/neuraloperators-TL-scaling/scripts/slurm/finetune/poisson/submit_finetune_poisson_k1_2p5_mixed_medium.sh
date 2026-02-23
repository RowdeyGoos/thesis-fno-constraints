#!/bin/bash
#SBATCH --job-name=neuralop-mixed-medium
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=6:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0

# Mixed Dataset Fine-Tuning: Medium Sample Sizes (4k, 8k samples)
# This script fine-tunes the mixed-pretrained model on Poisson k∈[1,2.5] domain
# with medium numbers of downstream examples: 4k, 8k
#
# Time allocation: 1 hour (sufficient for medium sample training)
#
# Prerequisites:
#   1. Completed mixed dataset pretraining (submit_mixed_pretrain.sh)
#   2. Updated checkpoint paths in config/operators_poisson.yaml
#
# Usage:
#   sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_medium.sh

echo "=========================================="
echo "Mixed Dataset Fine-Tuning - Medium Samples (Task $SLURM_ARRAY_TASK_ID)"
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

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="$SLURM_SUBMIT_DIR/$JOB_TMP_REL"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "$SLURM_SUBMIT_DIR/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT


# Configuration file
CONFIG_FILE="config/operators_poisson.yaml"

# Array of configurations for mixed fine-tuning
declare -a configs=(
    # "poisson-k1_2.5-finetune-mixed-4k:finetune-mixed-4k"
    "poisson-k1_2.5-finetune-mixed-8k:finetune-mixed-8k"
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
    echo "Please update the checkpoint path in $CONFIG_FILE"
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
             job_tmp="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" && \
             export TMPDIR="/workspace/${job_tmp}" && \
             mkdir -p wandb wandb/cache wandb/tmp tmp experiments "$TMPDIR" && \
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

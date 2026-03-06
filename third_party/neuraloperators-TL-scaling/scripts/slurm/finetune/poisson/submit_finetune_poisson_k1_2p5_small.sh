#!/bin/bash
#SBATCH --job-name=neuralop-poisson-k1_2p5-small
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=4:00:00
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G
#SBATCH --array=0-7

# Transfer Learning: Poisson k∈[1,2.5] - Small Sample Sizes (16, 64, 256, 1k samples)
# This script runs 8 experiments:
#   - 4 fine-tuning experiments (16, 64, 256, 1k samples) with pre-trained weights from k1_5
#   - 4 from-scratch experiments (16, 64, 256, 1k samples) without pre-training
# 
# Time allocation: 30 minutes (sufficient for small sample training)
#
# Prerequisites:
#   1. Pre-trained model checkpoint from poisson-scale-k1_5 pretraining
#   2. Generated data for poisson k1_2.5 domain (3-component tensor format)
#   3. Computed scales for k1_2.5 data
#
# Usage:
#   Before submitting, update PRETRAIN_CHECKPOINT with your actual pretrain job ID
#   sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_small.sh

echo "=========================================="
echo "Transfer Learning Experiment (Poisson k∈[1,2.5] - Small Samples)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Guard against running without an array task context.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit this script with sbatch so the array directive (#SBATCH --array=0-7) is applied."
    exit 1
fi

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


# -------- UPDATE THIS: Path to pre-trained checkpoint --------
# Replace JOBID with your actual poisson-scale-k1_5 pretraining job ID
PRETRAIN_CHECKPOINT="experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-12147812-0/checkpoints/ckpt_best.tar"

# Verify checkpoint exists for fine-tuning tasks
if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
        echo "WARNING: Pre-trained checkpoint not found at: $PRETRAIN_CHECKPOINT"
        echo "Please update PRETRAIN_CHECKPOINT variable in this script with the correct path"
        echo "Continuing anyway - training will start from scratch if weights file not found"
    else
        echo "✓ Pre-trained checkpoint found: $PRETRAIN_CHECKPOINT"
    fi
fi

# Define experiments
# Format: "config_name:experiment_description"
experiments=(
    # Fine-tuning experiments from k1_5 pretrained (tasks 0-3)
    "poisson-k1_2.5-finetune-16:finetune-k1_5-16-samples"
    "poisson-k1_2.5-finetune-64:finetune-k1_5-64-samples"
    "poisson-k1_2.5-finetune-256:finetune-k1_5-256-samples"
    "poisson-k1_2.5-finetune-1k:finetune-k1_5-1k-samples"
    # From-scratch experiments (tasks 4-7)
    "poisson-k1_2.5-scratch-16:scratch-16-samples"
    "poisson-k1_2.5-scratch-64:scratch-64-samples"
    "poisson-k1_2.5-scratch-256:scratch-256-samples"
    "poisson-k1_2.5-scratch-1k:scratch-1k-samples"
)

# Validate task index before reading the experiment entry.
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#experiments[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range (0-$((${#experiments[@]} - 1)))."
    exit 1
fi

# Get experiment for this task
IFS=':' read -r config_name exp_desc <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $config_name"
echo "Experiment: $exp_desc"
echo ""

# Determine if this is fine-tuning or from-scratch
if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    exp_type="finetune"
    echo "Type: Fine-tuning with k1_5 pre-trained weights"
else
    exp_type="scratch"
    echo "Type: Training from scratch"
fi

# Create directories
mkdir -p experiments

# Bind directories
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command
CMD="python /workspace/train.py \
    --yaml_config=/workspace/config/operators_poisson.yaml \
    --config=$config_name \
    --run_num=transfer-${exp_desc}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments"

echo "Running training..."
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
    echo "Task $SLURM_ARRAY_TASK_ID ($exp_desc) completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID ($exp_desc) FAILED with exit code $status."
fi
echo "Type: $exp_type"
echo "Config: $config_name"
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results: experiments/expts/$config_name/"
echo "=========================================="

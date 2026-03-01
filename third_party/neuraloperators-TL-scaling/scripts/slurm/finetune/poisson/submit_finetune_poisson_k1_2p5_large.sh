#!/bin/bash
#SBATCH --job-name=neuralop-poisson-k1_2p5-large
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=25:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0-3

# Transfer Learning: Poisson k∈[1,2.5] - Large Sample Sizes (16k, 32k samples)
# This script runs 4 experiments:
#   - 2 fine-tuning experiments (16k, 32k samples) with pre-trained weights from k1_5
#   - 2 from-scratch experiments (16k, 32k samples) without pre-training
# 
# Time allocation: 2 hours (sufficient for large sample training)
#
# Prerequisites:
#   1. Pre-trained model checkpoint from poisson-scale-k1_5 pretraining
#   2. Generated data for poisson k1_2.5 domain (3-component tensor format)
#   3. Computed scales for k1_2.5 data
#
# Usage:
#   Before submitting, update PRETRAIN_CHECKPOINT with your actual pretrain job ID
#   sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_large.sh

echo "=========================================="
echo "Transfer Learning Experiment (Poisson k∈[1,2.5] - Large Samples)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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


# -------- UPDATE THIS: Path to pre-trained checkpoint --------
# Replace JOBID with your actual poisson-scale-k1_5 pretraining job ID
PRETRAIN_CHECKPOINT="experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-12147812-0/checkpoints/ckpt_best.tar"

# Verify checkpoint exists for fine-tuning tasks
if [ $SLURM_ARRAY_TASK_ID -lt 2 ]; then
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
    # Fine-tuning experiments from k1_5 pretrained (tasks 0-1)
    "poisson-k1_2.5-finetune-16k:finetune-k1_5-16k-samples"
    "poisson-k1_2.5-finetune-32k:finetune-k1_5-32k-samples"
    # From-scratch experiments (tasks 2-3)
    "poisson-k1_2.5-scratch-16k:scratch-16k-samples"
    "poisson-k1_2.5-scratch-32k:scratch-32k-samples"
)

# Get experiment for this task
IFS=':' read -r config_name exp_desc <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $config_name"
echo "Experiment: $exp_desc"
echo ""

# Determine if this is fine-tuning or from-scratch
if [ $SLURM_ARRAY_TASK_ID -lt 2 ]; then
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

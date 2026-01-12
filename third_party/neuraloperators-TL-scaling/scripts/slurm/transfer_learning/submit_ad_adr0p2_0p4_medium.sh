#!/bin/bash
#SBATCH --job-name=neuralop-ad-0p2_0p4-medium
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=1:00:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --array=0-3

# Transfer Learning: AdvDiff adr∈[0.2,0.4] - Medium Sample Sizes (4k, 8k samples)
# This script runs 4 experiments:
#   - 2 fine-tuning experiments (4k, 8k samples) with pre-trained weights from adr0.2_1
#   - 2 from-scratch experiments (4k, 8k samples) without pre-training
# 
# Time allocation: 1 hour (sufficient for medium sample training)
#
# Prerequisites:
#   1. Pre-trained model checkpoint from ad-scale-adr0p2_1 pretraining
#   2. Generated data for AdvDiff adr∈[0.2,0.4] domain
#   3. Computed scales for adr∈[0.2,0.4] data
#
# Usage:
#   Before submitting, update PRETRAIN_CHECKPOINT with your actual pretrain job ID
#   sbatch scripts/slurm/transfer_learning/submit_ad_adr0p2_0p4_medium.sh

echo "=========================================="
echo "Transfer Learning Experiment (AdvDiff adr∈[0.2,0.4] - Medium Samples)"
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

# -------- UPDATE THIS: Path to pre-trained checkpoint --------
# Replace JOBID with your actual ad-scale-adr0p2_1 pretraining job ID
PRETRAIN_CHECKPOINT="experiments/expts/ad-scale-adr0p2_1/pretrain-ad-adr0p2_1-12147812-1/checkpoints/ckpt_best.tar"

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
    # Fine-tuning experiments from adr0.2_1 pretrained (tasks 0-1)
    "ad-adr0p2_0p4-finetune-4k:finetune-adr0p2_1-4k-samples"
    "ad-adr0p2_0p4-finetune-8k:finetune-adr0p2_1-8k-samples"
    # From-scratch experiments (tasks 2-3)
    "ad-adr0p2_0p4-scratch-4k:scratch-4k-samples"
    "ad-adr0p2_0p4-scratch-8k:scratch-8k-samples"
)

# Get experiment for this task
IFS=':' read -r config_name exp_desc <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $config_name"
echo "Experiment: $exp_desc"
echo ""

# Determine if this is fine-tuning or from-scratch
if [ $SLURM_ARRAY_TASK_ID -lt 2 ]; then
    exp_type="finetune"
    echo "Type: Fine-tuning with adr0.2_1 pre-trained weights"
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
    --yaml_config=/workspace/config/operators_ad.yaml \
    --config=$config_name \
    --run_num=transfer-${exp_desc}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments"

echo "Running training..."
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
    echo "Task $SLURM_ARRAY_TASK_ID ($exp_desc) completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID ($exp_desc) FAILED with exit code $status."
fi
echo "Type: $exp_type"
echo "Config: $config_name"
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results: experiments/expts/$config_name/"
echo "=========================================="

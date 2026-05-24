#!/bin/bash
#SBATCH --job-name=neuralop-helm-o1_5-small
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
#SBATCH --array=0-23

# Transfer Learning: Helmholtz o∈[1,5] - Small Sample Sizes (16, 64, 256, 1k samples)
# This script runs 8 experiments:
#   - 4 fine-tuning experiments (16, 64, 256, 1k samples) with pre-trained weights from o1_10
#   - 4 from-scratch experiments (16, 64, 256, 1k samples) without pre-training
#
# Usage:
#   Before submitting, update PRETRAIN_CHECKPOINT with your actual pretrain job ID
#   sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_small.sh

echo "=========================================="
echo "Transfer Learning Experiment (Helmholtz o∈[1,5] - Small Samples)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

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

source scripts/slurm/finetune/seed_grid.sh

# -------- UPDATE THIS: Path to pre-trained checkpoint --------
# Replace JOBID with your actual helm-scale-o1_10 pretraining job ID
PRETRAIN_CHECKPOINT="experiments/expts/helm-scale-o1_10/pretrain-helmholtz-o1_10-12147812-2/checkpoints/ckpt_best.tar"

if [ $SEED_EXPERIMENT_IDX -lt 4 ]; then
    if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
        echo "WARNING: Pre-trained checkpoint not found at: $PRETRAIN_CHECKPOINT"
        echo "Please update PRETRAIN_CHECKPOINT variable in this script with the correct path"
        echo "Continuing anyway - training will start from scratch if weights file not found"
    else
        echo "✓ Pre-trained checkpoint found: $PRETRAIN_CHECKPOINT"
    fi
fi

experiments=(
    "helm-o1_5-finetune-16:finetune-o1_10-16-samples"
    "helm-o1_5-finetune-64:finetune-o1_10-64-samples"
    "helm-o1_5-finetune-256:finetune-o1_10-256-samples"
    "helm-o1_5-finetune-1k:finetune-o1_10-1k-samples"
    "helm-o1_5-scratch-16:scratch-16-samples"
    "helm-o1_5-scratch-64:scratch-64-samples"
    "helm-o1_5-scratch-256:scratch-256-samples"
    "helm-o1_5-scratch-1k:scratch-1k-samples"
)

IFS=':' read -r config_name exp_desc <<< "${experiments[$SEED_EXPERIMENT_IDX]}"

echo "Configuration: $config_name"
echo "Experiment: $exp_desc"
echo "Seed: $SEED_VALUE"
echo ""

if [ $SEED_EXPERIMENT_IDX -lt 4 ]; then
    exp_type="finetune"
    echo "Type: Fine-tuning with o1_10 pre-trained weights"
else
    exp_type="scratch"
    echo "Type: Training from scratch"
fi

mkdir -p experiments

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

CMD="python /workspace/train.py \
    --yaml_config=/workspace/config/operators_helmholtz.yaml \
    --config=$config_name \
    --run_num=transfer-${exp_desc}-${SEED_RUN_SUFFIX}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${SEED_TRAIN_ARGS}"

echo "Running training..."
echo "Command: $CMD"
echo ""

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

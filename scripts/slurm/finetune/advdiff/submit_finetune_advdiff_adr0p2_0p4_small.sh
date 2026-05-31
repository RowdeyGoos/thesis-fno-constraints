#!/bin/bash
#SBATCH --job-name=neuralop-ad-0p2_0p4-small
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
#SBATCH --array=0-0

# Transfer Learning: AdvDiff adr∈[0.2,0.4] - Failed small-sample reruns only
# This script reruns the crashed tasks for:
#   - ad-adr0p2_0p4-finetune-1k (seed 1)

echo "=========================================="
echo "Transfer Learning Experiment (AdvDiff adr∈[0.2,0.4] - Small Samples)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif

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
# Replace JOBID with your actual ad-scale-adr0p2_1 pretraining job ID
PRETRAIN_CHECKPOINT="experiments/expts/ad-scale-adr0p2_1/pretrain-ad-adr0p2_1-12147812-1/checkpoints/ckpt_best.tar"

# Format: "config_name:experiment_description:seed:experiment_type"
failed_tasks=(
    "ad-adr0p2_0p4-finetune-1k:finetune-adr0p2_1-1k-samples:1:finetune"
)

if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#failed_tasks[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of range (0-$((${#failed_tasks[@]} - 1)))."
    exit 1
fi

IFS=':' read -r config_name exp_desc seed_value exp_type <<< "${failed_tasks[$SLURM_ARRAY_TASK_ID]}"
seed_run_suffix="seed${seed_value}"
seed_train_args="--seed=${seed_value} --train_shuffle --random_train_subset --subset_seed=${seed_value}"

if [ "$exp_type" = "finetune" ]; then
    if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
        echo "WARNING: Pre-trained checkpoint not found at: $PRETRAIN_CHECKPOINT"
        echo "Please update PRETRAIN_CHECKPOINT variable in this script with the correct path"
        echo "Continuing anyway - training will start from scratch if weights file not found"
    else
        echo "✓ Pre-trained checkpoint found: $PRETRAIN_CHECKPOINT"
    fi
fi

echo "Configuration: $config_name"
echo "Experiment: $exp_desc"
echo "Seed: $seed_value"
echo ""

if [ "$exp_type" = "finetune" ]; then
    echo "Type: Fine-tuning with adr0.2_1 pre-trained weights"
else
    echo "Type: Training from scratch"
fi

# Create directories
mkdir -p experiments

# Bind directories
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command
CMD="python /workspace/scripts/entrypoints/train.py \
    --yaml_config=/workspace/config/operators_ad.yaml \
    --config=$config_name \
    --run_num=transfer-${exp_desc}-${seed_run_suffix}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${seed_train_args}"

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

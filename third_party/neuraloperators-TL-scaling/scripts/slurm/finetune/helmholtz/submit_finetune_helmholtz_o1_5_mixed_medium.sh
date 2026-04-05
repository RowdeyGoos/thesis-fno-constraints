#!/bin/bash
#SBATCH --job-name=neuralop-helm-mixed-medium
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=7:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G
#SBATCH --array=0-1

# Mixed Dataset Fine-Tuning (Helmholtz): Medium Sample Sizes (reruns only)
# This rerun slice is intentionally narrowed to the still-missing Helmholtz
# mixed-zero-soft medium-sample transfer runs.
# Usage:
#   sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_medium.sh

echo "=========================================="
echo "Mixed Dataset Fine-Tuning (Helmholtz) - Medium Samples (Task $SLURM_ARRAY_TASK_ID)"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

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

MIXED_VARIANT="${MIXED_VARIANT:-mixed-zero-soft}"
RUN_VARIANT="${RUN_VARIANT:-$MIXED_VARIANT}"
CONFIG_FILE="${CONFIG_FILE:-config/operators_helmholtz.yaml}"

if [ "$MIXED_VARIANT" != "mixed-zero-soft" ]; then
    echo "Error: this rerun script is currently narrowed to MIXED_VARIANT=mixed-zero-soft."
    echo "Found: $MIXED_VARIANT"
    exit 1
fi

declare -a task_specs=(
    "helm-o1_5-finetune-${MIXED_VARIANT}-4k:finetune-${RUN_VARIANT}-4k:2"
    "helm-o1_5-finetune-${MIXED_VARIANT}-8k:finetune-${RUN_VARIANT}-8k:0"
)

if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#task_specs[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range for ${#task_specs[@]} rerun tasks."
    exit 1
fi

IFS=':' read -r CONFIG_NAME RUN_NAME SEED_VALUE <<< "${task_specs[$SLURM_ARRAY_TASK_ID]}"
SEED_RUN_SUFFIX="seed${SEED_VALUE}"
SEED_TRAIN_ARGS="--seed=${SEED_VALUE} --train_shuffle --random_train_subset --subset_seed=${SEED_VALUE}"

echo "Configuration: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Run name: $RUN_NAME"
echo "Seed: $SEED_VALUE"
echo ""

CHECKPOINT_LINE=$(grep -A 20 "^$CONFIG_NAME:" "$CONFIG_FILE" | grep "weights:" | head -n 1)
if [[ "$CHECKPOINT_LINE" == *"JOBID"* ]]; then
    echo "ERROR: Checkpoint path contains placeholder 'JOBID'!"
    echo "Please update the checkpoint path in $CONFIG_FILE"
    echo "Found line: $CHECKPOINT_LINE"
    exit 1
fi

echo "Checkpoint line: $CHECKPOINT_LINE"
echo ""

mkdir -p experiments

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

CMD="python /workspace/train.py \
    --yaml_config=/workspace/$CONFIG_FILE \
    --config=$CONFIG_NAME \
    --run_num=${RUN_NAME}-${SEED_RUN_SUFFIX}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${SEED_TRAIN_ARGS}"

echo "Running mixed fine-tuning (Task $SLURM_ARRAY_TASK_ID)..."
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
    echo "Task $SLURM_ARRAY_TASK_ID ($CONFIG_NAME) completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID ($CONFIG_NAME) FAILED with exit code $status."
fi
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results: experiments/expts/$CONFIG_NAME/"
echo "=========================================="

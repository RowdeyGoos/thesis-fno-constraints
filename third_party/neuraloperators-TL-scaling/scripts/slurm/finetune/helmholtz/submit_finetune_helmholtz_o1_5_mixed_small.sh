#!/bin/bash
#SBATCH --job-name=neuralop-helm-mixed-small
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=4:00:00
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G
#SBATCH --array=0-3

# Mixed Dataset Fine-Tuning (Helmholtz): Failed small-sample reruns only
# This script reruns the crashed tasks for:
#   - helm-o1_5-finetune-mixed-64 (seed 1)
#   - helm-o1_5-finetune-mixed-256 (seed 1)
#   - helm-o1_5-finetune-mixed-1k (seeds 0, 1)

echo "=========================================="
echo "Mixed Dataset Fine-Tuning (Helmholtz) - Small Samples (Task $SLURM_ARRAY_TASK_ID)"
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

MIXED_VARIANT="${MIXED_VARIANT:-mixed}"
RUN_VARIANT="${RUN_VARIANT:-$MIXED_VARIANT}"
CONFIG_FILE="${CONFIG_FILE:-config/operators_helmholtz.yaml}"

# Format: "config_name:run_name:seed"
failed_tasks=(
    "helm-o1_5-finetune-${MIXED_VARIANT}-64:finetune-${RUN_VARIANT}-64:1"
    "helm-o1_5-finetune-${MIXED_VARIANT}-256:finetune-${RUN_VARIANT}-256:1"
    "helm-o1_5-finetune-${MIXED_VARIANT}-1k:finetune-${RUN_VARIANT}-1k:0"
    "helm-o1_5-finetune-${MIXED_VARIANT}-1k:finetune-${RUN_VARIANT}-1k:1"
)

if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#failed_tasks[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of range (0-$((${#failed_tasks[@]} - 1)))."
    exit 1
fi

IFS=':' read -r CONFIG_NAME RUN_NAME seed_value <<< "${failed_tasks[$SLURM_ARRAY_TASK_ID]}"
seed_run_suffix="seed${seed_value}"
seed_train_args="--seed=${seed_value} --train_shuffle --random_train_subset --subset_seed=${seed_value}"

echo "Configuration: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Run name: $RUN_NAME"
echo "Seed: $seed_value"
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
    --run_num=${RUN_NAME}-${seed_run_suffix}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${seed_train_args}"

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

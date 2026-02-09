#!/bin/bash
#SBATCH --job-name=neuralop-helm-mixed-large
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=25:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0-1

# Mixed Dataset Fine-Tuning (Helmholtz): Large Sample Sizes (16k, 32k samples)
# Usage:
#   sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_large.sh

echo "=========================================="
echo "Mixed Dataset Fine-Tuning (Helmholtz) - Large Samples (Task $SLURM_ARRAY_TASK_ID)"
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

CONFIG_FILE="config/operators_helmholtz.yaml"

declare -a configs=(
    "helm-o1_5-finetune-mixed-16k:finetune-mixed-16k"
    "helm-o1_5-finetune-mixed-32k:finetune-mixed-32k"
)

IFS=':' read -r CONFIG_NAME RUN_NAME <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Run name: $RUN_NAME"
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
    --run_num=${RUN_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments"

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

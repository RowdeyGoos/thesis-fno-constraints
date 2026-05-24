#!/bin/bash
#SBATCH --job-name=neuralop-ad-ood-scratch
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
#SBATCH --array=0-23

echo "=========================================="
echo "OOD Scratch Training (AdvDiff) - 256 and 4k samples"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1
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
    rm -rf "${JOB_TMP_DIR}"
    rmdir "$SLURM_SUBMIT_DIR/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

CONFIG_FILE="${CONFIG_FILE:-config/operators_ad.yaml}"

source scripts/slurm/finetune/seed_grid.sh

declare -a configs=(
    "ad-adr1_1p2-scratch-256:scratch-adr1_1p2-256"
    "ad-adr1_1p2-scratch-4k:scratch-adr1_1p2-4k"
    "ad-adr1p2_1p4-scratch-256:scratch-adr1p2_1p4-256"
    "ad-adr1p2_1p4-scratch-4k:scratch-adr1p2_1p4-4k"
    "ad-adr1p4_1p6-scratch-256:scratch-adr1p4_1p6-256"
    "ad-adr1p4_1p6-scratch-4k:scratch-adr1p4_1p6-4k"
    "ad-adr1p6_1p8-scratch-256:scratch-adr1p6_1p8-256"
    "ad-adr1p6_1p8-scratch-4k:scratch-adr1p6_1p8-4k"
)

IFS=':' read -r CONFIG_NAME RUN_NAME <<< "${configs[$SEED_EXPERIMENT_IDX]}"

echo "Configuration: ${CONFIG_FILE}"
echo "Config name: ${CONFIG_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Seed: ${SEED_VALUE}"
echo "Type: Training from scratch"
echo ""

mkdir -p experiments

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"
CMD="python /workspace/scripts/entrypoints/train.py \
    --yaml_config=/workspace/${CONFIG_FILE} \
    --config=${CONFIG_NAME} \
    --run_num=${RUN_NAME}-${SEED_RUN_SUFFIX}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${SEED_TRAIN_ARGS}"

echo "Running OOD scratch training (Task ${SLURM_ARRAY_TASK_ID})..."
echo "Command: ${CMD}"
echo ""

apptainer exec --nv ${BIND} "${CONTAINER_PATH}" \
    bash -c 'cd /workspace && \
             job_tmp="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" && \
             export TMPDIR="/workspace/${job_tmp}" && \
             mkdir -p wandb wandb/cache wandb/tmp tmp experiments "$TMPDIR" && \
             '"${CMD}"

status=$?

echo ""
echo "=========================================="
if [ ${status} -eq 0 ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}) completed successfully."
else
    echo "Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}) FAILED with exit code ${status}."
fi
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results: experiments/expts/${CONFIG_NAME}/"
echo "=========================================="

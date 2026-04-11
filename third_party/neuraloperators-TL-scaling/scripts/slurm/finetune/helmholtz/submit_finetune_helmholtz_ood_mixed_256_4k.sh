#!/bin/bash
#SBATCH --job-name=neuralop-helm-ood-mixed
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

set -euo pipefail

echo "=========================================="
echo "Mixed OOD Fine-Tuning (Helmholtz) - 256 and 4k samples"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

resolve_project_dir() {
    local candidate
    if [ -n "${PROJECT_DIR:-}" ] && [ -f "${PROJECT_DIR}/train.py" ]; then
        printf '%s\n' "${PROJECT_DIR}"
        return
    fi
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/train.py" ]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}"
        return
    fi
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/third_party/neuraloperators-TL-scaling/train.py" ]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}/third_party/neuraloperators-TL-scaling"
        return
    fi
    candidate="$(cd "$(dirname "$0")/../../../.." && pwd)"
    printf '%s\n' "${candidate}"
}

PROJECT_DIR="$(resolve_project_dir)"
if [ ! -f "${PROJECT_DIR}/train.py" ]; then
    echo "Error: expected neuraloperators project root with train.py, got: ${PROJECT_DIR}"
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
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

cd "${PROJECT_DIR}"

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="${PROJECT_DIR}/${JOB_TMP_REL}"

cleanup_tmp_dir() {
    rm -rf "${JOB_TMP_DIR}"
    rmdir "${PROJECT_DIR}/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

CONFIG_FILE="${CONFIG_FILE:-config/operators_helmholtz.yaml}"

source scripts/slurm/finetune/seed_grid.sh

declare -a configs=(
    "helm-o10_15-finetune-mixed-256:finetune-mixed-o10_15-256"
    "helm-o10_15-finetune-mixed-4k:finetune-mixed-o10_15-4k"
    "helm-o15_20-finetune-mixed-256:finetune-mixed-o15_20-256"
    "helm-o15_20-finetune-mixed-4k:finetune-mixed-o15_20-4k"
    "helm-o20_25-finetune-mixed-256:finetune-mixed-o20_25-256"
    "helm-o20_25-finetune-mixed-4k:finetune-mixed-o20_25-4k"
    "helm-o25_30-finetune-mixed-256:finetune-mixed-o25_30-256"
    "helm-o25_30-finetune-mixed-4k:finetune-mixed-o25_30-4k"
)

IFS=':' read -r CONFIG_NAME RUN_NAME <<< "${configs[$SEED_EXPERIMENT_IDX]}"

echo "Project dir: ${PROJECT_DIR}"
echo "Configuration: ${CONFIG_FILE}"
echo "Config name: ${CONFIG_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Seed: ${SEED_VALUE}"
echo ""

CHECKPOINT_LINE="$(grep -A 20 "^${CONFIG_NAME}:" "${CONFIG_FILE}" | grep "weights:" | head -n 1)"
if [[ "${CHECKPOINT_LINE}" == *"JOBID"* ]]; then
    echo "ERROR: Checkpoint path contains placeholder 'JOBID'!"
    echo "Please update the checkpoint path in ${CONFIG_FILE}"
    echo "Found line: ${CHECKPOINT_LINE}"
    exit 1
fi

echo "Checkpoint line: ${CHECKPOINT_LINE}"
echo ""

mkdir -p experiments

BIND="--bind ${PROJECT_DIR}:/workspace"
CMD="python /workspace/train.py \
    --yaml_config=/workspace/${CONFIG_FILE} \
    --config=${CONFIG_NAME} \
    --run_num=${RUN_NAME}-${SEED_RUN_SUFFIX}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${SEED_TRAIN_ARGS}"

echo "Running mixed OOD fine-tuning (Task ${SLURM_ARRAY_TASK_ID})..."
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

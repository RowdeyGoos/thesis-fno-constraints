#!/bin/bash
#SBATCH --job-name=neuralop-poisson-ood-mixed
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
echo "OOD Fine-Tuning (Poisson, ${RUN_VARIANT:-mixed}) - 256 and 4k samples"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

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

MIXED_VARIANT="${MIXED_VARIANT:-mixed}"
RUN_VARIANT="${RUN_VARIANT:-$MIXED_VARIANT}"
CONFIG_FILE="${CONFIG_FILE:-config/operators_poisson.yaml}"
RERUN_MISSING_ONLY="${RERUN_MISSING_ONLY:-${MISSING_ONLY:-0}}"

source scripts/slurm/finetune/seed_grid.sh

if [ "${RERUN_MISSING_ONLY}" = "1" ]; then
    case "${MIXED_VARIANT}" in
        mixed-zero-hard)
            SEED_TASK_ALLOWLIST="${SEED_TASK_ALLOWLIST:-11,12,14}"
            ;;
        *)
            echo "Missing-only mode requested, but no Poisson OOD gaps are listed for ${MIXED_VARIANT}."
            exit 0
            ;;
    esac
fi

if [ -n "${SEED_TASK_ALLOWLIST:-}" ]; then
    seed_skip_unless_task_allowed "${SEED_TASK_ALLOWLIST}" "${SLURM_ARRAY_TASK_ID}" "Poisson ${MIXED_VARIANT} OOD"
fi

declare -a configs=(
    "poisson-k5_7p5-finetune-${MIXED_VARIANT}-256:finetune-${RUN_VARIANT}-k5_7p5-256"
    "poisson-k5_7p5-finetune-${MIXED_VARIANT}-4k:finetune-${RUN_VARIANT}-k5_7p5-4k"
    "poisson-k7p5_10-finetune-${MIXED_VARIANT}-256:finetune-${RUN_VARIANT}-k7p5_10-256"
    "poisson-k7p5_10-finetune-${MIXED_VARIANT}-4k:finetune-${RUN_VARIANT}-k7p5_10-4k"
    "poisson-k10_12p5-finetune-${MIXED_VARIANT}-256:finetune-${RUN_VARIANT}-k10_12p5-256"
    "poisson-k10_12p5-finetune-${MIXED_VARIANT}-4k:finetune-${RUN_VARIANT}-k10_12p5-4k"
    "poisson-k12p5_15-finetune-${MIXED_VARIANT}-256:finetune-${RUN_VARIANT}-k12p5_15-256"
    "poisson-k12p5_15-finetune-${MIXED_VARIANT}-4k:finetune-${RUN_VARIANT}-k12p5_15-4k"
)

IFS=':' read -r CONFIG_NAME RUN_NAME <<< "${configs[$SEED_EXPERIMENT_IDX]}"

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

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"
CMD="python /workspace/train.py \
    --yaml_config=/workspace/${CONFIG_FILE} \
    --config=${CONFIG_NAME} \
    --run_num=${RUN_NAME}-${SEED_RUN_SUFFIX}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments \
    ${SEED_TRAIN_ARGS}"

echo "Running OOD fine-tuning (Task ${SLURM_ARRAY_TASK_ID})..."
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

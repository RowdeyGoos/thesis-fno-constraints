#!/bin/bash
#SBATCH --job-name=neuralop-poisson-ood-scratch
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
echo "OOD Scratch Training (Poisson) - 256 and 4k samples"
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

CONFIG_FILE="${CONFIG_FILE:-config/operators_poisson.yaml}"

source scripts/slurm/finetune/seed_grid.sh

declare -a configs=(
    "poisson-k5_7p5-scratch-256:scratch-k5_7p5-256"
    "poisson-k5_7p5-scratch-4k:scratch-k5_7p5-4k"
    "poisson-k7p5_10-scratch-256:scratch-k7p5_10-256"
    "poisson-k7p5_10-scratch-4k:scratch-k7p5_10-4k"
    "poisson-k10_12p5-scratch-256:scratch-k10_12p5-256"
    "poisson-k10_12p5-scratch-4k:scratch-k10_12p5-4k"
    "poisson-k12p5_15-scratch-256:scratch-k12p5_15-256"
    "poisson-k12p5_15-scratch-4k:scratch-k12p5_15-4k"
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
CMD="python /workspace/train.py \
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

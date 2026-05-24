#!/bin/bash
#SBATCH --job-name=neuralop-mixed-bc-pretrain
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=21:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

echo "=========================================="
echo "Mixed BC Dataset Pretraining"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "=========================================="

# Container location
CONTAINER_PATH="${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif}"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: container not found at $CONTAINER_PATH"
    exit 1
fi

# Load Apptainer module
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

# W&B config
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/$JOB_TMP_REL"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "${SLURM_SUBMIT_DIR:-$(pwd)}/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

# Configuration (override via sbatch --export=ALL,...)
CONFIG_FILE="${CONFIG_FILE:-config/operators_mixed_bc.yaml}"
CONFIG_NAME="${CONFIG_NAME:-mixed-bc-scale-all-soft}"
RUN_NAME="${RUN_NAME:-pretrain-mixed-bc}"
ROOT_DIR="${ROOT_DIR:-experiments}"

MIXED_TRAIN="${MIXED_TRAIN:-data/bc/mixed/_train_mixed_32k_bc.h5}"
MIXED_SCALES="${MIXED_SCALES:-data/bc/mixed/_train_mixed_32k_bc_scales.npy}"

if [ ! -f "$MIXED_TRAIN" ]; then
    echo "Error: mixed BC training dataset not found at: $MIXED_TRAIN"
    echo "Build it first (from repo root) with:"
    echo "  bash scripts/workflows/run_build_mixed_bc.sh"
    exit 2
fi

if [ ! -f "$MIXED_SCALES" ]; then
    echo "Warning: scales file not found at: $MIXED_SCALES"
    echo "Training may fail if the selected config expects scales_path."
fi

echo "Configuration: $CONFIG_FILE"
echo "Config name:   $CONFIG_NAME"
echo "Run name:      $RUN_NAME"
echo "Mixed train:   $MIXED_TRAIN"
echo ""

mkdir -p "$ROOT_DIR"

# Bind directories
BIND="--bind $(pwd):/workspace"

CMD="python /workspace/scripts/entrypoints/train.py \
    --yaml_config=/workspace/${CONFIG_FILE} \
    --config=${CONFIG_NAME} \
    --run_num=${RUN_NAME}-${SLURM_JOB_ID}-0 \
    --root_dir=/workspace/${ROOT_DIR}"

echo "Running mixed BC pretraining..."
echo "Command: $CMD"
echo ""

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             job_tmp="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" && \
             export TMPDIR="/workspace/${job_tmp}" && \
             mkdir -p wandb wandb/cache wandb/tmp tmp "'"${ROOT_DIR}"'" "$TMPDIR" && \
             '"$CMD"

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Mixed BC pretraining completed successfully."
else
    echo "Mixed BC pretraining FAILED with exit code $status."
fi
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out / .err"
echo "Results: ${ROOT_DIR}/expts/${CONFIG_NAME}/"
echo "=========================================="

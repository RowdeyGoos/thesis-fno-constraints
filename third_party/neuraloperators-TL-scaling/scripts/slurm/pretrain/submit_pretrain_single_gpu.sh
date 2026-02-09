#!/bin/bash
#SBATCH --job-name=neuralop-poisson-single
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END         # Mail when job finishes
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short             # max 4 hours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G

echo "=========================================="
echo "Starting neuraloperators-TL-scaling (single GPU, container)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submit dir:  $SLURM_SUBMIT_DIR"
echo "=========================================="

# -------- Path to container on DAIC --------
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Load apptainer / singularity
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

# These will be seen *inside* the container too
# Make wandb + temp dirs live under /workspace so they are on the same filesystem
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp
JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
export TMPDIR="/workspace/$JOB_TMP_REL"

cd "$SLURM_SUBMIT_DIR"

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="$SLURM_SUBMIT_DIR/$JOB_TMP_REL"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "$SLURM_SUBMIT_DIR/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

# -------- Experiment config --------
CONFIG_FILE="config/operators_poisson.yaml"
CONFIG_NAME="poisson-scale-k1_5"
RUN_BASE="test"   # you can change this per experiment

# Results dir in your repo
SCRATCH="$SLURM_SUBMIT_DIR/experiments"
mkdir -p "$SCRATCH"

# Bind the whole repo into /workspace inside the container
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command to run inside the container
CMD="python /workspace/train.py \
    --yaml_config=/workspace/$CONFIG_FILE \
    --config=$CONFIG_NAME \
    --run_num=${RUN_BASE}-${SLURM_JOB_ID} \
    --root_dir=/workspace/experiments"

echo "Running single-GPU training..."
echo "Command: $CMD"
echo ""

# Run inside container:
# - cd /workspace so paths match
# - create wandb + tmp dirs on the same FS as experiments
apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             mkdir -p wandb wandb/cache wandb/tmp tmp experiments "$TMPDIR" && \
             '"$CMD"

STATUS=$?

echo ""
echo "=========================================="
if [ $STATUS -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training FAILED with exit code $STATUS."
fi
echo "Logs in: experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out / .err"
echo "Results in: $SCRATCH"
echo "=========================================="

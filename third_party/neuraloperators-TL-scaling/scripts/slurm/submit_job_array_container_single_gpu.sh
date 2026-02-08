#!/bin/bash
#SBATCH --job-name=neuralop-pretrain-array-1gpu
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=20:00:00
#SBATCH --qos=long
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # Single task (no DDP)
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a40:1             # 1 A40 GPU per array task
#SBATCH --mem=32G
#SBATCH --array=0-2

# Array job for pretraining all three PDE systems using Apptainer container (Single GPU)
# This runs 3 separate pretraining jobs (one per PDE system) in parallel using job arrays
# Each job uses 1 GPU (no Distributed Data Parallel)
# Submit from project root: sbatch scripts/slurm/submit_job_array_container_single_gpu.sh

echo "=========================================="
echo "Starting neuraloperators-TL-scaling Pretraining Array Job (Container, Single GPU)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node list: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

# Check if container exists
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

# These paths are *inside* the container
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

# -------- Define pretraining configurations for each PDE system --------
# Format: "config_file:config_name:run_name_base"
configs=(
    "config/operators_poisson.yaml:poisson-scale-k1_5:pretrain-poisson-k1_5"
    "config/operators_ad.yaml:ad-scale-adr0p2_1:pretrain-advdiff-adr0p2_1"
    "config/operators_helmholtz.yaml:helm-scale-o1_10:pretrain-helmholtz-o1_10"
)

# Get config for this array task
IFS=':' read -r config_file config_name run_base <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

echo "Configuration: $config_file"
echo "Config name: $config_name"
echo "Run base: $run_base"
echo ""

# Check if config exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found"
    exit 1
fi

# Create scratch directories
SCRATCH="$SLURM_SUBMIT_DIR/experiments"
mkdir -p "$SCRATCH"

# Bind whole repo and create necessary directories inside container
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command (inside container, paths under /workspace)
CMD="python /workspace/train.py \
    --yaml_config=/workspace/$config_file \
    --config=$config_name \
    --run_num=${run_base}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} \
    --root_dir=/workspace/experiments"

echo "Running single GPU training..."
echo "Command: $CMD"
echo ""

# Run with single GPU (no srun, direct apptainer exec)
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
    echo "Pretraining task $SLURM_ARRAY_TASK_ID completed successfully."
else
    echo "Pretraining task $SLURM_ARRAY_TASK_ID FAILED with exit code $status."
fi
echo "Logs in: experiments/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out / .err"
echo "Results in: $SCRATCH/expts"
echo "=========================================="

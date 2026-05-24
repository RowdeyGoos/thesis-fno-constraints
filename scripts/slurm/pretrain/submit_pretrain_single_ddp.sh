#!/bin/bash
#SBATCH --job-name=neuralop-poisson-ddp
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 🔴 One task per GPU
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:2             # 🔴 2 GPUs on the node
#SBATCH --mem=8G

echo "=========================================="
echo "Starting neuraloperators-TL-scaling DDP (Container, Slurm-managed)"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node list:   $SLURM_NODELIST"
echo "Submit dir:  $SLURM_SUBMIT_DIR"
echo "=========================================="

# -------- Path to container --------
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Load apptainer / singularity
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

# -------- W&B config (same idea as single-GPU) --------
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

# These paths are *inside* the container
export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp

# -------- NCCL debug / DAIC recommendation --------
# DAIC suggests this if multi-GPU hangs
export NCCL_P2P_DISABLE=1        # avoid direct GPU-GPU P2P if problematic
export NCCL_ASYNC_ERROR_HANDLING=1
# Optional extra debug if things still hang:
# export NCCL_DEBUG=INFO

cd "$SLURM_SUBMIT_DIR"

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="$SLURM_SUBMIT_DIR/$JOB_TMP_REL"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "$SLURM_SUBMIT_DIR/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

# -------- DDP + experiment config --------
export MASTER_ADDR=$(hostname)   # same as original repo script
export MASTER_PORT=29500         # matches export_DDP_vars.sh

ngpu=$SLURM_NTASKS               # == 2

CONFIG_FILE="./config/operators_poisson.yaml"
CONFIG_NAME="poisson-scale-k1_5"
RUN_BASE="ddp-poisson"

# host results dir; inside container it's /workspace/experiments
SCRATCH="$SLURM_SUBMIT_DIR/experiments"
mkdir -p "$SCRATCH"

# Python command (inside container, paths under /workspace)
CMD="python /workspace/train.py \
    --yaml_config=/workspace/config/operators_poisson.yaml \
    --config=$CONFIG_NAME \
    --run_num=${RUN_BASE}-${SLURM_JOB_ID} \
    --root_dir=/workspace/experiments"

# Bind whole repo into /workspace
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

echo "Running DDP training with $ngpu GPUs..."
echo "Command: $CMD"
echo ""

# This line is the key: same pattern as paper repo,
# just with apptainer exec injected.
srun -l -n $ngpu --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus-per-node=$ngpu \
    apptainer exec --nv $BIND "$CONTAINER_PATH" \
        bash -c 'cd /workspace && \
                 job_tmp="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" && \
                 export TMPDIR="/workspace/${job_tmp}" && \
                 mkdir -p wandb wandb/cache wandb/tmp tmp experiments "$TMPDIR" && \
                 echo "Rank: $SLURM_PROCID, Local rank: $SLURM_LOCALID, World size: $SLURM_NTASKS" && \
                 source export_DDP_vars.sh && \
                 '"$CMD"

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "DDP Training completed successfully."
else
    echo "DDP Training FAILED with exit code $status."
fi
echo "Logs in: experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out / .err"
echo "Results in: $SCRATCH"
echo "=========================================="

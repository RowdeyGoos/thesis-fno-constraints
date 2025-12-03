#!/bin/bash
#SBATCH --job-name=neuralop-poisson-ddp
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

echo "=========================================="
echo "Starting neuraloperators-TL-scaling DDP (Container)"
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
export WANDB_DIR=/workspace/wandb
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

cd "$SLURM_SUBMIT_DIR"

# -------- DDP + experiment config --------
export MASTER_ADDR=$(hostname)   # single-node DDP is fine with this
ngpu=4

config_file=./config/operators_poisson.yaml
config="poisson-scale-k1_5"
run_num="test"

# Store results in experiments/ inside the repo
scratch="$SLURM_SUBMIT_DIR/experiments"

mkdir -p "$scratch"

# Python command to run inside the container
cmd="python /workspace/train.py \
    --yaml_config=/workspace/config/operators_poisson.yaml \
    --config=$config \
    --run_num=$run_num \
    --root_dir=$scratch"

# Bind the whole repo into /workspace inside the container
BIND="--bind $SLURM_SUBMIT_DIR:/workspace \
      --bind $SLURM_SUBMIT_DIR/wandb:/workspace/wandb"

echo "Running DDP training with $ngpu GPUs..."
echo "Command: $cmd"
echo ""

# Each Slurm task runs: source DDP vars, then run train.py inside the container
srun -l -n $ngpu --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus-per-node=$ngpu \
    apptainer exec --nv $BIND "$CONTAINER_PATH" \
        bash -c "cd /workspace && source export_DDP_vars.sh && $cmd"

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "DDP Training completed successfully."
else
    echo "DDP Training FAILED with exit code $status."
fi
echo "Logs in: experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out / .err"
echo "Results in: $scratch"
echo "=========================================="

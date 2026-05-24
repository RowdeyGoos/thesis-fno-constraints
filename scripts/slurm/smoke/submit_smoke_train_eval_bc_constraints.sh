#!/bin/bash
#SBATCH --job-name=smoke-train-eval-bc-constraints
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=1:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

echo "=========================================="
echo "Smoke Train+Eval (BC constraints) starting"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node list:   ${SLURM_NODELIST:-local}"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=========================================="

CONTAINER_PATH="${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif}"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: container not found at $CONTAINER_PATH"
    echo "Set CONTAINER_PATH=/path/to/neuraloperators.sif and resubmit."
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
            # shellcheck disable=SC1091
            . /etc/profile.d/modules.sh
        elif [ -f /usr/share/Modules/init/bash ]; then
            # shellcheck disable=SC1091
            . /usr/share/Modules/init/bash
        fi
    fi

    if command -v module >/dev/null 2>&1; then
        if module load apptainer 2>/dev/null; then
            :
        elif module load singularity 2>/dev/null; then
            :
        fi
    fi
fi

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN="singularity"
else
    echo "Error: neither 'apptainer' nor 'singularity' is available on PATH."
    exit 1
fi

echo "Container image:   ${CONTAINER_PATH}"
echo "Container runtime: ${CONTAINER_BIN}"

export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p experiments

BIND="--bind ${WORKDIR}:/workspace"

BC_SMOKE_CMD='cd /workspace && bash scripts/utils/run_local_smoke_train_eval_bc_constraints.sh'

echo "Running BC smoke train+eval pipeline..."
echo "Command: ${BC_SMOKE_CMD}"
"${CONTAINER_BIN}" exec --nv $BIND "$CONTAINER_PATH" bash -lc "$BC_SMOKE_CMD"

echo ""
echo "=========================================="
echo "BC smoke train+eval completed successfully."
echo "=========================================="

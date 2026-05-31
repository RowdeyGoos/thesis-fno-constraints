#!/bin/bash
#SBATCH --job-name=smoke-train-eval-constraints
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=0:30:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

echo "=========================================="
echo "Smoke Train+Eval (constraints) starting"
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

echo "Container image: ${CONTAINER_PATH}"
echo "Resolving container runtime (apptainer/singularity)..."

# On some SLURM clusters `module` is not initialized in non-interactive shells.
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
    echo "If your cluster uses environment modules, ensure the module command is available in batch jobs."
    exit 1
fi

echo "Container runtime: ${CONTAINER_BIN}"

export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

mkdir -p experiments

BIND="--bind ${WORKDIR}:/workspace"

# Use a predefined smoke config that points at the real Poisson dataset and
# reduces runtime via `subsample`.
# Override at submission time if needed:
#   sbatch --export=ALL,YAML_CONFIG=config/operators_poisson.yaml,CONFIG_NAME=poisson-smoke-k1_5-constraints ...
YAML_CONFIG="${YAML_CONFIG:-${BASE_YAML:-config/operators_poisson.yaml}}"
CONFIG_NAME="${CONFIG_NAME:-${SMOKE_CONFIG:-poisson-smoke-k1_5-constraints}}"
RUN_TAG="smoke-${SLURM_JOB_ID:-local}"
ROOT_DIR="${ROOT_DIR:-experiments}"

echo "YAML config:  ${YAML_CONFIG}"
echo "Config name:  ${CONFIG_NAME}"

TRAIN_CMD="python /workspace/scripts/entrypoints/train.py \
  --yaml_config=/workspace/${YAML_CONFIG} \
  --config=${CONFIG_NAME} \
  --run_num=${RUN_TAG}-train \
  --root_dir=/workspace/${ROOT_DIR}"

echo "Running smoke training..."
echo "Command: ${TRAIN_CMD}"
"${CONTAINER_BIN}" exec --nv $BIND "$CONTAINER_PATH" bash -lc "cd /workspace && mkdir -p ${ROOT_DIR} && ${TRAIN_CMD}"

CKPT_BEST="${WORKDIR}/${ROOT_DIR}/expts/${CONFIG_NAME}/${RUN_TAG}-train/checkpoints/ckpt_best.tar"
CKPT_LAST="${WORKDIR}/${ROOT_DIR}/expts/${CONFIG_NAME}/${RUN_TAG}-train/checkpoints/ckpt.tar"
if [ -f "$CKPT_BEST" ]; then
    CKPT_PATH="$CKPT_BEST"
elif [ -f "$CKPT_LAST" ]; then
    CKPT_PATH="$CKPT_LAST"
else
    echo "Error: no checkpoint found in ${WORKDIR}/${ROOT_DIR}/expts/${CONFIG_NAME}/${RUN_TAG}-train/checkpoints"
    exit 2
fi

echo "Using checkpoint: ${CKPT_PATH}"

EVAL_CMD="python /workspace/scripts/entrypoints/eval.py \
  --yaml_config=/workspace/${YAML_CONFIG} \
  --config=${CONFIG_NAME} \
  --run_num=${RUN_TAG}-eval \
  --root_dir=/workspace/${ROOT_DIR} \
  --weights=/workspace/${CKPT_PATH#${WORKDIR}/}"

echo "Running smoke eval..."
echo "Command: ${EVAL_CMD}"
"${CONTAINER_BIN}" exec --nv $BIND "$CONTAINER_PATH" bash -lc "cd /workspace && ${EVAL_CMD}"

TRAIN_LOG="${WORKDIR}/${ROOT_DIR}/expts/${CONFIG_NAME}/${RUN_TAG}-train/logs_best.txt"
EVAL_LOG="${WORKDIR}/${ROOT_DIR}/expts/${CONFIG_NAME}/${RUN_TAG}-eval/logs_best.txt"

echo "Validating smoke metrics in logs..."
if ! grep -q "pde_residual_norm" "$TRAIN_LOG"; then
    echo "Error: training log missing pde_residual_norm (${TRAIN_LOG})"
    exit 3
fi
if ! grep -q "pde_al_lambda" "$TRAIN_LOG"; then
    echo "Error: training log missing pde_al_lambda (${TRAIN_LOG})"
    exit 4
fi
if ! grep -q "zero_mode_constraint_loss" "$TRAIN_LOG"; then
    echo "Error: training log missing zero_mode_constraint_loss (${TRAIN_LOG})"
    exit 5
fi
if ! grep -q "test_zero_mode_constraint_loss" "$EVAL_LOG"; then
    echo "Error: eval log missing test_zero_mode_constraint_loss (${EVAL_LOG})"
    exit 6
fi
if ! grep -q "test_pde_residual_norm" "$EVAL_LOG"; then
    echo "Error: eval log missing test_pde_residual_norm (${EVAL_LOG})"
    exit 7
fi
if ! grep -q "test_zero_mode_violation" "$EVAL_LOG"; then
    echo "Error: eval log missing test_zero_mode_violation (${EVAL_LOG})"
    exit 8
fi

echo ""
echo "=========================================="
echo "Smoke train+eval completed successfully."
echo "Train log:      ${TRAIN_LOG}"
echo "Eval log:       ${EVAL_LOG}"
echo "Checkpoint:     ${CKPT_PATH}"
echo "=========================================="

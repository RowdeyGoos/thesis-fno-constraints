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
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=24G

set -euo pipefail

echo "=========================================="
echo "Smoke Train+Eval (constraints) starting"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node list:   ${SLURM_NODELIST:-local}"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=========================================="

CONTAINER_PATH="${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif}"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: container not found at $CONTAINER_PATH"
    echo "Set CONTAINER_PATH=/path/to/neuraloperators.sif and resubmit."
    exit 1
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

mkdir -p experiments tmp

BIND="--bind ${WORKDIR}:/workspace"

# Base config can be overridden at submission time:
#   sbatch --export=ALL,BASE_YAML=config/operators_mixed.yaml,BASE_CONFIG=mixed-scale-all ...
BASE_YAML="${BASE_YAML:-config/operators_poisson.yaml}"
BASE_CONFIG="${BASE_CONFIG:-poisson-scale-k1_5}"
SMOKE_CONFIG="${SMOKE_CONFIG:-smoke-constraints-runtime}"
RUN_TAG="smoke-${SLURM_JOB_ID:-local}"
ROOT_DIR="experiments"

TMP_CFG_HOST="tmp/${RUN_TAG}-${SMOKE_CONFIG}.yaml"
TMP_CFG_CONT="/workspace/${TMP_CFG_HOST}"

echo "Base YAML:    ${BASE_YAML}"
echo "Base config:  ${BASE_CONFIG}"
echo "Smoke config: ${SMOKE_CONFIG}"

python_cfg_cmd="
from ruamel.yaml import YAML
from copy import deepcopy

base_yaml = '/workspace/${BASE_YAML}'
base_config = '${BASE_CONFIG}'
smoke_config = '${SMOKE_CONFIG}'
out_path = '${TMP_CFG_CONT}'

yaml = YAML()
with open(base_yaml, 'r') as f:
    cfg = yaml.load(f)

if base_config not in cfg:
    raise KeyError(f'Config {base_config} not found in {base_yaml}')

base = deepcopy(dict(cfg[base_config]))
base.update({
    'log_to_wandb': False,
    'plot_figs': False,
    'save_checkpoint': True,
    'scheduler': 'none',
    'max_epochs': 2,
    'batch_size': 8,
    'valid_batch_size': 8,
    'subsample': 4096,
    'constraint_zero_mode_enable': True,
    'constraint_zero_mode_mode': 'gauge_aware',
    'constraint_zero_mode_omega_tol': 1.0e-8,
    'constraint_pde_enable': True,
    'constraint_pde_method': 'augmented_lagrangian',
    'constraint_pde_weight': 0.1,
    'constraint_pde_warmup_fraction': 0.0,
    'constraint_pde_al_rho': 1.0,
    'constraint_pde_al_lambda0': 0.0,
    'constraint_pde_al_dual_clip': 1.0e6,
})
cfg[smoke_config] = base

with open(out_path, 'w') as f:
    yaml.dump(cfg, f)

print(f'Wrote smoke yaml: {out_path}')
"

echo "Generating temporary smoke config..."
apptainer exec --nv $BIND "$CONTAINER_PATH" bash -lc "cd /workspace && python - <<'PY'
${python_cfg_cmd}
PY"

TRAIN_CMD="python /workspace/train.py \
  --yaml_config=${TMP_CFG_CONT} \
  --config=${SMOKE_CONFIG} \
  --run_num=${RUN_TAG}-train \
  --root_dir=/workspace/${ROOT_DIR}"

echo "Running smoke training..."
echo "Command: ${TRAIN_CMD}"
apptainer exec --nv $BIND "$CONTAINER_PATH" bash -lc "cd /workspace && mkdir -p experiments tmp && ${TRAIN_CMD}"

CKPT_BEST="${WORKDIR}/${ROOT_DIR}/expts/${SMOKE_CONFIG}/${RUN_TAG}-train/checkpoints/ckpt_best.tar"
CKPT_LAST="${WORKDIR}/${ROOT_DIR}/expts/${SMOKE_CONFIG}/${RUN_TAG}-train/checkpoints/ckpt.tar"
if [ -f "$CKPT_BEST" ]; then
    CKPT_PATH="$CKPT_BEST"
elif [ -f "$CKPT_LAST" ]; then
    CKPT_PATH="$CKPT_LAST"
else
    echo "Error: no checkpoint found in ${WORKDIR}/${ROOT_DIR}/expts/${SMOKE_CONFIG}/${RUN_TAG}-train/checkpoints"
    exit 2
fi

echo "Using checkpoint: ${CKPT_PATH}"

EVAL_CMD="python /workspace/eval.py \
  --yaml_config=${TMP_CFG_CONT} \
  --config=${SMOKE_CONFIG} \
  --run_num=${RUN_TAG}-eval \
  --root_dir=/workspace/${ROOT_DIR} \
  --weights=/workspace/${CKPT_PATH#${WORKDIR}/}"

echo "Running smoke eval..."
echo "Command: ${EVAL_CMD}"
apptainer exec --nv $BIND "$CONTAINER_PATH" bash -lc "cd /workspace && ${EVAL_CMD}"

TRAIN_LOG="${WORKDIR}/${ROOT_DIR}/expts/${SMOKE_CONFIG}/${RUN_TAG}-train/logs_best.txt"
EVAL_LOG="${WORKDIR}/${ROOT_DIR}/expts/${SMOKE_CONFIG}/${RUN_TAG}-eval/logs_best.txt"

echo "Validating smoke metrics in logs..."
if ! grep -q "pde_residual_norm" "$TRAIN_LOG"; then
    echo "Error: training log missing pde_residual_norm (${TRAIN_LOG})"
    exit 3
fi
if ! grep -q "pde_al_lambda" "$TRAIN_LOG"; then
    echo "Error: training log missing pde_al_lambda (${TRAIN_LOG})"
    exit 4
fi
if ! grep -q "test_pde_residual_norm" "$EVAL_LOG"; then
    echo "Error: eval log missing test_pde_residual_norm (${EVAL_LOG})"
    exit 5
fi
if ! grep -q "test_zero_mode_violation" "$EVAL_LOG"; then
    echo "Error: eval log missing test_zero_mode_violation (${EVAL_LOG})"
    exit 6
fi

echo ""
echo "=========================================="
echo "Smoke train+eval completed successfully."
echo "Temporary YAML: ${TMP_CFG_HOST}"
echo "Train log:      ${TRAIN_LOG}"
echo "Eval log:       ${EVAL_LOG}"
echo "Checkpoint:     ${CKPT_PATH}"
echo "=========================================="

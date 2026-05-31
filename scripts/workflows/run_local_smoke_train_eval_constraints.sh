#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Local smoke train+eval (constraints)"
echo "=========================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Allow explicit override, e.g.:
#   PYTHON_BIN=/full/path/to/python bash scripts/workflows/run_local_smoke_train_eval_constraints.sh
if [ -n "${PYTHON_BIN:-}" ]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: PYTHON_BIN is set but not executable: $PYTHON_BIN"
    exit 1
  fi
else
  # Prefer active venv interpreter when available.
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/Scripts/python.exe" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: no python executable found."
    echo "Tip: activate your venv first or set PYTHON_BIN explicitly."
    exit 1
  fi
fi

echo "Using Python: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
required = ["torch", "h5py", "ruamel.yaml"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    print("Error: missing Python packages:", ", ".join(missing))
    print("Install dependencies and rerun. Example: pip install -r requirements.txt")
    sys.exit(1)
PY

DATA_DIR="data/local_smoke_poisson"
RESULTS_ROOT="${RESULTS_ROOT:-experiments_local_smoke}"
RUN_NAME="${RUN_NAME:-local-smoke}"
CONFIG_FILE="config/operators_local_smoke.yaml"
ZERO_MODE_ENFORCEMENT="${ZERO_MODE_ENFORCEMENT:-hard}"  # hard | soft

if [ -z "${CONFIG_NAME:-}" ]; then
  case "$ZERO_MODE_ENFORCEMENT" in
    hard)
      CONFIG_NAME="poisson-local-smoke-constraints"
      ;;
    soft)
      CONFIG_NAME="poisson-local-smoke-constraints-soft-zero"
      ;;
    *)
      echo "Error: ZERO_MODE_ENFORCEMENT must be 'hard' or 'soft' (got '$ZERO_MODE_ENFORCEMENT')."
      echo "Tip: set CONFIG_NAME explicitly to bypass automatic selection."
      exit 1
      ;;
  esac
fi

NTRAIN="${NTRAIN:-64}"
NVAL="${NVAL:-16}"
NTEST="${NTEST:-16}"
GRID_N="${GRID_N:-64}"
NG="${NG:-64}"

mkdir -p "$DATA_DIR" "$RESULTS_ROOT" tmp

echo "Generating tiny Poisson dataset..."
"$PYTHON_BIN" scripts/data/gen_data_poisson.py \
  --ntrain "$NTRAIN" \
  --nval "$NVAL" \
  --ntest "$NTEST" \
  --n "$GRID_N" \
  --ng "$NG" \
  --sparse \
  --datapath "$DATA_DIR" \
  --e1 1 \
  --e2 5

# Canonical filenames used by the smoke config.
TRAIN_CANON="${DATA_DIR}/_train_k1_5_32k.h5"
VAL_CANON="${DATA_DIR}/_val_k1_5_4k.h5"
TEST_CANON="${DATA_DIR}/_test_k1_5_4k.h5"

if [ ! -f "$TRAIN_CANON" ]; then
  TRAIN_GEN="$(find "$DATA_DIR" -maxdepth 1 -type f -name "_train_k*_32k.h5" | head -n 1)"
  [ -n "$TRAIN_GEN" ] || { echo "Error: train dataset file not found in $DATA_DIR"; exit 2; }
  cp "$TRAIN_GEN" "$TRAIN_CANON"
fi
if [ ! -f "$VAL_CANON" ]; then
  VAL_GEN="$(find "$DATA_DIR" -maxdepth 1 -type f -name "_val_k*_4k.h5" | head -n 1)"
  [ -n "$VAL_GEN" ] || { echo "Error: val dataset file not found in $DATA_DIR"; exit 2; }
  cp "$VAL_GEN" "$VAL_CANON"
fi
if [ ! -f "$TEST_CANON" ]; then
  TEST_GEN="$(find "$DATA_DIR" -maxdepth 1 -type f -name "_test_k*_4k.h5" | head -n 1)"
  [ -n "$TEST_GEN" ] || { echo "Error: test dataset file not found in $DATA_DIR"; exit 2; }
  cp "$TEST_GEN" "$TEST_CANON"
fi

echo "Computing scales..."
"$PYTHON_BIN" scripts/data/compute_scales.py \
  --datapath "$DATA_DIR" \
  --filename "$(basename "$TRAIN_CANON")" \
  --nx "$GRID_N" \
  --ny "$GRID_N" \
  --lx 1.0 \
  --ly 1.0 \
  --output "local_smoke_scales.npy"

echo "Running local smoke training..."
echo "Config: $CONFIG_NAME (ZERO_MODE_ENFORCEMENT=$ZERO_MODE_ENFORCEMENT)"
"$PYTHON_BIN" scripts/entrypoints/train.py \
  --yaml_config "$CONFIG_FILE" \
  --config "$CONFIG_NAME" \
  --run_num "${RUN_NAME}-train" \
  --root_dir "$RESULTS_ROOT"

CKPT_BEST="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${RUN_NAME}-train/checkpoints/ckpt_best.tar"
CKPT_LAST="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${RUN_NAME}-train/checkpoints/ckpt.tar"
if [ -f "$CKPT_BEST" ]; then
  CKPT="$CKPT_BEST"
elif [ -f "$CKPT_LAST" ]; then
  CKPT="$CKPT_LAST"
else
  echo "Error: no checkpoint found after training."
  exit 3
fi

echo "Running local smoke eval..."
"$PYTHON_BIN" scripts/entrypoints/eval.py \
  --yaml_config "$CONFIG_FILE" \
  --config "$CONFIG_NAME" \
  --run_num "${RUN_NAME}-eval" \
  --root_dir "$RESULTS_ROOT" \
  --weights "$CKPT"

TRAIN_LOG="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${RUN_NAME}-train/logs_best.txt"
EVAL_LOG="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${RUN_NAME}-eval/logs_best.txt"

echo "Checking expected metrics..."
grep -q "pde_residual_norm" "$TRAIN_LOG" || { echo "Missing pde_residual_norm in $TRAIN_LOG"; exit 4; }
grep -q "zero_mode_constraint_loss" "$TRAIN_LOG" || { echo "Missing zero_mode_constraint_loss in $TRAIN_LOG"; exit 5; }
grep -q "pde_al_lambda" "$TRAIN_LOG" || { echo "Missing pde_al_lambda in $TRAIN_LOG"; exit 6; }
grep -q "test_zero_mode_constraint_loss" "$EVAL_LOG" || { echo "Missing test_zero_mode_constraint_loss in $EVAL_LOG"; exit 7; }
grep -q "test_pde_residual_norm" "$EVAL_LOG" || { echo "Missing test_pde_residual_norm in $EVAL_LOG"; exit 8; }
grep -q "test_zero_mode_violation" "$EVAL_LOG" || { echo "Missing test_zero_mode_violation in $EVAL_LOG"; exit 9; }

echo ""
echo "Smoke run completed."
echo "Train log: $TRAIN_LOG"
echo "Eval log:  $EVAL_LOG"
echo "Checkpoint: $CKPT"

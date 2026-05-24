#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Local smoke train+eval (BC constraints)"
echo "=========================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: PYTHON_BIN is set but not executable: $PYTHON_BIN"
    exit 1
  fi
else
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
required = ["torch", "h5py", "ruamel.yaml", "numpy", "scipy"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    print("Error: missing Python packages:", ", ".join(missing))
    print("Install dependencies and rerun. Example: pip install -r requirements.txt")
    sys.exit(1)
PY

DATA_ROOT="${DATA_ROOT:-data/local_smoke_bc}"
POISSON_DIR="${DATA_ROOT}/poisson"
ADVDIFF_DIR="${DATA_ROOT}/advdiff"
HELMHOLTZ_DIR="${DATA_ROOT}/helmholtz"
MIXED_DIR="${MIXED_DIR:-data/local_smoke_mixed_bc}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments_local_smoke_bc}"
RUN_NAME="${RUN_NAME:-local-smoke-bc}"
CONFIG_FILE="config/operators_local_smoke_bc.yaml"
MODES="${MODES:-off soft hard hard+soft}"

# Tiny sizes for a quick local runtime check.
NTRAIN_PER_SYSTEM="${NTRAIN_PER_SYSTEM:-24}"
NVAL_PER_SYSTEM="${NVAL_PER_SYSTEM:-12}"
NTEST_PER_SYSTEM="${NTEST_PER_SYSTEM:-12}"
GRID_N="${GRID_N:-32}"
NG="${NG:-36}"

mkdir -p "$POISSON_DIR" "$ADVDIFF_DIR" "$HELMHOLTZ_DIR" "$MIXED_DIR" "$RESULTS_ROOT"

echo "Generating tiny BC Poisson/AdvDiff/Helmholtz datasets..."
"$PYTHON_BIN" utils/gen_data_poisson_bc.py \
  --ntrain "$NTRAIN_PER_SYSTEM" --nval "$NVAL_PER_SYSTEM" --ntest "$NTEST_PER_SYSTEM" \
  --n "$GRID_N" --ng "$NG" --sparse --datapath "$POISSON_DIR" --e1 1.0 --e2 5.0

"$PYTHON_BIN" utils/gen_data_advdiff_bc.py \
  --ntrain "$NTRAIN_PER_SYSTEM" --nval "$NVAL_PER_SYSTEM" --ntest "$NTEST_PER_SYSTEM" \
  --n "$GRID_N" --ng "$NG" --sparse --datapath "$ADVDIFF_DIR" --adr1 0.2 --adr2 1.0

"$PYTHON_BIN" utils/gen_data_helmholtz_bc.py \
  --ntrain "$NTRAIN_PER_SYSTEM" --nval "$NVAL_PER_SYSTEM" --ntest "$NTEST_PER_SYSTEM" \
  --n "$GRID_N" --ng "$NG" --sparse --datapath "$HELMHOLTZ_DIR" --o1 1 --o2 10

POISSON_TRAIN="${POISSON_DIR}/_train_k1p0_5p0_32k_bc.h5"
POISSON_VAL="${POISSON_DIR}/_val_k1p0_5p0_4k_bc.h5"
POISSON_TEST="${POISSON_DIR}/_test_k1p0_5p0_4k_bc.h5"

ADVDIFF_TRAIN="${ADVDIFF_DIR}/_train_adr0p2_1p0_32k_bc.h5"
ADVDIFF_VAL="${ADVDIFF_DIR}/_val_adr0p2_1p0_4k_bc.h5"
ADVDIFF_TEST="${ADVDIFF_DIR}/_test_adr0p2_1p0_4k_bc.h5"

HELMHOLTZ_TRAIN="${HELMHOLTZ_DIR}/_train_o1_10_32k_bc.h5"
HELMHOLTZ_VAL="${HELMHOLTZ_DIR}/_val_o1_10_4k_bc.h5"
HELMHOLTZ_TEST="${HELMHOLTZ_DIR}/_test_o1_10_4k_bc.h5"

for f in \
  "$POISSON_TRAIN" "$POISSON_VAL" "$POISSON_TEST" \
  "$ADVDIFF_TRAIN" "$ADVDIFF_VAL" "$ADVDIFF_TEST" \
  "$HELMHOLTZ_TRAIN" "$HELMHOLTZ_VAL" "$HELMHOLTZ_TEST"; do
  [ -f "$f" ] || { echo "Error: expected file not found: $f"; exit 2; }
done

echo "Running BC dataset sanity checks..."
"$PYTHON_BIN" scripts/utils/check_bc_dataset_sanity.py --input "$POISSON_DIR" --glob "*.h5"
"$PYTHON_BIN" scripts/utils/check_bc_dataset_sanity.py --input "$ADVDIFF_DIR" --glob "*.h5"
"$PYTHON_BIN" scripts/utils/check_bc_dataset_sanity.py --input "$HELMHOLTZ_DIR" --glob "*.h5"

echo "Building mixed BC datasets (train/val/test)..."
"$PYTHON_BIN" utils/create_mixed_dataset.py \
  --poisson_path "$POISSON_TRAIN" \
  --advdiff_path "$ADVDIFF_TRAIN" \
  --helmholtz_path "$HELMHOLTZ_TRAIN" \
  --output_path "${MIXED_DIR}/_train_mixed_bc_smoke.h5" \
  --samples_per_system "$NTRAIN_PER_SYSTEM" \
  --require_bc

"$PYTHON_BIN" utils/create_mixed_dataset.py \
  --poisson_path "$POISSON_VAL" \
  --advdiff_path "$ADVDIFF_VAL" \
  --helmholtz_path "$HELMHOLTZ_VAL" \
  --output_path "${MIXED_DIR}/_val_mixed_bc_smoke.h5" \
  --samples_per_system "$NVAL_PER_SYSTEM" \
  --require_bc

"$PYTHON_BIN" utils/create_mixed_dataset.py \
  --poisson_path "$POISSON_TEST" \
  --advdiff_path "$ADVDIFF_TEST" \
  --helmholtz_path "$HELMHOLTZ_TEST" \
  --output_path "${MIXED_DIR}/_test_mixed_bc_smoke.h5" \
  --samples_per_system "$NTEST_PER_SYSTEM" \
  --require_bc

"$PYTHON_BIN" scripts/utils/check_bc_dataset_sanity.py --input "$MIXED_DIR" --glob "*_mixed_bc_smoke.h5"

metric_leq () {
  local file="$1"
  local key="$2"
  local limit="$3"
  "$PYTHON_BIN" - "$file" "$key" "$limit" <<'PY'
import re
import sys

path, key, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
val = None
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith(key + ","):
            raw = line.split(",", 1)[1].strip()
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if m:
                val = float(m.group(0))
if val is None:
    print(f"Missing metric {key} in {path}")
    sys.exit(2)
if not (val <= limit):
    print(f"Metric {key}={val:.6e} exceeds limit {limit:.6e} in {path}")
    sys.exit(3)
print(f"{key}={val:.6e} <= {limit:.6e}")
PY
}

for mode in $MODES; do
  case "$mode" in
    off) CONFIG_NAME="mixed-bc-local-smoke-off" ;;
    soft) CONFIG_NAME="mixed-bc-local-smoke-soft" ;;
    hard) CONFIG_NAME="mixed-bc-local-smoke-hard" ;;
    hard+soft) CONFIG_NAME="mixed-bc-local-smoke-hard-soft" ;;
    *)
      echo "Error: unsupported mode '$mode'. Use: off soft hard hard+soft"
      exit 3
      ;;
  esac

  echo ""
  echo "Running smoke mode: $mode (config=$CONFIG_NAME)"

  TRAIN_RUN="${RUN_NAME}-${mode}-train"
  EVAL_RUN="${RUN_NAME}-${mode}-eval"

  "$PYTHON_BIN" train.py \
    --yaml_config "$CONFIG_FILE" \
    --config "$CONFIG_NAME" \
    --run_num "$TRAIN_RUN" \
    --root_dir "$RESULTS_ROOT"

  CKPT_BEST="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${TRAIN_RUN}/checkpoints/ckpt_best.tar"
  CKPT_LAST="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${TRAIN_RUN}/checkpoints/ckpt.tar"
  if [ -f "$CKPT_BEST" ]; then
    CKPT="$CKPT_BEST"
  elif [ -f "$CKPT_LAST" ]; then
    CKPT="$CKPT_LAST"
  else
    echo "Error: no checkpoint found for mode '$mode'."
    exit 4
  fi

  "$PYTHON_BIN" eval.py \
    --yaml_config "$CONFIG_FILE" \
    --config "$CONFIG_NAME" \
    --run_num "$EVAL_RUN" \
    --root_dir "$RESULTS_ROOT" \
    --weights "$CKPT"

  TRAIN_LOG="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${TRAIN_RUN}/logs_best.txt"
  EVAL_LOG="${RESULTS_ROOT}/expts/${CONFIG_NAME}/${EVAL_RUN}/logs_best.txt"

  grep -q "^bc_violation_raw," "$TRAIN_LOG" || { echo "Missing bc_violation_raw in $TRAIN_LOG"; exit 5; }
  grep -q "^bc_violation_final," "$TRAIN_LOG" || { echo "Missing bc_violation_final in $TRAIN_LOG"; exit 6; }
  grep -q "^val_err_interior," "$TRAIN_LOG" || { echo "Missing val_err_interior in $TRAIN_LOG"; exit 7; }
  grep -q "^test_bc_violation_raw," "$EVAL_LOG" || { echo "Missing test_bc_violation_raw in $EVAL_LOG"; exit 8; }
  grep -q "^test_bc_violation_final," "$EVAL_LOG" || { echo "Missing test_bc_violation_final in $EVAL_LOG"; exit 9; }
  grep -q "^test_err_interior," "$EVAL_LOG" || { echo "Missing test_err_interior in $EVAL_LOG"; exit 10; }

  # Hard projection should enforce final boundary values tightly.
  if [ "$mode" = "hard" ] || [ "$mode" = "hard+soft" ]; then
    metric_leq "$TRAIN_LOG" "bc_violation_final" "1e-6"
    metric_leq "$EVAL_LOG" "test_bc_violation_final" "1e-6"
  fi
done

echo ""
echo "BC smoke run completed."
echo "Results root: $RESULTS_ROOT"

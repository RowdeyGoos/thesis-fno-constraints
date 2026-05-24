#!/usr/bin/env bash
set -euo pipefail

# Submit BC mixed pretraining in SOFT mode.
#
# Usage:
#   bash scripts/utils/submit_bc_constraints_mode_soft.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh}"
CONFIG_NAME="${CONFIG_NAME:-mixed-bc-scale-all-soft}"
RUN_NAME="${RUN_NAME:-pretrain-mixed-bc-soft}"
EXTRA_EXPORT="${EXTRA_EXPORT:-}"
MODE_NAME="soft"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found on PATH"
  exit 2
fi

mkdir -p experiments results/constraints

export_args="ALL,CONFIG_NAME=${CONFIG_NAME},RUN_NAME=${RUN_NAME}"
if [[ -n "$EXTRA_EXPORT" ]]; then
  export_args="${export_args},${EXTRA_EXPORT}"
fi

job_raw="$(sbatch --parsable --export="${export_args}" "$SUBMIT_SCRIPT")"
job_id="${job_raw%%;*}"

echo "MODE=${MODE_NAME}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "JOB_ID=${job_id}"

{
  echo "timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "mode=${MODE_NAME}"
  echo "config_name=${CONFIG_NAME}"
  echo "run_name=${RUN_NAME}"
  echo "job_id=${job_id}"
  echo "---"
} >> results/constraints/bc_mode_submission_history.txt

echo ""
echo "Submitted BC SOFT mode run successfully."

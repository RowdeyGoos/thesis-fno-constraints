#!/usr/bin/env bash
set -euo pipefail

# Submit BC mixed evaluation in OFF mode.
#
# Usage:
#   bash scripts/experiments/submit_bc_eval_mode_off.sh <JOB_ID> [RUN_INDEX]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/slurm/eval/submit_eval_mixed_bc_bundle.sh}"
CONFIG_NAME="${CONFIG_NAME:-mixed-bc-scale-all-off}"
RUN_NAME="${RUN_NAME:-pretrain-mixed-bc-off}"
EXTRA_EXPORT="${EXTRA_EXPORT:-}"
MODE_NAME="off"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/experiments/submit_bc_eval_mode_off.sh <JOB_ID> [RUN_INDEX]" >&2
  exit 2
fi

TARGET_JOB_ID="$1"
TARGET_RUN_INDEX="${2:-0}"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found on PATH"
  exit 2
fi

mkdir -p experiments results/constraints

spec="${CONFIG_NAME}:${RUN_NAME}:${TARGET_JOB_ID}:${TARGET_RUN_INDEX}"
export_args="ALL"
if [[ -n "$EXTRA_EXPORT" ]]; then
  export_args="${export_args},${EXTRA_EXPORT}"
fi

job_raw="$(sbatch --parsable --export="${export_args}" "$SUBMIT_SCRIPT" "$spec")"
eval_job_id="${job_raw%%;*}"

echo "MODE=${MODE_NAME}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "TARGET_JOB_ID=${TARGET_JOB_ID}"
echo "TARGET_RUN_INDEX=${TARGET_RUN_INDEX}"
echo "EVAL_JOB_ID=${eval_job_id}"

{
  echo "timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "mode=${MODE_NAME}"
  echo "config_name=${CONFIG_NAME}"
  echo "run_name=${RUN_NAME}"
  echo "target_job_id=${TARGET_JOB_ID}"
  echo "target_run_index=${TARGET_RUN_INDEX}"
  echo "eval_job_id=${eval_job_id}"
  echo "---"
} >> results/constraints/bc_eval_submission_history.txt

echo ""
echo "Submitted BC OFF mode eval successfully."

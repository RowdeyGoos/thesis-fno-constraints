#!/usr/bin/env bash
set -euo pipefail

# Submit the BC soft vs hard pretraining pair.
#
# Usage:
#   bash scripts/experiments/submit_bc_constraints_soft_hard_compare.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh}"
RUN_PREFIX="${RUN_PREFIX:-pretrain-mixed-bc}"
EXTRA_EXPORT="${EXTRA_EXPORT:-}"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SUBMIT_SCRIPT"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found on PATH"
  exit 2
fi

mkdir -p experiments results/constraints

submit_one() {
  local mode="$1"
  local config_name="$2"
  local run_name="$3"

  local export_args="ALL,CONFIG_NAME=${config_name},RUN_NAME=${run_name}"
  if [[ -n "$EXTRA_EXPORT" ]]; then
    export_args="${export_args},${EXTRA_EXPORT}"
  fi

  job_raw="$(sbatch --parsable --export="${export_args}" "$SUBMIT_SCRIPT")"
  job_id="${job_raw%%;*}"
  echo "${mode^^}_JOB_ID=${job_id}"

  {
    echo "timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "mode=${mode}"
    echo "config_name=${config_name}"
    echo "run_name=${run_name}"
    echo "job_id=${job_id}"
    echo "---"
  } >> results/constraints/bc_soft_hard_submission_history.txt
}

echo "=========================================="
echo "Submitting BC soft vs hard runs"
echo "Submit script: ${SUBMIT_SCRIPT}"
echo "=========================================="

submit_one "soft" "mixed-bc-scale-all-soft" "${RUN_PREFIX}-soft"
submit_one "hard" "mixed-bc-scale-all-hard" "${RUN_PREFIX}-hard"

echo ""
echo "Submitted BC soft/hard comparison runs."

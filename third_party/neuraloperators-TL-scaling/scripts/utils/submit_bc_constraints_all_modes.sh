#!/usr/bin/env bash
set -euo pipefail

# Submit all BC enforcement pretraining runs (off, soft, hard, hard+soft)
# using the BC mixed pretrain SLURM entrypoint.
#
# Usage:
#   bash scripts/utils/submit_bc_constraints_all_modes.sh
#
# Optional environment overrides:
#   SUBMIT_SCRIPT     (default: scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh)
#   RUN_PREFIX        (default: pretrain-mixed-bc)
#   ROOT_DIR          (passed through to sbatch)
#   CONFIG_FILE       (passed through to sbatch)
#   EXTRA_EXPORT      (comma-separated additional --export vars)

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

declare -a MODES=(
  "off:mixed-bc-scale-all-off:off"
  "soft:mixed-bc-scale-all-soft:soft"
  "hard:mixed-bc-scale-all-hard:hard"
  "hard+soft:mixed-bc-scale-all-hard-soft:hard-soft"
)

echo "=========================================="
echo "Submitting BC constraint mode runs"
echo "Submit script: ${SUBMIT_SCRIPT}"
echo "Run prefix:    ${RUN_PREFIX}"
echo "=========================================="

for entry in "${MODES[@]}"; do
  IFS=':' read -r mode config_name run_suffix <<< "$entry"
  run_name="${RUN_PREFIX}-${run_suffix}"

  export_args="ALL,CONFIG_NAME=${config_name},RUN_NAME=${run_name}"
  if [[ -n "$EXTRA_EXPORT" ]]; then
    export_args="${export_args},${EXTRA_EXPORT}"
  fi

  echo ""
  echo "Submitting mode=${mode}"
  echo "  config=${config_name}"
  echo "  run_name=${run_name}"

  job_raw="$(sbatch --parsable --export="${export_args}" "$SUBMIT_SCRIPT")"
  job_id="${job_raw%%;*}"
  echo "  JOB_ID=${job_id}"

  {
    echo "timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "mode=${mode}"
    echo "config_name=${config_name}"
    echo "run_name=${run_name}"
    echo "job_id=${job_id}"
    echo "---"
  } >> results/constraints/bc_mode_submission_history.txt
done

echo ""
echo "Submitted all BC mode runs."

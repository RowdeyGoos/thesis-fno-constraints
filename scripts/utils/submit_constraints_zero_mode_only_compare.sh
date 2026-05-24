#!/usr/bin/env bash
set -euo pipefail

# Zero-mode-only comparison launcher (PDE residual off):
#   1) Hard zero-mode single run
#   2) Soft zero-mode sweep
#
# Usage:
#   bash scripts/utils/submit_constraints_zero_mode_only_compare.sh [agent_array]
#
# Example:
#   bash scripts/utils/submit_constraints_zero_mode_only_compare.sh
#   bash scripts/utils/submit_constraints_zero_mode_only_compare.sh 0-7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

AGENT_ARRAY="${1:-0-3}"
HARD_CONFIG_NAME="${HARD_CONFIG_NAME:-mixed-scale-all-constraints-zero-hard-only}"
HARD_RUN_NAME="${HARD_RUN_NAME:-pretrain-mixed-zero-hard-only}"
SOFT_SWEEP_YAML="${SOFT_SWEEP_YAML:-config/sweep_constraints_pretrain_zero_soft_only.yaml}"

echo "=========================================="
echo "Zero-Mode-Only Comparison (PDE off)"
echo "Hard config: ${HARD_CONFIG_NAME}"
echo "Hard run name: ${HARD_RUN_NAME}"
echo "Soft sweep: ${SOFT_SWEEP_YAML}"
echo "Soft agent array: ${AGENT_ARRAY}"
echo "=========================================="

echo "Submitting hard zero-mode single run..."
HARD_JOB_RAW="$(sbatch --parsable --export=ALL,CONFIG_NAME=${HARD_CONFIG_NAME},RUN_NAME=${HARD_RUN_NAME} scripts/slurm/pretrain/submit_pretrain_mixed.sh)"
HARD_JOB_ID="${HARD_JOB_RAW%%;*}"
echo "HARD_JOB_ID=${HARD_JOB_ID}"

echo ""
echo "Submitting soft zero-mode sweep..."
bash scripts/utils/submit_constraints_sweep.sh "${SOFT_SWEEP_YAML}" "${AGENT_ARRAY}"

echo ""
echo "Submitted zero-mode-only comparison jobs."
echo "  Hard single-run job id: ${HARD_JOB_ID}"
echo "  Soft sweep details are printed above (SWEEP_ID + AGENT_JOB_ID)."

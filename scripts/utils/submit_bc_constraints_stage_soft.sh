#!/usr/bin/env bash
set -euo pipefail

# BC Stage Soft launcher:
# soft BC sweep with BC mixed pretraining config.
#
# Usage:
#   bash scripts/utils/submit_bc_constraints_stage_soft.sh [agent_array]
#
# Example:
#   bash scripts/utils/submit_bc_constraints_stage_soft.sh
#   bash scripts/utils/submit_bc_constraints_stage_soft.sh 0-7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

AGENT_ARRAY="${1:-0-3}"
SWEEP_YAML="${SWEEP_YAML:-config/sweep_constraints_pretrain_bc_soft.yaml}"
LAUNCH_SCRIPT="scripts/utils/submit_bc_constraints_sweep.sh"

echo "=========================================="
echo "BC Stage Soft: BC soft-mode sweep"
echo "Sweep yaml:  ${SWEEP_YAML}"
echo "Agent array: ${AGENT_ARRAY}"
echo "=========================================="

bash "$LAUNCH_SCRIPT" "$SWEEP_YAML" "$AGENT_ARRAY"

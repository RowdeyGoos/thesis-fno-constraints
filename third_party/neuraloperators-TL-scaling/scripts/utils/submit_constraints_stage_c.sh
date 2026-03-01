#!/usr/bin/env bash
set -euo pipefail

# Stage C launcher:
# Soft zero-mode sweep using winner PDE method from Stage B.
#
# Usage:
#   bash scripts/utils/submit_constraints_stage_c.sh <penalty|al> [agent_array]
#
# Example:
#   bash scripts/utils/submit_constraints_stage_c.sh penalty
#   bash scripts/utils/submit_constraints_stage_c.sh al 0-7

usage() {
  cat <<'EOF'
Usage: bash scripts/utils/submit_constraints_stage_c.sh <penalty|al> [agent_array]
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

PDE_METHOD="$1"
AGENT_ARRAY="${2:-0-3}"
LAUNCH_SCRIPT="scripts/utils/submit_constraints_sweep.sh"

case "$PDE_METHOD" in
  penalty)
    SWEEP_YAML="config/sweep_constraints_pretrain_penalty_soft.yaml"
    ;;
  al)
    SWEEP_YAML="config/sweep_constraints_pretrain_al_soft.yaml"
    ;;
  *)
    echo "Error: unknown PDE method '$PDE_METHOD'. Use 'penalty' or 'al'."
    exit 2
    ;;
esac

echo "=========================================="
echo "Stage C: soft zero-mode sweep"
echo "PDE method: ${PDE_METHOD}"
echo "Sweep yaml: ${SWEEP_YAML}"
echo "Agent array: ${AGENT_ARRAY}"
echo "=========================================="

bash "$LAUNCH_SCRIPT" "$SWEEP_YAML" "$AGENT_ARRAY"


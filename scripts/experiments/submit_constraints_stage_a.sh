#!/usr/bin/env bash
set -euo pipefail

# Stage A launcher:
# PDE-only control (zero-mode off): penalty vs augmented Lagrangian.
#
# Usage:
#   bash scripts/experiments/submit_constraints_stage_a.sh [agent_array]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

AGENT_ARRAY="${1:-0-3}"
LAUNCH_SCRIPT="scripts/experiments/submit_constraints_sweep.sh"

echo "=========================================="
echo "Stage A: PDE-only control sweeps"
echo "Agent array: ${AGENT_ARRAY}"
echo "=========================================="

bash "$LAUNCH_SCRIPT" config/sweep_constraints_pretrain_penalty_pde_only.yaml "$AGENT_ARRAY"
echo ""
bash "$LAUNCH_SCRIPT" config/sweep_constraints_pretrain_al_pde_only.yaml "$AGENT_ARRAY"


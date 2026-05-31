#!/usr/bin/env bash
set -euo pipefail

# Submit a tiny sanity sweep to validate SLURM + W&B sweep plumbing.
#
# Usage:
#   bash scripts/experiments/submit_constraints_sanity_sweep.sh [agent_array]
#
# Example:
#   bash scripts/experiments/submit_constraints_sanity_sweep.sh
#   bash scripts/experiments/submit_constraints_sanity_sweep.sh 0-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

AGENT_ARRAY="${1:-0-1}"

export TRAIN_YAML="config/operators_poisson.yaml"
export TRAIN_CONFIG="poisson-smoke-k1_5-constraints"
export ROOT_DIR="experiments_sanity_sweep"
export RUN_PREFIX="constraints-sanity"

# Two runs total (lr grid with two values), so array 0-1 is enough.
export AGENT_COUNT_PER_TASK="${AGENT_COUNT_PER_TASK:-1}"

bash scripts/experiments/submit_constraints_sweep.sh config/sweep_constraints_sanity_smoke.yaml "$AGENT_ARRAY"

echo ""
echo "Sanity sweep submitted."
echo "Inspect local trial logs under: ${ROOT_DIR}/sweeps/<SWEEP_ID>/"

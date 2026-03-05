#!/usr/bin/env bash
set -euo pipefail

# Submit one BC constraints sweep end-to-end:
# 1) submit MODE=create job
# 2) wait until SWEEP_ID appears in create log
# 3) submit MODE=agent array job
#
# Usage:
#   bash scripts/utils/submit_bc_constraints_sweep.sh [sweep_yaml] [agent_array]
#
# Example:
#   bash scripts/utils/submit_bc_constraints_sweep.sh
#   bash scripts/utils/submit_bc_constraints_sweep.sh config/sweep_constraints_pretrain_bc_soft.yaml 0-7

usage() {
  cat <<'EOF'
Usage: bash scripts/utils/submit_bc_constraints_sweep.sh [sweep_yaml] [agent_array]

Arguments:
  sweep_yaml   Optional sweep yaml path relative to repo root
               (default: config/sweep_constraints_pretrain_bc_soft.yaml)
  agent_array  Optional SLURM array spec for agents (default: 0-3)

Environment overrides:
  TRAIN_YAML            Train config yaml (default: config/operators_mixed_bc.yaml)
  TRAIN_CONFIG          Config name inside TRAIN_YAML (default: mixed-bc-scale-all-soft)
  RUN_PREFIX            Sweep run prefix (default: bc-soft-sweep)
  ROOT_DIR              Training root dir (default: experiments)
  AGENT_COUNT_PER_TASK  Number of agents per array task (default: 1)
  SWEEP_WAIT_SECONDS    Max seconds to wait for SWEEP_ID (default: 600)
  SWEEP_POLL_SECONDS    Poll interval in seconds (default: 5)
EOF
}

if [[ $# -gt 2 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP_YAML="${1:-config/sweep_constraints_pretrain_bc_soft.yaml}"
AGENT_ARRAY="${2:-0-3}"
TRAIN_YAML="${TRAIN_YAML:-config/operators_mixed_bc.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-mixed-bc-scale-all-soft}"
RUN_PREFIX="${RUN_PREFIX:-bc-soft-sweep}"
ROOT_DIR="${ROOT_DIR:-experiments}"
AGENT_COUNT_PER_TASK="${AGENT_COUNT_PER_TASK:-1}"
SWEEP_WAIT_SECONDS="${SWEEP_WAIT_SECONDS:-600}"
SWEEP_POLL_SECONDS="${SWEEP_POLL_SECONDS:-5}"
SWEEP_SUBMIT_SCRIPT="${SWEEP_SUBMIT_SCRIPT:-scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh}"
SWEEP_JOB_NAME="${SWEEP_JOB_NAME:-neuralop-constraints-sweep}"

if [[ ! -f "$SWEEP_YAML" ]]; then
  echo "Error: sweep yaml not found: $SWEEP_YAML"
  exit 2
fi

if [[ ! -f "$SWEEP_SUBMIT_SCRIPT" ]]; then
  echo "Error: submit script not found: $SWEEP_SUBMIT_SCRIPT"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found on PATH"
  exit 2
fi

mkdir -p experiments results/constraints

echo "Submitting BC sweep create job for: $SWEEP_YAML"
CREATE_JOB_RAW="$(sbatch --parsable --export=ALL,MODE=create,SWEEP_YAML="$SWEEP_YAML",TRAIN_YAML="$TRAIN_YAML",TRAIN_CONFIG="$TRAIN_CONFIG",RUN_PREFIX="$RUN_PREFIX",ROOT_DIR="$ROOT_DIR" "$SWEEP_SUBMIT_SCRIPT")"
CREATE_JOB_ID="${CREATE_JOB_RAW%%;*}"
echo "CREATE_JOB_ID=$CREATE_JOB_ID"

LOG_GLOB="experiments/${SWEEP_JOB_NAME}-${CREATE_JOB_ID}-*.out"
SWEEP_ID=""
ELAPSED=0

while [[ -z "$SWEEP_ID" && "$ELAPSED" -lt "$SWEEP_WAIT_SECONDS" ]]; do
  SWEEP_ID="$( (grep -ho 'SWEEP_ID=.*' ${LOG_GLOB} 2>/dev/null || true) | tail -n1 | cut -d= -f2 )"
  if [[ -n "$SWEEP_ID" ]]; then
    break
  fi
  sleep "$SWEEP_POLL_SECONDS"
  ELAPSED=$((ELAPSED + SWEEP_POLL_SECONDS))
done

if [[ -z "$SWEEP_ID" ]]; then
  echo "Error: timed out waiting for SWEEP_ID in create logs."
  echo "Look at: ${LOG_GLOB}"
  exit 3
fi

echo "SWEEP_ID=$SWEEP_ID"
echo "Submitting BC agent array: $AGENT_ARRAY (AGENT_COUNT_PER_TASK=$AGENT_COUNT_PER_TASK)"

AGENT_JOB_RAW="$(sbatch --parsable --array="$AGENT_ARRAY" --export=ALL,MODE=agent,SWEEP_ID="$SWEEP_ID",SWEEP_YAML="$SWEEP_YAML",TRAIN_YAML="$TRAIN_YAML",TRAIN_CONFIG="$TRAIN_CONFIG",RUN_PREFIX="$RUN_PREFIX",ROOT_DIR="$ROOT_DIR",AGENT_COUNT_PER_TASK="$AGENT_COUNT_PER_TASK" "$SWEEP_SUBMIT_SCRIPT")"
AGENT_JOB_ID="${AGENT_JOB_RAW%%;*}"
echo "AGENT_JOB_ID=$AGENT_JOB_ID"

{
  echo "timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "sweep_yaml=${SWEEP_YAML}"
  echo "train_yaml=${TRAIN_YAML}"
  echo "train_config=${TRAIN_CONFIG}"
  echo "run_prefix=${RUN_PREFIX}"
  echo "create_job_id=${CREATE_JOB_ID}"
  echo "sweep_id=${SWEEP_ID}"
  echo "agent_job_id=${AGENT_JOB_ID}"
  echo "agent_array=${AGENT_ARRAY}"
  echo "agent_count_per_task=${AGENT_COUNT_PER_TASK}"
  echo "---"
} >> results/constraints/bc_sweep_submission_history.txt

echo ""
echo "Submitted BC sweep successfully."
echo "  SWEEP_ID:      ${SWEEP_ID}"
echo "  CREATE_JOB_ID: ${CREATE_JOB_ID}"
echo "  AGENT_JOB_ID:  ${AGENT_JOB_ID}"

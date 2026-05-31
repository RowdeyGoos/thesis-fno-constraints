#!/bin/bash
#SBATCH --job-name=neuralop-mixed-constraints-locked
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=21:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G
#SBATCH --array=0-2

# Submit the three locked mixed constrained pretraining runs as one array:
#   0 -> mixed-scale-all-constraints-penalty-pde-only
#   1 -> mixed-scale-all-constraints-al-pde-only-conservative
#   2 -> mixed-scale-all-constraints-zero-soft-only
#
# Usage:
#   sbatch scripts/slurm/pretrain/submit_pretrain_mixed_constraints_locked_array.sh

echo "=========================================="
echo "Locked Mixed Constraints Pretraining Array"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

configs=(
  "mixed-scale-all-constraints-penalty-pde-only:pretrain-mixed-penalty-pde-only"
  "mixed-scale-all-constraints-al-pde-only-conservative:pretrain-mixed-al-pde-only-conservative"
  "mixed-scale-all-constraints-zero-soft-only:pretrain-mixed-zero-soft-only"
)

IFS=':' read -r config_name run_name <<< "${configs[$SLURM_ARRAY_TASK_ID]}"

export CONFIG_FILE="${CONFIG_FILE:-config/operators_mixed.yaml}"
export CONFIG_NAME="${CONFIG_NAME:-$config_name}"
export RUN_NAME="${RUN_NAME:-$run_name}"

echo "Delegating to scripts/slurm/pretrain/submit_pretrain_mixed.sh"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo ""

exec bash scripts/slurm/pretrain/submit_pretrain_mixed.sh

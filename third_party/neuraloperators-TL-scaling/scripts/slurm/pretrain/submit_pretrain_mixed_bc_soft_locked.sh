#!/bin/bash
#SBATCH --job-name=neuralop-mixed-bc-soft-locked
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=21:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

# Submit the locked BC-soft mixed pretraining run.
#
# Usage:
#   sbatch scripts/slurm/pretrain/submit_pretrain_mixed_bc_soft_locked.sh

echo "=========================================="
echo "Locked Mixed BC Soft Pretraining"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export CONFIG_FILE="${CONFIG_FILE:-config/operators_mixed_bc.yaml}"
export CONFIG_NAME="${CONFIG_NAME:-mixed-bc-scale-all-soft}"
export RUN_NAME="${RUN_NAME:-pretrain-mixed-bc-soft}"

echo "Delegating to scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo ""

exec bash scripts/slurm/pretrain/submit_pretrain_mixed_bc.sh

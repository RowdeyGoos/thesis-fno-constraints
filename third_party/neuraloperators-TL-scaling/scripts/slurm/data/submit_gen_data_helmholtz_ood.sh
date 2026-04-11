#!/usr/bin/env bash
#SBATCH --job-name=data-helmholtz-ood
#SBATCH --output=experiments/%x-%A_%a.out
#SBATCH --error=experiments/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --time=4:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ ! -f "${PROJECT_DIR}/run_gen_data_ood.sh" ]; then
    echo "Error: expected project root with run_gen_data_ood.sh, got: ${PROJECT_DIR}"
    exit 1
fi

# Default OOD bins beyond the Helmholtz source/pretrain range omega in [1, 10].
RANGES_CSV="${OOD_RANGES:-10:15,15:20,20:25,25:30}"
IFS=',' read -r -a RANGES <<< "${RANGES_CSV}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${#RANGES[@]}" ]; then
    echo "Error: task index ${TASK_ID} is out of range for ${#RANGES[@]} Helmholtz OOD bins."
    echo "RANGES=${RANGES_CSV}"
    echo "Submit with a matching array, for example: sbatch --array=0-$((${#RANGES[@]} - 1)) $0"
    exit 1
fi

IFS=':' read -r RANGE_MIN RANGE_MAX <<< "${RANGES[${TASK_ID}]}"
if [ -z "${RANGE_MIN:-}" ] || [ -z "${RANGE_MAX:-}" ]; then
    echo "Error: invalid Helmholtz range '${RANGES[${TASK_ID}]}' (expected min:max)."
    exit 1
fi

export DATASET="helmholtz"
export RANGE_SET="transfer"
export DATA_ROOT="${DATA_ROOT:-data}"
export GENERATE_MIXED_FORMAT="${GENERATE_MIXED_FORMAT:-1}"
export TRANSFER_HELMHOLTZ_O1="${RANGE_MIN}"
export TRANSFER_HELMHOLTZ_O2="${RANGE_MAX}"

echo "=========================================="
echo "Helmholtz OOD data generation"
echo "Task ID:         ${TASK_ID}"
echo "Range list:      ${RANGES_CSV}"
echo "Selected omega:  [${TRANSFER_HELMHOLTZ_O1}, ${TRANSFER_HELMHOLTZ_O2}]"
echo "Data root:       ${DATA_ROOT}"
echo "Mixed copy:      ${GENERATE_MIXED_FORMAT}"
echo "Project dir:     ${PROJECT_DIR}"
echo "=========================================="

cd "${PROJECT_DIR}"
export SLURM_SUBMIT_DIR="${PROJECT_DIR}"
bash "${PROJECT_DIR}/scripts/slurm/data/submit_gen_data_ood.sh"

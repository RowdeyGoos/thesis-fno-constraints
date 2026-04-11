#!/usr/bin/env bash
#SBATCH --job-name=data-ood-all
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=11:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

if [ ! -f "${PROJECT_DIR}/run_gen_data_ood.sh" ]; then
    echo "Error: expected project root with run_gen_data_ood.sh, got: ${PROJECT_DIR}"
    echo "Submit from third_party/neuraloperators-TL-scaling or set SLURM_SUBMIT_DIR accordingly."
    exit 1
fi

cd "$PROJECT_DIR"

DATASET="${DATASET:-all}"
DATA_ROOT="${DATA_ROOT:-data}"
RANGE_SET="${RANGE_SET:-transfer}"
GENERATE_MIXED_FORMAT="${GENERATE_MIXED_FORMAT:-1}"

echo "=========================================="
echo "Normal OOD data generation starting"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node list:      ${SLURM_NODELIST:-local}"
echo "Project dir:    ${PROJECT_DIR}"
echo "Dataset:        ${DATASET}"
echo "Data root:      ${DATA_ROOT}"
echo "Range set:      ${RANGE_SET}"
echo "Mixed copies:   ${GENERATE_MIXED_FORMAT}"
echo "=========================================="

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
            # shellcheck disable=SC1091
            . /etc/profile.d/modules.sh
        elif [ -f /usr/share/Modules/init/bash ]; then
            # shellcheck disable=SC1091
            . /usr/share/Modules/init/bash
        fi
    fi

    if command -v module >/dev/null 2>&1; then
        if module load apptainer 2>/dev/null; then
            :
        elif module load singularity 2>/dev/null; then
            :
        fi
    fi
fi

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN="singularity"
else
    CONTAINER_BIN=""
fi

if [ -n "$CONTAINER_BIN" ]; then
    CONTAINER_PATH="${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif}"

    if [ ! -f "$CONTAINER_PATH" ]; then
        echo "Error: container not found at $CONTAINER_PATH"
        echo "Set CONTAINER_PATH=/path/to/neuraloperators.sif and resubmit."
        exit 1
    fi

    export PYTHONUNBUFFERED=1
    echo "Container runtime: ${CONTAINER_BIN}"
    echo "Container image:   ${CONTAINER_PATH}"
    "${CONTAINER_BIN}" exec --bind "${PROJECT_DIR}:/workspace" "$CONTAINER_PATH" \
        bash -lc 'cd /workspace && bash run_gen_data_ood.sh'
else
    echo "No apptainer/singularity runtime found; using host Python."
    export PYTHONUNBUFFERED=1
    bash run_gen_data_ood.sh
fi

echo "=========================================="
echo "Normal OOD data generation completed successfully."
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=eval-mixed-zs-bundle
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

# Submit mixed zero-shot bundle evaluation on the cluster.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_mixed_zeroshot_bundle.sh \
#     mixed-scale-all:pretrain-mixed:12173564 \
#     mixed-scale-all-constraints-zero-hard-only:pretrain-mixed-zero-hard-only:12239220
#
# Optional environment overrides:
#   ROOT_DIR=experiments_zeroshot
#   SUMMARY_FILE=results/constraints/zeroshot_bundle_my_run.tsv
#   BUNDLE_PYTHON_BIN=python
#   CONTAINER_PATH=/path/to/neuraloperators.sif
#   ZERO_SHOT_SPECS="spec1 spec2 ..."
#
# Notes:
#   - Positional arguments take precedence for specs. If none are passed, the script
#     falls back to ZERO_SHOT_SPECS from the environment.
#   - The underlying bundle script only supports standard 7-channel mixed checkpoints,
#     not mixed-bc-* checkpoints.

set -euo pipefail

echo "=========================================="
echo "Mixed Zero-Shot Bundle Eval"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node list:   ${SLURM_NODELIST:-local}"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=========================================="

ROOT_DIR="${ROOT_DIR:-experiments_zeroshot}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
BUNDLE_PYTHON_BIN="${BUNDLE_PYTHON_BIN:-}"

SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ] && [ -n "${ZERO_SHOT_SPECS:-}" ]; then
    read -r -a SPECS <<< "${ZERO_SHOT_SPECS}"
fi

if [ ${#SPECS[@]} -eq 0 ]; then
    echo "Error: provide at least one bundle spec."
    echo ""
    echo "Example:"
    echo "  sbatch scripts/slurm/eval/submit_eval_mixed_zeroshot_bundle.sh \\"
    echo "    mixed-scale-all-constraints-zero-hard-only:pretrain-mixed-zero-hard-only:12239220"
    echo ""
    echo "Or:"
    echo "  sbatch --export=ALL,ZERO_SHOT_SPECS=\"mixed-scale-all:pretrain-mixed:12173564\" \\"
    echo "    scripts/slurm/eval/submit_eval_mixed_zeroshot_bundle.sh"
    exit 2
fi

ZERO_SHOT_SPECS_STR="${SPECS[*]}"

CONTAINER_PATH="${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif}"

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: container not found at $CONTAINER_PATH"
    echo "Set CONTAINER_PATH=/path/to/neuraloperators.sif and resubmit."
    exit 1
fi

echo "Container image: ${CONTAINER_PATH}"
echo "Resolving container runtime (apptainer/singularity)..."

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
    echo "Error: neither 'apptainer' nor 'singularity' is available on PATH."
    exit 1
fi

echo "Container runtime: ${CONTAINER_BIN}"

echo "Root dir:           ${ROOT_DIR}"
if [ -n "${SUMMARY_FILE}" ]; then
    echo "Summary file:       ${SUMMARY_FILE}"
else
    echo "Summary file:       bundle default (timestamped TSV)"
fi
if [ -n "${BUNDLE_PYTHON_BIN}" ]; then
    echo "Bundle python:      ${BUNDLE_PYTHON_BIN}"
fi
echo "Bundle specs:       ${#SPECS[@]}"
for spec in "${SPECS[@]}"; do
    echo "  - ${spec}"
done

export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"

JOB_TMP_REL="tmp/${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="${WORKDIR}/${JOB_TMP_REL}"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "${WORKDIR}/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

mkdir -p experiments "$JOB_TMP_DIR"

export ROOT_DIR
export SUMMARY_FILE
export BUNDLE_PYTHON_BIN
export ZERO_SHOT_SPECS_STR

echo ""
echo "Launching zero-shot bundle inside container..."

if "${CONTAINER_BIN}" exec --nv --bind "${WORKDIR}:/workspace" "$CONTAINER_PATH" \
    bash -lc '
        set -euo pipefail

        cd /workspace
        export TMPDIR="/workspace/'"$JOB_TMP_REL"'"
        mkdir -p wandb wandb/cache wandb/tmp tmp experiments "$TMPDIR"

        declare -a specs=()
        read -r -a specs <<< "${ZERO_SHOT_SPECS_STR}"

        cmd=(bash scripts/utils/run_mixed_zeroshot_bundle.sh --root-dir "${ROOT_DIR}")
        if [[ -n "${SUMMARY_FILE}" ]]; then
            cmd+=(--summary-file "${SUMMARY_FILE}")
        fi
        if [[ -n "${BUNDLE_PYTHON_BIN}" ]]; then
            cmd+=(--python "${BUNDLE_PYTHON_BIN}")
        fi
        cmd+=("${specs[@]}")

        echo "Command:"
        printf "  %q" "${cmd[@]}"
        printf "\n\n"

        "${cmd[@]}"
    '; then
    status=0
else
    status=$?
fi

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Mixed zero-shot bundle completed successfully."
    if [ -n "${SUMMARY_FILE}" ]; then
        echo "Summary TSV: ${SUMMARY_FILE}"
    else
        echo "Summary TSV: see bundle output above for the timestamped path."
    fi
    echo "Eval root:   ${ROOT_DIR}"
else
    echo "Mixed zero-shot bundle FAILED with exit code ${status}."
fi
echo "=========================================="

exit $status

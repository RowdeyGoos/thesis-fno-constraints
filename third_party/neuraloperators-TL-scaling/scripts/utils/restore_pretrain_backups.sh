#!/bin/bash
# Restore pretraining experiment runs and matching Slurm logs from the cluster
# backup repo into the active repo.
#
# Default behavior is dry-run. Pass --apply to actually copy files.
#
# Usage:
#   bash scripts/utils/restore_pretrain_backups.sh
#   bash scripts/utils/restore_pretrain_backups.sh --apply
#
# Optional environment overrides:
#   SRC_REPO=/path/to/backup/repo
#   DST_REPO=/path/to/active/repo
#
# Notes:
# - Restores every `pretrain-*` run under the configured setup folders.
# - Also restores matching top-level `experiments/*.out` and `*.err` files.
# - Excludes nested `wandb/` run artifacts inside experiment directories.
# - In apply mode the script uses `--ignore-existing` by default so it will
#   fill in missing files without overwriting anything already restored.

set -euo pipefail

SRC_REPO="${SRC_REPO:-/tudelft.net/staff-umbrella/MscThesisRGoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling}"
DST_REPO="${DST_REPO:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling}"

SRC_EXP="${SRC_REPO}/experiments"
DST_EXP="${DST_REPO}/experiments"

MODE="dry-run"
OVERWRITE=0

usage() {
    cat <<EOF
Usage: bash scripts/utils/restore_pretrain_backups.sh [--apply] [--overwrite]

Options:
  --apply       Actually copy files. Default is dry-run.
  --overwrite   In apply mode, allow rsync to overwrite existing files.
  -h, --help    Show this help text.

Environment overrides:
  SRC_REPO      Source backup repo root.
  DST_REPO      Destination active repo root.

Current defaults:
  SRC_REPO=${SRC_REPO}
  DST_REPO=${DST_REPO}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            MODE="apply"
            ;;
        --overwrite)
            OVERWRITE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ ! -d "${SRC_EXP}" ]]; then
    echo "Error: source experiments directory does not exist: ${SRC_EXP}"
    exit 1
fi

mkdir -p "${DST_EXP}"

if [[ "${MODE}" == "dry-run" ]]; then
    RSYNC_OPTS=(-avP --dry-run)
else
    RSYNC_OPTS=(-avP)
    if [[ "${OVERWRITE}" -eq 0 ]]; then
        RSYNC_OPTS+=(--ignore-existing)
    fi
fi

configs=(
  helm-scale-o1_10
  mixed-scale-all
  poisson-scale-k1_5
  ad-scale-adr0p2_1
  mixed-scale-all-constraints-al-pde-only-conservative
  mixed-scale-all-constraints-penalty-pde-only
  mixed-scale-all-constraints-zero-hard-only
  mixed-scale-all-constraints-zero-soft-only
  mixed-bc-scale-all-hard
  mixed-bc-scale-all-off
  mixed-bc-scale-all-soft
)

declare -A synced_logs=()
restored_runs=0
restored_logs=0

sync_path() {
    local src="$1"
    local dst="$2"

    rsync "${RSYNC_OPTS[@]}" --exclude='wandb/' "$src" "$dst"
}

sync_log_if_present() {
    local log_path="$1"

    [[ -e "${log_path}" ]] || return 0
    if [[ -n "${synced_logs["$log_path"]:-}" ]]; then
        return 0
    fi

    synced_logs["$log_path"]=1
    sync_path "${log_path}" "${DST_EXP}/"
    restored_logs=$((restored_logs + 1))
}

sync_logs_for_run() {
    local run_name="$1"
    local jobid=""
    local taskid=""
    local log=""

    if [[ "${run_name}" =~ -([0-9]+)-([0-9]+)$ ]]; then
        jobid="${BASH_REMATCH[1]}"
        taskid="${BASH_REMATCH[2]}"

        for log in \
            "${SRC_EXP}"/*-"${jobid}"-"${taskid}".out \
            "${SRC_EXP}"/*-"${jobid}"-"${taskid}".err \
            "${SRC_EXP}"/*-"${jobid}"_"${taskid}".out \
            "${SRC_EXP}"/*-"${jobid}"_"${taskid}".err
        do
            sync_log_if_present "${log}"
        done

        if [[ "${taskid}" == "0" ]]; then
            for log in "${SRC_EXP}"/*-"${jobid}".out "${SRC_EXP}"/*-"${jobid}".err; do
                sync_log_if_present "${log}"
            done
        fi
        return 0
    fi

    if [[ "${run_name}" =~ -([0-9]+)$ ]]; then
        jobid="${BASH_REMATCH[1]}"
        for log in "${SRC_EXP}"/*-"${jobid}".out "${SRC_EXP}"/*-"${jobid}".err; do
            sync_log_if_present "${log}"
        done
    fi
}

echo "Source repo: ${SRC_REPO}"
echo "Destination repo: ${DST_REPO}"
echo "Mode: ${MODE}"
if [[ "${MODE}" == "apply" && "${OVERWRITE}" -eq 0 ]]; then
    echo "Overwrite behavior: ignore existing files"
elif [[ "${MODE}" == "apply" ]]; then
    echo "Overwrite behavior: allow overwrite"
fi
echo ""

shopt -s nullglob

for config in "${configs[@]}"; do
    src_cfg="${SRC_EXP}/expts/${config}"
    dst_cfg="${DST_EXP}/expts/${config}"

    if [[ ! -d "${src_cfg}" ]]; then
        echo "Skipping missing folder: ${src_cfg}"
        continue
    fi

    mkdir -p "${dst_cfg}"

    found_any_runs=0
    for src_run in "${src_cfg}"/pretrain-*; do
        [[ -d "${src_run}" ]] || continue
        found_any_runs=1

        run_name="$(basename "${src_run}")"
        echo "Restoring ${config}/${run_name}"

        sync_path "${src_run}/" "${dst_cfg}/${run_name}/"
        restored_runs=$((restored_runs + 1))
        sync_logs_for_run "${run_name}"
    done

    if [[ "${found_any_runs}" -eq 0 ]]; then
        echo "No pretrain runs found under: ${src_cfg}"
    fi
done

echo ""
echo "Summary:"
echo "  Pretrain run directories processed: ${restored_runs}"
echo "  Top-level log files processed: ${restored_logs}"

if [[ "${MODE}" == "dry-run" ]]; then
    echo ""
    echo "Dry-run only. Re-run with --apply to actually restore files."
fi

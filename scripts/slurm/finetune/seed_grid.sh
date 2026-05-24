#!/bin/bash

# Shared downstream seed-grid settings.
# Every downstream training experiment is repeated for seeds 0, 1, and 2.

SEED_VALUES=(0 1 2)
SEED_COUNT=${#SEED_VALUES[@]}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    echo "Submit this script with sbatch so the array directive is applied."
    exit 1
fi

SEED_EXPERIMENT_IDX=$((SLURM_ARRAY_TASK_ID / SEED_COUNT))
SEED_SLOT_IDX=$((SLURM_ARRAY_TASK_ID % SEED_COUNT))
SEED_VALUE="${SEED_VALUES[$SEED_SLOT_IDX]}"
SEED_RUN_SUFFIX="seed${SEED_VALUE}"
SEED_TRAIN_ARGS="--seed=${SEED_VALUE} --train_shuffle --random_train_subset --subset_seed=${SEED_VALUE}"

seed_task_in_allowlist() {
    local task_id="$1"
    local allowlist="$2"
    local entry start end

    if [ -z "${allowlist}" ]; then
        return 0
    fi

    IFS=',' read -ra entries <<< "${allowlist}"
    for entry in "${entries[@]}"; do
        entry="${entry//[[:space:]]/}"
        if [ -z "${entry}" ]; then
            continue
        fi

        if [[ "${entry}" == *-* ]]; then
            start="${entry%-*}"
            end="${entry#*-}"
            if [[ "${start}" =~ ^[0-9]+$ && "${end}" =~ ^[0-9]+$ ]] \
                && [ "${task_id}" -ge "${start}" ] \
                && [ "${task_id}" -le "${end}" ]; then
                return 0
            fi
        elif [ "${task_id}" = "${entry}" ]; then
            return 0
        fi
    done

    return 1
}

seed_skip_unless_task_allowed() {
    local allowlist="$1"
    local task_id="${2:-${SLURM_ARRAY_TASK_ID}}"
    local label="${3:-seed task}"

    if ! seed_task_in_allowlist "${task_id}" "${allowlist}"; then
        echo "Skipping ${label} task ${task_id}; allowed tasks: ${allowlist}"
        exit 0
    fi
}

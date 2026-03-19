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

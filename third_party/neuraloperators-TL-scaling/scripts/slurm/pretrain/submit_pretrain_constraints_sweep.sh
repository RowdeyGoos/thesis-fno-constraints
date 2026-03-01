#!/bin/bash
#SBATCH --job-name=neuralop-constraints-sweep
#SBATCH --output=experiments/%x-%A-%a.out
#SBATCH --error=experiments/%x-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --time=8:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --array=0-3

# Constraint sweep launcher for strict-FM mixed pretraining.
#
# Modes:
#   MODE=create : create a W&B sweep from SWEEP_YAML and print SWEEP_ID, then exit
#   MODE=agent  : run one or more sweep agents for an existing SWEEP_ID
#
# Examples:
#   sbatch --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
#     scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
#
#   sbatch --export=ALL,MODE=create,SWEEP_YAML=config/sweep_constraints_pretrain_al_pde_only.yaml \
#     scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh
#
#   sbatch --array=0-3 --export=ALL,MODE=agent,SWEEP_ID=<id>,SWEEP_YAML=config/sweep_constraints_pretrain_al_hard.yaml \
#     scripts/slurm/pretrain/submit_pretrain_constraints_sweep.sh

echo "=========================================="
echo "Constraint Sweep Launcher (Mixed Pretraining)"
echo "Mode: ${MODE:-agent}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "=========================================="

CONTAINER_PATH=${CONTAINER_PATH:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif}

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BIN=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BIN=singularity
else
    echo "Error: neither apptainer nor singularity found on PATH"
    exit 1
fi

export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp

MODE=${MODE:-agent}
SWEEP_YAML=${SWEEP_YAML:-config/sweep_constraints_pretrain_al_hard.yaml}
TRAIN_YAML=${TRAIN_YAML:-config/operators_mixed.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-mixed-scale-all}
ROOT_DIR=${ROOT_DIR:-experiments}
RUN_PREFIX=${RUN_PREFIX:-constraints-sweep}
AGENT_COUNT_PER_TASK=${AGENT_COUNT_PER_TASK:-1}

WORKDIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WORKDIR"

if [ ! -f "$SWEEP_YAML" ]; then
    echo "Error: Sweep YAML not found: $SWEEP_YAML"
    exit 1
fi

mkdir -p experiments
BIND="--bind $WORKDIR:/workspace"

JOB_TMP_REL="tmp/${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}"
JOB_TMP_DIR="$WORKDIR/$JOB_TMP_REL"

cleanup_tmp_dir() {
    rm -rf "$JOB_TMP_DIR"
    rmdir "$WORKDIR/tmp" 2>/dev/null || true
}
trap cleanup_tmp_dir EXIT

if [ "$MODE" = "create" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ]; then
        echo "MODE=create is only executed on array task 0. Exiting task ${SLURM_ARRAY_TASK_ID}."
        exit 0
    fi

    echo "Creating sweep from: $SWEEP_YAML"
    ENTITY_ARG=${WANDB_ENTITY:-}
    PROJECT_ARG=${WANDB_PROJECT:-}

    CREATE_CMD="python /workspace/scripts/utils/create_wandb_sweep.py --sweep_yaml=/workspace/${SWEEP_YAML}"
    if [ -n "$ENTITY_ARG" ]; then
        CREATE_CMD="$CREATE_CMD --entity=$ENTITY_ARG"
    fi
    if [ -n "$PROJECT_ARG" ]; then
        CREATE_CMD="$CREATE_CMD --project=$PROJECT_ARG"
    fi

    CREATE_OUTPUT=$($CONTAINER_BIN exec --nv $BIND "$CONTAINER_PATH" \
        bash -c "cd /workspace && mkdir -p wandb wandb/cache wandb/tmp tmp '$ROOT_DIR' '/workspace/$JOB_TMP_REL' && export TMPDIR='/workspace/$JOB_TMP_REL' && ${CREATE_CMD}")

    echo "$CREATE_OUTPUT"
    SWEEP_ID=$(echo "$CREATE_OUTPUT" | sed -n 's/^SWEEP_ID=//p' | tail -n 1)

    if [ -z "$SWEEP_ID" ]; then
        echo "Error: failed to parse SWEEP_ID from create output"
        exit 2
    fi

    echo ""
    echo "Created sweep successfully."
    echo "SWEEP_ID=$SWEEP_ID"
    exit 0
fi

if [ "$MODE" != "agent" ]; then
    echo "Error: MODE must be 'create' or 'agent' (got '$MODE')"
    exit 1
fi

if [ -z "${SWEEP_ID:-}" ]; then
    echo "Error: MODE=agent requires SWEEP_ID"
    echo "Hint: first run MODE=create, then pass SWEEP_ID in --export"
    exit 1
fi

echo "Using SWEEP_ID: $SWEEP_ID"
echo "Sweep YAML: $SWEEP_YAML"
echo "Train YAML/config: $TRAIN_YAML / $TRAIN_CONFIG"
echo "Agents per task: $AGENT_COUNT_PER_TASK"

for i in $(seq 1 "$AGENT_COUNT_PER_TASK"); do
    RUN_NUM="${RUN_PREFIX}-${SWEEP_ID}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}-a${i}"
    CMD="python /workspace/train.py \
      --yaml_config=/workspace/${TRAIN_YAML} \
      --config=${TRAIN_CONFIG} \
      --run_num=${RUN_NUM} \
      --root_dir=/workspace/${ROOT_DIR} \
      --sweep_id=${SWEEP_ID}"

    echo "------------------------------------------"
    echo "Launching sweep agent ${i}/${AGENT_COUNT_PER_TASK}"
    echo "Run num: $RUN_NUM"
    echo "Command: $CMD"

    $CONTAINER_BIN exec --nv $BIND "$CONTAINER_PATH" \
        bash -c 'cd /workspace && \
                 export TMPDIR="/workspace/'"$JOB_TMP_REL"'" && \
                 mkdir -p wandb wandb/cache wandb/tmp tmp '"$ROOT_DIR"' "$TMPDIR" && \
                 '"$CMD"

    status=$?
    if [ $status -ne 0 ]; then
        echo "Agent run failed with exit code $status"
        exit $status
    fi
done

echo ""
echo "=========================================="
echo "Sweep agent task completed successfully."
echo "Sweep ID: $SWEEP_ID"
echo "Results root: $ROOT_DIR/sweeps/$SWEEP_ID"
echo "=========================================="

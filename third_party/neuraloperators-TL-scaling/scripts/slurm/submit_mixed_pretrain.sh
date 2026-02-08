#!/bin/bash
#SBATCH --job-name=neuralop-mixed-pretrain
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=21:00:00
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G

# Mixed Dataset Pretraining Job
# This script trains on a mixed dataset combining Poisson, Advection-Diffusion, and Helmholtz
# The model learns from all three PDE systems simultaneously (multi-task learning)
#
# Prerequisites:
#   1. Generated data for all three PDE systems
#   2. Created mixed dataset using create_mixed_dataset.py
#   3. Computed scales for mixed dataset
#
# Usage:
#   sbatch scripts/slurm/submit_mixed_pretrain.sh

echo "=========================================="
echo "Mixed Dataset Pretraining (Multi-Task Learning)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Container location
CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Load Apptainer module
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

# -------- W&B config --------
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export WANDB_DIR=/workspace/wandb
export WANDB_DATA_DIR=/workspace/wandb
export WANDB_CACHE_DIR=/workspace/wandb/cache
export WANDB_TEMP_DIR=/workspace/wandb/tmp

cd "$SLURM_SUBMIT_DIR"

# Configuration
CONFIG_FILE="config/operators_mixed.yaml"
CONFIG_NAME="mixed-scale-all"
RUN_NAME="pretrain-mixed"

# Verify mixed dataset exists
MIXED_TRAIN="data/mixed/_train_mixed_32k.h5"
if [ ! -f "$MIXED_TRAIN" ]; then
    echo "Error: Mixed training dataset not found at: $MIXED_TRAIN"
    echo "Please create mixed dataset first using:"
    echo "  python utils/create_mixed_dataset.py \\"
    echo "    --poisson_path data/poisson/_train_k1_5_32k.h5 \\"
    echo "    --advdiff_path data/advdiff/_train_adr0.2_1_32k.h5 \\"
    echo "    --helmholtz_path data/helmholtz/_train_o1_10_32k.h5 \\"
    echo "    --output_path $MIXED_TRAIN"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Config name: $CONFIG_NAME"
echo "Run name: $RUN_NAME"
echo ""

# Create directories
mkdir -p experiments

# Bind directories
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

# Python command
CMD="python /workspace/train.py \
    --yaml_config=/workspace/$CONFIG_FILE \
    --config=$CONFIG_NAME \
    --run_num=${RUN_NAME}-${SLURM_JOB_ID}-0 \
    --root_dir=/workspace/experiments"

echo "Running mixed dataset pretraining..."
echo "Command: $CMD"
echo ""

# Run training
apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             mkdir -p wandb wandb/cache wandb/tmp tmp experiments && \
             export TMPDIR=/workspace/tmp && \
             '"$CMD"

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Mixed dataset pretraining completed successfully."
    echo ""
    echo "Next steps:"
    echo "  1. Note the checkpoint path for transfer learning"
    echo "  2. Use the checkpoint for fine-tuning on downstream tasks"
    echo "  3. Compare with single-domain pretraining results"
else
    echo "Mixed dataset pretraining FAILED with exit code $status."
fi
echo "Logs: experiments/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out / .err"
echo "Results: experiments/expts/$CONFIG_NAME/"
echo "=========================================="

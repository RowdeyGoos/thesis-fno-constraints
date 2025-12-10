#!/bin/bash
# Helper script to update checkpoint paths in transfer learning configs and scripts
# Usage: bash scripts/utils/update_checkpoint_path.sh <pretrain_job_id>

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/utils/update_checkpoint_path.sh <pretrain_job_id>"
    echo ""
    echo "Example:"
    echo "  bash scripts/utils/update_checkpoint_path.sh 12345"
    echo ""
    echo "This will update the checkpoint path to:"
    echo "  experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-12345-0/checkpoints/ckpt_best.tar"
    exit 1
fi

JOBID=$1
CHECKPOINT_PATH="experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-${JOBID}-0/checkpoints/ckpt_best.tar"

echo "Updating checkpoint path to: $CHECKPOINT_PATH"
echo ""

# Update config file
CONFIG_FILE="config/operators_poisson.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating $CONFIG_FILE..."
    sed -i "s|weights: 'experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-JOBID-0/checkpoints/ckpt_best.tar'|weights: '$CHECKPOINT_PATH'|g" "$CONFIG_FILE"
    echo "✓ Config file updated"
else
    echo "✗ Config file not found: $CONFIG_FILE"
fi

# Update SLURM script
SLURM_SCRIPT="scripts/slurm/submit_transfer_learning_array.sh"
if [ -f "$SLURM_SCRIPT" ]; then
    echo "Updating $SLURM_SCRIPT..."
    sed -i "s|PRETRAIN_CHECKPOINT=\"experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-JOBID-0/checkpoints/ckpt_best.tar\"|PRETRAIN_CHECKPOINT=\"$CHECKPOINT_PATH\"|g" "$SLURM_SCRIPT"
    echo "✓ SLURM script updated"
else
    echo "✗ SLURM script not found: $SLURM_SCRIPT"
fi

echo ""
echo "Done! You can now submit the transfer learning jobs:"
echo "  sbatch scripts/slurm/submit_transfer_learning_array.sh"

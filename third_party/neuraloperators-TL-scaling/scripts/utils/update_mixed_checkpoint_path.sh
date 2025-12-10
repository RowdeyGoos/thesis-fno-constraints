#!/bin/bash
# Helper script to update checkpoint paths for mixed dataset transfer learning configs
# Usage: bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>"
    echo ""
    echo "Example:"
    echo "  bash scripts/utils/update_mixed_checkpoint_path.sh 12345"
    echo ""
    echo "This will update the checkpoint path to:"
    echo "  experiments/expts/mixed-scale-all/pretrain-mixed-12345-0/checkpoints/ckpt_best.tar"
    exit 1
fi

JOBID=$1
CHECKPOINT_PATH="experiments/expts/mixed-scale-all/pretrain-mixed-${JOBID}-0/checkpoints/ckpt_best.tar"

echo "Updating mixed checkpoint path to: $CHECKPOINT_PATH"
echo ""

# Update config file
CONFIG_FILE="config/operators_mixed.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Updating $CONFIG_FILE..."
    sed -i "s|weights: 'experiments/expts/mixed-scale-all/pretrain-mixed-JOBID-0/checkpoints/ckpt_best.tar'|weights: '$CHECKPOINT_PATH'|g" "$CONFIG_FILE"
    echo "✓ Config file updated"
else
    echo "✗ Config file not found: $CONFIG_FILE"
fi

echo ""
echo "Done! You can now submit the mixed fine-tuning jobs:"
echo "  sbatch scripts/slurm/submit_mixed_finetune_array.sh"

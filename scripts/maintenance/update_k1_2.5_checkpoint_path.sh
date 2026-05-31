#!/bin/bash
# Update checkpoint paths in operators_poisson.yaml for k1_2.5 transfer learning
# Usage: bash scripts/maintenance/update_k1_2.5_checkpoint_path.sh <JOBID>

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/maintenance/update_k1_2.5_checkpoint_path.sh <JOBID>"
    echo ""
    echo "Example:"
    echo "  bash scripts/maintenance/update_k1_2.5_checkpoint_path.sh 12345678"
    echo ""
    echo "This will update the checkpoint path in config/operators_poisson.yaml"
    echo "for k1_2.5 fine-tuning experiments to use the specified pretrain job ID."
    exit 1
fi

JOBID=$1
CONFIG_FILE="config/operators_poisson.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Updating checkpoint paths in $CONFIG_FILE"
echo "New JOBID: $JOBID"

# Use sed to replace JOBID placeholder in k1_2.5 configs
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS sed requires backup extension
    sed -i '' "s|pretrain-poisson-k1_5-JOBID-0|pretrain-poisson-k1_5-${JOBID}-0|g" "$CONFIG_FILE"
else
    # Linux sed
    sed -i "s|pretrain-poisson-k1_5-JOBID-0|pretrain-poisson-k1_5-${JOBID}-0|g" "$CONFIG_FILE"
fi

if [ $? -eq 0 ]; then
    echo "✓ Successfully updated checkpoint paths"
    echo ""
    echo "New path: experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-${JOBID}-0/checkpoints/ckpt_best.tar"
    echo ""
    echo "Updated configs:"
    echo "  - poisson-k1_2.5-finetune-16"
    echo "  - poisson-k1_2.5-finetune-64"
    echo "  - poisson-k1_2.5-finetune-256"
    echo "  - poisson-k1_2.5-finetune-1k"
    echo "  - poisson-k1_2.5-finetune-4k"
    echo ""
    echo "You can now submit the transfer learning job array:"
    echo "  sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_array.sh"
else
    echo "✗ Error updating checkpoint paths"
    exit 1
fi

#!/bin/bash
# Batch convert Poisson k∈[1,2.5] datasets to mixed-compatible format
# This script converts train, val, and test datasets in place
#
# Usage:
#   bash scripts/utils/convert_k1_2.5_to_mixed_format.sh

echo "=============================================="
echo "Converting Poisson k∈[1,2.5] datasets"
echo "to mixed-compatible format (5-component tensors)"
echo "=============================================="
echo ""

# Dataset base names
DATASETS=(
    "data/poisson/_train_k1.0_2.5_32k.h5"
    "data/poisson/_val_k1.0_2.5_4k.h5"
    "data/poisson/_test_k1.0_2.5_4k.h5"
)

# Check if datasets exist
missing=0
for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$dataset" ]; then
        echo "❌ Missing: $dataset"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo ""
    echo "Error: One or more datasets not found."
    echo "Please generate them first using gen_data_poisson.py with k∈[1,2.5]:"
    echo ""
    echo "  python utils/gen_data_poisson.py --train_samples 32768 --val_samples 4096 --test_samples 4096 \\"
    echo "    --e1 1.0 --e2 2.5 --save_name k1_2.5"
    echo ""
    exit 1
fi

echo "All datasets found. Starting conversion..."
echo ""

# Convert each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Converting: $dataset"
    python utils/convert_poisson_to_mixed_format.py \
        --input_path "$dataset" \
        --in_place
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error converting $dataset"
        exit 1
    fi
    echo ""
done

echo "=============================================="
echo "✓ All datasets converted successfully!"
echo "=============================================="
echo ""
echo "Your Poisson k∈[1,2.5] datasets now have 5-component tensors"
echo "and are compatible with the mixed-pretrained model."
echo ""
echo "Backup files created:"
for dataset in "${DATASETS[@]}"; do
    echo "  ${dataset}.backup"
done
echo ""
echo "Next steps:"
echo "  1. Update checkpoint path:"
echo "     bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>"
echo "  2. Submit fine-tuning jobs:"
echo "     sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh"
echo ""

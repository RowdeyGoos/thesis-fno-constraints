#!/bin/bash
# Batch convert Advection-Diffusion adr∈[0.2,0.4] datasets to mixed-compatible format
# This script converts train, val, and test datasets in place
#
# Usage:
#   bash scripts/utils/convert_adr0p2_1_to_mixed_format.sh

echo "=============================================="
echo "Converting AdvDiff adr∈[0.2,0.4] datasets"
echo "to mixed-compatible format (6-component tensors)"
echo "=============================================="
echo ""

# Dataset base names
DATASETS=(
    "data/advdiff/_train_adr0.2_0.4_32k.h5"
    "data/advdiff/_val_adr0.2_0.4_4k.h5"
    "data/advdiff/_test_adr0.2_0.4_4k.h5"
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
    echo "Please generate them first using gen_data_advdiff.py with adr∈[0.2,0.4]:"
    echo ""
    echo "  python utils/gen_data_advdiff.py --train_samples 32768 --val_samples 4096 --test_samples 4096 \\"
    echo "    --e1 0.2 --e2 0.4 --save_name adr0p2_0p4"
    echo ""
    exit 1
fi

echo "All datasets found. Starting conversion..."
echo ""

# Convert each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Converting: $dataset"
    python utils/convert_ad_to_mixed_format.py \
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
echo "Your AdvDiff adr∈[0.2,0.4] datasets now have 6-component tensors"
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
echo "     sbatch scripts/slurm/submit_mixed_finetune_array.sh"
echo ""

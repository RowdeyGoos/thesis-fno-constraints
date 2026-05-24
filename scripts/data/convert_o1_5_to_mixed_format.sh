#!/bin/bash
# Batch convert Helmholtz o∈[1,5] datasets to mixed-compatible format
# This script converts train, val, and test datasets in place
#
# Usage:
#   bash scripts/data/convert_o1_5_to_mixed_format.sh

echo "=============================================="
echo "Converting Helmholtz o∈[1,5] datasets"
echo "to mixed-compatible format (6-component tensors)"
echo "=============================================="
echo ""

# Dataset base names
DATASETS=(
    "data/helmholtz/_train_o1_5_32k.h5"
    "data/helmholtz/_val_o1_5_4k.h5"
    "data/helmholtz/_test_o1_5_4k.h5"
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
    echo "Please generate them first using gen_data_helmholtz.py with o∈[1,5]:"
    echo ""
    echo "  python scripts/data/gen_data_helmholtz.py --ntrain 32768 --nval 4096 --ntest 4096 \\"
    echo "    --ng 144 --n 128 --sparse --datapath data/helmholtz --o1 1 --o2 5"
    echo ""
    exit 1
fi

echo "All datasets found. Starting conversion..."
echo ""

# Convert each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Converting: $dataset"
    python scripts/data/convert_helmholtz_to_mixed_format.py \
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
echo "Your Helmholtz o∈[1,5] datasets now have 6-component tensors"
echo "and are compatible with the mixed-pretrained model."
echo ""
echo "Backup files created:"
for dataset in "${DATASETS[@]}"; do
    echo "  ${dataset}.backup"
done
echo ""
echo "Next steps:"
echo "  1. Update checkpoint path(s) in config/operators_helmholtz.yaml"
echo "  2. Submit mixed fine-tuning jobs:"
echo "     sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh"
echo ""

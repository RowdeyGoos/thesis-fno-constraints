#!/bin/bash
# Batch evaluate all transfer learning experiments and generate comparison plot
# 
# This script evaluates models from three training approaches:
#   1. Fine-tuned from mixed-domain pretraining
#   2. Fine-tuned from k1_5 Poisson pretraining  
#   3. Trained from scratch
#
# Usage:
#   bash scripts/eval/eval_all_transfer_learning.sh

echo "=========================================="
echo "Transfer Learning Batch Evaluation"
echo "=========================================="

# Configuration
EXPERIMENT_DIR="experiments"
OUTPUT_DIR="results/transfer_learning_k1_2.5"
SIZES="16 64 256 1k 4k"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Step 1: Evaluating mixed-pretrained models..."
echo "----------------------------------------------"

python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_mixed.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-mixed-16 \
             poisson-k1_2.5-finetune-mixed-64 \
             poisson-k1_2.5-finetune-mixed-256 \
             poisson-k1_2.5-finetune-mixed-1k \
             poisson-k1_2.5-finetune-mixed-4k \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR/mixed" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "⚠️  Mixed evaluation failed, continuing..."
fi

echo ""
echo "Step 2: Evaluating k1_5-pretrained models..."
echo "----------------------------------------------"

python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-finetune-16 \
             poisson-k1_2.5-finetune-64 \
             poisson-k1_2.5-finetune-256 \
             poisson-k1_2.5-finetune-1k \
             poisson-k1_2.5-finetune-4k \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR/k1_5" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "⚠️  k1_5 evaluation failed, continuing..."
fi

echo ""
echo "Step 3: Evaluating from-scratch models..."
echo "----------------------------------------------"

python scripts/entrypoints/eval_transfer_learning.py \
    --yaml_config config/operators_poisson.yaml \
    --experiment_type poisson \
    --configs poisson-k1_2.5-scratch-16 \
             poisson-k1_2.5-scratch-64 \
             poisson-k1_2.5-scratch-256 \
             poisson-k1_2.5-scratch-1k \
             poisson-k1_2.5-scratch-4k \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR/scratch" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "⚠️  Scratch evaluation failed, continuing..."
fi

echo ""
echo "Step 4: Generating combined comparison plot..."
echo "----------------------------------------------"

python scripts/eval/plot_transfer_learning_comparison.py \
    --mixed_results "$OUTPUT_DIR/mixed/poisson_results.json" \
    --k1_5_results "$OUTPUT_DIR/k1_5/poisson_results.json" \
    --scratch_results "$OUTPUT_DIR/scratch/poisson_results.json" \
    --output_dir "$OUTPUT_DIR" \
    --output_name "poisson_k1_2p5_transfer_learning_comparison" \
    --title "Transfer Learning: Poisson k∈[1,2.5]"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - Mixed results:     $OUTPUT_DIR/mixed/"
echo "  - k1_5 results:      $OUTPUT_DIR/k1_5/"
echo "  - Scratch results:   $OUTPUT_DIR/scratch/"
echo "  - Combined plot:     $OUTPUT_DIR/poisson_k1_2p5_transfer_learning_comparison.png"
echo ""

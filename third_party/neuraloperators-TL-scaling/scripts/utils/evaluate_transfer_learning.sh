#!/bin/bash
# Quick evaluation script for transfer learning experiments

set -e

YAML_CONFIG="config/operators_poisson.yaml"
EXPERIMENT_DIR="experiments"
OUTPUT_DIR="results/transfer_learning"
EXPERIMENT_TYPE="poisson"

echo "=========================================="
echo "Transfer Learning Evaluation Script"
echo "=========================================="
echo ""

# Parse command line arguments
INCLUDE_MIXED=""
SKIP_EVAL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mixed)
            INCLUDE_MIXED="--include_mixed"
            echo "Including mixed-pretrained models"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL="--skip_evaluation"
            echo "Skipping evaluation, regenerating plots only"
            shift
            ;;
        --type)
            EXPERIMENT_TYPE="$2"
            echo "Experiment type: $EXPERIMENT_TYPE"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mixed] [--skip-eval] [--type poisson|advdiff|helmholtz]"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  YAML Config: $YAML_CONFIG"
echo "  Experiment Dir: $EXPERIMENT_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Experiment Type: $EXPERIMENT_TYPE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
echo ""

python eval_transfer_learning.py \
    --yaml_config "$YAML_CONFIG" \
    --experiment_type "$EXPERIMENT_TYPE" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $INCLUDE_MIXED \
    $SKIP_EVAL

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - ${EXPERIMENT_TYPE}_results.json"
echo "  - ${EXPERIMENT_TYPE}_transfer_learning.png"
echo "  - ${EXPERIMENT_TYPE}_transfer_learning.pdf"
echo ""

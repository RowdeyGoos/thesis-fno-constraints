#!/bin/bash
#SBATCH --job-name=eval-transfer-learning
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=2:00:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G

# Batch Evaluation: Transfer Learning Experiments
# 
# This job evaluates all trained models from the three approaches:
#   1. Fine-tuned from mixed-domain pretraining
#   2. Fine-tuned from k1_5 Poisson pretraining
#   3. Trained from scratch
#   4. Zero-shot from pretrained checkpoints (no downstream training)
#
# And generates a comparison plot showing test error vs number of training samples.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_poisson_transfer.sh

echo "=========================================="
echo "Transfer Learning Batch Evaluation"
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

cd "$SLURM_SUBMIT_DIR"

# Configuration
EXPERIMENT_DIR="experiments"
OUTPUT_DIR="results/transfer_learning_k1_2.5"

# Create output directories
mkdir -p "$OUTPUT_DIR/mixed"
mkdir -p "$OUTPUT_DIR/k1_5"
mkdir -p "$OUTPUT_DIR/scratch"
mkdir -p experiments/logs

# Bind directories
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

echo ""
echo "Step 1: Evaluating mixed-pretrained models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_mixed.yaml \
                --experiment_type poisson \
                --configs poisson-k1_2.5-zeroshot-mixed \
                         poisson-k1_2.5-finetune-mixed-16 \
                         poisson-k1_2.5-finetune-mixed-64 \
                         poisson-k1_2.5-finetune-mixed-256 \
                         poisson-k1_2.5-finetune-mixed-1k \
                         poisson-k1_2.5-finetune-mixed-4k \
                         poisson-k1_2.5-finetune-mixed-8k \
                         poisson-k1_2.5-finetune-mixed-16k \
                         poisson-k1_2.5-finetune-mixed-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_k1_2.5/mixed \
                --device cuda:0'

echo ""
echo "Step 2: Evaluating k1_5-pretrained models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_poisson.yaml \
                --experiment_type poisson \
                --configs poisson-k1_2.5-zeroshot \
                         poisson-k1_2.5-finetune-16 \
                         poisson-k1_2.5-finetune-64 \
                         poisson-k1_2.5-finetune-256 \
                         poisson-k1_2.5-finetune-1k \
                         poisson-k1_2.5-finetune-4k \
                         poisson-k1_2.5-finetune-8k \
                         poisson-k1_2.5-finetune-16k \
                         poisson-k1_2.5-finetune-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_k1_2.5/k1_5 \
                --device cuda:0'

echo ""
echo "Step 3: Evaluating from-scratch models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_poisson.yaml \
                --experiment_type poisson \
                --configs poisson-k1_2.5-scratch-zeroshot \
                         poisson-k1_2.5-scratch-16 \
                         poisson-k1_2.5-scratch-64 \
                         poisson-k1_2.5-scratch-256 \
                         poisson-k1_2.5-scratch-1k \
                         poisson-k1_2.5-scratch-4k \
                         poisson-k1_2.5-scratch-8k \
                         poisson-k1_2.5-scratch-16k \
                         poisson-k1_2.5-scratch-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_k1_2.5/scratch \
                --device cuda:0'

echo ""
echo "Step 4: Generating combined comparison plot..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python utils/plot_transfer_learning_comparison.py \
                --mixed_results results/transfer_learning_k1_2.5/mixed/poisson_results.json \
                --k1_5_results results/transfer_learning_k1_2.5/k1_5/poisson_results.json \
                --scratch_results results/transfer_learning_k1_2.5/scratch/poisson_results.json \
                --output_dir results/transfer_learning_k1_2.5 \
                --title "Transfer Learning: Poisson k∈[1,2.5]"'

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: results/transfer_learning_k1_2.5/"
    echo "  - Mixed results:     results/transfer_learning_k1_2.5/mixed/"
    echo "  - k1_5 results:      results/transfer_learning_k1_2.5/k1_5/"
    echo "  - Scratch results:   results/transfer_learning_k1_2.5/scratch/"
    echo "  - Comparison plot:   results/transfer_learning_k1_2.5/transfer_learning_comparison.png"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

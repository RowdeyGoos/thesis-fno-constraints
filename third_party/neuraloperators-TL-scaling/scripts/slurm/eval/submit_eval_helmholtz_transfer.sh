#!/bin/bash
#SBATCH --job-name=eval-tl-helmholtz
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=1:15:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

# Batch Evaluation: Transfer Learning (Helmholtz)
#
# This job evaluates trained Helmholtz transfer learning experiments for:
#   1) Fine-tuned from Mixed-Domain pretraining
#   2) Fine-tuned from Helmholtz pretraining (o∈[1,10])
#   3) Trained from scratch
#   4) Zero-shot from pretrained checkpoints (no downstream training)
#
# It then generates a combined comparison plot.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_helmholtz_transfer.sh

echo "=========================================="
echo "Transfer Learning Batch Evaluation (Helmholtz)"
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
OUTPUT_DIR="results/transfer_learning_helmholtz_o1_5"

# Create output directories
mkdir -p "$OUTPUT_DIR/mixed"
mkdir -p "$OUTPUT_DIR/helmholtz"
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
                --yaml_config config/operators_helmholtz.yaml \
                --experiment_type helmholtz \
                --configs helm-o1_5-zeroshot-mixed \
                         helm-o1_5-finetune-mixed-16 \
                         helm-o1_5-finetune-mixed-64 \
                         helm-o1_5-finetune-mixed-256 \
                         helm-o1_5-finetune-mixed-1k \
                         helm-o1_5-finetune-mixed-4k \
                         helm-o1_5-finetune-mixed-8k \
                         helm-o1_5-finetune-mixed-16k \
                         helm-o1_5-finetune-mixed-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_helmholtz_o1_5/mixed \
                --aggregate_runs \
                --run_pattern "*-seed*" \
                --device cuda:0'

echo ""
echo "Step 2: Evaluating Helmholtz-pretrained models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_helmholtz.yaml \
                --experiment_type helmholtz \
                --configs helm-o1_5-zeroshot \
                         helm-o1_5-finetune-16 \
                         helm-o1_5-finetune-64 \
                         helm-o1_5-finetune-256 \
                         helm-o1_5-finetune-1k \
                         helm-o1_5-finetune-4k \
                         helm-o1_5-finetune-8k \
                         helm-o1_5-finetune-16k \
                         helm-o1_5-finetune-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_helmholtz_o1_5/helmholtz \
                --aggregate_runs \
                --run_pattern "*-seed*" \
                --device cuda:0'

echo ""
echo "Step 3: Evaluating from-scratch models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_helmholtz.yaml \
                --experiment_type helmholtz \
                --configs helm-o1_5-scratch-zeroshot \
                         helm-o1_5-scratch-16 \
                         helm-o1_5-scratch-64 \
                         helm-o1_5-scratch-256 \
                         helm-o1_5-scratch-1k \
                         helm-o1_5-scratch-4k \
                         helm-o1_5-scratch-8k \
                         helm-o1_5-scratch-16k \
                         helm-o1_5-scratch-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_helmholtz_o1_5/scratch \
                --aggregate_runs \
                --run_pattern "*-seed*" \
                --device cuda:0'

echo ""
echo "Step 4: Generating combined comparison plot..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python utils/plot_transfer_learning_comparison.py \
                --mixed_results results/transfer_learning_helmholtz_o1_5/mixed/helmholtz_results.json \
                --k1_5_results results/transfer_learning_helmholtz_o1_5/helmholtz/helmholtz_results.json \
                --scratch_results results/transfer_learning_helmholtz_o1_5/scratch/helmholtz_results.json \
                --output_dir results/transfer_learning_helmholtz_o1_5 \
                --title "Transfer Learning: Helmholtz o∈[1,5]"'

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - Mixed results:      $OUTPUT_DIR/mixed/"
    echo "  - Helmholtz results:  $OUTPUT_DIR/helmholtz/"
    echo "  - Scratch results:    $OUTPUT_DIR/scratch/"
    echo "  - Comparison plot:    $OUTPUT_DIR/transfer_learning_comparison.png"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

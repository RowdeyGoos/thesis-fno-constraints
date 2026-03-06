#!/bin/bash
#SBATCH --job-name=eval-tl-advdiff
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

# Batch Evaluation: Transfer Learning (Advection-Diffusion)
#
# This job evaluates trained AdvDiff transfer learning experiments for:
#   1) Fine-tuned from Mixed-Domain pretraining
#   2) Fine-tuned from AdvDiff pretraining (adr∈[0.2,1])
#   3) Trained from scratch
#   4) Zero-shot from pretrained checkpoints (no downstream training)
#
# It then generates a combined comparison plot.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_advdiff_transfer.sh

echo "=========================================="
echo "Transfer Learning Batch Evaluation (AdvDiff)"
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
OUTPUT_DIR="results/transfer_learning_advdiff_adr0.2_0.4"

# Create output directories
mkdir -p "$OUTPUT_DIR/mixed"
mkdir -p "$OUTPUT_DIR/advdiff"
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
                --yaml_config config/operators_ad.yaml \
                --experiment_type advdiff \
                --configs ad-adr0p2_0p4-zeroshot-mixed \
                         ad-adr0p2_0p4-finetune-mixed-16 \
                         ad-adr0p2_0p4-finetune-mixed-64 \
                         ad-adr0p2_0p4-finetune-mixed-256 \
                         ad-adr0p2_0p4-finetune-mixed-1k \
                         ad-adr0p2_0p4-finetune-mixed-4k \
                         ad-adr0p2_0p4-finetune-mixed-8k \
                         ad-adr0p2_0p4-finetune-mixed-16k \
                         ad-adr0p2_0p4-finetune-mixed-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_advdiff_adr0.2_0.4/mixed \
                --device cuda:0'

echo ""
echo "Step 2: Evaluating AdvDiff-pretrained models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_ad.yaml \
                --experiment_type advdiff \
                --configs ad-adr0p2_0p4-zeroshot \
                         ad-adr0p2_0p4-finetune-16 \
                         ad-adr0p2_0p4-finetune-64 \
                         ad-adr0p2_0p4-finetune-256 \
                         ad-adr0p2_0p4-finetune-1k \
                         ad-adr0p2_0p4-finetune-4k \
                         ad-adr0p2_0p4-finetune-8k \
                         ad-adr0p2_0p4-finetune-16k \
                         ad-adr0p2_0p4-finetune-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_advdiff_adr0.2_0.4/advdiff \
                --device cuda:0'

echo ""
echo "Step 3: Evaluating from-scratch models (including zero-shot)..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_transfer_learning.py \
                --yaml_config config/operators_ad.yaml \
                --experiment_type advdiff \
                --configs ad-adr0p2_0p4-scratch-zeroshot \
                         ad-adr0p2_0p4-scratch-16 \
                         ad-adr0p2_0p4-scratch-64 \
                         ad-adr0p2_0p4-scratch-256 \
                         ad-adr0p2_0p4-scratch-1k \
                         ad-adr0p2_0p4-scratch-4k \
                         ad-adr0p2_0p4-scratch-8k \
                         ad-adr0p2_0p4-scratch-16k \
                         ad-adr0p2_0p4-scratch-32k \
                --experiment_dir experiments \
                --output_dir results/transfer_learning_advdiff_adr0.2_0.4/scratch \
                --device cuda:0'

echo ""
echo "Step 4: Generating combined comparison plot..."
echo "----------------------------------------------"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python utils/plot_transfer_learning_comparison.py \
                --mixed_results results/transfer_learning_advdiff_adr0.2_0.4/mixed/advdiff_results.json \
                --k1_5_results results/transfer_learning_advdiff_adr0.2_0.4/advdiff/advdiff_results.json \
                --scratch_results results/transfer_learning_advdiff_adr0.2_0.4/scratch/advdiff_results.json \
                --output_dir results/transfer_learning_advdiff_adr0.2_0.4 \
                --title "Transfer Learning: AdvDiff adr∈[0.2,0.4]"'

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - Mixed results:     $OUTPUT_DIR/mixed/"
    echo "  - AdvDiff results:   $OUTPUT_DIR/advdiff/"
    echo "  - Scratch results:   $OUTPUT_DIR/scratch/"
    echo "  - Comparison plot:   $OUTPUT_DIR/transfer_learning_comparison.png"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

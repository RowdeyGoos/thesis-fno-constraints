#!/bin/bash
#SBATCH --job-name=eval-ood-helmholtz-constraints
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

# Batch Evaluation: Helmholtz OOD constraint comparison
#
# Evaluates the OOD Helmholtz runs across all default OOD bins for:
#   1) Mixed baseline
#   2) Mixed zero-hard pretraining
#   3) Mixed zero-soft pretraining
#   4) Mixed PDE-penalty pretraining
#
# Each variant is evaluated at the 256-sample and 4K-sample downstream budgets.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_helmholtz_ood_constraints.sh

echo "=========================================="
echo "Helmholtz OOD Constraint Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/third_party/neuraloperators-TL-scaling/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="results/ood_constraints_helmholtz"
mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/logs

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_ood_constraint_comparison.py \
                --experiment_type helmholtz \
                --experiment_dir experiments \
                --output_dir results/ood_constraints_helmholtz \
                --aggregate_runs \
                --run_pattern "*-seed*" \
                --device cuda:0'

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - JSON:             $OUTPUT_DIR/helmholtz_ood_constraints_results.json"
    echo "  - Comparison plot:  $OUTPUT_DIR/helmholtz_ood_constraints_comparison.png"
    echo "  - PDF plot:         $OUTPUT_DIR/helmholtz_ood_constraints_comparison.pdf"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

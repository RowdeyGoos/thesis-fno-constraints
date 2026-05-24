#!/bin/bash
#SBATCH --job-name=eval-ood-poisson
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=1:45:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

# Batch Evaluation: Poisson OOD degradation
#
# Evaluates the mixed-pretrained and scratch OOD runs across all Poisson OOD bins
# and generates one plot with OOD datasets on the x-axis.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_poisson_ood.sh

echo "=========================================="
echo "Poisson OOD Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

CONTAINER_PATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/rgoos/thesis-fno-constraints/containers/neuraloperators.sif

if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

module load apptainer 2>/dev/null || module load singularity 2>/dev/null

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="results/ood_poisson"
mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/logs

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python scripts/entrypoints/eval_ood_comparison.py \
                --yaml_config config/operators_poisson.yaml \
                --experiment_type poisson \
                --experiment_dir experiments \
                --output_dir results/ood_poisson \
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
    echo "  - JSON:             $OUTPUT_DIR/poisson_ood_results.json"
    echo "  - Comparison plot:  $OUTPUT_DIR/poisson_ood_comparison.png"
    echo "  - PDF plot:         $OUTPUT_DIR/poisson_ood_comparison.pdf"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

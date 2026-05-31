#!/bin/bash
#SBATCH --job-name=eval-tl-poisson-constraints
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

# Batch Evaluation: Transfer Learning Constraint Comparison (Poisson)
#
# This job evaluates trained Poisson transfer-learning experiments for:
#   1) Fine-tuned from the mixed baseline checkpoint
#   2) Fine-tuned from mixed zero-hard pretraining
#   3) Fine-tuned from mixed zero-soft pretraining
#   4) Fine-tuned from mixed PDE-penalty pretraining
#
# It then generates a combined comparison plot.
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_poisson_transfer_constraints.sh

echo "=========================================="
echo "Transfer Learning Constraint Evaluation (Poisson)"
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

OUTPUT_DIR="results/transfer_learning_constraints_poisson_k1_2.5"

mkdir -p "$OUTPUT_DIR/mixed"
mkdir -p "$OUTPUT_DIR/mixed-zero-hard"
mkdir -p "$OUTPUT_DIR/mixed-zero-soft"
mkdir -p "$OUTPUT_DIR/mixed-penalty-pde"
mkdir -p experiments/logs

BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

run_eval() {
    local series_name="$1"
    local yaml_config="$2"
    shift 2
    local configs=("$@")
    local cmd=(
        python scripts/entrypoints/eval_transfer_learning.py
        --yaml_config "$yaml_config"
        --experiment_type poisson
        --configs
        "${configs[@]}"
        --experiment_dir experiments
        --output_dir "$OUTPUT_DIR/$series_name"
        --aggregate_runs
        --run_pattern "*-seed*"
        --device cuda:0
    )

    echo ""
    echo "Evaluating ${series_name}..."
    echo "----------------------------------------------"

    apptainer exec --nv $BIND "$CONTAINER_PATH" \
        bash -lc "cd /workspace && $(printf '%q ' "${cmd[@]}")"
}

plot_comparison() {
    local cmd=(
        python scripts/eval/plot_transfer_learning_comparison.py
        --series "mixed=results/transfer_learning_constraints_poisson_k1_2.5/mixed/poisson_results.json"
        --series "mixed-zero-hard=results/transfer_learning_constraints_poisson_k1_2.5/mixed-zero-hard/poisson_results.json"
        --series "mixed-zero-soft=results/transfer_learning_constraints_poisson_k1_2.5/mixed-zero-soft/poisson_results.json"
        --series "mixed-penalty-pde=results/transfer_learning_constraints_poisson_k1_2.5/mixed-penalty-pde/poisson_results.json"
        --output_dir "$OUTPUT_DIR"
        --output_name "poisson_k1_2p5_transfer_learning_constraints_comparison"
        --title "Transfer Learning Constraints: Poisson k in [1,2.5]"
    )

    echo ""
    echo "Generating combined comparison plot..."
    echo "----------------------------------------------"

    apptainer exec --nv $BIND "$CONTAINER_PATH" \
        bash -lc "cd /workspace && $(printf '%q ' "${cmd[@]}")"
}

mixed_configs=(
    poisson-k1_2.5-zeroshot-mixed
    poisson-k1_2.5-finetune-mixed-16
    poisson-k1_2.5-finetune-mixed-64
    poisson-k1_2.5-finetune-mixed-256
    poisson-k1_2.5-finetune-mixed-1k
    poisson-k1_2.5-finetune-mixed-4k
    poisson-k1_2.5-finetune-mixed-8k
    poisson-k1_2.5-finetune-mixed-16k
    poisson-k1_2.5-finetune-mixed-32k
)

zero_hard_configs=(
    poisson-k1_2.5-zeroshot-mixed-zero-hard
    poisson-k1_2.5-finetune-mixed-zero-hard-16
    poisson-k1_2.5-finetune-mixed-zero-hard-64
    poisson-k1_2.5-finetune-mixed-zero-hard-256
    poisson-k1_2.5-finetune-mixed-zero-hard-1k
    poisson-k1_2.5-finetune-mixed-zero-hard-4k
    poisson-k1_2.5-finetune-mixed-zero-hard-8k
    poisson-k1_2.5-finetune-mixed-zero-hard-16k
    poisson-k1_2.5-finetune-mixed-zero-hard-32k
)

zero_soft_configs=(
    poisson-k1_2.5-zeroshot-mixed-zero-soft
    poisson-k1_2.5-finetune-mixed-zero-soft-16
    poisson-k1_2.5-finetune-mixed-zero-soft-64
    poisson-k1_2.5-finetune-mixed-zero-soft-256
    poisson-k1_2.5-finetune-mixed-zero-soft-1k
    poisson-k1_2.5-finetune-mixed-zero-soft-4k
    poisson-k1_2.5-finetune-mixed-zero-soft-8k
    poisson-k1_2.5-finetune-mixed-zero-soft-16k
    poisson-k1_2.5-finetune-mixed-zero-soft-32k
)

penalty_pde_configs=(
    poisson-k1_2.5-zeroshot-mixed-penalty-pde
    poisson-k1_2.5-finetune-mixed-penalty-pde-16
    poisson-k1_2.5-finetune-mixed-penalty-pde-64
    poisson-k1_2.5-finetune-mixed-penalty-pde-256
    poisson-k1_2.5-finetune-mixed-penalty-pde-1k
    poisson-k1_2.5-finetune-mixed-penalty-pde-4k
    poisson-k1_2.5-finetune-mixed-penalty-pde-8k
    poisson-k1_2.5-finetune-mixed-penalty-pde-16k
    poisson-k1_2.5-finetune-mixed-penalty-pde-32k
)

run_eval "mixed" "config/operators_poisson.yaml" "${mixed_configs[@]}" || exit $?
run_eval "mixed-zero-hard" "config/operators_poisson_mixed_constraints.yaml" "${zero_hard_configs[@]}" || exit $?
run_eval "mixed-zero-soft" "config/operators_poisson_mixed_constraints.yaml" "${zero_soft_configs[@]}" || exit $?
run_eval "mixed-penalty-pde" "config/operators_poisson_mixed_constraints.yaml" "${penalty_pde_configs[@]}" || exit $?

plot_comparison
status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - Mixed baseline:     $OUTPUT_DIR/mixed/"
    echo "  - Mixed zero-hard:    $OUTPUT_DIR/mixed-zero-hard/"
    echo "  - Mixed zero-soft:    $OUTPUT_DIR/mixed-zero-soft/"
    echo "  - Mixed PDE penalty:  $OUTPUT_DIR/mixed-penalty-pde/"
    echo "  - Comparison plot:    $OUTPUT_DIR/poisson_k1_2p5_transfer_learning_constraints_comparison.png"
else
    echo "Evaluation FAILED with exit code $status"
fi
echo "=========================================="

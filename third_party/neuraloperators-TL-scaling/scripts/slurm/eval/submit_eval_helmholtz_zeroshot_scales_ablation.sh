#!/bin/bash
#SBATCH --job-name=eval-helm-zs-scales
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --time=0:45:00
#SBATCH --partition=insy,general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=8G

# Evaluate Helmholtz zero-shot performance with a scales-path ablation:
#   1) downstream scales (current transfer-eval behavior)
#   2) source pretraining scales
#
# Covers both:
#   - single-domain Helmholtz pretraining -> Helmholtz o∈[1,5]
#   - mixed-domain pretraining -> Helmholtz o∈[1,5] (mixed format)
#
# Usage:
#   sbatch scripts/slurm/eval/submit_eval_helmholtz_zeroshot_scales_ablation.sh

echo "=========================================="
echo "Helmholtz Zero-Shot Scales Ablation Eval"
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

# Output locations
OUTPUT_DIR="results/transfer_learning_helmholtz_o1_5/zeroshot_scales_ablation"
EVAL_ROOT_DIR="./evaluation_tmp_zeroshot_scales_ablation"

mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/logs

# Bind repo root into container
BIND="--bind $SLURM_SUBMIT_DIR:/workspace"

echo ""
echo "Running zero-shot scales ablation..."
echo "  Output dir: $OUTPUT_DIR"
echo "  Eval tmp:   $EVAL_ROOT_DIR"
echo ""

apptainer exec --nv $BIND "$CONTAINER_PATH" \
    bash -c 'cd /workspace && \
             python eval_helmholtz_zeroshot_scales_ablation.py \
                --helm_yaml config/operators_helmholtz.yaml \
                --mixed_yaml config/operators_mixed.yaml \
                --output_dir '"$OUTPUT_DIR"' \
                --eval_root_dir '"$EVAL_ROOT_DIR"' \
                --device cuda:0'

status=$?

echo ""
echo "=========================================="
if [ $status -eq 0 ]; then
    echo "Ablation evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/"
    echo "JSON: $OUTPUT_DIR/helmholtz_zeroshot_scales_ablation.json"
else
    echo "Ablation evaluation FAILED with exit code $status"
fi
echo "=========================================="

exit $status

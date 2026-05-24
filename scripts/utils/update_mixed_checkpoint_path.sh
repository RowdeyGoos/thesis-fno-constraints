#!/bin/bash
# Helper script to update mixed-pretraining checkpoint paths in transfer configs.
#
# Usage:
#   bash scripts/utils/update_mixed_checkpoint_path.sh <job_id>
#
# Optional environment overrides:
#   PRETRAIN_CONFIG_NAME  (default: mixed-scale-all)
#   PRETRAIN_RUN_PREFIX   (default: pretrain-mixed)
#   PRETRAIN_RUN_INDEX    (default: 0)
#
# Example:
#   PRETRAIN_CONFIG_NAME=mixed-scale-all-constraints-al-hard \\
#   PRETRAIN_RUN_PREFIX=pretrain-mixed-al-hard \\
#   bash scripts/utils/update_mixed_checkpoint_path.sh 12345

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/utils/update_mixed_checkpoint_path.sh <mixed_pretrain_job_id>"
    echo ""
    echo "Optional env vars:"
    echo "  PRETRAIN_CONFIG_NAME (default: mixed-scale-all)"
    echo "  PRETRAIN_RUN_PREFIX  (default: pretrain-mixed)"
    echo "  PRETRAIN_RUN_INDEX   (default: 0)"
    echo ""
    echo "Example:"
    echo "  bash scripts/utils/update_mixed_checkpoint_path.sh 12345"
    echo "  PRETRAIN_CONFIG_NAME=mixed-scale-all-constraints-al-hard PRETRAIN_RUN_PREFIX=pretrain-mixed-al-hard bash scripts/utils/update_mixed_checkpoint_path.sh 12345"
    exit 1
fi

JOBID="$1"
PRETRAIN_CONFIG_NAME="${PRETRAIN_CONFIG_NAME:-mixed-scale-all}"
PRETRAIN_RUN_PREFIX="${PRETRAIN_RUN_PREFIX:-pretrain-mixed}"
PRETRAIN_RUN_INDEX="${PRETRAIN_RUN_INDEX:-0}"

CHECKPOINT_PATH="experiments/expts/${PRETRAIN_CONFIG_NAME}/${PRETRAIN_RUN_PREFIX}-${JOBID}-${PRETRAIN_RUN_INDEX}/checkpoints/ckpt_best.tar"

echo "Updating mixed transfer checkpoint path to:"
echo "  $CHECKPOINT_PATH"
echo ""

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "Error: neither python3 nor python found on PATH"
    exit 1
fi

"$PYTHON_BIN" - "$CHECKPOINT_PATH" <<'PY'
import re
import sys
from pathlib import Path

new_ckpt = sys.argv[1]

repo_root = Path(".")
config_files = [
    repo_root / "config" / "operators_poisson.yaml",
    repo_root / "config" / "operators_ad.yaml",
    repo_root / "config" / "operators_helmholtz.yaml",
]

pattern = re.compile(
    r"weights:\s*'experiments/expts/mixed-[^']*/pretrain-mixed-[^']*/checkpoints/ckpt_best\.tar'"
)

replacement = f"weights: '{new_ckpt}'"

total_updates = 0
for path in config_files:
    if not path.exists():
        print(f"✗ Config not found: {path}")
        continue

    text = path.read_text(encoding="utf-8")
    new_text, count = pattern.subn(replacement, text)
    if count > 0:
        path.write_text(new_text, encoding="utf-8")
        print(f"✓ Updated {path} ({count} entries)")
        total_updates += count
    else:
        print(f"• No mixed-pretrain weights matched in {path}")

print("")
print(f"Total updated entries: {total_updates}")
PY

echo ""
echo "Done. You can now run mixed-pretraining transfer jobs:"
echo "  sbatch scripts/slurm/finetune/poisson/submit_finetune_poisson_k1_2p5_mixed_array.sh"
echo "  sbatch scripts/slurm/finetune/advdiff/submit_finetune_advdiff_adr0p2_0p4_mixed_small.sh"
echo "  sbatch scripts/slurm/finetune/helmholtz/submit_finetune_helmholtz_o1_5_mixed_small.sh"

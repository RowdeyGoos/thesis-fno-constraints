#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utils/run_mixed_zeroshot_bundle.sh [--root-dir DIR] [--summary-file PATH] [--python BIN] \
    <spec> [<spec> ...]

Each <spec> must be:
  CONFIG_NAME:RUN_PREFIX:JOB_ID[:RUN_INDEX]

Examples:
  bash scripts/utils/run_mixed_zeroshot_bundle.sh \
    mixed-scale-all:pretrain-mixed:12345 \
    mixed-scale-all-constraints-al-hard:pretrain-mixed-al-hard:12346 \
    mixed-scale-all-constraints-zero-hard-only:pretrain-mixed-zero-hard:12347

Options:
  --root-dir DIR
      Root dir passed to eval.py. Default: experiments_zeroshot

  --summary-file PATH
      TSV summary output path. Default:
      results/constraints/zeroshot_bundle_<UTC timestamp>.tsv

  --python BIN
      Python executable to use. Default: auto-detect from PYTHON_BIN, venv, python3, python

Notes:
  - This script runs the existing downstream mixed zero-shot configs:
      poisson-k1_2.5-zeroshot-mixed
      ad-adr0p2_0p4-zeroshot-mixed
      helm-o1_5-zeroshot-mixed
  - It is intended for standard mixed checkpoints with 7 input channels.
  - For mixed-scale-all-constraints-zero-hard-only, Poisson/AdvDiff use dedicated
    zero-shot configs that keep hard zero-mode projection enabled at inference.
  - BC-conditioned mixed checkpoints (mixed-bc-*) are not directly supported by the current
    downstream zero-shot configs, which expect 7-channel mixed inputs rather than 9-channel BC inputs.
EOF
}

sanitize_id() {
  printf '%s' "$1" | tr '/:[:space:]' '-' | tr -cd 'A-Za-z0-9._-'
}

choose_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/Scripts/python.exe" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/Scripts/python.exe"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return 0
  fi
  return 1
}

metric_from_log() {
  local log_path="$1"
  local key="$2"

  awk -F, -v want="$key" '
    $1 == want {
      gsub(/^[ \t]+|[ \t]+$/, "", $2)
      print $2
      found = 1
      exit 0
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "$log_path"
}

append_summary_row() {
  local label="$1"
  local config_name="$2"
  local run_prefix="$3"
  local job_id="$4"
  local run_index="$5"
  local dataset="$6"
  local downstream_config="$7"
  local checkpoint_path="$8"
  local log_path="$9"
  local status="${10}"
  local test_err="${11}"
  local test_loss="${12}"
  local test_zero_mode_violation="${13}"
  local test_pde_residual_norm="${14}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$dataset" \
    "$downstream_config" "$checkpoint_path" "$log_path" "$status" "$test_err" \
    "$test_loss" "$test_zero_mode_violation" "$test_pde_residual_norm" >> "$SUMMARY_FILE"
}

RUN_STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ROOT_DIR="experiments_zeroshot"
SUMMARY_FILE="results/constraints/zeroshot_bundle_${RUN_STAMP}.tsv"
PYTHON_OVERRIDE=""
SPECS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --summary-file)
      SUMMARY_FILE="$2"
      shift 2
      ;;
    --python)
      PYTHON_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
    *)
      SPECS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SPECS[@]} -eq 0 ]]; then
  echo "Error: provide at least one checkpoint spec." >&2
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "$PYTHON_OVERRIDE" ]]; then
  PYTHON_BIN="$PYTHON_OVERRIDE"
else
  PYTHON_BIN="$(choose_python)" || {
    echo "Error: no Python executable found." >&2
    exit 1
  }
fi

if [[ ! -f "eval.py" ]]; then
  echo "Error: eval.py not found. Run this from inside the neuraloperators-TL-scaling repo." >&2
  exit 1
fi

mkdir -p "$ROOT_DIR" "$(dirname "$SUMMARY_FILE")"

printf 'label\tpretrain_config\trun_prefix\tjob_id\trun_index\tdataset\tdownstream_config\tcheckpoint_path\tlog_path\tstatus\ttest_err\ttest_loss\ttest_zero_mode_violation\ttest_pde_residual_norm\n' > "$SUMMARY_FILE"

DATASETS=(
  "poisson|config/operators_poisson.yaml|poisson-k1_2.5-zeroshot-mixed"
  "advdiff|config/operators_ad.yaml|ad-adr0p2_0p4-zeroshot-mixed"
  "helmholtz|config/operators_helmholtz.yaml|helm-o1_5-zeroshot-mixed"
)

echo "=========================================="
echo "Mixed Zero-Shot Bundle Evaluation"
echo "Timestamp:   $RUN_STAMP"
echo "Python:      $PYTHON_BIN"
echo "Root dir:    $ROOT_DIR"
echo "Summary:     $SUMMARY_FILE"
echo "Specs:       ${#SPECS[@]}"
echo "=========================================="

for spec in "${SPECS[@]}"; do
  IFS=':' read -r config_name run_prefix job_id run_index extra <<< "$spec"

  if [[ -n "${extra:-}" || -z "${config_name:-}" || -z "${run_prefix:-}" || -z "${job_id:-}" ]]; then
    echo ""
    echo "Skipping invalid spec: $spec"
    echo "Expected format: CONFIG_NAME:RUN_PREFIX:JOB_ID[:RUN_INDEX]"
    append_summary_row "$spec" "${config_name:-}" "${run_prefix:-}" "${job_id:-}" "${run_index:-0}" "all" "-" "-" "-" "invalid_spec" "-" "-" "-" "-"
    continue
  fi

  run_index="${run_index:-0}"
  label="$(sanitize_id "${config_name}-${job_id}-${run_index}")"
  checkpoint_path="experiments/expts/${config_name}/${run_prefix}-${job_id}-${run_index}/checkpoints/ckpt_best.tar"

  echo ""
  echo "------------------------------------------"
  echo "Spec:        $spec"
  echo "Label:       $label"
  echo "Checkpoint:  $checkpoint_path"
  echo "------------------------------------------"

  if [[ "$config_name" == mixed-bc-* ]]; then
    echo "Skipping $config_name: BC-conditioned mixed checkpoints are not compatible with current 7-channel downstream zero-shot configs."
    append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "all" "-" "$checkpoint_path" "-" "unsupported_bc_channels" "-" "-" "-" "-"
    continue
  fi

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Skipping $config_name: checkpoint not found."
    append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "all" "-" "$checkpoint_path" "-" "missing_checkpoint" "-" "-" "-" "-"
    continue
  fi

  for dataset_info in "${DATASETS[@]}"; do
    IFS='|' read -r dataset yaml_config downstream_config <<< "$dataset_info"
    selected_downstream_config="$downstream_config"

    # Zero-hard checkpoints were trained with the hard projection active in the
    # forward pass for Poisson/AdvDiff. Use matching downstream configs so
    # zero-shot evaluation preserves that model behavior.
    if [[ "$config_name" == "mixed-scale-all-constraints-zero-hard-only" ]]; then
      case "$dataset" in
        poisson)
          selected_downstream_config="poisson-k1_2.5-zeroshot-mixed-zero-hard"
          ;;
        advdiff)
          selected_downstream_config="ad-adr0p2_0p4-zeroshot-mixed-zero-hard"
          ;;
      esac
    fi

    run_name="$(sanitize_id "zeroshot-${dataset}-${label}-${RUN_STAMP}")"
    log_path="${ROOT_DIR}/expts/${selected_downstream_config}/${run_name}/logs_best.txt"

    echo ""
    echo "Running ${dataset} zero-shot with config ${selected_downstream_config}"

    if "$PYTHON_BIN" eval.py \
      --yaml_config "$yaml_config" \
      --config "$selected_downstream_config" \
      --run_num "$run_name" \
      --root_dir "$ROOT_DIR" \
      --weights "$checkpoint_path"; then

      test_err="-"
      test_loss="-"
      test_zero_mode_violation="-"
      test_pde_residual_norm="-"

      if [[ -f "$log_path" ]]; then
        test_err="$(metric_from_log "$log_path" "test_err" 2>/dev/null || printf '%s' '-')"
        test_loss="$(metric_from_log "$log_path" "test_loss" 2>/dev/null || printf '%s' '-')"
        test_zero_mode_violation="$(metric_from_log "$log_path" "test_zero_mode_violation" 2>/dev/null || printf '%s' '-')"
        test_pde_residual_norm="$(metric_from_log "$log_path" "test_pde_residual_norm" 2>/dev/null || printf '%s' '-')"
        append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$dataset" "$selected_downstream_config" "$checkpoint_path" "$log_path" "ok" "$test_err" "$test_loss" "$test_zero_mode_violation" "$test_pde_residual_norm"
      else
        append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$dataset" "$selected_downstream_config" "$checkpoint_path" "$log_path" "missing_log" "-" "-" "-" "-"
      fi
    else
      echo "Evaluation failed for ${dataset} / ${config_name}"
      append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$dataset" "$selected_downstream_config" "$checkpoint_path" "$log_path" "eval_failed" "-" "-" "-" "-"
    fi
  done
done

echo ""
echo "=========================================="
echo "Zero-shot bundle complete"
echo "Summary file: $SUMMARY_FILE"
echo "=========================================="

if command -v column >/dev/null 2>&1; then
  echo ""
  column -ts $'\t' "$SUMMARY_FILE"
fi

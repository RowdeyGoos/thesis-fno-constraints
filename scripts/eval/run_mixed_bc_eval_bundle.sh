#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/run_mixed_bc_eval_bundle.sh [--root-dir DIR] [--summary-file PATH] [--yaml-config PATH] [--python BIN] \
    <spec> [<spec> ...]

Each <spec> must be:
  CONFIG_NAME:RUN_PREFIX:JOB_ID[:RUN_INDEX]

Examples:
  bash scripts/eval/run_mixed_bc_eval_bundle.sh \
    mixed-bc-scale-all-off:pretrain-mixed-bc-off:12345 \
    mixed-bc-scale-all-soft:pretrain-mixed-bc-soft:12346 \
    mixed-bc-scale-all-hard:pretrain-mixed-bc-hard:12347

Options:
  --root-dir DIR
      Root dir passed to scripts/entrypoints/eval.py. Default: experiments_bc_eval

  --summary-file PATH
      TSV summary output path. Default:
      results/constraints/bc_eval_bundle_<UTC timestamp>.tsv

  --yaml-config PATH
      YAML config passed to scripts/entrypoints/eval.py. Default: config/operators_mixed_bc.yaml

  --python BIN
      Python executable to use. Default: auto-detect from PYTHON_BIN, venv, python3, python

Notes:
  - This script evaluates BC-conditioned mixed checkpoints on the BC mixed test set.
  - The eval config is the same as the pretrain config name inside operators_mixed_bc.yaml.
  - Only mixed-bc-* checkpoints are supported by this bundle.
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
  local yaml_config="$6"
  local checkpoint_path="$7"
  local log_path="$8"
  local status="${9}"
  local test_err="${10}"
  local test_loss="${11}"
  local test_err_interior="${12}"
  local test_bc_violation_raw="${13}"
  local test_bc_violation_final="${14}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$yaml_config" \
    "$checkpoint_path" "$log_path" "$status" "$test_err" "$test_loss" \
    "$test_err_interior" "$test_bc_violation_raw" "$test_bc_violation_final" >> "$SUMMARY_FILE"
}

RUN_STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
ROOT_DIR="experiments_bc_eval"
SUMMARY_FILE="results/constraints/bc_eval_bundle_${RUN_STAMP}.tsv"
YAML_CONFIG="config/operators_mixed_bc.yaml"
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
    --yaml-config)
      YAML_CONFIG="$2"
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

if [[ ! -f "scripts/entrypoints/eval.py" ]]; then
  echo "Error: scripts/entrypoints/eval.py not found. Run this from the repository root." >&2
  exit 1
fi

if [[ ! -f "$YAML_CONFIG" ]]; then
  echo "Error: YAML config not found: $YAML_CONFIG" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR" "$(dirname "$SUMMARY_FILE")"

printf 'label\tpretrain_config\trun_prefix\tjob_id\trun_index\tyaml_config\tcheckpoint_path\tlog_path\tstatus\ttest_err\ttest_loss\ttest_err_interior\ttest_bc_violation_raw\ttest_bc_violation_final\n' > "$SUMMARY_FILE"

echo "=========================================="
echo "Mixed BC Eval Bundle"
echo "Timestamp:   $RUN_STAMP"
echo "Python:      $PYTHON_BIN"
echo "YAML:        $YAML_CONFIG"
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
    append_summary_row "$spec" "${config_name:-}" "${run_prefix:-}" "${job_id:-}" "${run_index:-0}" "$YAML_CONFIG" "-" "-" "invalid_spec" "-" "-" "-" "-" "-"
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

  if [[ "$config_name" != mixed-bc-* ]]; then
    echo "Skipping $config_name: only mixed-bc-* checkpoints are supported by this bundle."
    append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$YAML_CONFIG" "$checkpoint_path" "-" "unsupported_non_bc_config" "-" "-" "-" "-" "-"
    continue
  fi

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Skipping $config_name: checkpoint not found."
    append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$YAML_CONFIG" "$checkpoint_path" "-" "missing_checkpoint" "-" "-" "-" "-" "-"
    continue
  fi

  run_name="$(sanitize_id "bc-eval-${label}-${RUN_STAMP}")"
  log_path="${ROOT_DIR}/expts/${config_name}/${run_name}/logs_best.txt"

  if "$PYTHON_BIN" scripts/entrypoints/eval.py \
    --yaml_config "$YAML_CONFIG" \
    --config "$config_name" \
    --run_num "$run_name" \
    --root_dir "$ROOT_DIR" \
    --weights "$checkpoint_path"; then

    test_err="-"
    test_loss="-"
    test_err_interior="-"
    test_bc_violation_raw="-"
    test_bc_violation_final="-"

    if [[ -f "$log_path" ]]; then
      test_err="$(metric_from_log "$log_path" "test_err" 2>/dev/null || printf '%s' '-')"
      test_loss="$(metric_from_log "$log_path" "test_loss" 2>/dev/null || printf '%s' '-')"
      test_err_interior="$(metric_from_log "$log_path" "test_err_interior" 2>/dev/null || printf '%s' '-')"
      test_bc_violation_raw="$(metric_from_log "$log_path" "test_bc_violation_raw" 2>/dev/null || printf '%s' '-')"
      test_bc_violation_final="$(metric_from_log "$log_path" "test_bc_violation_final" 2>/dev/null || printf '%s' '-')"
      append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$YAML_CONFIG" "$checkpoint_path" "$log_path" "ok" "$test_err" "$test_loss" "$test_err_interior" "$test_bc_violation_raw" "$test_bc_violation_final"
    else
      append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$YAML_CONFIG" "$checkpoint_path" "$log_path" "missing_log" "-" "-" "-" "-" "-"
    fi
  else
    echo "Evaluation failed for ${config_name}"
    append_summary_row "$label" "$config_name" "$run_prefix" "$job_id" "$run_index" "$YAML_CONFIG" "$checkpoint_path" "$log_path" "eval_failed" "-" "-" "-" "-" "-"
  fi
done

echo ""
echo "=========================================="
echo "BC eval bundle complete"
echo "Summary file: $SUMMARY_FILE"
echo "=========================================="

if command -v column >/dev/null 2>&1; then
  echo ""
  column -ts $'\t' "$SUMMARY_FILE"
fi

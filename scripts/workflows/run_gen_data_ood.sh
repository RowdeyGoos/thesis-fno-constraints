#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "Error: PYTHON_BIN is set but not executable: $PYTHON_BIN"
        exit 1
    fi
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: no python executable found. Set PYTHON_BIN=/path/to/python and rerun."
    exit 1
fi

ntrain="${NTRAIN:-32768}"
nval="${NVAL:-4096}"
ntest="${NTEST:-4096}"
n="${GRID_N:-128}"
ng="${NG:-144}"

data_root="${DATA_ROOT:-data}"
poisson_datapath="${POISSON_DATAPATH:-${data_root}/poisson}"
advdiff_datapath="${ADVDIFF_DATAPATH:-${data_root}/advection-diffusion}"
helmholtz_datapath="${HELMHOLTZ_DATAPATH:-${data_root}/helmholtz}"

poisson_source_e1="${POISSON_E1:-${E1:-1.0}}"
poisson_source_e2="${POISSON_E2:-${E2:-5.0}}"
poisson_transfer_e1="${TRANSFER_POISSON_E1:-5.0}"
poisson_transfer_e2="${TRANSFER_POISSON_E2:-10.0}"

advdiff_source_adr1="${ADVDIFF_ADR1:-${ADR1:-0.2}}"
advdiff_source_adr2="${ADVDIFF_ADR2:-${ADR2:-1.0}}"
advdiff_transfer_adr1="${TRANSFER_ADVDIFF_ADR1:-1.0}"
advdiff_transfer_adr2="${TRANSFER_ADVDIFF_ADR2:-1.2}"

helmholtz_source_o1="${HELMHOLTZ_O1:-${O1:-1}}"
helmholtz_source_o2="${HELMHOLTZ_O2:-${O2:-10}}"
helmholtz_transfer_o1="${TRANSFER_HELMHOLTZ_O1:-10}"
helmholtz_transfer_o2="${TRANSFER_HELMHOLTZ_O2:-15}"

compute_scales="${COMPUTE_SCALES:-1}"
generate_mixed_format="${GENERATE_MIXED_FORMAT:-1}"
h5_chunk_samples="${H5_CHUNK_SAMPLES:-256}"
progress_every="${PROGRESS_EVERY:-1000}"

dataset="${DATASET:-all}"
range_set="${RANGE_SET:-transfer}"

case "$dataset" in
    all|poisson|advdiff|helmholtz)
        ;;
    *)
        echo "Error: unsupported DATASET='$dataset'. Use one of: all, poisson, advdiff, helmholtz."
        exit 2
        ;;
esac

case "$range_set" in
    all|source|transfer)
        ;;
    *)
        echo "Error: unsupported RANGE_SET='$range_set'. Use one of: all, source, transfer."
        exit 2
        ;;
esac

py_float_str() {
    "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"
}

py_int_str() {
    "$PYTHON_BIN" -c 'import sys; print(int(float(sys.argv[1])))' "$1"
}

compute_train_scales() {
    local datapath="$1"
    local filename="$2"

    if [ "$compute_scales" = "0" ]; then
        return
    fi

    echo "Computing scales for ${datapath}/${filename}"
    "$PYTHON_BIN" scripts/data/compute_scales.py \
        --datapath "$datapath" \
        --filename "$filename" \
        --nx "$n" \
        --ny "$n" \
        --lx 1.0 \
        --ly 1.0
}

mixed_generation_enabled() {
    case "${generate_mixed_format}" in
        1|true|TRUE|yes|YES)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

generate_mixed_copies() {
    local label="$1"
    local datapath="$2"
    local converter_script="$3"
    shift 3
    local raw_file
    local mixed_file
    local train_mixed_file

    if ! mixed_generation_enabled; then
        return
    fi

    echo "Generating mixed-format copies for ${label} in: $datapath"
    for raw_file in "$@"; do
        mixed_file="${raw_file%.h5}_mixed.h5"
        echo "Converting ${datapath}/${raw_file} -> ${datapath}/${mixed_file}"
        "$PYTHON_BIN" "${converter_script}" \
            --input_path "${datapath}/${raw_file}" \
            --output_path "${datapath}/${mixed_file}"
    done

    train_mixed_file="${1%.h5}_mixed.h5"
    compute_train_scales "$datapath" "$train_mixed_file"
}

generate_poisson() {
    local label="$1"
    local datapath="$2"
    local e1="$3"
    local e2="$4"
    local e1_str
    local e2_str
    local train_file
    local val_file
    local test_file

    e1_str="$(py_float_str "$e1")"
    e2_str="$(py_float_str "$e2")"

    echo "Generating Poisson data (${label}) in: $datapath"
    "$PYTHON_BIN" scripts/data/gen_data_poisson.py \
        --ntrain="$ntrain" \
        --nval="$nval" \
        --ntest="$ntest" \
        --ng="$ng" \
        --sparse \
        --n "$n" \
        --datapath "$datapath" \
        --e1 "$e1" \
        --e2 "$e2" \
        --h5_chunk_samples "$h5_chunk_samples" \
        --progress_every "$progress_every"

    train_file="_train_k${e1_str}_${e2_str}_32k.h5"
    val_file="_val_k${e1_str}_${e2_str}_4k.h5"
    test_file="_test_k${e1_str}_${e2_str}_4k.h5"
    compute_train_scales "$datapath" "$train_file"
    generate_mixed_copies \
        "Poisson data (${label})" \
        "$datapath" \
        "scripts/data/convert_poisson_to_mixed_format.py" \
        "$train_file" "$val_file" "$test_file"
}

generate_advdiff() {
    local label="$1"
    local datapath="$2"
    local adr1="$3"
    local adr2="$4"
    local adr1_str
    local adr2_str
    local train_file
    local val_file
    local test_file

    adr1_str="$(py_float_str "$adr1")"
    adr2_str="$(py_float_str "$adr2")"

    echo "Generating AdvDiff data (${label}) in: $datapath"
    "$PYTHON_BIN" scripts/data/gen_data_advdiff.py \
        --ntrain="$ntrain" \
        --nval="$nval" \
        --ntest="$ntest" \
        --ng="$ng" \
        --sparse \
        --n "$n" \
        --datapath "$datapath" \
        --adr1 "$adr1" \
        --adr2 "$adr2" \
        --h5_chunk_samples "$h5_chunk_samples" \
        --progress_every "$progress_every"

    train_file="_train_adr${adr1_str}_${adr2_str}_32k.h5"
    val_file="_val_adr${adr1_str}_${adr2_str}_4k.h5"
    test_file="_test_adr${adr1_str}_${adr2_str}_4k.h5"
    compute_train_scales "$datapath" "$train_file"
    generate_mixed_copies \
        "AdvDiff data (${label})" \
        "$datapath" \
        "scripts/data/convert_ad_to_mixed_format.py" \
        "$train_file" "$val_file" "$test_file"
}

generate_helmholtz() {
    local label="$1"
    local datapath="$2"
    local o1="$3"
    local o2="$4"
    local o1_str
    local o2_str
    local train_file
    local val_file
    local test_file

    o1_str="$(py_int_str "$o1")"
    o2_str="$(py_int_str "$o2")"

    echo "Generating Helmholtz data (${label}) in: $datapath"
    "$PYTHON_BIN" scripts/data/gen_data_helmholtz.py \
        --ntrain="$ntrain" \
        --nval="$nval" \
        --ntest="$ntest" \
        --ng="$ng" \
        --sparse \
        --n "$n" \
        --datapath "$datapath" \
        --o1 "$o1" \
        --o2 "$o2" \
        --h5_chunk_samples "$h5_chunk_samples" \
        --progress_every "$progress_every"

    train_file="_train_o${o1_str}_${o2_str}_32k.h5"
    val_file="_val_o${o1_str}_${o2_str}_4k.h5"
    test_file="_test_o${o1_str}_${o2_str}_4k.h5"
    compute_train_scales "$datapath" "$train_file"
    generate_mixed_copies \
        "Helmholtz data (${label})" \
        "$datapath" \
        "scripts/data/convert_helmholtz_to_mixed_format.py" \
        "$train_file" "$val_file" "$test_file"
}

mkdir -p "$poisson_datapath" "$advdiff_datapath" "$helmholtz_datapath"

echo "=========================================="
echo "Normal data generation configuration"
echo "Data root:    $data_root"
echo "Dataset:      $dataset"
echo "Range set:    $range_set"
echo "Grid:         ${n}x${n}"
echo "Split sizes:  train=${ntrain}, val=${nval}, test=${ntest}"
echo "Scales:       ${compute_scales}"
echo "Mixed copies: ${generate_mixed_format}"
echo "Chunk size:   ${h5_chunk_samples}"
echo "Progress:     ${progress_every}"
echo "=========================================="

if [ "$dataset" = "all" ] || [ "$dataset" = "poisson" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_poisson "source/pretrain k in [${poisson_source_e1}, ${poisson_source_e2}]" \
            "$poisson_datapath" "$poisson_source_e1" "$poisson_source_e2"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_poisson "downstream/OOD k in [${poisson_transfer_e1}, ${poisson_transfer_e2}]" \
            "$poisson_datapath" "$poisson_transfer_e1" "$poisson_transfer_e2"
    fi
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "advdiff" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_advdiff "source/pretrain adr in [${advdiff_source_adr1}, ${advdiff_source_adr2}]" \
            "$advdiff_datapath" "$advdiff_source_adr1" "$advdiff_source_adr2"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_advdiff "downstream/OOD adr in [${advdiff_transfer_adr1}, ${advdiff_transfer_adr2}]" \
            "$advdiff_datapath" "$advdiff_transfer_adr1" "$advdiff_transfer_adr2"
    fi
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "helmholtz" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_helmholtz "source/pretrain omega in [${helmholtz_source_o1}, ${helmholtz_source_o2}]" \
            "$helmholtz_datapath" "$helmholtz_source_o1" "$helmholtz_source_o2"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_helmholtz "downstream/OOD omega in [${helmholtz_transfer_o1}, ${helmholtz_transfer_o2}]" \
            "$helmholtz_datapath" "$helmholtz_transfer_o1" "$helmholtz_transfer_o2"
    fi
fi

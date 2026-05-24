#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

data_root="${DATA_ROOT:-data/bc}"
poisson_datapath="${POISSON_DATAPATH:-${data_root}/poisson}"
advdiff_datapath="${ADVDIFF_DATAPATH:-${data_root}/advdiff}"
helmholtz_datapath="${HELMHOLTZ_DATAPATH:-${data_root}/helmholtz}"

poisson_source_e1="${POISSON_E1:-${E1:-1}}"
poisson_source_e2="${POISSON_E2:-${E2:-5}}"
poisson_transfer_e1="${TRANSFER_POISSON_E1:-1.0}"
poisson_transfer_e2="${TRANSFER_POISSON_E2:-2.5}"

advdiff_source_adr1="${ADVDIFF_ADR1:-${ADR1:-0.2}}"
advdiff_source_adr2="${ADVDIFF_ADR2:-${ADR2:-1}}"
advdiff_transfer_adr1="${TRANSFER_ADVDIFF_ADR1:-0.2}"
advdiff_transfer_adr2="${TRANSFER_ADVDIFF_ADR2:-0.4}"
advdiff_source_e1="${ADVDIFF_E1:-${E1:-1}}"
advdiff_source_e2="${ADVDIFF_E2:-${E2:-5}}"
advdiff_transfer_e1="${TRANSFER_ADVDIFF_E1:-${advdiff_source_e1}}"
advdiff_transfer_e2="${TRANSFER_ADVDIFF_E2:-${advdiff_source_e2}}"
# ADR->lambda mapping is loaded from utils/*.npy inside gen_data_advdiff_bc.py

helmholtz_source_o1="${HELMHOLTZ_O1:-${O1:-1}}"
helmholtz_source_o2="${HELMHOLTZ_O2:-${O2:-10}}"
helmholtz_transfer_o1="${TRANSFER_HELMHOLTZ_O1:-1}"
helmholtz_transfer_o2="${TRANSFER_HELMHOLTZ_O2:-5}"

bc_modes="${BC_MODES:-5}"
bc_amplitude="${BC_AMPLITUDE:-1.0}"
bc_width="${BC_WIDTH:-1}"
h5_chunk_samples="${H5_CHUNK_SAMPLES:-64}"
progress_every="${PROGRESS_EVERY:-1000}"
helmholtz_max_sample_attempts="${HELMHOLTZ_MAX_SAMPLE_ATTEMPTS:-100}"
# For BC Helmholtz, interior amplitudes can naturally exceed 2 due boundary forcing.
# Set <=0 to disable amplitude rejection; only non-finite solves are retried.
helmholtz_max_abs_solution="${HELMHOLTZ_MAX_ABS_SOLUTION:-0.0}"
compute_scales="${COMPUTE_SCALES:-1}"

poisson_seed="${POISSON_SEED:-0}"
advdiff_seed="${ADVDIFF_SEED:-0}"
helmholtz_seed="${HELMHOLTZ_SEED:-0}"
poisson_transfer_seed="${TRANSFER_POISSON_SEED:-${poisson_seed}}"
advdiff_transfer_seed="${TRANSFER_ADVDIFF_SEED:-${advdiff_seed}}"
helmholtz_transfer_seed="${TRANSFER_HELMHOLTZ_SEED:-${helmholtz_seed}}"

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

fmt_stamp() {
    local value="$1"
    echo "${value//./p}"
}

compute_train_scales() {
    local datapath="$1"
    local filename="$2"

    if [ "$compute_scales" = "0" ]; then
        return
    fi

    echo "Computing scales for ${datapath}/${filename}"
    "$PYTHON_BIN" utils/compute_scales.py \
        --datapath "$datapath" \
        --filename "$filename" \
        --nx "$n" \
        --ny "$n" \
        --lx 1.0 \
        --ly 1.0
}

generate_poisson_bc() {
    local label="$1"
    local datapath="$2"
    local e1="$3"
    local e2="$4"
    local seed="$5"
    local train_file

    echo "Generating Poisson BC data (${label}) in: $datapath"
    "$PYTHON_BIN" utils/gen_data_poisson_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$datapath" \
                        --e1 "$e1" --e2 "$e2" --bc_modes "$bc_modes" \
                        --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                        --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                        --seed "$seed"

    train_file="_train_k$(fmt_stamp "$e1")_$(fmt_stamp "$e2")_32k_bc.h5"
    compute_train_scales "$datapath" "$train_file"
}

generate_advdiff_bc() {
    local label="$1"
    local datapath="$2"
    local adr1="$3"
    local adr2="$4"
    local e1="$5"
    local e2="$6"
    local seed="$7"
    local train_file

    echo "Generating AdvDiff BC data (${label}) in: $datapath"
    "$PYTHON_BIN" utils/gen_data_advdiff_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$datapath" \
                        --adr1 "$adr1" --adr2 "$adr2" --e1 "$e1" --e2 "$e2" \
                        --bc_modes "$bc_modes" --bc_amplitude "$bc_amplitude" \
                        --bc_width "$bc_width" --h5_chunk_samples "$h5_chunk_samples" \
                        --progress_every "$progress_every" \
                        --seed "$seed"

    train_file="_train_adr$(fmt_stamp "$adr1")_$(fmt_stamp "$adr2")_32k_bc.h5"
    compute_train_scales "$datapath" "$train_file"
}

generate_helmholtz_bc() {
    local label="$1"
    local datapath="$2"
    local o1="$3"
    local o2="$4"
    local seed="$5"
    local train_file

    echo "Generating Helmholtz BC data (${label}) in: $datapath"
    "$PYTHON_BIN" utils/gen_data_helmholtz_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$datapath" \
                        --o1 "$o1" --o2 "$o2" --bc_modes "$bc_modes" \
                        --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                        --max_abs_solution "$helmholtz_max_abs_solution" \
                        --max_sample_attempts "$helmholtz_max_sample_attempts" \
                        --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                        --seed "$seed"

    train_file="_train_o$(fmt_stamp "$o1")_$(fmt_stamp "$o2")_32k_bc.h5"
    compute_train_scales "$datapath" "$train_file"
}

mkdir -p "$poisson_datapath" "$advdiff_datapath" "$helmholtz_datapath"

echo "=========================================="
echo "BC data generation configuration"
echo "Data root:    $data_root"
echo "Dataset:      $dataset"
echo "Range set:    $range_set"
echo "Grid:         ${n}x${n}"
echo "Split sizes:  train=${ntrain}, val=${nval}, test=${ntest}"
echo "Scales:       ${compute_scales}"
echo "=========================================="

if [ "$dataset" = "all" ] || [ "$dataset" = "poisson" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_poisson_bc "source/pretrain k in [${poisson_source_e1}, ${poisson_source_e2}]" \
            "$poisson_datapath" "$poisson_source_e1" "$poisson_source_e2" "$poisson_seed"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_poisson_bc "downstream/transfer k in [${poisson_transfer_e1}, ${poisson_transfer_e2}]" \
            "$poisson_datapath" "$poisson_transfer_e1" "$poisson_transfer_e2" "$poisson_transfer_seed"
    fi
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "advdiff" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_advdiff_bc "source/pretrain adr in [${advdiff_source_adr1}, ${advdiff_source_adr2}]" \
            "$advdiff_datapath" "$advdiff_source_adr1" "$advdiff_source_adr2" \
            "$advdiff_source_e1" "$advdiff_source_e2" "$advdiff_seed"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_advdiff_bc "downstream/transfer adr in [${advdiff_transfer_adr1}, ${advdiff_transfer_adr2}]" \
            "$advdiff_datapath" "$advdiff_transfer_adr1" "$advdiff_transfer_adr2" \
            "$advdiff_transfer_e1" "$advdiff_transfer_e2" "$advdiff_transfer_seed"
    fi
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "helmholtz" ]; then
    if [ "$range_set" = "all" ] || [ "$range_set" = "source" ]; then
        generate_helmholtz_bc "source/pretrain omega in [${helmholtz_source_o1}, ${helmholtz_source_o2}]" \
            "$helmholtz_datapath" "$helmholtz_source_o1" "$helmholtz_source_o2" "$helmholtz_seed"
    fi

    if [ "$range_set" = "all" ] || [ "$range_set" = "transfer" ]; then
        generate_helmholtz_bc "downstream/transfer omega in [${helmholtz_transfer_o1}, ${helmholtz_transfer_o2}]" \
            "$helmholtz_datapath" "$helmholtz_transfer_o1" "$helmholtz_transfer_o2" "$helmholtz_transfer_seed"
    fi
fi

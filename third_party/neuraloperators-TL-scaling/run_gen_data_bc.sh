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

data_root="${DATA_ROOT:-/path/to/data/bc}"
poisson_datapath="${POISSON_DATAPATH:-${data_root}/poisson}"
advdiff_datapath="${ADVDIFF_DATAPATH:-${data_root}/advdiff}"
helmholtz_datapath="${HELMHOLTZ_DATAPATH:-${data_root}/helmholtz}"

e1="${E1:-1}"      # Poisson diffusion eigenvalue range
e2="${E2:-5}"

adr1="${ADR1:-0.2}"  # AdvDiff advection-to-diffusion ratio range
adr2="${ADR2:-1}"
# ADR->lambda mapping is loaded from utils/*.npy inside gen_data_advdiff_bc.py

o1="${O1:-1}"      # Helmholtz wave number range
o2="${O2:-10}"

bc_modes="${BC_MODES:-5}"
bc_amplitude="${BC_AMPLITUDE:-1.0}"
bc_width="${BC_WIDTH:-1}"
h5_chunk_samples="${H5_CHUNK_SAMPLES:-64}"
progress_every="${PROGRESS_EVERY:-1000}"
helmholtz_max_sample_attempts="${HELMHOLTZ_MAX_SAMPLE_ATTEMPTS:-100}"
# For BC Helmholtz, interior amplitudes can naturally exceed 2 due boundary forcing.
# Set <=0 to disable amplitude rejection; only non-finite solves are retried.
helmholtz_max_abs_solution="${HELMHOLTZ_MAX_ABS_SOLUTION:-0.0}"

poisson_seed="${POISSON_SEED:-0}"
advdiff_seed="${ADVDIFF_SEED:-0}"
helmholtz_seed="${HELMHOLTZ_SEED:-0}"

dataset="${DATASET:-all}"

case "$dataset" in
    all|poisson|advdiff|helmholtz)
        ;;
    *)
        echo "Error: unsupported DATASET='$dataset'. Use one of: all, poisson, advdiff, helmholtz."
        exit 2
        ;;
esac

mkdir -p "$poisson_datapath" "$advdiff_datapath" "$helmholtz_datapath"

if [ "$dataset" = "all" ] || [ "$dataset" = "poisson" ]; then
    echo "Generating Poisson BC data in: $poisson_datapath"
    "$PYTHON_BIN" utils/gen_data_poisson_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$poisson_datapath" \
                        --e1 "$e1" --e2 "$e2" --bc_modes "$bc_modes" \
                        --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                        --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                        --seed "$poisson_seed"
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "advdiff" ]; then
    echo "Generating AdvDiff BC data in: $advdiff_datapath"
    "$PYTHON_BIN" utils/gen_data_advdiff_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$advdiff_datapath" \
                        --adr1 "$adr1" --adr2 "$adr2" --e1 "$e1" --e2 "$e2" \
                        --bc_modes "$bc_modes" --bc_amplitude "$bc_amplitude" \
                        --bc_width "$bc_width" --h5_chunk_samples "$h5_chunk_samples" \
                        --progress_every "$progress_every" \
                        --seed "$advdiff_seed"
fi

if [ "$dataset" = "all" ] || [ "$dataset" = "helmholtz" ]; then
    echo "Generating Helmholtz BC data in: $helmholtz_datapath"
    "$PYTHON_BIN" utils/gen_data_helmholtz_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                        --ng="$ng" --sparse --n "$n" --datapath "$helmholtz_datapath" \
                        --o1 "$o1" --o2 "$o2" --bc_modes "$bc_modes" \
                        --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                        --max_abs_solution "$helmholtz_max_abs_solution" \
                        --max_sample_attempts "$helmholtz_max_sample_attempts" \
                        --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                        --seed "$helmholtz_seed"
fi

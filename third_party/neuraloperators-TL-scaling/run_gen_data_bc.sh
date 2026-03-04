#!/bin/bash
set -euo pipefail

ntrain=32768
nval=4096
ntest=4096
n=128
ng=144

data_root='/path/to/data/bc'
poisson_datapath="${data_root}/poisson"
advdiff_datapath="${data_root}/advdiff"
helmholtz_datapath="${data_root}/helmholtz"

e1=1      # Poisson diffusion eigenvalue range
e2=5

adr1=0.2  # AdvDiff advection-to-diffusion ratio range
adr2=1
# ADR->lambda mapping is loaded from utils/*.npy inside gen_data_advdiff_bc.py

o1=1      # Helmholtz wave number range
o2=10

bc_modes=5
bc_amplitude=1.0
bc_width=1
h5_chunk_samples=64
progress_every=1000
helmholtz_max_abs_solution=2.0
helmholtz_max_sample_attempts=100

poisson_seed=0
advdiff_seed=1
helmholtz_seed=2

mkdir -p "$poisson_datapath" "$advdiff_datapath" "$helmholtz_datapath"

# Create Poisson BC examples
python utils/gen_data_poisson_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                    --ng="$ng" --sparse --n "$n" --datapath "$poisson_datapath" \
                    --e1 "$e1" --e2 "$e2" --bc_modes "$bc_modes" \
                    --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                    --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                    --seed "$poisson_seed"

# Create AdvDiff BC examples
python utils/gen_data_advdiff_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                    --ng="$ng" --sparse --n "$n" --datapath "$advdiff_datapath" \
                    --adr1 "$adr1" --adr2 "$adr2" --e1 "$e1" --e2 "$e2" \
                    --bc_modes "$bc_modes" --bc_amplitude "$bc_amplitude" \
                    --bc_width "$bc_width" --h5_chunk_samples "$h5_chunk_samples" \
                    --progress_every "$progress_every" \
                    --seed "$advdiff_seed"

# Create Helmholtz BC examples
python utils/gen_data_helmholtz_bc.py --ntrain="$ntrain" --nval="$nval" --ntest="$ntest" \
                    --ng="$ng" --sparse --n "$n" --datapath "$helmholtz_datapath" \
                    --o1 "$o1" --o2 "$o2" --bc_modes "$bc_modes" \
                    --bc_amplitude "$bc_amplitude" --bc_width "$bc_width" \
                    --max_abs_solution "$helmholtz_max_abs_solution" \
                    --max_sample_attempts "$helmholtz_max_sample_attempts" \
                    --h5_chunk_samples "$h5_chunk_samples" --progress_every "$progress_every" \
                    --seed "$helmholtz_seed"

#!/bin/bash
set -euo pipefail

# Input BC dataset roots (from scripts/workflows/run_gen_data_bc.sh outputs)
data_root='/path/to/data/bc'
poisson_datapath="${data_root}/poisson"
advdiff_datapath="${data_root}/advdiff"
helmholtz_datapath="${data_root}/helmholtz"

# Output mixed BC dataset root
mixed_datapath="${data_root}/mixed"

# Parameter stamps used in BC generator output filenames.
e1=1.0
e2=5.0
adr1=0.2
adr2=1.0
o1=1
o2=10

# Mixed sizes: 10922*3=32766 (~32k), 1365*3=4095 (~4k)
samples_per_system_train=10922
samples_per_system_val=1365
samples_per_system_test=1365

run_sanity_checks=1

fmt_float() {
  local value="$1"
  echo "${value//./p}"
}

e1f="$(fmt_float "$e1")"
e2f="$(fmt_float "$e2")"
adr1f="$(fmt_float "$adr1")"
adr2f="$(fmt_float "$adr2")"

poisson_train="${poisson_datapath}/_train_k${e1f}_${e2f}_32k_bc.h5"
poisson_val="${poisson_datapath}/_val_k${e1f}_${e2f}_4k_bc.h5"
poisson_test="${poisson_datapath}/_test_k${e1f}_${e2f}_4k_bc.h5"

advdiff_train="${advdiff_datapath}/_train_adr${adr1f}_${adr2f}_32k_bc.h5"
advdiff_val="${advdiff_datapath}/_val_adr${adr1f}_${adr2f}_4k_bc.h5"
advdiff_test="${advdiff_datapath}/_test_adr${adr1f}_${adr2f}_4k_bc.h5"

helmholtz_train="${helmholtz_datapath}/_train_o${o1}_${o2}_32k_bc.h5"
helmholtz_val="${helmholtz_datapath}/_val_o${o1}_${o2}_4k_bc.h5"
helmholtz_test="${helmholtz_datapath}/_test_o${o1}_${o2}_4k_bc.h5"

for file in \
  "$poisson_train" "$poisson_val" "$poisson_test" \
  "$advdiff_train" "$advdiff_val" "$advdiff_test" \
  "$helmholtz_train" "$helmholtz_val" "$helmholtz_test"; do
  [ -f "$file" ] || { echo "Error: expected input file not found: $file"; exit 2; }
done

mkdir -p "$mixed_datapath"

python scripts/data/create_mixed_dataset.py \
  --poisson_path "$poisson_train" \
  --advdiff_path "$advdiff_train" \
  --helmholtz_path "$helmholtz_train" \
  --output_path "${mixed_datapath}/_train_mixed_32k_bc.h5" \
  --samples_per_system "$samples_per_system_train" \
  --require_bc

python scripts/data/create_mixed_dataset.py \
  --poisson_path "$poisson_val" \
  --advdiff_path "$advdiff_val" \
  --helmholtz_path "$helmholtz_val" \
  --output_path "${mixed_datapath}/_val_mixed_4k_bc.h5" \
  --samples_per_system "$samples_per_system_val" \
  --require_bc

python scripts/data/create_mixed_dataset.py \
  --poisson_path "$poisson_test" \
  --advdiff_path "$advdiff_test" \
  --helmholtz_path "$helmholtz_test" \
  --output_path "${mixed_datapath}/_test_mixed_4k_bc.h5" \
  --samples_per_system "$samples_per_system_test" \
  --require_bc

if [ "$run_sanity_checks" -eq 1 ]; then
  python scripts/data/check_bc_dataset_sanity.py --input "$mixed_datapath" --glob "*_mixed_*_bc.h5"
fi

echo "Built BC mixed datasets in: $mixed_datapath"
echo "Expected scales file: ${mixed_datapath}/_train_mixed_32k_bc_scales.npy"

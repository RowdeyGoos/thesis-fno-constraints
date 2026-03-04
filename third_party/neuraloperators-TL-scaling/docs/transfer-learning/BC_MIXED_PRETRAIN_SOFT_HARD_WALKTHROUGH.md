# BC Mixed Pretraining Walkthrough (Soft and Hard BC Constraints)

This walkthrough is a command-first runbook for:

1. generating BC-enabled datasets for Poisson, AdvDiff, and Helmholtz,
2. building BC-enabled mixed train/val/test datasets, and
3. running mixed pretraining experiments with `soft` and `hard` boundary constraints.

The target training configs are in `config/operators_mixed_bc.yaml`:

- `mixed-bc-scale-all-soft`
- `mixed-bc-scale-all-hard`

## 1) Prerequisites

Run from:

```bash
cd /mnt/c/Users/rowde/Documents/GitHub/thesis-fno-constraints-boundary/third_party/neuraloperators-TL-scaling
```

Confirm key packages:

```bash
python3 - <<'PY'
import importlib.util
required = ["torch", "numpy", "h5py", "scipy", "ruamel.yaml"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
print("missing:", missing)
PY
```

## 2) Set paths and sizes

The config expects these mixed files:

- `data/mixed/_train_mixed_32k_bc.h5`
- `data/mixed/_val_mixed_4k_bc.h5`
- `data/mixed/_test_mixed_4k_bc.h5`
- `data/mixed/_train_mixed_32k_bc_scales.npy`

Use these shell variables:

```bash
export DATA_ROOT="data"
export POISSON_DIR="${DATA_ROOT}/poisson_bc"
export ADVDIFF_DIR="${DATA_ROOT}/advdiff_bc"
export HELMHOLTZ_DIR="${DATA_ROOT}/helmholtz_bc"
export MIXED_DIR="${DATA_ROOT}/mixed"

mkdir -p "${POISSON_DIR}" "${ADVDIFF_DIR}" "${HELMHOLTZ_DIR}" "${MIXED_DIR}"

# Base per-system split sizes
export NTRAIN_SYS=32768
export NVAL_SYS=4096
export NTEST_SYS=4096
export N=128
export NG=144
```

## 3) Generate BC datasets for each PDE system

### 3.1 Poisson BC

```bash
python3 utils/gen_data_poisson_bc.py \
  --ntrain "${NTRAIN_SYS}" --nval "${NVAL_SYS}" --ntest "${NTEST_SYS}" \
  --n "${N}" --ng "${NG}" --sparse \
  --datapath "${POISSON_DIR}" \
  --e1 1.0 --e2 5.0 \
  --diff_coef_scale 0.01
```

### 3.2 AdvDiff BC

```bash
python3 utils/gen_data_advdiff_bc.py \
  --ntrain "${NTRAIN_SYS}" --nval "${NVAL_SYS}" --ntest "${NTEST_SYS}" \
  --n "${N}" --ng "${NG}" --sparse \
  --datapath "${ADVDIFF_DIR}" \
  --adr1 0.2 --adr2 1.0 \
  --e1 1.0 --e2 5.0 \
  --adv_coef_scale 1.0 --diff_coef_scale 0.01
```

### 3.3 Helmholtz BC

```bash
python3 utils/gen_data_helmholtz_bc.py \
  --ntrain "${NTRAIN_SYS}" --nval "${NVAL_SYS}" --ntest "${NTEST_SYS}" \
  --n "${N}" --ng "${NG}" --sparse \
  --datapath "${HELMHOLTZ_DIR}" \
  --o1 1 --o2 10 \
  --diff_coef_scale 0.01
```

## 4) Optional BC sanity checks on system datasets

```bash
python3 scripts/utils/check_bc_dataset_sanity.py --input "${POISSON_DIR}" --glob "*.h5"
python3 scripts/utils/check_bc_dataset_sanity.py --input "${ADVDIFF_DIR}" --glob "*.h5"
python3 scripts/utils/check_bc_dataset_sanity.py --input "${HELMHOLTZ_DIR}" --glob "*.h5"
```

## 5) Build BC mixed train/val/test datasets

Use `--require_bc` so mixed creation enforces `bc` in each input dataset.

```bash
# Expected generated names from the BC generators
export P_TRAIN="${POISSON_DIR}/_train_k1p0_5p0_32k_bc.h5"
export P_VAL="${POISSON_DIR}/_val_k1p0_5p0_4k_bc.h5"
export P_TEST="${POISSON_DIR}/_test_k1p0_5p0_4k_bc.h5"

export A_TRAIN="${ADVDIFF_DIR}/_train_adr0p2_1p0_32k_bc.h5"
export A_VAL="${ADVDIFF_DIR}/_val_adr0p2_1p0_4k_bc.h5"
export A_TEST="${ADVDIFF_DIR}/_test_adr0p2_1p0_4k_bc.h5"

export H_TRAIN="${HELMHOLTZ_DIR}/_train_o1_10_32k_bc.h5"
export H_VAL="${HELMHOLTZ_DIR}/_val_o1_10_4k_bc.h5"
export H_TEST="${HELMHOLTZ_DIR}/_test_o1_10_4k_bc.h5"
```

```bash
# 32k-ish mixed train: 10922 x 3 = 32766
python3 utils/create_mixed_dataset.py \
  --poisson_path "${P_TRAIN}" \
  --advdiff_path "${A_TRAIN}" \
  --helmholtz_path "${H_TRAIN}" \
  --output_path "${MIXED_DIR}/_train_mixed_32k_bc.h5" \
  --samples_per_system 10922 \
  --require_bc

# 4k-ish mixed val: 1365 x 3 = 4095
python3 utils/create_mixed_dataset.py \
  --poisson_path "${P_VAL}" \
  --advdiff_path "${A_VAL}" \
  --helmholtz_path "${H_VAL}" \
  --output_path "${MIXED_DIR}/_val_mixed_4k_bc.h5" \
  --samples_per_system 1365 \
  --require_bc

# 4k-ish mixed test: 1365 x 3 = 4095
python3 utils/create_mixed_dataset.py \
  --poisson_path "${P_TEST}" \
  --advdiff_path "${A_TEST}" \
  --helmholtz_path "${H_TEST}" \
  --output_path "${MIXED_DIR}/_test_mixed_4k_bc.h5" \
  --samples_per_system 1365 \
  --require_bc
```

Notes:

- `create_mixed_dataset.py` automatically computes and saves scales:
  - `data/mixed/_train_mixed_32k_bc_scales.npy` (used by config)

## 6) Optional sanity check on mixed datasets

```bash
python3 scripts/utils/check_bc_dataset_sanity.py --input "${MIXED_DIR}" --glob "*_mixed_*_bc.h5"
```

## 7) Run mixed pretraining experiments (soft and hard)

## 7.1 Soft BC run

```bash
python3 train.py \
  --yaml_config config/operators_mixed_bc.yaml \
  --config mixed-bc-scale-all-soft \
  --run_num pretrain-mixed-bc-soft-0 \
  --root_dir experiments
```

## 7.2 Hard BC run

```bash
python3 train.py \
  --yaml_config config/operators_mixed_bc.yaml \
  --config mixed-bc-scale-all-hard \
  --run_num pretrain-mixed-bc-hard-0 \
  --root_dir experiments
```

## 8) Evaluate both runs

```bash
python3 eval.py \
  --yaml_config config/operators_mixed_bc.yaml \
  --config mixed-bc-scale-all-soft \
  --run_num eval-mixed-bc-soft-0 \
  --root_dir experiments \
  --weights experiments/expts/mixed-bc-scale-all-soft/pretrain-mixed-bc-soft-0/checkpoints/ckpt_best.tar

python3 eval.py \
  --yaml_config config/operators_mixed_bc.yaml \
  --config mixed-bc-scale-all-hard \
  --run_num eval-mixed-bc-hard-0 \
  --root_dir experiments \
  --weights experiments/expts/mixed-bc-scale-all-hard/pretrain-mixed-bc-hard-0/checkpoints/ckpt_best.tar
```

## 9) Compare key metrics

Training logs:

- `experiments/expts/mixed-bc-scale-all-soft/pretrain-mixed-bc-soft-0/logs_best.txt`
- `experiments/expts/mixed-bc-scale-all-hard/pretrain-mixed-bc-hard-0/logs_best.txt`

Eval logs:

- `experiments/expts/mixed-bc-scale-all-soft/eval-mixed-bc-soft-0/logs_best.txt`
- `experiments/expts/mixed-bc-scale-all-hard/eval-mixed-bc-hard-0/logs_best.txt`

Useful keys to compare:

- `val_err`, `val_err_interior`
- `val_bc_violation_raw`, `val_bc_violation_final`
- `test_err`, `test_err_interior`
- `test_bc_violation_raw`, `test_bc_violation_final`

Expected behavior:

- hard mode should drive `*_bc_violation_final` close to zero
- soft mode should reduce boundary violations versus BC-off baseline without exact projection

## 10) Quick smoke alternative (small local test)

For a fast sanity pass before full-scale runs:

```bash
MODES="soft hard" bash scripts/utils/run_local_smoke_train_eval_bc_constraints.sh
```

This runs tiny datasets and one-epoch train/eval checks for selected modes.

## 11) SLURM note

`scripts/slurm/pretrain/submit_pretrain_mixed.sh` currently has a legacy mixed-dataset existence check (`data/mixed/_train_mixed_32k.h5`).  
For BC runs, either:

1. run `train.py` directly in your SLURM job command, or
2. adapt that script’s dataset check for `_train_mixed_32k_bc.h5` before using it for BC configs.

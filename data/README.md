# Dataset Information

This directory contains the datasets used in the thesis experiments.

## Structure

- `raw/`: Raw downloaded datasets (gitignored)
- `processed/`: Preprocessed datasets ready for training (gitignored)

## Datasets

### PDEBench

PDEBench is a benchmark dataset for PDE solving with machine learning.

**Download:**
```bash
# Visit: https://github.com/pdebench/PDEBench
# Follow instructions to download specific PDE datasets
```

**Location:** Place downloaded data in `data/raw/pdebench/`

### Navier-Stokes

2D incompressible Navier-Stokes equations dataset.

**Download:**
```bash
# Option 1: Generate using scripts in this repo (recommended)
python scripts/generate_navier_stokes.py

# Option 2: Download from source
# Visit: [Add source URL]
```

**Location:** Place data in `data/raw/navier_stokes/`

## Data Format

All datasets should be stored in HDF5 format with the following structure:
- `/train`: Training samples
- `/val`: Validation samples
- `/test`: Test samples

Each sample should contain:
- `input`: Initial conditions or input fields
- `output`: Solution or target fields
- `metadata`: Any additional information (timesteps, parameters, etc.)

## Preprocessing

Preprocessing is handled automatically by the dataloaders in `src/thesis_fno/data/`.

To manually preprocess data:
```bash
python scripts/preprocess_data.py --dataset pdebench --config configs/datasets/pdebench.yaml
```

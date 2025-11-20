# Thesis: Fourier Neural Operators with Physical Constraints

This repository contains the implementation and experiments for a thesis on incorporating physical constraints into Fourier Neural Operators (FNOs) for solving partial differential equations.

## Project Structure

- `configs/`: Configuration files for models, datasets, and training
- `src/thesis_fno/`: Main source code
  - `data/`: Dataset loaders and transforms
  - `models/`: FNO model implementations
  - `constraints/`: Physical constraint implementations
  - `training/`: Training loops and utilities
  - `utils/`: Helper utilities
- `scripts/`: Executable scripts for training and evaluation
- `experiments/`: Experiment logs and configurations
- `notebooks/`: Jupyter notebooks for analysis
- `data/`: Dataset storage (see data/README.md)
- `models/`: Model checkpoints
- `tests/`: Unit tests

## Installation

### PyTorch with CUDA Support

For GPU acceleration, install PyTorch with CUDA support first:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install Package

Then install the package and dependencies:

```bash
pip install -e .

# With development tools
pip install -e ".[dev]"

# With notebook support
pip install -e ".[notebooks]"

# Install everything
pip install -e ".[dev,notebooks]"
```

## Usage

See individual configuration files in `configs/` for training different model variants.

## License

See LICENSE file for details.

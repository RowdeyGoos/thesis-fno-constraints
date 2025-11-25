#!/bin/bash
# Setup script for DAIC cluster environment
# Run this once to set up your environment

echo "=========================================="
echo "Setting up environment for DAIC cluster"
echo "=========================================="

# Load required modules
module purge
module load python/3.10
module load cuda/11.8
module load cudnn/8.6

# Create virtual environment
VENV_DIR=~/venv/thesis-fno
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo "Installing project dependencies..."
pip install -e ".[dev,notebooks]"

# Create necessary directories
mkdir -p experiments/logs
mkdir -p experiments/runs
mkdir -p experiments/configs_used
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints

echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  module load python/3.10 cuda/11.8 cudnn/8.6"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To submit a job, use:"
echo "  sbatch scripts/submit_job.sh"

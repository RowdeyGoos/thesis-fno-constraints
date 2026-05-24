#!/bin/bash
# Setup Python virtual environment for the flattened neuraloperators project on DAIC

set -e

echo "=========================================="
echo "Setting up neuraloperators-TL-scaling Environment on DAIC"
echo "=========================================="

# Load required modules
echo "Loading modules..."
module purge
module load python/3.10
module load cuda/11.3
module load cudnn/8.2

# Create virtual environment
VENV_DIR=~/venv/neuraloperators
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Exiting..."
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_DIR..."
python -m venv "$VENV_DIR"

# Activate environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch 1.12.0 with CUDA 11.3 support
echo "Installing PyTorch 1.12.0 with CUDA 11.3..."
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Navigate to project directory (if we're in it)
if [ -f "requirements.txt" ]; then
    PROJECT_DIR=$(pwd)
elif [ -f "../../requirements.txt" ]; then
    PROJECT_DIR="$(cd ../.. && pwd)"
else
    echo "Warning: requirements.txt not found. Assuming you'll install dependencies manually."
    PROJECT_DIR="~/thesis-fno-constraints"
fi

# Install project requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Installing project dependencies from requirements.txt..."
    # Skip torch and torchvision since we already installed specific versions
    grep -v "^torch" "$PROJECT_DIR/requirements.txt" | grep -v "^torchvision" | pip install -r /dev/stdin
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: $VENV_DIR"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  alias neuralop-env='source $VENV_DIR/bin/activate'"
echo ""
echo "Test PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

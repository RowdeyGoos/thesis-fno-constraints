#!/bin/bash
# Build Apptainer/Singularity container directly (alternative to Docker build)
# Only use this if you have Apptainer installed locally

set -e

echo "=========================================="
echo "Building Apptainer Container for neuraloperators-TL-scaling"
echo "=========================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check if apptainer.def exists
if [ ! -f "apptainer.def" ]; then
    echo "Error: apptainer.def not found"
    exit 1
fi

# Build with Apptainer
echo "Building Apptainer container..."
apptainer build --fakeroot neuraloperators.sif apptainer.def

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Apptainer container built successfully!"
    echo ""
    ls -lh neuraloperators.sif
    echo ""
    echo "Test the container:"
    echo "  apptainer exec neuraloperators.sif python -c 'import torch; print(torch.__version__)'"
else
    echo "✗ Build failed"
    exit 1
fi

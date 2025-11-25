#!/bin/bash
# Build Apptainer image directly (requires Apptainer/Singularity installed)
# This can be run on a system with Apptainer, or on DAIC itself

set -e  # Exit on error

echo "=========================================="
echo "Building Apptainer Container"
echo "=========================================="

IMAGE_NAME="thesis-fno"
DEF_FILE="apptainer.def"
OUTPUT_FILE="${IMAGE_NAME}.sif"

# Check if Apptainer is available
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not installed or not in PATH"
    echo "Please install Apptainer or use Docker build method"
    exit 1
fi

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "Error: $DEF_FILE not found"
    exit 1
fi

echo ""
echo "Building from: $DEF_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Build the container
# Note: Building requires sudo/fakeroot on most systems
if [ "$EUID" -eq 0 ]; then
    # Running as root
    apptainer build --force $OUTPUT_FILE $DEF_FILE
else
    # Try with fakeroot (if available), otherwise prompt for sudo
    if apptainer build --help | grep -q "fakeroot"; then
        echo "Using fakeroot for build..."
        apptainer build --fakeroot --force $OUTPUT_FILE $DEF_FILE
    else
        echo "Note: Building may require sudo privileges"
        sudo apptainer build --force $OUTPUT_FILE $DEF_FILE
    fi
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Container: $OUTPUT_FILE"
echo ""
echo "Test the container:"
echo "  apptainer exec $OUTPUT_FILE python -c 'import torch; print(torch.__version__)'"
echo ""
echo "Run training:"
echo "  apptainer exec --nv $OUTPUT_FILE python scripts/train.py --config configs/training/default.yaml"

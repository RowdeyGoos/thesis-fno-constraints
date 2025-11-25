#!/bin/bash
# Build Docker container for neuraloperators-TL-scaling project
# This is separate from the main thesis project to avoid dependency conflicts

set -e  # Exit on error

echo "=========================================="
echo "Building neuraloperators-TL-scaling Container"
echo "=========================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found in current directory"
    exit 1
fi

# Build Docker image
echo "Building Docker image (this may take a few minutes)..."
docker build -t neuraloperators:latest .

# Check build success
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Container built successfully!"
    echo ""
    echo "Container details:"
    docker images neuraloperators:latest
    echo ""
    
    # Export to tar file for transfer
    echo "Exporting container to tar file..."
    docker save neuraloperators:latest -o neuraloperators_latest.tar
    
    if [ $? -eq 0 ]; then
        echo "✓ Container exported to neuraloperators_latest.tar"
        echo ""
        echo "File size:"
        ls -lh neuraloperators_latest.tar
        echo ""
        echo "=========================================="
        echo "Next Steps:"
        echo "=========================================="
        echo "1. Test locally (optional):"
        echo "   docker run --rm neuraloperators:latest python -c 'import torch; print(torch.__version__)'"
        echo ""
        echo "2. Transfer to DAIC:"
        echo "   bash scripts/container/transfer_container.sh <your-netid>"
        echo ""
        echo "3. Or use Makefile:"
        echo "   make transfer-container NETID=<your-netid>"
        echo ""
    else
        echo "✗ Error exporting container"
        exit 1
    fi
else
    echo "✗ Container build failed"
    exit 1
fi

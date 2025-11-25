#!/bin/bash
# Transfer Docker image to DAIC and convert to Apptainer
# Run this from your local machine

set -e

echo "=========================================="
echo "Transfer Container to DAIC"
echo "=========================================="

# Configuration
IMAGE_TAR="thesis-fno_latest.tar"
NETID=${1:-$USER}
REMOTE_HOST="login.daic.tudelft.nl"

# Check if tar file exists
if [ ! -f "$IMAGE_TAR" ]; then
    echo "Error: $IMAGE_TAR not found"
    echo "Please run: bash scripts/build_container.sh first"
    exit 1
fi

echo ""
echo "Transferring to: ${NETID}@${REMOTE_HOST}"
echo "File: $IMAGE_TAR"
echo ""

# Transfer the tar file
echo "Step 1: Uploading container image (this may take a while)..."
scp $IMAGE_TAR ${NETID}@${REMOTE_HOST}:~/

echo ""
echo "Step 2: Converting to Apptainer format on DAIC..."

# SSH to DAIC and convert
ssh ${NETID}@${REMOTE_HOST} << 'ENDSSH'
    cd ~
    
    # Check if Apptainer is available
    if ! command -v apptainer &> /dev/null; then
        echo "Error: Apptainer not available on DAIC"
        echo "Loading module..."
        module load apptainer 2>/dev/null || module load singularity 2>/dev/null
    fi
    
    # Convert Docker tar to Apptainer SIF
    echo "Converting Docker archive to Apptainer SIF..."
    apptainer build thesis-fno.sif docker-archive://thesis-fno_latest.tar
    
    # Clean up tar file
    echo "Cleaning up tar file..."
    rm thesis-fno_latest.tar
    
    echo ""
    echo "Container ready at: ~/thesis-fno.sif"
    
    # Test the container
    echo ""
    echo "Testing container..."
    apptainer exec thesis-fno.sif python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
ENDSSH

echo ""
echo "=========================================="
echo "Transfer complete!"
echo "=========================================="
echo ""
echo "The container is now available on DAIC at: ~/thesis-fno.sif"
echo ""
echo "To use it:"
echo "  1. SSH to DAIC: ssh ${NETID}@${REMOTE_HOST}"
echo "  2. Move to project: cd thesis-fno-constraints"
echo "  3. Use container scripts: bash scripts/submit_job_container.sh"

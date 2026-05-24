#!/bin/bash
# Transfer neuraloperators container to DAIC cluster

set -e

# Check if netid is provided
if [ -z "$1" ]; then
    echo "Error: NetID not provided"
    echo "Usage: bash transfer_container.sh <your-netid>"
    exit 1
fi

NETID=$1
CONTAINER_TAR="neuraloperators_latest.tar"
REMOTE_PATH="~/neuraloperators"
REMOTE_HOST="${NETID}@login.daic.tudelft.nl"

echo "=========================================="
echo "Transferring Container to DAIC"
echo "=========================================="
echo "NetID: $NETID"
echo "Container: $CONTAINER_TAR"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Check if container tar exists
if [ ! -f "$CONTAINER_TAR" ]; then
    echo "Error: $CONTAINER_TAR not found"
    echo "Please build the container first:"
    echo "  bash scripts/container/build_container.sh"
    exit 1
fi

# Transfer tar file
echo "Transferring container (this may take several minutes)..."
scp "$CONTAINER_TAR" "${REMOTE_HOST}:${REMOTE_PATH}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Transfer complete!"
    echo ""
    echo "=========================================="
    echo "Next Steps (on DAIC):"
    echo "=========================================="
    echo "1. SSH to DAIC:"
    echo "   ssh ${REMOTE_HOST}"
    echo ""
    echo "2. Convert to Apptainer/Singularity format:"
    echo "   cd ${REMOTE_PATH}"
    echo "   module load apptainer"
    echo "   apptainer build neuraloperators.sif docker-archive://neuraloperators_latest.tar"
    echo ""
    echo "3. Test the container:"
    echo "   apptainer exec neuraloperators.sif python -c 'import torch; print(torch.__version__)'"
    echo ""
    echo "4. Submit a job:"
    echo "   sbatch scripts/slurm/pretrain/submit_pretrain_single_ddp.sh"
    echo ""
else
    echo "✗ Transfer failed"
    exit 1
fi

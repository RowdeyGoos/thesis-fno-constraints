#!/bin/bash
# Build Apptainer/Singularity container locally
# This should be run on your local machine with Docker installed

set -e  # Exit on error

echo "=========================================="
echo "Building FNO Training Container"
echo "=========================================="

# Configuration
IMAGE_NAME="thesis-fno"
VERSION="latest"
DOCKERFILE="Dockerfile"
APPTAINER_DEF="apptainer.def"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo ""
echo "Step 1: Building Docker image..."
docker build -t ${IMAGE_NAME}:${VERSION} -f ${DOCKERFILE} .

echo ""
echo "Step 2: Saving Docker image to tar..."
docker save ${IMAGE_NAME}:${VERSION} -o ${IMAGE_NAME}_${VERSION}.tar

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Docker image: ${IMAGE_NAME}:${VERSION}"
echo "Tar file: ${IMAGE_NAME}_${VERSION}.tar"
echo ""
echo "Next steps:"
echo "1. Test locally:"
echo "   docker run -it --gpus all ${IMAGE_NAME}:${VERSION}"
echo ""
echo "2. Transfer to DAIC:"
echo "   scp ${IMAGE_NAME}_${VERSION}.tar <netid>@login.daic.tudelft.nl:~/"
echo ""
echo "3. On DAIC, convert to Apptainer:"
echo "   apptainer build ${IMAGE_NAME}.sif docker-archive://${IMAGE_NAME}_${VERSION}.tar"
echo ""
echo "Or use the convenience script:"
echo "   bash scripts/transfer_container.sh"

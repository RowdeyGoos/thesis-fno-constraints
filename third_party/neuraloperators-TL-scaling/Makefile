# Makefile for neuraloperators-TL-scaling DAIC cluster setup

.PHONY: help build-container transfer-container test-container clean

# Default target
help:
	@echo "neuraloperators-TL-scaling - DAIC Cluster Setup"
	@echo "================================================"
	@echo ""
	@echo "Container Management:"
	@echo "  make build-container      - Build Docker container locally"
	@echo "  make transfer-container   - Transfer container to DAIC (requires NETID=<your-netid>)"
	@echo "  make test-container       - Test container locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               - Remove temporary container files"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make build-container"
	@echo "  make transfer-container NETID=mynetid"
	@echo ""

# Build Docker container
build-container:
	@echo "Building Docker container for neuraloperators..."
	bash scripts/container/build_container.sh

# Transfer container to DAIC
transfer-container:
ifndef NETID
	@echo "Error: NETID not set"
	@echo "Usage: make transfer-container NETID=<your-netid>"
	@exit 1
endif
	@echo "Transferring container to DAIC (netid: $(NETID))..."
	bash scripts/container/transfer_container.sh $(NETID)

# Test container locally
test-container:
	@echo "Testing container locally..."
	@docker run --rm neuraloperators:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
	@docker run --rm neuraloperators:latest python -c "import h5py, matplotlib, numpy, scipy, wandb; print('All packages OK')"

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	@rm -f neuraloperators_latest.tar
	@echo "Done!"

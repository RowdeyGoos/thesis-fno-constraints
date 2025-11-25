.PHONY: help build-container transfer-container clean check-container test-container

# Default target
help:
	@echo "DAIC Cluster - Make Commands"
	@echo "=============================="
	@echo ""
	@echo "Container Management:"
	@echo "  make build-container      - Build Docker container locally"
	@echo "  make transfer-container   - Transfer container to DAIC (requires NETID=<your-netid>)"
	@echo "  make test-container       - Test container locally"
	@echo "  make check-container      - Check if container files exist"
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
	@echo "Building Docker container..."
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
	@docker run --rm thesis-fno:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if container files exist
check-container:
	@echo "Checking container files..."
	@if [ -f "Dockerfile" ]; then echo "✓ Dockerfile found"; else echo "✗ Dockerfile missing"; fi
	@if [ -f "apptainer.def" ]; then echo "✓ apptainer.def found"; else echo "✗ apptainer.def missing"; fi
	@if [ -f "thesis-fno_latest.tar" ]; then echo "✓ Container tar found"; else echo "✗ Container tar not built yet"; fi

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	@rm -f thesis-fno_latest.tar
	@echo "Done!"

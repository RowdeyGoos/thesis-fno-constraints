#!/bin/bash
# Quick test script to verify the reproduction pipeline works
# This runs a minimal example with a small dataset and few epochs

set -e  # Exit on error

echo "=========================================="
echo "Neural Operators Reproduction Quick Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
TEST_DATA_DIR="data/test_poisson"
NTRAIN=100
NVAL=20
NTEST=20
MAX_EPOCHS=5

echo -e "${YELLOW}Step 1: Creating test data directory${NC}"
mkdir -p $TEST_DATA_DIR
echo "✓ Created $TEST_DATA_DIR"
echo ""

echo -e "${YELLOW}Step 2: Generating minimal test dataset${NC}"
echo "  - Training samples: $NTRAIN"
echo "  - Validation samples: $NVAL"
echo "  - Test samples: $NTEST"
echo "  - Grid resolution: 128x128"
echo ""

python utils/gen_data_poisson.py \
    --ntrain $NTRAIN \
    --nval $NVAL \
    --ntest $NTEST \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath $TEST_DATA_DIR \
    --e1 1 \
    --e2 5

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset generation completed${NC}"
else
    echo -e "${RED}✗ Dataset generation failed${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 3: Computing normalization scales${NC}"
python compute_scales.py \
    $TEST_DATA_DIR/poissons_train_k1_5_${NTRAIN}.h5 \
    $TEST_DATA_DIR/poissons_train_k1_5_${NTRAIN}_scales.npy

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Scales computed successfully${NC}"
else
    echo -e "${RED}✗ Scale computation failed${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 4: Creating test configuration${NC}"
cat > config/operators_poisson_quicktest.yaml << 'EOF'
default: &DEFAULT
  num_data_workers: 1
  model: 'fno'
  depth: 5
  in_dim: 2
  out_dim: 1
  dropout: 0
  Lx: !!float 1.0
  Ly: !!float 1.0
  nx: 256
  ny: 256
  loss_style: 'mean'
  loss_func: 'mse'
  optimizer: 'adam'
  scheduler: 'none'
  lr: !!float 1.0
  max_epochs: 500
  batch_size: 25
  log_to_screen: !!bool True
  save_checkpoint: !!bool False
  seed: 0
  plot_figs: !!bool False
  pack_data: !!bool False
  entity: 'test'
  project: 'test-reproduction'
  log_to_wandb: !!bool False
  distill: !!bool False
  subsample: 1

poisson: &poisson
  <<: *DEFAULT
  batch_size: 512
  valid_batch_size: 512
  nx: 128
  ny: 128
  log_to_wandb: !!bool False
  save_checkpoint: !!bool False
  max_epochs: 500
  scheduler: 'cosine'
  plot_figs: !!bool False
  loss_style: 'sum'
  system: 'poisson'
  model: 'fno'
  layers: [64, 64, 64, 64, 64]
  modes1: [65, 65, 65, 65]
  modes2: [65, 65, 65, 65]
  fc_dim: 128
  in_dim: 4
  out_dim: 1
  mode_cut: 16
  embed_cut: 64
  fc_cut: 2
  optimizer: 'adam'
  lr: 1E-3
  pack_data: !!bool False

poisson-quicktest:
  <<: *poisson
  train_path: 'data/test_poisson/poissons_train_k1_5_100.h5'
  val_path: 'data/test_poisson/poissons_val_k1_5_20.h5'
  test_path: 'data/test_poisson/poissons_test_k1_5_20.h5'
  scales_path: 'data/test_poisson/poissons_train_k1_5_100_scales.npy'
  batch_size: 10
  valid_batch_size: 10
  max_epochs: 5
  log_to_wandb: !!bool False
  save_checkpoint: !!bool False
  plot_figs: !!bool False
  mode_cut: 16
  embed_cut: 32
  fc_cut: 2
EOF

echo -e "${GREEN}✓ Configuration file created${NC}"
echo ""

echo -e "${YELLOW}Step 5: Running quick training test (${MAX_EPOCHS} epochs)${NC}"
echo "This may take a few minutes depending on your hardware..."
echo ""

python train.py \
    --yaml_config config/operators_poisson_quicktest.yaml \
    --config poisson-quicktest \
    --run_num quicktest \
    --root_dir ./results

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 6: Checking results${NC}"
RESULTS_DIR="./results/expts/poisson-quicktest/quicktest"
if [ -d "$RESULTS_DIR" ]; then
    echo "Results directory: $RESULTS_DIR"
    echo ""
    echo "Directory contents:"
    ls -lh $RESULTS_DIR
    echo ""
    
    if [ -f "$RESULTS_DIR/logs/train.log" ]; then
        echo "Last 15 lines of training log:"
        echo "----------------------------------------"
        tail -n 15 $RESULTS_DIR/logs/train.log
        echo "----------------------------------------"
        echo ""
    fi
    
    echo -e "${GREEN}✓ All checks passed!${NC}"
else
    echo -e "${RED}✗ Results directory not found${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Quick test completed successfully! ✓${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check the training log for decreasing loss"
echo "2. Generate full dataset (32k samples) for paper reproduction"
echo "3. Run full training (500 epochs)"
echo ""
echo "See REPRODUCTION_GUIDE.md for detailed instructions."

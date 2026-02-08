# Reproduction Guide for Neural Operators Transfer Learning Paper

This guide provides step-by-step instructions to reproduce results from "Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior of Neural Operators" by Subramanian et al. (2023).

## 📋 Overview

The paper investigates:
- **Scaling behavior** of Fourier Neural Operators (FNOs) on three PDE systems
- **Transfer learning** capabilities across different downstream tasks
- **Data efficiency** through pre-training on diverse PDE problems

**PDE Systems studied:**
1. **Poisson's equation** - Elliptic PDE for steady-state diffusion
2. **Advection-Diffusion** - Transport phenomena with diffusion
3. **Helmholtz equation** - Wave propagation problems

---

## 🚀 Quick Start (Minimal Verification)

Let's start with a minimal example to verify everything works correctly.

### Step 1: Setup Environment

#### Option A: Using pip (Local/Development)

```bash
cd third_party/neuraloperators-TL-scaling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

#### Option B: Using Container (Recommended for DAIC)

See `DAIC_SETUP.md` for container-based setup on the cluster.

### Step 2: Generate Minimal Test Dataset

We'll generate a small Poisson dataset for quick verification:

```bash
# Create data directory
mkdir -p data/test_poisson

# Generate minimal dataset (100 train, 20 val, 20 test samples)
python utils/gen_data_poisson.py \
    --ntrain 100 \
    --nval 20 \
    --ntest 20 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/test_poisson \
    --e1 1 \
    --e2 5

# This creates:
# - data/test_poisson/poissons_train_k1_5_100.h5
# - data/test_poisson/poissons_val_k1_5_20.h5
# - data/test_poisson/poissons_test_k1_5_20.h5
```

**Parameters explained:**
- `--ntrain`: Number of training samples
- `--nval`: Number of validation samples
- `--ntest`: Number of test samples
- `--ng`: Number of RBF centers for source function (144 = 12x12 grid)
- `--n`: Grid resolution (128x128)
- `--sparse`: Use sparse source functions (more realistic)
- `--e1, --e2`: Range for diffusion tensor eigenvalues [1, 5]

### Step 3: Generate Normalization Scales

The FNO requires input normalization. We'll compute median-based scales from the training data:

```bash
# Compute scales for the test dataset
python utils/compute_scales.py \
    --datapath data/test_poisson \
    --filename poissons_train_k1_5_100.h5

# This will automatically save to: data/test_poisson/poissons_train_k1_5_100_scales.npy
```

**About the compute_scales script:**
- Computes median-based normalization scales from training data
- Handles source function norms, tensor components, and solution field maxima
- Automatically generates output filename (`_scales.npy`)
- Can be run standalone for any dataset

For more options:
```bash
python utils/compute_scales.py --help
```

### Step 4: Create Test Configuration

Create a minimal config for testing:

```bash
cat > config/operators_poisson_test.yaml << 'EOF'
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
  save_checkpoint: !!bool True
  max_epochs: 500
  scheduler: 'cosine'
  plot_figs: !!bool True
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

poisson-test: &poisson_test
  <<: *poisson
  train_path: 'data/test_poisson/poissons_train_k1_5_100.h5'
  val_path: 'data/test_poisson/poissons_val_k1_5_20.h5'
  test_path: 'data/test_poisson/poissons_test_k1_5_20.h5'
  scales_path: 'data/test_poisson/poissons_train_k1_5_100_scales.npy'
  batch_size: 10
  valid_batch_size: 10
  max_epochs: 5  # Just 5 epochs for testing
  log_to_wandb: !!bool False
  save_checkpoint: !!bool False
  mode_cut: 16
  embed_cut: 32
  fc_cut: 2
EOF
```

### Step 5: Run Minimal Training Test

```bash
# Single GPU test (no distributed training)
python train.py \
    --yaml_config config/operators_poisson_test.yaml \
    --config poisson-test \
    --run_num test_run \
    --root_dir ./results

# Expected output:
# - Training progress for 5 epochs
# - Validation loss printed each epoch
# - Model should overfit on small dataset (loss decreasing)
```

### Step 6: Verify Results

```bash
# Check if training completed
ls -lh results/expts/poisson-test/test_run/

# Expected files:
# - logs/train.log (training logs)
# - figures/ (if plot_figs enabled)

# Check the log for decreasing loss
tail -n 20 results/expts/poisson-test/test_run/logs/train.log
```

---

## 📊 Full Reproduction (Paper Results)

Once the minimal test works, proceed to full-scale reproduction:

### Step 1: Generate Full Datasets

#### Poisson's Equation (k ∈ [1, 5])

```bash
# Create full data directory
mkdir -p data/poisson_full

# Generate 32k train, 4k val, 4k test samples
python utils/gen_data_poisson.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/poisson_full \
    --e1 1 \
    --e2 5

# Compute normalization scales for the training data
python utils/compute_scales.py \
    --datapath data/poisson_full \
    --filename poissons_train_k1_5_32768.h5

# Expected time: ~30-60 minutes for data generation, ~5-10 minutes for scale computation
```

**Output files:**
- `data/poisson_full/poissons_train_k1_5_32768.h5` (training data)
- `data/poisson_full/poissons_val_k1_5_4096.h5` (validation data)
- `data/poisson_full/poissons_test_k1_5_4096.h5` (test data)
- `data/poisson_full/poissons_train_k1_5_32768_scales.npy` (scales for normalization)

#### Advection-Diffusion (ADR ∈ [0.2, 1])

```bash
mkdir -p data/advdiff_full

python utils/gen_data_advdiff.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/advdiff_full \
    --adr1 0.2 \
    --adr2 1.0

# Compute normalization scales
python utils/compute_scales.py \
    --datapath data/advdiff_full \
    --filename advdiff_train_adr0.2_1.0_32768.h5
```

#### Helmholtz Equation (ω ∈ [1, 10])

```bash
mkdir -p data/helmholtz_full

python utils/gen_data_helmholtz.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/helmholtz_full \
    --o1 1 \
    --o2 10

# Compute normalization scales
python utils/compute_scales.py \
    --datapath data/helmholtz_full \
    --filename helmholtz_train_o1_10_32768.h5
```

### Step 2: Update Full Configuration Files

Edit the existing config files to point to your generated data:

**For Poisson** (`config/operators_poisson.yaml`):
```yaml
poisson-scale-k1_5: &poisson_scale_k1_5
  <<: *poisson
  train_path: 'data/poisson_full/poissons_train_k1_5_32768.h5'
  val_path: 'data/poisson_full/poissons_val_k1_5_4096.h5'
  test_path: 'data/poisson_full/poissons_test_k1_5_4096.h5'
  scales_path: 'data/poisson_full/poissons_train_k1_5_32768_scales.npy'
  batch_size: 128
  valid_batch_size: 128
  log_to_wandb: !!bool True  # Enable if you have W&B setup
  mode_cut: 32
  embed_cut: 64
  fc_cut: 2
```

Repeat for `config/operators_ad.yaml` and `config/operators_helmholtz.yaml`.

### Step 3: Run Full Training

#### Single PDE System (4 GPU example)

```bash
# Setup environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run with 4 GPUs
srun -l -n 4 --cpus-per-task=10 --gpus-per-node 4 \
    bash -c "source export_DDP_vars.sh && \
    python train.py \
        --yaml_config=config/operators_poisson.yaml \
        --config=poisson-scale-k1_5 \
        --run_num=run_0 \
        --root_dir=./results"

# Expected time: ~8-12 hours for 500 epochs on 4x A100 GPUs
```

#### All Three Systems (Job Array)

If on a SLURM cluster:

```bash
# Submit array job for all three PDE systems
sbatch scripts/slurm/submit_job_array_container.sh
```

### Step 4: Evaluation and Analysis

#### Evaluate Trained Model

```bash
# Run evaluation on test set
python eval.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k1_5 \
    --run_num run_0 \
    --root_dir ./results \
    --weights ./results/expts/poisson-scale-k1_5/run_0/checkpoints/ckpt_best.tar

# This generates:
# - Test set predictions
# - Error metrics (MSE, relative L2 error)
# - Visualization plots (if enabled)
```

#### Expected Results (Paper Table 1)

| PDE System | Test L2 Error | Training Time (500 epochs) |
|------------|---------------|----------------------------|
| Poisson (k∈[1,5]) | ~0.02-0.03 | ~10 hours (4 GPUs) |
| Advection-Diffusion | ~0.03-0.05 | ~10 hours (4 GPUs) |
| Helmholtz (ω∈[1,10]) | ~0.05-0.08 | ~10 hours (4 GPUs) |

---

## 🔬 Advanced: Scaling and Transfer Learning Experiments

### Data Scaling Experiments (Figure 3a)

Test model performance with different training dataset sizes to characterize scaling behavior.

#### Method 1: Using `subsample` parameter

The `subsample` parameter allows you to train on a fraction of your dataset without regenerating data:

```yaml
# In config file, e.g., config/operators_poisson.yaml
poisson-scale-k1_5-subsample2: &poisson_k1_5_sub2
  <<: *poisson_scale_k1_5
  subsample: 2  # Uses 50% of data (32768 / 2 = 16384 samples)
  
poisson-scale-k1_5-subsample4: &poisson_k1_5_sub4
  <<: *poisson_scale_k1_5
  subsample: 4  # Uses 25% of data (32768 / 4 = 8192 samples)
```

Then run experiments:

```bash
# Full data (32k samples)
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k1_5 \
    --run_num scale_full \
    --root_dir ./results

# 50% data (16k samples)
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k1_5-subsample2 \
    --run_num scale_50pct \
    --root_dir ./results

# 25% data (8k samples)
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-scale-k1_5-subsample4 \
    --run_num scale_25pct \
    --root_dir ./results
```

#### Method 2: Generate datasets of different sizes

For more control, generate separate datasets:

```bash
# 4k training samples
python utils/gen_data_poisson.py \
    --ntrain 4096 --nval 512 --ntest 1024 \
    --ng 144 --n 128 --sparse \
    --datapath data/poisson_4k \
    --e1 1 --e2 5

# 8k training samples
python utils/gen_data_poisson.py \
    --ntrain 8192 --nval 1024 --ntest 1024 \
    --ng 144 --n 128 --sparse \
    --datapath data/poisson_8k \
    --e1 1 --e2 5

# 16k training samples
python utils/gen_data_poisson.py \
    --ntrain 16384 --nval 2048 --ntest 2048 \
    --ng 144 --n 128 --sparse \
    --datapath data/poisson_16k \
    --e1 1 --e2 5
```

Then create separate configs for each and train.

### Model Scaling Experiments

Test different FNO architectures:

```yaml
# Small model
poisson-scale-k1_5-small:
  <<: *poisson_scale_k1_5
  mode_cut: 16
  embed_cut: 32
  fc_cut: 1

# Medium model (baseline)
poisson-scale-k1_5-medium:
  <<: *poisson_scale_k1_5
  mode_cut: 32
  embed_cut: 64
  fc_cut: 2

# Large model
poisson-scale-k1_5-large:
  <<: *poisson_scale_k1_5
  mode_cut: 64
  embed_cut: 128
  fc_cut: 4
```

### Transfer Learning Experiments (Data Efficiency Study)

This section demonstrates how pre-training improves data efficiency by comparing fine-tuned models against models trained from scratch on limited downstream data.

**Experiment Setup:**
- **Source Domain**: Poisson k∈[1,5] (32k training samples)
- **Target Domain**: Poisson k∈[5,10] (varying sample sizes)
- **Sample Sizes**: 16, 64, 256, 1k, 4k downstream examples
- **Comparison**: Fine-tuned (with pre-training) vs From-scratch (no pre-training)

#### Step 1: Pre-train on Source Domain

First, complete the pretraining on Poisson k∈[1,5] using the array script:

```bash
# Submit pretraining array job (includes all three PDE systems)
sbatch scripts/slurm/submit_job_array_container_single_gpu.sh

# Or submit just Poisson pretraining manually:
sbatch scripts/slurm/submit_job_container.sh
```

Wait for the job to complete and note the job ID (e.g., `12345`).

#### Step 2: Generate Target Domain Dataset

Generate the downstream dataset for Poisson k∈[5,10]:

```bash
mkdir -p data/poisson

# Generate full dataset (32k training samples)
python utils/gen_data_poisson.py \
    --ntrain 32768 \
    --nval 4096 \
    --ntest 4096 \
    --ng 144 \
    --n 128 \
    --sparse \
    --datapath data/poisson \
    --e1 5 \
    --e2 10

# Output files will be named: _train_k5_10_32k.h5, _val_k5_10_4k.h5, _test_k5_10_4k.h5

# Compute normalization scales
python utils/compute_scales.py \
    --datapath data/poisson \
    --filename _train_k5_10_32k.h5
```

#### Step 3: Update Checkpoint Path

Update the config and script with your actual pretraining job ID:

```bash
# Replace JOBID with your actual pretraining job ID (e.g., 12345)
bash scripts/utils/update_checkpoint_path.sh <JOBID>

# Example:
bash scripts/utils/update_checkpoint_path.sh 12345
```

This automatically updates:
- `config/operators_poisson.yaml` - Fine-tuning configs with correct checkpoint path
- `scripts/slurm/submit_transfer_learning_array.sh` - SLURM script checkpoint verification

#### Step 4: Verify Configuration

The config file now contains all necessary configurations:

```yaml
# Fine-tuning experiments (5 configs)
poisson-k5_10-finetune-16    # 16 samples, with pre-training
poisson-k5_10-finetune-64    # 64 samples, with pre-training
poisson-k5_10-finetune-256   # 256 samples, with pre-training
poisson-k5_10-finetune-1k    # 1k samples, with pre-training
poisson-k5_10-finetune-4k    # 4k samples, with pre-training

# From-scratch experiments (5 configs)
poisson-k5_10-scratch-16     # 16 samples, no pre-training
poisson-k5_10-scratch-64     # 64 samples, no pre-training
poisson-k5_10-scratch-256    # 256 samples, no pre-training
poisson-k5_10-scratch-1k     # 1k samples, no pre-training
poisson-k5_10-scratch-4k     # 4k samples, no pre-training
```

Each config uses the `subsample` parameter to train on a subset of the 32k dataset:
- 16 samples: `subsample: 2048` (32768 / 2048)
- 64 samples: `subsample: 512` (32768 / 512)
- 256 samples: `subsample: 128` (32768 / 128)
- 1k samples: `subsample: 32` (32768 / 32)
- 4k samples: `subsample: 8` (32768 / 8)

#### Step 5: Submit Transfer Learning Experiments

```bash
# Submit array job for all 10 experiments (5 fine-tuning + 5 from-scratch)
sbatch scripts/slurm/submit_transfer_learning_array.sh

# This will launch 10 array tasks (0-9):
# Tasks 0-4: Fine-tuning experiments
# Tasks 5-9: From-scratch experiments
```

#### Step 6: Monitor and Analyze Results

Check job status:
```bash
squeue -u $USER | grep transfer-learning
```

View logs:
```bash
# Check specific task log
cat experiments/neuralop-transfer-learning-<JOBID>-<TASKID>.out

# Check all logs
ls -lh experiments/neuralop-transfer-learning-*
```

Results will be saved in:
```
experiments/expts/
├── poisson-k5_10-finetune-16/
├── poisson-k5_10-finetune-64/
├── poisson-k5_10-finetune-256/
├── poisson-k5_10-finetune-1k/
├── poisson-k5_10-finetune-4k/
├── poisson-k5_10-scratch-16/
├── poisson-k5_10-scratch-64/
├── poisson-k5_10-scratch-256/
├── poisson-k5_10-scratch-1k/
└── poisson-k5_10-scratch-4k/
```

#### Expected Results

According to the paper, fine-tuned models should significantly outperform from-scratch models, especially with limited data:

| Samples | From-Scratch L2 Error | Fine-tuned L2 Error | Improvement |
|---------|----------------------|---------------------|-------------|
| 16      | ~0.15-0.20          | ~0.05-0.08         | ~60-70%     |
| 64      | ~0.10-0.15          | ~0.04-0.06         | ~50-60%     |
| 256     | ~0.08-0.12          | ~0.03-0.05         | ~40-50%     |
| 1k      | ~0.05-0.08          | ~0.03-0.04         | ~30-40%     |
| 4k      | ~0.04-0.06          | ~0.02-0.03         | ~20-30%     |

The gap narrows as more data becomes available, demonstrating the value of pre-training for low-data regimes.

---

### Advanced: Manual Transfer Learning Configuration

If you prefer manual control instead of using the array script:

#### Step 3 (Alternative): Add Fine-tuning Config Manually

Add to `config/operators_poisson.yaml`:

```yaml
poisson-k5_10-finetune-custom: &poisson_k5_10_ft_custom
  <<: *poisson
  train_path:    'data/poisson/_train_k5_10_32k.h5'
  val_path:      'data/poisson/_val_k5_10_4k.h5'
  test_path:     'data/poisson/_test_k5_10_4k.h5'
  scales_path:   'data/poisson/train_k5_10_scales.npy'
  weights:       'experiments/expts/poisson-scale-k1_5/pretrain-poisson-k1_5-JOBID-0/checkpoints/ckpt_best.tar'
  batch_size: 128
  valid_batch_size: 128
  log_to_wandb: !!bool True
  mode_cut: 32
  embed_cut: 64
  fc_cut: 2
  max_epochs: 200
  subsample: 32  # Use 1024 samples (32768 / 32)
```

#### Step 4 (Alternative): Run Fine-tuning Manually

```bash
# Fine-tune with 1024 samples
python train.py \
    --yaml_config config/operators_poisson.yaml \
    --config poisson-k5_10-finetune-custom \
    --run_num finetune-1k \
    --root_dir ./experiments
```

The trainer will automatically load the pre-trained weights from the path specified in the config before training begins.

---

## 📈 Weights & Biases Integration

To track experiments with W&B:

1. **Setup W&B:**
```bash
pip install wandb
wandb login
```

2. **Update config:**
```yaml
entity: 'your-wandb-username'
project: 'neuraloperators-reproduction'
log_to_wandb: !!bool True
```

3. **Run HPO Sweep:**
```bash
# Create sweep
wandb sweep config/sweep_config.yaml

# Run agent
wandb agent your-entity/your-project/sweep-id
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
```yaml
# Reduce batch size
batch_size: 64  # instead of 128
valid_batch_size: 64
```

**2. Data Generation Takes Too Long:**
```bash
# Generate smaller dataset first
--ntrain 1000 --nval 100 --ntest 100
```

**3. Import Errors:**
```bash
# Ensure you're in the correct directory
cd third_party/neuraloperators-TL-scaling
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**4. DDP Hangs:**
```bash
# Check network setup
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Use single GPU for debugging
python train.py ... # without srun
```

---

## 📚 References

- Paper: Subramanian et al., "Towards Foundation Models for Scientific Machine Learning", 2023
- FNO Original Paper: Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

---

## ✅ Verification Checklist

- [ ] Environment setup complete
- [ ] Dependencies installed correctly
- [ ] Test dataset generated (100 samples)
- [ ] Normalization scales computed
- [ ] Minimal training runs successfully (5 epochs)
- [ ] Loss decreases over epochs
- [ ] Full dataset generated (32k samples)
- [ ] Full training runs (500 epochs)
- [ ] Results match paper (±10% tolerance)
- [ ] Evaluation script works
- [ ] Plots generated successfully

---

## � Multi-Task Learning Experiments (Mixed Dataset)

This section demonstrates multi-task learning by training a single model on multiple PDE systems simultaneously. This reproduces **Figure 6a** from the paper, showing that mixed-domain pretraining can match or exceed single-domain pretraining effectiveness.

**Experiment Setup:**
- **Training Systems**: Poisson k∈[1,5] + AdvDiff αdr∈[0.2,1.0] + Helmholtz ω∈[1,10]
- **Mixed Dataset**: ~10,922 samples per system (total ~32k combined)
- **Model Input**: 6 channels (zero-padded to handle different PDE input dimensions)
- **Target Domain**: Poisson k∈[5,10] (for comparison with single-domain pretraining)
- **Comparison**: Mixed-domain pretrained vs Single-domain pretrained

### Channel Normalization Strategy

Different PDEs have different tensor coefficients:
- **Poisson**: 3 components (k11, k12, k22) → diffusion tensor
- **Advection-Diffusion**: 5 components (k11, k12, k22, vx, vy) → diffusion + advection
- **Helmholtz**: 2 components (k_constant, omega) → constant diffusion + wavenumber

As stated in the paper:
> "When pre-training a single model on this 'mixed' dataset, we simply use zero channels 
> for those coefficients that do not exist when using examples from a specific operator."

To train a unified model, we **zero-pad all tensor coefficients to 5 components**:
- Poisson: `[k11, k12, k22, 0, 0]` → 3 diffusion coefficients + 2 zeros
- AdvDiff: `[k11, k12, k22, vx, vy]` → 3 diffusion + 2 advection (no change)
- Helmholtz: `[k_constant, omega, 0, 0, 0]` → constant diffusion + wavenumber + 3 zeros

During data loading, the PDESolns class expands each tensor component to a spatial channel:
- **Input to model**: 1 (source) + 5 (tensor expanded) = **6 channels total**
- The zero-padded values effectively signal which operator is being used

This simple approach allows the model to learn representations across multiple PDE systems
without requiring explicit operator selection or advanced prompting mechanisms.

### Step 1: Generate Individual PDE Datasets

Ensure you have generated the three pretraining datasets:

```bash
# Poisson k∈[1,5] (32k samples)
python utils/gen_data_poisson.py \
    --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse \
    --datapath data/poisson \
    --e1 1 --e2 5

# Advection-Diffusion αdr∈[0.2,1.0] (32k samples)
python utils/gen_data_advdiff.py \
    --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse \
    --datapath data/advdiff \
    --adr1 0.2 --adr2 1.0

# Helmholtz ω∈[1,10] (32k samples)
python utils/gen_data_helmholtz.py \
    --ntrain 32768 --nval 4096 --ntest 4096 \
    --ng 144 --n 128 --sparse \
    --datapath data/helmholtz \
    --om1 1 --om2 10
```

### Step 2: Create Mixed Dataset

Combine the three PDE datasets into a unified mixed dataset with zero-padding:

```bash
# Create output directory
mkdir -p data/mixed

# Create training set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_train_k1_5_32k.h5 \
    --advdiff_path data/advdiff/_train_adr0p2_1_32k.h5 \
    --helmholtz_path data/helmholtz/_train_o1_10_32k.h5 \
    --output_path data/mixed/_train_mixed_32k.h5 \
    --samples_per_system 10922

# Create validation set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_val_k1_5_4k.h5 \
    --advdiff_path data/advdiff/_val_adr0p2_1_4k.h5 \
    --helmholtz_path data/helmholtz/_val_o1_10_4k.h5 \
    --output_path data/mixed/_val_mixed_4k.h5 \
    --samples_per_system 1365

# Create test set
python utils/create_mixed_dataset.py \
    --poisson_path data/poisson/_test_k1_5_4k.h5 \
    --advdiff_path data/advdiff/_test_adr0p2_1_4k.h5 \
    --helmholtz_path data/helmholtz/_test_o1_10_4k.h5 \
    --output_path data/mixed/_test_mixed_4k.h5 \
    --samples_per_system 1365
```

### Step 3: Compute Normalization Scales

```bash
python utils/compute_scales.py \
    --datapath data/mixed \
    --filename _train_mixed_32k.h5
```

Expected output file: `data/mixed/_mixed_scales.txt`

### Step 4: Pre-train on Mixed Dataset

```bash
# Submit mixed dataset pretraining job
sbatch scripts/slurm/submit_mixed_pretrain.sh

# Note the job ID (e.g., 67890)
```

This trains a single FNO model on all three PDE systems simultaneously. The model has:
- **Input dimension**: 6 (accommodates zero-padded inputs)
- **Architecture**: Same as single-domain pretraining (5 layers, mode_cut=32, etc.)
- **Training samples**: ~32k mixed (balanced across three systems)

### Step 5: Update Checkpoint Path

After mixed pretraining completes, update the checkpoint paths for fine-tuning:

```bash
# Update checkpoint paths with your mixed pretraining job ID
bash scripts/utils/update_checkpoint_path.sh 67890

# This updates config/operators_poisson.yaml to point to the correct checkpoint
```

### Step 6: Fine-tune on Target Domain

Submit the mixed fine-tuning experiments to compare with single-domain pretraining:

```bash
# Submit array job for 5 fine-tuning experiments (16, 64, 256, 1k, 4k samples)
sbatch scripts/slurm/submit_mixed_finetune_array.sh
```

This launches 5 jobs:
- **Task 0**: Fine-tune on 16 Poisson k∈[5,10] samples
- **Task 1**: Fine-tune on 64 samples
- **Task 2**: Fine-tune on 256 samples
- **Task 3**: Fine-tune on 1k samples
- **Task 4**: Fine-tune on 4k samples

### Step 7: Compare Results

Compare mixed-domain pretraining against single-domain pretraining:

| **Downstream Samples** | **Single-Domain Pretrain** | **Mixed-Domain Pretrain** | **From Scratch** |
|-------------------------|----------------------------|---------------------------|------------------|
| 16                      | W&B: `poisson-k5_10-finetune-16` | W&B: `finetune-mixed-16` | W&B: `scratch-16` |
| 64                      | W&B: `poisson-k5_10-finetune-64` | W&B: `finetune-mixed-64` | W&B: `scratch-64` |
| 256                     | W&B: `poisson-k5_10-finetune-256` | W&B: `finetune-mixed-256` | W&B: `scratch-256` |
| 1k                      | W&B: `poisson-k5_10-finetune-1k` | W&B: `finetune-mixed-1k` | W&B: `scratch-1k` |
| 4k                      | W&B: `poisson-k5_10-finetune-4k` | W&B: `finetune-mixed-4k` | W&B: `scratch-4k` |

**Expected Findings (from Figure 6a in paper):**
- Mixed-domain pretraining achieves similar or better performance than single-domain
- Both pretraining approaches significantly outperform training from scratch
- The benefit is most pronounced in low-data regimes (16-256 samples)
- Mixed pretraining provides a more general representation across PDE systems

**Analysis Questions:**
1. Does mixed pretraining match single-domain performance on Poisson downstream task?
2. How much data efficiency gain comes from multi-task learning?
3. Which sample sizes show the largest improvement from mixed pretraining?

### Verification Checklist

- [ ] All three individual PDE datasets generated (Poisson, AdvDiff, Helmholtz)
- [ ] Mixed dataset created with correct zero-padding (6 channels)
- [ ] Normalization scales computed for mixed dataset
- [ ] Mixed pretraining completed successfully (~50 epochs)
- [ ] Checkpoint path updated in config files
- [ ] Mixed fine-tuning jobs submitted (5 array tasks)
- [ ] Results logged to W&B for comparison
- [ ] Performance compared against single-domain baseline

---

## �💡 Tips for Success

1. **Start Small**: Always test with minimal datasets first
2. **Monitor GPU**: Use `nvidia-smi` to check GPU utilization
3. **Save Checkpoints**: Enable `save_checkpoint: !!bool True`
4. **Use Logging**: Check logs regularly for issues
5. **Version Control**: Track config changes for reproducibility
6. **Document**: Keep notes on hyperparameter changes and results

---

## 📞 Getting Help

If you encounter issues:
1. Check logs in `results/expts/[config]/[run_num]/logs/`
2. Verify data shapes with `h5py` inspection
3. Test individual components (data generation, model forward pass)
4. Compare with original paper's hyperparameters

Good luck with reproducing the results! 🚀

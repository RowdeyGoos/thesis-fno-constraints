# Reproduction Workflow

This document provides a visual overview of the reproduction workflow.

## 🔄 Overall Process Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     REPRODUCTION PIPELINE                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Phase 1: Environment Setup             │
        │  • Install dependencies (requirements.txt)│
        │  • Setup PyTorch 1.12 + CUDA 11.3      │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Phase 2: Quick Verification            │
        │  • Run quick_test.sh                   │
        │  • Generate 100 samples                │
        │  • Train 5 epochs                      │
        │  • Verify loss decreases               │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Phase 3: Data Generation               │
        │  • Poisson: 32k train samples          │
        │  • Advection-Diffusion: 32k samples    │
        │  • Helmholtz: 32k samples              │
        │  • Compute normalization scales        │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Phase 4: Full Training                 │
        │  • Train FNO on each PDE (500 epochs)  │
        │  • Save checkpoints                    │
        │  • Log metrics (W&B optional)          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  Phase 5: Evaluation                    │
        │  • Test set inference                  │
        │  • Compute metrics                     │
        │  • Generate plots                      │
        │  • Compare with paper results          │
        └─────────────────────────────────────────┘
```

## 📊 Data Generation Pipeline

```
┌──────────────────┐
│  PDE Parameters  │
│  • Domain: [0,1]²│
│  • Grid: 128×128 │
│  • RBF sources   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│        Generate Training Examples           │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐│
│  │ Poisson  │  │ Adv-Diff  │  │Helmholtz ││
│  │ k∈[1,5]  │  │ADR∈[0.2,1]│  │ ω∈[1,10] ││
│  └──────────┘  └───────────┘  └──────────┘│
└─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Solve PDEs      │
│  • Spectral      │
│  • FFT-based     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Save HDF5       │
│  fields: [N,2,H,W]│
│  tensor: [N,d]   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Compute Scales   │
│ • Mean/Std       │
│ • Save .npy      │
└──────────────────┘
```

## 🧠 Training Pipeline

```
┌────────────────────────────────────────────────┐
│              Training Loop (FNO)               │
└────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ Load Batch   │            │  Forward     │
│ • Source     │──────────▶│  • Lift      │
│ • PDE coeff  │            │  • Fourier   │
│ • Target     │            │  • Project   │
└──────────────┘            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Loss (MSE)   │
                            │ Σ‖u_pred-u‖² │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Backward    │
                            │  • Adam      │
                            │  • lr=1e-3   │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Update      │
                            │  • Weights   │
                            │  • Scheduler │
                            └──────────────┘
```

## 🎯 Model Architecture (FNO)

```
Input: (batch, 4, 128, 128)
  ├─ Source function (1 channel)
  ├─ PDE coefficients (3 channels)
  │
  ▼
┌─────────────────────────────────┐
│  Lifting Layer (P)              │
│  Conv: 4 → 64 channels          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Fourier Layer 1                │
│  • FFT → Fourier modes          │
│  • Linear transform (low modes) │
│  • IFFT → Physical space        │
│  • + Skip connection            │
│  • Activation (GELU)            │
└────────────┬────────────────────┘
             │
             ▼
        (Repeat 3x)
             │
             ▼
┌─────────────────────────────────┐
│  Projection Layer (Q)           │
│  Linear: 64 → 128 → 1           │
└────────────┬────────────────────┘
             │
             ▼
Output: (batch, 1, 128, 128)
  └─ Predicted solution
```

## 🔬 Experiment Matrix

### Scaling Experiments

```
┌─────────────────────────────────────────────────┐
│              Model Scaling                      │
├──────────────┬──────────┬──────────┬──────────┤
│ Size         │ Modes    │ Embed    │ FC_mult  │
├──────────────┼──────────┼──────────┼──────────┤
│ Small        │    16    │    32    │    1     │
│ Medium       │    32    │    64    │    2     │
│ Large        │    64    │   128    │    4     │
└──────────────┴──────────┴──────────┴──────────┘

┌─────────────────────────────────────────────────┐
│              Data Scaling                       │
├──────────────┬──────────────────────────────────┤
│ Subsample    │ Effective Training Size          │
├──────────────┼──────────────────────────────────┤
│      1       │ 32k (full)                       │
│      2       │ 16k (50%)                        │
│      4       │  8k (25%)                        │
│      8       │  4k (12.5%)                      │
└──────────────┴──────────────────────────────────┘
```

### Transfer Learning Experiments

```
Pre-training Domain    →    Fine-tuning Domain
═══════════════════════════════════════════════════
Poisson k∈[1,5]        →    Poisson k∈[5,10]
Poisson k∈[1,5]        →    Poisson k∈[10,20]
Advection-Diffusion    →    Pure Advection
Helmholtz ω∈[1,10]     →    Helmholtz ω∈[10,20]
Multi-PDE Mix          →    Individual PDEs
```

## 📈 Expected Training Curves

```
Loss vs. Epoch (Poisson k∈[1,5])

Train Loss
  │
1.0│  ●
   │    ●
0.5│      ●●
   │         ●●●
0.1│             ●●●●●●
   │                   ●●●●●●●●●___________
0.01│                                      
   └────────────────────────────────────────
   0   50   100   150   200   250   300...500
                    Epoch

Val Loss (similar curve, slightly higher)
Test L2 Error: ~0.02-0.03 (final)
```

## ⏱️ Time Estimates

```
┌────────────────────────────────────────────────┐
│  Component        │  Time (4x A100 GPUs)       │
├───────────────────┼────────────────────────────┤
│  Quick test       │  5-10 minutes              │
│  Data gen (1 PDE) │  30-60 minutes             │
│  Data gen (3 PDEs)│  2-3 hours                 │
│  Training (1 PDE) │  8-12 hours (500 epochs)   │
│  Training (3 PDEs)│  24-36 hours               │
│  Evaluation       │  10-20 minutes             │
│                   │                            │
│  TOTAL (full)     │  ~2-3 days                 │
└───────────────────┴────────────────────────────┘
```

## ✅ Validation Checkpoints

After each phase, verify:

**Phase 1 (Setup):**
- [ ] `python --version` shows 3.8+
- [ ] `python -c "import torch; print(torch.__version__)"` shows 1.12.0
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` shows True

**Phase 2 (Quick Test):**
- [ ] Dataset files exist in `data/test_poisson/`
- [ ] Training completes without errors
- [ ] Loss decreases over 5 epochs
- [ ] Results directory created

**Phase 3 (Data Generation):**
- [ ] Each dataset is ~1-2 GB in size
- [ ] HDF5 files have correct shapes
- [ ] Scale files generated (.npy)

**Phase 4 (Training):**
- [ ] Training runs for 500 epochs
- [ ] Validation loss improves
- [ ] Checkpoints saved
- [ ] W&B logs uploaded (if enabled)

**Phase 5 (Evaluation):**
- [ ] Test metrics computed
- [ ] Plots generated
- [ ] Results within ±10% of paper

## 🎓 Key Paper Results to Reproduce

1. **Base Performance (Table 1)**
   - Poisson: Test L2 ~0.02-0.03
   - Advection-Diffusion: Test L2 ~0.03-0.05
   - Helmholtz: Test L2 ~0.05-0.08

2. **Scaling Laws (Figures 2-3)**
   - Power law relationship between data size and error
   - Model size vs. performance trade-offs

3. **Transfer Learning (Figure 4)**
   - Pre-trained models converge faster
   - Better final performance with less data
   - Cross-PDE transfer effectiveness

4. **Data Efficiency (Figure 5)**
   - Pre-trained: 50% data for same performance
   - From scratch: Full dataset needed

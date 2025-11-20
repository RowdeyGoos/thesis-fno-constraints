# Model Checkpoints

This directory contains trained model checkpoints.

## Structure

- `checkpoints/`: Saved model weights and training state (gitignored)

## Checkpoint Format

Checkpoints are saved as PyTorch state dictionaries with the following structure:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
    'config': dict,
}
```

## Loading Checkpoints

```python
from thesis_fno.models import FNOBaseline
import torch

# Load model
model = FNOBaseline(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Pre-trained Models

Pre-trained models for thesis experiments will be made available upon publication.

**Download:** [Add URL when available]

**Available models:**
- `fno_baseline_pdebench.pt`: Baseline FNO on PDEBench
- `fno_divfree_ns.pt`: Divergence-free FNO on Navier-Stokes
- `fno_conservation_advection.pt`: Conservation FNO on advection equation
- `fno_multi_ns.pt`: Multi-constraint FNO on Navier-Stokes

#!/usr/bin/env python3
"""
Diagnostic script to check Poisson k1.0_2.5 dataset compatibility with mixed-pretrained model.

This script helps identify why fine-tuning might be slow by checking:
1. Dataset tensor dimensions
2. Model input/output dimensions
3. Data loading speed
4. GPU utilization potential

Usage:
    python utils/diagnose_finetuning_slowness.py \
        --data_path data/poisson/_train_k1.0_2.5_32k.h5 \
        --checkpoint_path experiments/expts/mixed-scale-all/pretrain-mixed-XXXXX-0/checkpoints/ckpt_best.tar
"""

import argparse
import h5py
import numpy as np
import torch
import time
from pathlib import Path


def check_dataset_format(data_path):
    """Check if dataset has correct tensor format"""
    print("\n" + "="*60)
    print("DATASET FORMAT CHECK")
    print("="*60)
    
    if not Path(data_path).exists():
        print(f"❌ ERROR: Dataset not found at {data_path}")
        return False
    
    with h5py.File(data_path, 'r') as f:
        print(f"✓ Dataset found: {data_path}")
        print(f"  Keys: {list(f.keys())}")
        
        fields_shape = f['fields'].shape
        tensor_shape = f['tensor'].shape
        
        print(f"\n  Fields shape: {fields_shape}")
        print(f"  Tensor shape: {tensor_shape}")
        
        # Check tensor dimensions
        if tensor_shape[1] == 3:
            print(f"\n  ⚠️  WARNING: Dataset has 3-component tensors")
            print(f"      Mixed model expects 5-component tensors!")
            print(f"      This will cause dimension mismatch and slow performance.")
            print(f"\n  Solution: Run conversion script:")
            print(f"      bash scripts/utils/convert_k1_2.5_to_mixed_format.sh")
            return False
        elif tensor_shape[1] == 5:
            print(f"\n  ✓ Dataset has 5-component tensors (correct for mixed model)")
            
            # Show first sample
            tensor_sample = f['tensor'][0]
            print(f"\n  First sample tensor: [{tensor_sample[0]:.4f}, {tensor_sample[1]:.4f}, "
                  f"{tensor_sample[2]:.4f}, {tensor_sample[3]:.4f}, {tensor_sample[4]:.4f}]")
            
            # Check if last two components are zero (should be for Poisson)
            if np.allclose(tensor_sample[3:], 0.0):
                print(f"  ✓ Last two components are zero (correct for Poisson)")
            else:
                print(f"  ⚠️  Last two components are non-zero (unexpected for Poisson)")
            
            return True
        else:
            print(f"\n  ❌ ERROR: Unexpected tensor shape {tensor_shape}")
            return False


def check_checkpoint(checkpoint_path):
    """Check checkpoint dimensions"""
    print("\n" + "="*60)
    print("CHECKPOINT CHECK")
    print("="*60)
    
    if not Path(checkpoint_path).exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   (This is OK if you haven't updated the path yet)")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        
        # Try to find input dimension info
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Look for first layer to determine input channels
            first_layer_keys = [k for k in state_dict.keys() if 'conv' in k.lower() or 'linear' in k.lower()]
            if first_layer_keys:
                first_key = sorted(first_layer_keys)[0]
                first_layer = state_dict[first_key]
                if hasattr(first_layer, 'shape') and len(first_layer.shape) >= 2:
                    in_channels = first_layer.shape[1]
                    print(f"\n  Model input channels: {in_channels}")
                    
                    if in_channels == 6:
                        print(f"  ✓ Model expects 6 input channels (correct for mixed model)")
                    elif in_channels == 4:
                        print(f"  ⚠️  Model expects 4 input channels (single-domain Poisson)")
                    else:
                        print(f"  ⚠️  Unexpected input channels: {in_channels}")
                    
                    return in_channels
        
        print(f"  ℹ️  Could not determine input dimensions from checkpoint")
        return None
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def benchmark_data_loading(data_path, batch_size=16, num_batches=10):
    """Benchmark data loading speed"""
    print("\n" + "="*60)
    print("DATA LOADING BENCHMARK")
    print("="*60)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            fields = f['fields'][:]
            tensor = f['tensor'][:]
        
        n_samples = fields.shape[0]
        print(f"  Total samples: {n_samples}")
        
        # Simulate batch loading
        start_time = time.time()
        for i in range(num_batches):
            idx = np.random.choice(n_samples, batch_size, replace=False)
            batch_fields = fields[idx]
            batch_tensor = tensor[idx]
            
            # Simulate some processing
            _ = torch.from_numpy(batch_fields).float()
            _ = torch.from_numpy(batch_tensor).float()
        
        elapsed = time.time() - start_time
        batches_per_sec = num_batches / elapsed
        samples_per_sec = (num_batches * batch_size) / elapsed
        
        print(f"\n  Time for {num_batches} batches: {elapsed:.2f} seconds")
        print(f"  Batches per second: {batches_per_sec:.1f}")
        print(f"  Samples per second: {samples_per_sec:.1f}")
        
        if samples_per_sec < 100:
            print(f"  ⚠️  Slow data loading detected!")
            print(f"     Consider increasing num_data_workers")
        else:
            print(f"  ✓ Data loading speed looks good")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        return False


def estimate_training_time(data_path, batch_size=16, max_epochs=50):
    """Estimate training time based on dataset size"""
    print("\n" + "="*60)
    print("TRAINING TIME ESTIMATE")
    print("="*60)
    
    try:
        with h5py.File(data_path, 'r') as f:
            n_samples = f['fields'].shape[0]
        
        steps_per_epoch = n_samples // batch_size
        total_steps = steps_per_epoch * max_epochs
        
        print(f"  Dataset size: {n_samples} samples")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total epochs: {max_epochs}")
        print(f"  Total training steps: {total_steps}")
        
        # Rough estimates (assuming ~0.5s per step for small batches)
        if batch_size <= 16:
            time_per_step = 0.5
        elif batch_size <= 32:
            time_per_step = 0.3
        elif batch_size <= 64:
            time_per_step = 0.2
        else:
            time_per_step = 0.15
        
        estimated_time_min = (total_steps * time_per_step) / 60
        estimated_time_hr = estimated_time_min / 60
        
        print(f"\n  Estimated time per step: ~{time_per_step}s")
        print(f"  Estimated total training time: {estimated_time_min:.1f} minutes ({estimated_time_hr:.2f} hours)")
        
        if estimated_time_hr > 2:
            print(f"\n  ⚠️  Training may take a long time!")
            print(f"     Consider:")
            print(f"     - Reducing max_epochs")
            print(f"     - Increasing batch_size (if memory allows)")
            print(f"     - Reducing validation frequency")
        else:
            print(f"\n  ✓ Training time looks reasonable")
        
        return True
    
    except Exception as e:
        print(f"❌ Error estimating time: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose potential issues causing slow fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/poisson/_train_k1.0_2.5_32k.h5',
        help='Path to training dataset'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to mixed-pretrained checkpoint (optional)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size to test with'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Number of epochs for time estimate'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FINE-TUNING DIAGNOSTIC TOOL")
    print("="*60)
    print(f"\nData path: {args.data_path}")
    if args.checkpoint_path:
        print(f"Checkpoint: {args.checkpoint_path}")
    
    # Run checks
    dataset_ok = check_dataset_format(args.data_path)
    
    if args.checkpoint_path:
        checkpoint_dims = check_checkpoint(args.checkpoint_path)
    
    if dataset_ok:
        benchmark_data_loading(args.data_path, args.batch_size)
        estimate_training_time(args.data_path, args.batch_size, args.max_epochs)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not dataset_ok:
        print("\n❌ CRITICAL ISSUE: Dataset needs conversion to 5-component format")
        print("\n   Run this command:")
        print("   bash scripts/utils/convert_k1_2.5_to_mixed_format.sh")
        return 1
    else:
        print("\n✓ Dataset format is correct")
        print("\n  If training is still slow, check:")
        print("  1. GPU utilization: nvidia-smi")
        print("  2. Set plot_figs: False in config")
        print("  3. Check job logs for warnings")
        print("  4. Verify checkpoint path is correct")
        return 0


if __name__ == '__main__':
    exit(main())

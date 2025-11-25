#!/bin/bash
# Request an interactive GPU session on DAIC
# Usage: bash scripts/interactive_gpu.sh

echo "Requesting interactive GPU session..."

srun --partition=gpu \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gpus=1 \
     --mem=32G \
     --time=04:00:00 \
     --pty bash -i

# After the session starts, remember to:
# module load python/3.10 cuda/11.8 cudnn/8.6
# source ~/venv/thesis-fno/bin/activate

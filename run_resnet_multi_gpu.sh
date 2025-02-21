#!/bin/bash

#SBATCH --job-name=resnet_multi_gpu

#SBATCH --output=resnet_multi_gpu.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:2  # Change this for more GPUs

#SBATCH --time=10:00

#SBATCH --mem=32G

#SBATCH --partition=ice-gpu



# Load Modules

module purge

module load python/3.10.10

module load cuda/11.7


# Debugging: Print Python Path
which python3
python3 --version

# Activate Virtual Environment

source ~/venv/bin/activate

# Ensure Correct NumPy Version
pip install --no-cache-dir numpy==1.26.4

# Install Required Packages

#pip install --upgrade pip

#pip install torch torchvision deepspeed requests pillow  # Add fsdp if needed

# Force GPU Usage in DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs

# Debugging: Check NumPy in Compute Node
#python -c "import numpy as np; print('NumPy Version:', np.__version__)"

# Ensure NumPy is Installed
#pip install --no-cache-dir numpy

# Run Multi-GPU Inference

deepspeed run_resnet_multi_gpu.py



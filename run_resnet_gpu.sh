#!/bin/bash

#SBATCH --job-name=resnet_gpu

#SBATCH --output=resnet_gpu.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --time=10:00

#SBATCH --mem=16G

#SBATCH --partition=ice-gpu  # Adjust based on your available GPU partition



# Load Required Modules

#module load python/3.10

#module load cuda/11.7  # Load the correct CUDA version



# Activate Virtual Environment

source ~/venv/bin/activate



# Ensure Required Packages Are Installed

#pip install --upgrade pip

#pip install torch torchvision pillow requests psutil



# Run the Model

python run_resnet_gpu.py



#!/bin/bash

#SBATCH --job-name=resnet_cpu

#SBATCH --output=resnet_cpu.out

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4  # Request 4 CPU cores

#SBATCH --time=10:00  # Max 10 minutes

#SBATCH --mem=16G  # Request 16GB RAM

#SBATCH --partition=ice-cpu  # Use the correct CPU partition



# Load Python module (if needed)

module load python/3.10



# Activate virtual environment (if applicable)

source ~/venv/bin/activate

# Verify Installation
python -c "import torch; print('PyTorch Installed:', torch.__version__)"

# Run the model

python run_resnet_cpu.py



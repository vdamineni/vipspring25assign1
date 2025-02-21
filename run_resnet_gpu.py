import time

import torch

from torchvision import models, transforms

from PIL import Image

import requests

import psutil 

import numpy as np

import threading



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Load ResNet-50 Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Preprocess Image

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

])



# Load an Image

# Load an Image (Ensure it's converted to RGB)

url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png"

image = Image.open(requests.get(url, stream=True).raw).convert("RGB")  # Ensure RGB format

image = transform(image).unsqueeze(0).to(device)  # Add batch dimension



# Debugging: Print shape to verify channels

print(f"Image tensor shape: {image.shape}")  # Expected output: [1, 3, 224, 224]



# Ensure image has exactly 3 channels

if image.shape[1] != 3:

    raise ValueError(f"Incorrect number of channels: Expected 3, but got {image.shape[1]}")




# Measure Execution Time

start_time = time.time()

with torch.no_grad():

    output = model(image)

elapsed_time = time.time() - start_time




print(f"Inference Time on CPU: {elapsed_time:.4f} seconds")

gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # Convert to MB
gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # Convert to MB

# Print Results
print(f"Inference Time on GPU: {elapsed_time:.4f} seconds")
print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")



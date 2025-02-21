import time

import torch

from torchvision import models, transforms

from PIL import Image

import requests

import psutil 

import numpy as np

import threading



# Load ResNet-50 on CPU

device = torch.device("cpu")

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



# Measure CPU Utilization Before Execution

cpu_usage_before = psutil.cpu_percent(interval=1)  # 1-second average CPU usage



# Function to track CPU usage during execution

cpu_usage = []

running = True  # Flag to stop monitoring when execution ends



def monitor_cpu():

    """Monitor CPU usage in a separate thread."""

    while running:

        cpu_usage.append(psutil.cpu_percent(interval=0.1))  # Sample CPU every 0.1s



# Start CPU monitoring in a separate thread

monitor_thread = threading.Thread(target=monitor_cpu)

monitor_thread.start()



# Measure Execution Time

start_time = time.time()

with torch.no_grad():

    output = model(image)

elapsed_time = time.time() - start_time



# Stop monitoring CPU usage

running = False

monitor_thread.join()  # Wait for monitoring thread to finish



print(f"Inference Time on CPU: {elapsed_time:.4f} seconds")

print(f"CPU Utilization During Execution: {cpu_usage}")

if cpu_usage:

    avg_cpu_usage = sum(cpu_usage) / len(cpu_usage)

    print(f"Average CPU Usage During Execution: {avg_cpu_usage:.2f}%")

cpu_usage_after = psutil.cpu_percent(interval=1)



process = psutil.Process()

mem_info = process.memory_info()

print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")  # Convert to MB



# Measure CPU Utilization After Execution

cpu_usage_after = psutil.cpu_percent(interval=1)



# Print Results

print(f"Inference Time on CPU: {elapsed_time:.4f} seconds")

print(f"CPU Usage Before Execution: {cpu_usage_before}%")

print(f"CPU Usage After Execution: {cpu_usage_after}%")



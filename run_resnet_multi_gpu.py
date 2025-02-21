import torch

import deepspeed

from torchvision import models, transforms

from PIL import Image

import time

import requests

import numpy as np

print("NumPy Version:", np.__version__)



# Initialize DeepSpeed

deepspeed.init_distributed()



# Assign GPUs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {torch.cuda.device_count()} GPUs")



# Load ResNet-50 Model and Move to GPUs

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

model = model.to(device)

model = deepspeed.init_inference(model, dtype=torch.float16, replace_method="auto")



# Image Preprocessing

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

])



# Load Image

url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png"

image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image = transform(image).unsqueeze(0).to(device).half()  # Move image to GPU



# Measure Execution Time

start_time = time.time()

with torch.no_grad():

    output = model(image)

elapsed_time = time.time() - start_time



# Print Execution Time

print(f"Inference Time on {torch.cuda.device_count()} GPUs: {elapsed_time:.4f} seconds")



import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision
import os
import torch
from model import TinyVGG

device  = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = torch.load("/Users/shoumik20/Shoumik_work/Repos/Pothole_Hackathon/final_model.pth", map_location=torch.device('cpu'))

input_shape = 3
hidden_units = 10
output_shape = 2  
model = TinyVGG(input_shape, hidden_units, output_shape)

model.load_state_dict(state_dict)

model.eval()

custom_image_transform = transforms.Compose([
    transforms.Resize(size=(224,224))
])

class_names = ['normal', 'potholes']

def pred_and_plot(model: torch.nn.Module,
                  image_path: str,
                  class_names = class_names,
                  transform = custom_image_transform,
                  device = device):
  
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
  target_image = target_image/255
  if transform:
    target_image = transform(target_image)
  model.to(device)
  with torch.inference_mode():
    target_image = target_image.unsqueeze(0)
    target_image_pred = model(target_image.to(device))
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
  return class_names[target_image_pred_label.cpu()]

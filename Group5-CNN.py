# import torch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f

# import for graphing and matrix operations
import matplotlib.pyplot as plt
import numpy as np

# use gpu if available
if(torch.cuda.is_available()):
  dev = torch.device("cuda:0")
else:
  dev = torch.device("cpu")

transform = transforms.Compose(torchvision.transforms.ToTensor())

# probably has to change
dataPath = '/baseline methods/Self-Trans/LUNA/train'

lRate = 0.001

# model
class CNN(nn.Module):
  # constructor
  def __init__(self):
    super(CNN, self).__init__()
    
    # Convolution operations
    self.conv1 = nn.Conv2d(1, 224, 5)
    self.pool = nn.MaxPool2d(3,3)
    self.conv2 = nn.Conv2d(224,224,3)
    
    # ANN operations
    self.fc1 = nn.Linear(224*224*3,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,2)
  
  # ID sequence:
  # conv1 -> pool -> conv2 -> pool -> fc1 -> fc2 -> fc3
  def forward(self, run):
    run = self.pool(f.relu(self.conv1(run)))
    run = self.pool(f.relu(self.conv2(run)))
    run = run.view(-1, 224*224*3)
    run = f.relu(self.fc1(run))
    run = f.relu(self.fc2(run))
    run = self.fc3(run)
    return run
 
CNN1 = CNN()
print("Model created!")


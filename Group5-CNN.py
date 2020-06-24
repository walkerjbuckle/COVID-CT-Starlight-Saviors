# import torch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
import tprch.optim as op

# import for graphing and matrix operations
import matplotlib.pyplot as plt
import numpy as np

# use gpu if available
if(torch.cuda.is_available()):
  dev = torch.device("cuda:0")
else:
  dev = torch.device("cpu")

print(dev)
  
# probably has to change
dataPath = '/baseline methods/Self-Trans/LUNA/train'

# modify
lRate = 0.001
epochs = 

# data transformations
transform = transforms.Compose(torchvision.transforms.ToTensor())
trainset = 
testset = 
classes = ('COVID', 'Non-COVID')

# model
class CNN(nn.Module):
  # constructor
  def __init__(self):
    super(CNN, self).__init__()
    
    # Convolution operations
    self.conv1 = nn.Conv2d(1, 224, 5)
    self.pool = nn.MaxPool2d(2,2)
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
    
    # make tensor
    run = run.view(-1, 224*224*3)
    
    # run through ANN
    run = f.relu(self.fc1(run))
    run = f.relu(self.fc2(run))
    run = self.fc3(run)
    return run
 
CNN1 = CNN()
print("Model created!")


#optimizer
crit = nn.CrossEntropyLoss()
optimizer = op.SGD(CNN1.parameters(), lr = lRate, momentum = 0.9)

# train
def train():
  for epoch in range(2):

train()

print("Training complete!")
print("Model saved!")

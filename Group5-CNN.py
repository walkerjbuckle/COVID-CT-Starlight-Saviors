# import torch libraries
import torch
import torchvision
import torchvision.transforms as transforms

# import for graphing and matrix operations
imprt matplotlib.pyplot as plt
import numpy as np

# use gpu if available
if(torch.cuda.is_available()):
  dev = torch.device("cuda:0")
else:
  dev = torch.device("cpu")
 
lRate = 0.001
 
class CNN(nn.Module):
  # constructor
  def __init__(self):
    super(CNN, self).__init__()

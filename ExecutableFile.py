import torch
import torch.nn as nn
import torchvision as vision

# Load saved model and run with test images
exec = CNN()
exec.load_state_dict(torch.load('model.pth'))

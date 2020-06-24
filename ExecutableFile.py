import torch
import torch.nn as nn
import torchvision as vision

# Load saved model and run with test images
exec = CNN()
exec.load_state_dict(torch.load('model.pt'))
exec.eval()

# run to test images
while true:
  id_img = input("Enter a test file name with extension")

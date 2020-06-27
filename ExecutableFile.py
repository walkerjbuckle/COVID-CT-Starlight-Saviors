import torch
import torch.nn as nn
import torchvision as vision
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image
import torch.nn.functional as f
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution operations - test other values for 32 and 64
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # ANN operations - test other values for 1000 and 100
        self.fc1 = nn.Linear(55 * 55 * 64, 1000)  # 64 should match above's value (2nd arg in conv2)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    # ID sequence:
    # conv1 -> pool -> conv2 -> pool -> fc1 -> fc2 -> fc3
    def forward(self, run):
        run = self.pool(f.relu(self.conv1(run)))  # 224 x 224 -> 112 x 112
        run = self.pool(f.relu(self.conv2(run)))  # 112 x 112 -> 55 x 55

        # make tensor
        run = run.view(-1, 55 * 55 * 64)  # again, 64 should match above's value

        # run through ANN
        run = f.relu(self.fc1(run))
        run = f.relu(self.fc2(run))
        run = self.fc3(run)
        return run

# Load saved model and run with test images
exec = CNN()
exec.load_state_dict(torch.load('model.pt'))
#exec.eval()

transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((-20, 20)), transforms.RandomAffine(0, translate=None, scale=[0.7, 1.3], shear=None, resample=False, fillcolor=0), transforms.ToTensor()])

run = True

# run to test images
while (run):
    id_img = input('Enter a test file name with extension: ')
    if(id_img == 'exit'):
        run = False
    else:
        print(id_img)
        plt.figure()
        plt.imshow(io.imread(os.path.join('dataset2', id_img)))
        plt.show()
        output = exec(transform1(Image.open(os.path.join('dataset2', id_img)).convert('RGB')))
        print(output)

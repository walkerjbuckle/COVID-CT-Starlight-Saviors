import torch
import torch.nn as nn
import torchvision as vision
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image
import torch.nn.functional as f
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class runData():
    def __init__(self, csvFile, transform = None):
        self.transform = transform
        self.CT = pd.read_csv(csvFile)

    def __len__(self):
        return len(self.CT)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgName = os.path.join('dataset2/', self.CT.iloc[idx,1])
        img = Image.open(imgName).convert('L')
        cS = self.CT.iloc[idx,2]

        if self.transform:
            img = self.transform(img)

        return (img, cS)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution operations - test other values for 32 and 64
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # ANN operations - test other values for 1000 and 100
        self.fc1 = nn.Linear(55 * 55 * 64, 100)  # 64 should match above's value (2nd arg in conv2)
        self.fc2 = nn.Linear(100, 100)
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

transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

run = True

# run to test images
while (run):
    id_img = input('Enter a test file name with extension: ')
    if(id_img == 'exit'):
        run = False
    else:
        print(id_img)
        dataRun = open("run.csv", "x")
        dataRun.write("id,img,status")
        dataRun.write("\n")
        dataRun.write("1,"+ id_img + ",unknown")
        dataRun.close()
        runner = runData('run.csv', transform = transform1)
        runLoader = torch.utils.data.DataLoader(runner, batch_size = 1, shuffle=True)
        plt.figure()
        plt.imshow(io.imread(os.path.join('dataset2', id_img)))
        plt.show()
        with torch.no_grad():
            for i, (img, status) in enumerate(runLoader):
                output = exec(img)
                print(output)
        os.remove("run.csv")

#!/usr/bin/env python

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as op
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataLoader import data, EngineeredData

trainDIR = 'dataset/'
batchSize = 50
lRate = 0.001
epochs = 200

# Dataset creation
#imgName = []
#label = []

#for root, directories, files in os.walk(trainDIR):
#    for file in files:
#        imgName.append(root + "/" + file)
#        label.append(int(root.split("/")[-1]))

#imgTrain, imgTest, labelTrain, labelTest = train_test_split(imgName, label, test_size=0.3, random_state=50,
#                                                          stratify=label)


trainAug = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.RandomRotation((-20, 20)),
     torchvision.transforms.RandomAffine(0, translate=None, scale=[0.7, 1.3], shear=None, resample=False, fillcolor=0),
     torchvision.transforms.ToTensor()])
testAug = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])


class CustomDatasetFromImages(torch.utils.data.Dataset):
    def __init__(self, imgName, label, transforms=None):
        self.image_arr = np.asarray(imgName)
        self.label_arr = np.asarray(label)
        self.data_len = len(imgName)
        self.transforms = transforms

    def __getitem__(self, index):
        single_img_name = self.image_arr[index]
        img_array = Image.open(single_img_name).convert('RGB')
        if self.transforms is not None:
            img_array = self.transforms(img_array)
        image_label = self.label_arr[index]
        return (single_img_name, img_array, image_label)

    def __len__(self):
        return self.data_len


#trainData = CustomDatasetFromImages(imgTrain, labelTrain, transforms=trainAug)
#testData = CustomDatasetFromImages(imgTest, labelTest, transforms=testAug)

#testLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, shuffle=False)

trainData = data('data.csv', 'dataset2', transform=trainAug)
trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, shuffle=True)

# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution operations - test other values for 32 and 64
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # ANN operations - test other values for 1000 and 100
        self.fc1 = nn.Linear(55 * 55 * 64, 1000)  # 64 should match above's value (2nd arg in conv2)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 2)

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


# backup to try
class CNNBackup(nn.Module):
    def __init__(self):
        super(CNNBackup, self).__init__()  # test other values for 32, 64, and 1000
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(55 * 55 * 64, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)  # 224 x 224 -> 112 x 112
        out = self.layer2(out)   # 112 x 112 -> 55 x 55
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


CNN1 = CNN()
CNN2 = CNNBackup()
print("Model created!")

# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(CNN1.parameters(), lr=lRate, momentum=0.9)

# train
totalStep = len(trainLoader)
lossList = []
accList = []


def train(trainer):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainer):
            images, labels = images.to(dev), labels.to(dev)
            # Run the forward pass
            outputs = CNN1(images)  # use either CCN1 or CNN2
            loss = criterion(outputs, labels)
            lossList.append(loss.item())

            # Backprop and perform optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            stepTotal = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            stepCorrect = (predicted == labels).sum().item()
            accList.append(stepCorrect / stepTotal)

            if (i + 1) % batchSize == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy this step: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, totalStep, loss.item(),
                          (stepCorrect / stepTotal) * 100))


# backup to try
def backupTrain():
    for epoch in range(epochs):
        # add up losses to get average
        rloss = 0.0
        total = 0
        correct = 0
        for i, (image, labels) in enumerate(trainLoader):
            image, labels = image.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = CNN1(image)  # use either CCN1 or CNN2
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = total + 1
            _, predicted = torch.max(outputs.data, 1)
            correct = correct + (predicted == labels).sum().item()
            accList.append(correct / total)

            if (i + 1) % batchSize == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, totalStep, loss.item(),
                              (correct / total) * 100))


# use gpu if available
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

print(dev)

CNN1.to(dev)  # use either CCN1 or CNN2, based on what you set in the training method being used

train(trainLoader)

for i in enumerate(30):
    trainer = EngineeredData('data.csv', 'dataset2', transform = trainAug, 2)
    loader = torch.utils.data.DataLoader(trainer, batch_size=500, shuffle=True)
    train(loader)

trainer = EngineeredData('data.csv', 'dataset2', transform = trainAug, 1)
loader = torch.utils.data.DataLoader(trainer, batch_size=500, shuffle=True)

train(loader)
    
print("Training complete!")

# saves cnn to model.pt for use
torch.save(CNN1.state_dict(), 'model.pt')
print("Model saved!")

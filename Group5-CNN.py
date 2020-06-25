#!/usr/bin/env python

# import torch libraries
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as op

# import for graphing and matrix operations
import matplotlib.pyplot as plt
import numpy as np

# use gpu if available
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

print(dev)

# probably has to change
dataPath = '/baseline methods/Self-Trans/LUNA/train'  # what is this for?

# modify
lRate = 0.001
epochs = 5
batchSize = 100

# data transformations
transform = transforms.Compose(transforms.ToTensor())  # do we need to normalize the dataset?
trainSet = None  # we should ask about how to define these two data sets
testSet = None
classes = ('COVID', 'Non-COVID')  # how do we link these classes to the images?

trainLoader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=testSet, batch_size=batchSize, shuffle=False)


# model
class CNN(nn.Module):
    # constructor
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution operations
        self.conv1 = nn.Conv2d(1, 224, 5)  # any reason for 224 output channels and kernel size of 5?
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(224, 224, 3)

        # ANN operations
        self.fc1 = nn.Linear(224 * 224 * 3, 120)  # how did you find the size of the images afterwards to be 224x224?
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        ''' ALTERNATE METHOD
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(256 * 128 * 64, 1000)  # how do we find the size of the images afterwards (ex. 256x128)?
        self.fc2 = nn.Linear(1000, 2)
        '''

    # ID sequence:
    # conv1 -> pool -> conv2 -> pool -> fc1 -> fc2 -> fc3
    def forward(self, run):
        run = self.pool(f.relu(self.conv1(run)))
        run = self.pool(f.relu(self.conv2(run)))

        # make tensor
        run = run.view(-1, 224 * 224 * 3)

        # run through ANN
        run = f.relu(self.fc1(run))
        run = f.relu(self.fc2(run))
        run = self.fc3(run)
        return run

    ''' ALTERNATE METHOD
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    '''


CNN1 = CNN()
print("Model created!")

# optimizer
crit = nn.CrossEntropyLoss()
optimizer = op.SGD(CNN1.parameters(), lr=lRate, momentum=0.9)

# train
totalStep = len(trainLoader)
lossList = []
accList = []


# not sure if this will work
def train():
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainLoader):  # where are the labels defined for each image?
            # Run the forward pass
            outputs = CNN1(images)
            loss = crit(outputs, labels)
            lossList.append(loss.item())

            # Backprop and perform optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accList.append(correct / total)

            if (i + 1) % batchSize == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, epochs, i + 1, totalStep, loss.item(),
                              (correct / total) * 100))


train()

print("Training complete!")
print("Model saved!")

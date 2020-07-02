#!/usr/bin/env python

import torch
import torchvision
from torch.utils.data import DataLoader
from dataLoader import data
import torch.nn as nn
import torch.nn.functional as f

testAug = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
testData = data('testData.csv', 'testDataset/', transform=testAug)
testLoader = torch.utils.data.DataLoader(testData, shuffle=False)

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

print(dev)


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


CNN = CNN()
CNN.load_state_dict(torch.load('model.pt'))
CNN.eval()

if torch.cuda.is_available():
    CNN.cuda()

# totalStep = len(testLoader)


def test():
    totalTotal = 0
    totalCorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, (images, labels) in enumerate(testLoader):
        images, labels = images.to(dev), labels.to(dev)
        # Run the forward pass
        outputs = CNN(images)

        # Track the accuracy
        stepTotal = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        totalTotal += stepTotal
        totalCorrect += correct
       
        #print(predicted)
        #print(labels)
        if predicted == labels:
            if predicted == 1:
                TP += 1
            elif predicted == 0:
                TN += 1
        elif predicted != labels:
            if predicted == 1:
                FP += 1
            elif predicted == 0:
                FN += 1
        #print(stepTotal)
        #print(totalTotal)
        #print(correct)
        #print(totalCorrect)
        #print("")

    print('Accuracy: {:.2f}%'
          .format((totalCorrect / totalTotal) * 100))
    print("correct: " + str(totalCorrect))
    print("total: " + str(totalTotal))
    print("true positives: " + str(TP))
    print("true negatives: " + str(TN))
    print("false positives: " + str(FP))
    print("false negatives: " + str(FN))


test()

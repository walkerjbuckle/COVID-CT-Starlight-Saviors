#!/usr/bin/env python

import os
import csv

testDIR = 'testDataset/'

with open('testData.csv', mode='w') as testData:
    idx = 0
    fieldnames = ['idx', 'images', 'blank', 'labels']
    testData = csv.DictWriter(testData, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
    testData.writeheader()

    for root, directories, files in os.walk(testDIR):
        for file in files:
            if int(root.split("/")[-1]) == 0:
                testData.writerow({'idx': idx, 'images': root.split("/")[-1] + "/" + file, 'blank': '', 'labels': 0})
            elif int(root.split("/")[-1]) == 1:
                testData.writerow({'idx': idx, 'images': root.split("/")[-1] + "/" + file, 'blank': '', 'labels': 1})
            idx += 1

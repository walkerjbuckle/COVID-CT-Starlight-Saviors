"""
This script is just meant to unzip the four zip files of images into the two directories called 'CT_COVID' and 'CT_NonCOVID'
"""
import os
import zipfile

if __name__ == '__main__':

    os.mkdir('CT_COVID')
    os.mkdir('CT_NonCOVID')

    covidMain = 'CT_COVID'
    nonCovidMain = 'CT_NonCOVID'

    covidZips = [zipfile.ZipFile(
        'CT_COVID_1.zip', 'r'), zipfile.ZipFile('CT_COVID_2.zip', 'r')]
    nonCovidZips = [zipfile.ZipFile(
        'CT_NonCOVID_1.zip', 'r'), zipfile.ZipFile('CT_NonCOVID_2.zip', 'r')]

    for file in covidZips:
        file.extractall(covidMain)

    for file in nonCovidZips:
        file.extractall(nonCovidMain)
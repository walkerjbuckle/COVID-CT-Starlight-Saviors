"""
This script can be used to load in new datasets. Essentially, it takes images
from a COVID directory and a NONCOVID directory, randomly assigns them to train,
val, and test groups by adding their names to the text files in Data-split, and
then moves the images to the COVID and NONCOVID directories in Images-processed

Before using it make sure you have the following:
    Directories named 'CT_COVID' and 'CT_NonCOVID' in COVID-CT/Images-processed
    A directory of images with COVID to be loaded in
    A directory of images without COVID to be loaded in

"""
import os
import shutil
import random
import argparse

def standardizeTextFile(textFile):
    """
    Parameters:
        textFile (str):         the path to the text file to be checked
    Purpose:
        this function should be run on all six Data-split text files to make sure that the end with a newline character
        returns nothing
    """
    
    text = open(textFile, 'r')
    words = text.read()

    while words.find('\n') != -1:
        words = words[words.find('\n')+1:]

    text.close()

    if words != '':
        text = open(textFile, 'r+')
        text.read()
        text.write('\n')


def getGroupSize(textFile):
    """
    Parameters:
        textFile (str):         the path to the text file to be read from
    Purpose:
        this counts how many lines of text are in a text file
        it is useful for counting how many images are loaded into the dataset already
        returns an integer for the number of lines 
    """
    
    scans = 0
    text = open(textFile, 'r')
    words = text.read()

    while words.find('\n') != -1:
        words = words[words.find('\n')+1:]
        scans += 1

    text.close()
    return(scans)

def randomlyAssign(numSelecting, imgList, textFilename, startPath, endPath):
    """
    Parameters:
        numSelecting (int):     the number of images that need to be added to this split
        imgList (list):         the list of either covid or noncovid images that are to be added
        textFilename (str):     the filepath from Data-split to the text file where the names of each image in
                                a group are stored
        startPath (str):        the filepath to the directory containing the images to be added
        endPath (str):          the folder in Images-processed that the images will go into
    Purpose:
        this effectively loads images in by adding their names to the different text files for the train, val, and test
            sets while also moving the images
        generates a text file called temp.txt that can be deleted
        returns nothing
    """
    
    tempText = open('temp.txt', 'w+')
    # temp.txt ins't a necessary file, I just wanted to edit the files in Data-split as few times as possible
    for image in range(0, numSelecting):
        index = random.randint(0, len(imgList)-1)
        filename = imgList[index]
        del imgList[index]
        tempText.write(filename + "\n")
        shutil.move(startPath+'/'+filename, '../Images-processed/'+endPath)
    tempText.close()
    tempText = open('temp.txt')
    textFile = open(textFilename, 'r+')
    textFile.read()
    textFile.write(tempText.read())

    textFile.close()
    tempText.close()
    
def checkImgCounts(counts, totalCovid, totalNonCovid):
    """
    Parameters:
        counts (list):          a list containing all counts of images to be added for each split in this order:
                                [trainCOVID, valCOVID, testCOVID, trainNONCOVID, valNONCOVID, testNONCOVID]
        totalCovid (int):       the total number of covid images to be added to the dataset
        totalNonCovid (int):    the total number of noncovid images to be added to the dataset
    Purpose:
        checks and corrects the counts for each split so that they reflect the amount of images to be loaded in
        returns a list containing the updated counts in the order:
            [trainCOVID, valCOVID, testCOVID, trainNONCOVID, valNONCOVID, testNONCOVID]
    """
    
    covidCount = counts[0] + counts[1] + counts[2]
    nonCovidCount = counts[3] + counts[4] + counts[5]
    
    # these are the differences of the images the program wants and the actual amount
    # for example, a value of 3 means that the program is looking for 3 more scans than we have
    extraCovid = covidCount - totalCovid
    extraNonCovid = nonCovidCount - totalNonCovid

    # explanation of what is happening here because it looks bad but this was the best working version I had:
    
    # first, the outer if statement is finding out if the current splits require more images than exist
    # to resolve this issue, this will try to remove the same amount of images from each group
    # this number of images removed from each group is extraCovid // 3 or extraNonCovid // 3
    # there is one case where the same amount of images can be removed from all three groups (extraCovid % 3 == 0)
    # however, it can't remove fractions of an image, so there are the two cases where there leftover images
    # in these cases, the extra image(s) are removed from the larger groups
    # with 1 extra image, it takes it from the train group (either count[0] or count[3])
    # with 2 extra images, it takes one from the train group and one from the test group (either count[2] or count[5])
    # going back to the outer if, the elif is to find out if the current splits don't use all the images
    # the number of images added to each group is abs(extraCovid) // 3 or abs(extraNonCovid) // 3
    # otherwise, the same three cases exist when adding the missing images as above, but with addition

    # corrections for covid splits
    if(extraCovid > 0):
        if(extraCovid % 3 == 0):
            counts[0] -= extraCovid // 3
            counts[1] -= extraCovid // 3
            counts[2] -= extraCovid // 3
        elif(extraCovid % 3 == 1):
            counts[0] -= extraCovid // 3 + 1
            counts[1] -= extraCovid // 3
            counts[2] -= extraCovid // 3
        else:
            counts[0] -= extraCovid // 3 + 1
            counts[1] -= extraCovid // 3
            counts[2] -= extraCovid // 3 + 1
    elif(extraCovid < 0):
        if(abs(extraCovid) % 3 == 0):
            counts[0] += abs(extraCovid) // 3
            counts[1] += abs(extraCovid) // 3
            counts[2] += abs(extraCovid) // 3
        elif(abs(extraCovid) % 3 == 1):
            counts[0] += abs(extraCovid) // 3 + 1
            counts[1] += abs(extraCovid) // 3
            counts[2] += abs(extraCovid) // 3
        else:
            counts[0] += abs(extraCovid) // 3 + 1
            counts[1] += abs(extraCovid) // 3
            counts[2] += abs(extraCovid) // 3 + 1
    
    # corrections for noncovid splits
    if(extraNonCovid > 0):
        if(extraNonCovid % 3 == 0):
            counts[3] -= extraNonCovid // 3
            counts[4] -= extraNonCovid // 3 
            counts[5] -= extraNonCovid // 3
        elif(extraNonCovid % 3 == 1):
            counts[3] -= extraNonCovid // 3 + 1
            counts[4] -= extraNonCovid // 3
            counts[5] -= extraNonCovid // 3
        else:
            counts[3] -= extraNonCovid // 3 + 1
            counts[4] -= extraNonCovid // 3
            counts[5] -= extraNonCovid // 3 + 1
    elif(extraNonCovid < 0):
        if(abs(extraNonCovid) % 3 == 0):
            counts[3] += abs(extraNonCovid) // 3
            counts[4] += abs(extraNonCovid) // 3
            counts[5] += abs(extraNonCovid) // 3
        elif(abs(extraNonCovid) % 3 == 1):
            counts[3] += abs(extraNonCovid) // 3 + 1
            counts[4] += abs(extraNonCovid) // 3
            counts[5] += abs(extraNonCovid) // 3
        else:
            counts[3] += abs(extraNonCovid) // 3 + 1
            counts[4] += abs(extraNonCovid) // 3
            counts[5] += abs(extraNonCovid) // 3 + 1

    return counts

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load in a dataset from two folders")
    
    parser.add_argument('--c', default='../../../../Downloads/615374_1199870_bundle_archive/COVID',
                        help='File path for directory of COVID images from {}'.format(os.getcwd()))

    parser.add_argument('--nc', default='../../../../Downloads/615374_1199870_bundle_archive/non-COVID',
                        help='File path for directory of Non-COVID images from {}'.format(os.getcwd()))

    args = parser.parse_args()

    # saves the filepaths to the dataset to be loaded in
    covidPath = args.c
    nonCovidPath = args.nc

    # creates lists containg all of the image filenames
    covidList = os.listdir(covidPath)
    nonCovidList = os.listdir(nonCovidPath)
    
    # the proportion of images in each group
    trainProp = 425.0 / 746
    valProp = 118.0 / 746
    testProp = 203.0 / 746

    # finds the total amount of images after the datasets are merged
    totalImages = len(os.listdir('../Images-processed/CT_COVID')) \
                  + len(os.listdir('../Images-processed/CT_NonCOVID')) \
                  + len(covidList) \
                  + len(nonCovidList)

    # finds the total amount of images in the dataset to be loaded in
    totalNewImages = len(covidList) + len(nonCovidList)

    # these variables are the number of images in each group with both new and old images
    trainSize = int(totalImages * trainProp)
    valSize = int(totalImages * valProp)
    testSize = int(totalImages * testProp)

    # rounding issues caused the sum of the groups to be one higher than desired
    if trainSize + valSize + testSize == totalImages - 1:
        trainSize += 1

    # standardizing all six text files to make sure that they all end with a newline character
    standardizeTextFile('COVID/trainCT_COVID.txt')
    standardizeTextFile('COVID/valCT_COVID.txt')
    standardizeTextFile('COVID/testCT_COVID.txt')
    standardizeTextFile('NonCOVID/trainCT_NonCOVID.txt')
    standardizeTextFile('NonCOVID/valCT_NonCOVID.txt')
    standardizeTextFile('NonCOVID/testCT_NonCOVID.txt')

    # this need to be subtracting the number of lines in the text document, removing the old images from the counts
    trainSize -= getGroupSize('COVID/trainCT_COVID.txt') + getGroupSize('NonCOVID/trainCT_NonCOVID.txt')
    valSize -= getGroupSize('COVID/valCT_COVID.txt') + getGroupSize('NonCOVID/valCT_NonCOVID.txt')
    testSize -= getGroupSize('COVID/testCT_COVID.txt') + getGroupSize('NonCOVID/testCT_NonCOVID.txt')

    # these variables are the number of scans from either covid or noncovid that need to be added to each group
    trainNewCovid = int( trainSize/float(totalNewImages) * len(covidList) + 0.5)
    trainNewNonCovid = int( trainSize/float(totalNewImages) * len(nonCovidList) + 0.5)
    valNewCovid = int( valSize/float(totalNewImages) * len(covidList) + 0.5)
    valNewNonCovid = int( valSize/float(totalNewImages) * len(nonCovidList) + 0.5)
    testNewCovid = int( testSize/float(totalNewImages) * len(covidList) + 0.5)
    testNewNonCovid = int( testSize/float(totalNewImages) * len(nonCovidList) + 0.5)

    # this checks to see that we are using all of the images in the new dataset
    newCounts = checkImgCounts([trainNewCovid, valNewCovid, testNewCovid, trainNewNonCovid, valNewNonCovid, testNewNonCovid], len(covidList), len(nonCovidList))
    trainNewCovid = newCounts[0]
    valNewCovid = newCounts[1]
    testNewCovid = newCounts[2]
    trainNewNonCovid = newCounts[3]
    valNewNonCovid = newCounts[4]
    testNewNonCovid = newCounts[5]

    # this is now where we randomly assign the scans to groups
    randomlyAssign(trainNewCovid, covidList, 'COVID/trainCT_COVID.txt', covidPath, 'CT_COVID')
    randomlyAssign(valNewCovid, covidList, 'COVID/valCT_COVID.txt', covidPath, 'CT_COVID')
    randomlyAssign(testNewCovid, covidList, 'COVID/testCT_COVID.txt', covidPath, 'CT_COVID')
    randomlyAssign(trainNewNonCovid, nonCovidList, 'NonCOVID/trainCT_NonCOVID.txt', nonCovidPath, 'CT_NonCOVID')
    randomlyAssign(valNewNonCovid, nonCovidList, 'NonCOVID/valCT_NonCOVID.txt', nonCovidPath, 'CT_NonCOVID')
    randomlyAssign(testNewNonCovid, nonCovidList, 'NonCOVID/testCT_NonCOVID.txt', nonCovidPath, 'CT_NonCOVID')

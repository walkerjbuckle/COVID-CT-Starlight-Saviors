import os
import zipfile

if __name__ == '__main__':

    os.mkdir('CT_COVID')
    os.mkdir('CT_NonCOVID')

    covidMain = 'CT_COVID'
    nonCovidMain = 'CT_NonCOVID'
    
    covidZips = [zipfile.ZipFile('CT_COVID_1.zip', 'r'), zipfile.ZipFile('CT_COVID_2.zip', 'r')]
    nonCovidZips = [zipfile.ZipFile('CT_NonCOVID_1.zip', 'r'), zipfile.ZipFile('CT_NonCOVID_2.zip', 'r')]

    for file in covidZips:
        file.extractall(covidMain)

    for file in nonCovidZips:
        file.extractall(nonCovidMain)

    


"""
I tried something here that works if all four zip files are unzipped into a folder
of the same name, with empty folders called CT_COVID and CT_NonCOVID, but I didn't think
that this was the most effective way to sort the images given that the files are zipped at the start.
"""

#import shutil
#import os
#
#def createImageLists(folders, images):
#    for folder in folders:
#        images.append(os.listdir(folder))
#
#def moveImages(folders, images, mainFolder):
#    for index in range(0, len(folders)):
#        for image in images[index]:
#            shutil.move(folders[index]+'/'+image, mainFolder)
#            #print(folders[index]+'/'+image)
#
#if __name__ == '__main__':
#    covidMain = 'CT_COVID'
#    nonCovidMain = 'CT_NonCOVID'
#    covidFolders = ['CT_COVID_1', 'CT_COVID_2']
#    nonCovidFolders = ['CT_NonCOVID_1', 'CT_NonCOVID_2']
#
#    covidImages = []
#    nonCovidImages = []
#
#    createImageLists(covidFolders, covidImages)
#    createImageLists(nonCovidFolders, nonCovidImages)
#
#    moveImages(covidFolders, covidImages, covidMain)
#    moveImages(nonCovidFolders, nonCovidImages, nonCovidMain)

    


# Some notes:

`CT_COVID.zip` and `CT_NonCOVID.zip` contain the previous dataset. These files are not necessary to unzip, and they will be unused.

In the process of adding the second dataset to this project, the zip files became too large for GitHub to handle. As such, we broke them into two smaller ones. These new files are `CT_COVID_1.zip` and `CT_COVID_2.zip` for the COVID half of the dataset and `CT_NonCOVID_1.zip` and `CT_COVID_2.zip` for the NonCOVID half of the dataset.

In order to properly set up the project in a way that the images can be accessed, you need to run the filesort.py script that is included in this folder. This script will unzip all four zip files into two folders, `CT_COVID` and `CT_NonCOVID`.

This can be done in a terminal by changing directories to this folder (`Images-processed`) and then running the command `python filesort.py`

If everything worked, you should have 1601 files in the `CT_COVID` folder and 1626 files in the `CT_NonCOVID` folder.

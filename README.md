This repository is a fork of [a machine learning(ML) model made by UCSD](https://github.com/UCSD-AI4H/COVID-CT).
**Our goal is to create a reliable ML model that can determine if a patient has COVID-19 based on computed tomography(CT) scans from a sample of patients that have tested positive and patients that have tested negative.**

# Citations

The additional dataset that we used can be found at https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset#. The citations for this dataset are:
Angelov, Plamen, and Eduardo Almeida Soares. "EXPLAINABLE-BY-DESIGN APPROACH FOR COVID-19 CLASSIFICATION VIA CT-SCAN." medRxiv (2020).
Soares, Eduardo, Angelov, Plamen, Biaso, Sarah, Higa Froes, Michele, and Kanda Abe, Daniel. "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification." medRxiv (2020). doi: https://doi.org/10.1101/2020.04.24.20078584.
Link:
https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v2

# How to use

Clone the repository

```
git clone https://github.com/walkerjbuckle/COVID-CT-Starlight-Saviors.git
```
Create an envirement (using environment.yml or requirements.txt)

-Using Conda environment

Requires Anaconda running python 3.7 or newer

```
cd COVID-CT-Starlight-Saviors
conda env create -f environment.yml
conda activate starlightenv
```

-Using pip requirements.txt

```
pip install -r requiremennts.txt
pip install torch torchvision
```

Lastly, install git pre-commit hook

```
cd COVID-CT-Starlight-Saviors
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

# Who are we?
This project is being done by Interns as part of the [MISI Internship program](https://www.misiacademy.tech/). Interns divided into six teams in order to work on the project. Team 1 was responsible for the command line Linux app and a PyQt cross-platform GUI app. Team 2 was responsible for managing the Git repository, assisting other teams with it, and writing the README. Team 3 was responsible for adding datasets for training, making a script for partitioning the datasets and making an API for data loading. Team 4 was responsible for making an API for training and researching training options. Team 5 was responsible for researching the options for layers and making a new CNN class model. Team 6 was responsible for searching for possible models to apply with transfer learning that have not yet been tried, and to train and validate those models.

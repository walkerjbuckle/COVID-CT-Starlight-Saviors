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

Run the install script

```
cd COVID-CT-Starlight-Saviors
pip install pre-commit
pip install autopep8
pre-commit install
pre-commit run --all-files
```

# Who are we?
This project is being done by Interns as part of the [MISI Internship program](https://www.misiacademy.tech/)

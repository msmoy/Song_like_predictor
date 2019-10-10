# Kaggle: Spotify Classifier
--------

## Table of Contents:

* [Introduction](#introduction)
* [Technologies](#technologies)
* [Data Description](#data-description)
* [Process Overview](#process-overview)
* [Project Status](#project-status)
* [Source](#source)

---
### Introduction:

This is based off a dataset that was obtained via [Kaggle](https://www.kaggle.com/geomack/spotifyclassification). It's a project to create a classifier that will predict whether the user likes or doesn't like a specific song.
---
### Technologies:

- **Programming Language**
    - Python 3.7
    
- **Imported Libraries**
    - Pandas
    - Numpy
    - Seaborn
    - Matplotlib.pyplot
    - sklearn.model_selection
        - train_test_split, cross_val_score
    - sklearn.linear_model
        - LogisticRegression
    - sklearn.ensemble 
        - RandomForestClassifier
    - sklearn.cluster
        - KMeans
    - sklearn.metrics
        - accuracy_score
        
--- 
### Data Description:

- **data.csv**
    - Dataset that user obtained on their song history from Spotify. Includes song attributes and a target column (1 = liked song, 0 = unliked song)

- **liked_songs.py**
    - code for reviewing data and creating classifier to predict whether the user liked a specific song
        
- **README.md**
    
---
### Process Overview:

- **Data exploration and analyzing**
    - Checking for nulls or data that appears invalid
    - Feature selection

- **Modeling**
    - Testing models that will accurately predict whether user liked specific songs
        
---
### Project Status:

Created as a way to practice python skills and modeling

---
### Source:

- dataset download
    - https://www.kaggle.com/geomack/spotifyclassification

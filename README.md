# ISIC-2024---Skin-Cancer-Detection-with-3D-TBP

This repository contains the implementation of a machine learning pipeline for **skin cancer prediction**, leveraging the ISIC 2024 dataset. The project explores advanced data preprocessing, visualization, and predictive modeling techniques to classify skin lesions and detect cancerous abnormalities effectively.

[Competiton Page](https://www.kaggle.com/competitions/isic-2024-challenge/leaderboard)

Public Rank - 950 out of 3410 Participants
Private Rank - 640 out of 3410 Participants

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)

---

## Overview
Skin cancer is one of the most common and severe health conditions globally. Early detection significantly improves treatment outcomes. This project employs machine learning models to analyze dermoscopic images and classify skin lesions into malignant or benign categories.

**Key Features:**
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) for better feature understanding
- Machine learning model training and hyperparameter tuning
- Performance evaluation with Parital AUC Score 

---

## Dataset
This project uses the **ISIC 2024 dataset**, a well-known collection of dermoscopic images with over 400000+ lesion images. The dataset includes labeled images of various skin lesions, with annotations for malignant and benign cases along with patient meta data from 28000 unique patients. 

**Dataset Highlights:**
#### Image Data
1. **Benign Images**: 400,666  
2. **Malignant Images**: 393  

#### Meta Data
1. **No of features**: 55
2. **Numerical Features**: 34
3. **Categorical Features**: 21


---

## Methodology
1. **Data Preprocessing**
   - Handling missing values using Median value replacement and class imbalances using  Random Under Sampling in meta data
   - Feature Encoding using OHE for categorical Features and Feature Scaling using MinMaxScaler
   - Feature Engineering for extracting useful features from given set of features
   - Image resizing and normalization for model input compatibility

2. **Exploratory Data Analysis (EDA)**
   - Visualizing class distributions and lesion patterns
   - Analyzing correlations and feature importance

3. **Model Development**
   - Training the EfficientNetV1 for image classifcation and appending its probability predictions as a feature in the Meta data
   - An Ensemble Model of XGBoost, CatBoost and LightGBM with confidence of (0.28, 0.47, 0.3) respectively 
   - Bayesian Optimization was used for Hyperparameter Optimization.

4. **Evaluation**
   - Validating model performance using Stratified Cross Validation 
   - Generating classification reports and confusion matrices

---

## Results
The trained model achieved:
- **Partial AUC Score (With TRP threshold =0.8):0.16753**%

Detailed performance analysis and visualizations are included in the notebook.

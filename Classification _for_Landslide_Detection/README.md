# Integrated Landslide Detection Pipeline

## Overview
This project is built for the **Classification for Landslide Detection** challenge on Zindi.  
The goal is to build a classification model that can accurately distinguish between landslide‑affected areas and unaffected regions using both optical imagery and Synthetic Aperture Radar (SAR) data.

Competition link: [Zindi - Classification for Landslide Detection](https://zindi.africa/competitions/classification-for-landslide-detection)

## Challenge Context
- **Objective:** Predict landslide occurrences by analyzing changes captured in multi-modal satellite data, including optical and post-event radar imagery (change detection bands, backscatter intensities).
- **Data:** The dataset includes post-event SAR images and change detection bands that highlight ground changes after events.

## Notebook Structure
The pipeline is organized into several key sections:
- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing**
- **Feature Engineering**
- **Model Development and Training** (e.g., Random Forest, XGBoost, CatBoost)
- **Model Evaluation and Results**
- **Feature Importance and Interpretation**

## Requirements
Make sure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- catboost

You can install them with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost
```

## Running the Notebook
1. Clone or download this repository.
2. Install the dependencies.
3. Launch Jupyter Notebook or JupyterLab.
4. Open the notebook `integratedlandslidedetectionpipeline.ipynb`.
5. Run all cells in sequence.

## Results & Outputs
The notebook generates:
- **Performance Metrics:** Classification scores such as accuracy, F1-score, ROC-AUC.
- **Visualizations:** Confusion matrix, feature importance plots.
- **Predictions:** Model outputs ready for submission to Zindi.

## License
This project is licensed under the **MIT License**.

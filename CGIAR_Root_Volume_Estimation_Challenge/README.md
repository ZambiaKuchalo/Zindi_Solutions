
# CGIAR Root Volume Estimation Challenge

Welcome to the repository for the [CGIAR Root Volume Estimation Challenge](https://zindi.africa/competitions/cgiar-root-volume-estimation-challenge) on Zindi. This project aims to estimate plant root volume from multi-modal data (images and metadata) using advanced image segmentation, feature extraction, and ensemble machine learning techniques.

## Overview

The challenge involves processing aerial or field images of plant roots along with associated metadata. Our solution consists of several stages:

- **Data Extraction & Preparation:**  
  Extracting and organizing the dataset and pre-trained models from ZIP archives. Loading and cleaning the training and testing CSV files.

- **Image Segmentation & Feature Extraction:**  
  Using YOLOv8 segmentation models to extract root masks from images, followed by morphological, texture, fractal, and deep feature extraction. Stereo image processing is also applied to obtain disparity-based features.

- **Advanced Feature Engineering:**  
  Generating interaction, polynomial, temporal, and domain-specific features. Transformations and encoding strategies are applied to capture complex relationships in the data.

- **Model Optimization & Ensemble Learning:**  
  Hyperparameter tuning with Optuna is performed on base learners (XGBoost and Random Forest). A meta-model is built from cross-validated meta-features to create a robust ensemble.

- **Submission Generation:**  
  Final predictions are post-processed and saved for submission.

## Repository Structure

- **README.md**  
  This file provides an overview and instructions for reproducing the challenge pipeline.

- **Main Code Files:**  
  The code implements the full pipeline:
  - **Data extraction and CSV loading**
  - **Dataset structure verification and visualization**
  - **YOLO segmentation and model loading**
  - **Feature extraction (morphological, texture, fractal, deep features)**
  - **Stereo processing and complete plant feature estimation**
  - **Advanced feature engineering and interaction feature creation**
  - **Optuna-based hyperparameter optimization for XGBoost and Random Forest**
  - **Ensemble building and meta-feature generation**
  - **Submission file generation**

- **Data and Model Archives:**  
  - `data.zip`: Contains the main dataset.
  - `Models.zip`: Contains pre-trained segmentation models (`best_early.pt`, `best_late.pt`).

## Environment Setup and Data Preparation

1. **Data Extraction:**  
   The script extracts data and model archives stored in Google Drive:
   - The main dataset is extracted into `/content/data`.
   - Pre-trained YOLO segmentation models are extracted into `/content/Models`.

2. **CSV Loading & Cleaning:**  
   - Train and test CSV files are loaded from the designated directory.
   - Training data rows with zero `RootVolume` are dropped.
   - A new column `PlantID` is created for consistency.

3. **Dataset Structure Verification:**  
   The code lists top-level folders (e.g., `train`, `test`), verifies subfolder contents, and visualizes sample images to ensure the dataset is organized correctly.

## Feature Extraction & Processing Pipeline

### 1. Segmentation and Mask Extraction

- **YOLO Segmentation:**  
  - Pre-trained YOLOv8 segmentation models are loaded.
  - `segment_roots` extracts binary masks from input images using confidence thresholds and combines available masks.

- **Morphological and Texture Features:**  
  - Morphological features: Skeleton length, contour area, thickness, shape eccentricity, and complexity.
  - Texture features: GLCM (contrast, homogeneity, energy) and Local Binary Pattern (LBP) analyses.
  - Fractal dimension calculations are implemented to capture root complexity.

- **Deep Feature Extraction:**  
  - A pre-trained ResNet18 extracts deep features from images to complement classical image features.

### 2. Stereo Processing

- **Disparity Calculation:**  
  - Computes the disparity between paired left and right images using a StereoSGBM matcher.
  
- **Stereo Feature Aggregation:**  
  - Deep features from both views are aggregated.
  - A mean deep feature is calculated, and additional stereo-specific features are appended.

### 3. Advanced Feature Engineering

- **Interaction and Polynomial Features:**  
  - Interaction terms, polynomial expansions, and derived ratios (e.g., Root Density, Volume Efficiency) are generated.
  
- **Quantile Transformation and Encoding:**  
  - Continuous features are transformed using `QuantileTransformer`.
  - Categorical features (e.g., `Stage`, `Genotype`) are encoded with cyclical features and one-hot encoding.

- **Temporal and Biological Features:**  
  - Temporal gradients and domain-specific ratios are calculated.
  - Features such as area growth rate, acceleration, and estimated volume are computed.

## Hyperparameter Optimization & Ensemble Building

### 1. Model Optimization with Optuna

- **XGBoost Optimization:**  
  - Hyperparameters (n_estimators, learning_rate, max_depth, etc.) are tuned using a TPE sampler.
  
- **Random Forest Optimization:**  
  - Hyperparameters are tuned with repeated K-Fold cross-validation and pruning.
  
- **Meta-Feature Generation:**  
  - Base models (XGBoost and Random Forest) generate cross-validated meta-features which are used to train an ensemble meta-model.

### 2. Ensemble Generation

- The final ensemble combines predictions from the tuned base models and a meta-learner, optimized to minimize mean squared error.

## Submission Generation

- **Prediction and Post-Processing:**  
  - The ensemble produces raw predictions.
  - The output undergoes an inverse log transformation and is clipped to ensure non-negative root volume values.
  - The final submission is saved as `best-score.csv`.

## Running the Pipeline

1. **Setup Environment:**  
   - Install required packages listed in your `requirements.txt`.
   - Adjust file paths for data and model archives if necessary (e.g., Google Drive paths).

2. **Data Extraction & Verification:**  
   - Execute the initial cells to extract ZIP files and verify dataset structure.
   - Visualize sample images to ensure data integrity.

3. **Feature Extraction & Model Training:**  
   - Run the segmentation, feature extraction, and advanced feature engineering modules.
   - Execute the Optuna optimization routines for hyperparameter tuning.
   - Build the ensemble and generate predictions.

4. **Generate Submission:**  
   - Run the final pipeline to create the submission CSV file (`best-score.csv`).

## Acknowledgments

This project leverages several powerful libraries including:
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html)
- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Optuna](https://optuna.org/)
- [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/)

Special thanks to the open-source community for their invaluable contributions.

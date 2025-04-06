
# Lacuna Solar Survey Challenge

Welcome to the Lacuna Solar Survey Challenge repository. This project aims to tackle the challenge of accurately predicting solar panel survey counts by leveraging state-of-the-art deep learning techniques combined with metadata features.

## 📊 Problem Statement

In this challenge, the goal is to predict key solar panel survey metrics from aerial images and associated metadata. The predictions focus on two targets per image:
- **boil_nbr**: A count related to the survey,
- **pan_nbr**: Another key count extracted from the imagery.

Participants are provided with a training dataset (images and tabular metadata) and are expected to generate predictions on a test set, achieving high accuracy while maintaining robust generalization.

## 🧠 Solution Approach

Our solution involves several steps:

- **Data Preparation & Cleaning:**  
  - Reading the training and test data.
  - Aggregating metadata (e.g., image origin and placement details) for effective feature representation.

- **Dataset & Data Augmentation:**  
  - Custom `SolarPanelDataset` class to handle image reading, metadata extraction, and transformation.
  - Extensive use of Albumentations for data augmentation, including random cropping, flips, rotations, and normalization to increase model robustness.

- **Model Architecture:**  
  - **Backbone:** We utilize the `tf_efficientnetv2_b3` model (pretrained on ImageNet) for feature extraction from images.
  - **Metadata Processing:** A small neural network processes metadata, which is further enhanced with a multi-head attention mechanism.
  - **Regression Head:** A regressor combines both image and metadata features to predict the two target values using layers with non-linear activations and a final Softplus function to ensure positivity for count predictions.

- **Training Strategy:**  
  - Multi-fold cross-validation using K-Fold splitting to ensure robust performance.
  - Mixed precision training with gradient scaling for faster convergence.
  - Use of a Huber loss function and a cosine annealing scheduler for effective learning rate management.
  - Saving the best-performing model from each fold based on validation Mean Absolute Error (MAE).

- **Inference with Test Time Augmentation (TTA):**  
  - Averaging predictions over multiple augmented passes to improve prediction reliability on the test dataset.

## 🚀 Results

- **Rank:** Top 50% (placeholder for your ranking)

The final predictions are saved in two formats:
- **Original Submission:** Floating point predictions.
- **Integer Submission:** Rounded predictions (for challenges requiring integer outputs).

## 📁 Files

- **`notebook.ipynb`**  
  Main Jupyter notebook detailing the solution pipeline, including data processing, model training, validation, and inference steps.

- **`README.md`**  
  This file – a comprehensive guide for understanding and running the challenge solution.

- **`data_description.md`**  
  A summary of the dataset features and detailed explanations of the metadata used.

- **`requirements.txt`**  
  A list of Python packages and dependencies required to run the code.

- **Additional Scripts:**  
  The repository includes scripts for training (`train.py`), inference (`predict.py`), and utility functions for data loading and augmentation.

## ⚙️ How to Run

1. **Setup Environment:**
   - Install the required packages using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Dataset Organization:**
   - Ensure the dataset is organized with a base directory containing the images and CSV files (`Train.csv` and `Test.csv`).
   - Update the `BASE_DIR` and `IMAGE_DIR` paths in the code if necessary.

3. **Training the Model:**
   - Run the main execution block to train the model on the provided folds:
     ```bash
     python train.py
     ```
   - The best model weights for each fold will be saved automatically.

4. **Generating Predictions:**
   - Once the models are trained, run the inference script to generate predictions:
     ```bash
     python predict.py
     ```
   - Two submission files will be created: one with the original predictions and one with integer-rounded values.

## 📝 Acknowledgments

This solution leverages several open-source libraries such as [PyTorch](https://pytorch.org/), [timm](https://github.com/rwightman/pytorch-image-models), and [Albumentations](https://albumentations.ai/), whose contributions are gratefully acknowledged.


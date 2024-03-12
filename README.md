# Breast Cancer Classification using DataFrame Processing

This repository contains Python code for processing the Breast Cancer dataset using DataFrame operations, data visualization, and classification techniques. The main steps include:

## Data Loading and Preprocessing:

- Importing the breast cancer dataset.
- Exploring the dataset and handling missing values (skipped as there are no missing values).
- Normalizing the data and applying Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) for visualization.

## Model Training and Evaluation:

- Splitting the preprocessed dataset into training and testing sets.
- Training different classifiers including Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Multi-layer Perceptron Classifier, Decision Tree Classifier, and Gaussian Naive Bayes.
- Evaluating each classifier's performance using cross-validation, ROC AUC, accuracy, and classification reports on both training and testing sets.
- Visualizing the classification results using confusion matrices.

## Feature Selection:

- Trying to improve classification results by adding a Feature Selection preprocessing stage using the VarianceThreshold method.

## Model Retraining and Evaluation with Feature Selection:

- Retraining the classifiers after feature selection.
- Evaluating the classifiers' performance on both training and testing sets with the new feature-selected data.
- Visualizing the updated classification results using confusion matrices.

## Repository Structure

- `README.md`: This file containing information about the project, dataset, methods, and results.
- `breast_cancer_classification.ipynb`: Jupyter notebook containing the Python code for data processing, classification, and visualization.

## Instructions for Running the Code

1. Ensure you have Python installed on your system along with the necessary libraries specified in the notebook.
2. Open and run the `breast_cancer_classification.ipynb` notebook using Jupyter or any compatible environment.
3. Follow the instructions and execute the code cells sequentially to perform data processing, model training, and evaluation.

## Results Summary

- The repository provides a comprehensive analysis of breast cancer classification using various classifiers.
- Support Vector Classifier and Multi-layer Perceptron Classifier exhibit consistent performance across multiple metrics, making them the preferred choices.
- Logistic Regression and K-Nearest Neighbors also perform well in accurately predicting the correct category.
- Decision Tree Classifier shows signs of potential overfitting, particularly with higher ROC AUC on the training set compared to the test set.
- Feature selection via VarianceThreshold does not significantly improve classification results.

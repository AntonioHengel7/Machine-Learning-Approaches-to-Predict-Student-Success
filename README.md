# Machine Learning Approaches to Predict Student Success

This repository compares multiple machine learning approaches for predicting student final grades (`G3`) using the UCI Student Performance dataset. The project focuses on three model families:

- Support Vector Regression (SVR)
- Linear models (Linear Regression, Ridge, and Lasso)
- Feedforward Neural Networks (FNN)

The goal was to treat student grade prediction as a **regression problem** and compare how well different models capture the relationship between demographic, social, school, and academic factors and final performance.

## Project Goal

Given student attributes and prior school-related information, predict the final grade (`G3`) and compare model performance through:

- preprocessing and feature engineering,
- hyperparameter tuning,
- residual analysis,
- learning curves,
- and ablation studies.

## Dataset

This project uses the **UCI Student Performance Dataset**, which includes two related datasets:

- `student-mat.csv` — Mathematics course data
- `student-por.csv` — Portuguese course data

The dataset contains:

- demographic features,
- family and social background,
- school-related variables,
- study habits and support information,
- and academic grades (`G1`, `G2`, `G3`).

The prediction target is:

- `G3`: final grade

## Models Compared

### 1) Support Vector Regression (SVR)

The SVM/SVR experiments test:

- Linear kernel
- Polynomial kernel
- RBF kernel

This workflow uses one-hot encoding, feature scaling, rare-grade filtering, and SMOTE-based rebalancing. In the current SVM script, `G1` and `G2` are dropped to make the task more challenging.

### 2) Linear Models

The linear-model experiments compare:

- Linear Regression
- Ridge Regression
- Lasso Regression

This workflow combines the math and Portuguese datasets, applies standard scaling to numerical features and one-hot encoding to categorical features, and compares several regularization strengths for Ridge and Lasso.

### 3) Feedforward Neural Network (FNN)

The neural-network experiments use fully connected feedforward networks with:

- different hidden-layer sizes,
- different activation functions,
- different learning rates,
- dropout,
- and batch normalization.

The neural-network workflow also applies scaling, rare-class filtering, and SMOTE.

## Reported Results

The main reported results from the project are:

| Model | Best configuration | MSE | MAE | R² |
|---|---|---:|---:|---:|
| SVR | RBF kernel | 2.60 | — | 0.89 |
| Linear model | Lasso (`alpha = 0.1`) | 1.80 | 0.83 | 0.88 |
| FNN | Best tuned configuration | 2.91* | — | 0.86–0.88 |

\* The neural-network section reports a validation MSE of about `2.9058` for the best tanh-based model, while the report summary rounds the best FNN R² to about `0.88`.

## Key Findings

- The **RBF SVR** achieved the best overall predictive performance.
- **Lasso Regression** was the strongest linear baseline and improved interpretability through feature selection.
- The **FNN** performed competitively and captured non-linear relationships well.
- Data preprocessing, scaling, and handling grade imbalance were important for model performance.
- Earlier grades (`G1`, `G2`) were especially important in the linear-model analysis, while the non-linear models still performed well even when those intermediate grades were removed.

## Current Repository Structure

Current project folders:

- `Final Project (2)/Final Project/` — SVM / SVR code
- `Final Project Linear Regression/Final Project Linear Regression/` — Linear / Ridge / Lasso code
- `Final_Project_Neural_Network/Final Project/` — Neural-network code

Main scripts:

- `main_svm_final.py`
- `main_linear_final.py`
- `main_snn_final.py`

## How to Run

Because the project is currently organized by model, each script is run from its own folder.

### SVM / SVR

```bash
cd "Final Project (2)/Final Project"
python main_svm_final.py

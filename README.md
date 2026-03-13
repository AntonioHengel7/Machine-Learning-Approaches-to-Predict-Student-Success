# Machine Learning Approaches to Predict Student Success

This repository explores how machine learning can be used to predict **student academic success**, measured by the final course grade (`G3`), using demographic, academic, and lifestyle variables from the Student Performance dataset.

## Project Summary

The main goal of this project is to predict student outcomes using features such as:

- study time
- number of past failures
- family and school support
- absences
- health and lifestyle variables
- parental background and education

The primary implementation in this repository uses a **feedforward neural network (FNN)** built with TensorFlow/Keras.

---

## Main Project

The core student-success workflow is implemented in:

- `Final_Project_Neural_Network/Final Project/main_snn_final.py`

This script:

1. loads the student dataset
2. removes `G1` and `G2` to reduce target leakage
3. one-hot encodes categorical variables
4. standardizes features
5. removes rare target classes
6. applies SMOTE to balance the dataset
7. trains multiple neural-network configurations
8. evaluates performance using:
   - Mean Squared Error (MSE)
   - R² score
9. generates plots for:
   - training vs validation loss
   - predicted vs actual grades
   - residuals
   - learning curves
   - feature ablation results

### Target variable

- `G3` = final grade

---

## Dataset Files

This repository includes the Student Performance dataset files:

- `student-mat.csv` — Mathematics course data
- `student-por.csv` — Portuguese course data
- `student.txt` — attribute and dataset description
- `student-merge.R` — script related to merging the two datasets

The current Python pipeline appears to use **`student-por.csv`** as the main training dataset.

---

## Additional Neural Network Reference Code

This repository also contains:

- `Final_Project_Neural_Network/Final Project/code_FNN_TF/`

That folder contains TensorFlow code for a **fully connected neural network on CIFAR-10**, including:

- `main_fnn_cifar10.py`
- `func_two_layer_fc.py`
- `data_helpers.py`

This appears to be supporting/reference neural-network coursework code and is **not the main student-success experiment**.

If you are here for the student-success project, start with:

- `main_snn_final.py`

---

## Repository Structure

```text
Machine-Learning-Approaches-to-Predict-Student-Success/
└── Final_Project_Neural_Network/
    └── Final Project/
        ├── main_snn_final.py
        ├── util.py
        ├── student-mat.csv
        ├── student-por.csv
        ├── student.txt
        ├── student-merge.R
        ├── code_FNN_TF/
        │   ├── data_helpers.py
        │   ├── func_two_layer_fc.py
        │   └── main_fnn_cifar10.py
        └── ...

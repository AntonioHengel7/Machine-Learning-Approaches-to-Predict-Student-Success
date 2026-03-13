import numpy as np
import pandas as pd
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from imblearn.over_sampling import SMOTE

# Define plot learning curve function
def plot_learning_curve(model, X_train, y_train, title, cv=5):
    """Plots the learning curve for a given model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    plt.plot(train_sizes, val_mean, label="Validation Score", color="green")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="green")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("R^2 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Define plot convergence curve function
def plot_convergence_curve(model, X_train, y_train, title, cv=5):
    """Plots the convergence curve (validation scores) for a given model."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, val_mean, label="Validation Score", color="green")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="green")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Validation R^2 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Step 1: Load and Inspect Data
# data = dl.download_data('student-mat.csv')  # Load Mathematics dataset
data = dl.download_data('student-por.csv')  # Load Portuguese dataset

# Step 2: Preprocess Data
# Drop G1 and G2 for challenging predictions (optional)
data = data.drop(columns=['G1', 'G2'])

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop(columns=['G3'])
y = data['G3']

# print(y.value_counts())

# Identify rare grades
rare_grades = y.value_counts()[y.value_counts() < 2].index

# Filter out rare grades, one student with grade = 5, one student with grade = 1
X = X[~y.isin(rare_grades)]
y = y[~y.isin(rare_grades)]

# print(y.value_counts())

print(f"After filtering, dataset size: {X.shape}, {y.shape}")

# Normalize features
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply SMOTE to Handle Imbalance
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42, k_neighbors=1)  # Apply SMOTE with a fixed random_state for reproducibility
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data
# First split: Training (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.4)

# Second split: Validation (20%) and Test (20%) from temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

print(f"After SMOTE - Training set: {X_train.shape}, {y_train.shape}")
print(f"After SMOTE - Validation set: {X_val.shape}, {y_val.shape}")
print(f"After SMOTE - Test set: {X_test.shape}, {y_test.shape}")


# Function to train, validate, and test SVM models with different kernels
def evaluate_svm(kernel, degree=None, gamma='scale', C=1.0):
    if kernel == 'poly':
        model = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, C=C)
    else:
        model = svm.SVR(kernel=kernel, gamma=gamma, C=C)

    # Train on the training data
    model.fit(X_train, y_train)

    # Validate the model
    y_val_pred = model.predict(X_val)
    mse_val = metrics.mean_squared_error(y_val, y_val_pred)
    r2_val = metrics.r2_score(y_val, y_val_pred)

    # Retrain with 80% (Training + Validation) and Evaluate on Test Set
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    final_model = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, C=C) if kernel == 'poly' else svm.SVR(kernel=kernel, gamma=gamma, C=C)
    final_model.fit(X_train_val, y_train_val)

    # Test the model
    y_test_pred = final_model.predict(X_test)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)

    return {
        'kernel': kernel,
        'C': C,
        'degree': degree,
        'mse_test': mse_test,
        'r2_test': r2_test,
        'model': final_model,
        'y_test_pred': y_test_pred
    }

# Store the best results for each kernel
best_results = {
    'linear': {'r2_test': -float('inf')},  # Initialize with very low R^2
    'poly': {'r2_test': -float('inf')},
    'rbf': {'r2_test': -float('inf')}
}

# Evaluate Linear Kernel
for C in [1.0, 2.0, 4.0, 6.0, 8.0]:
    result = evaluate_svm(kernel='linear', C=C)
    if result['r2_test'] > best_results['linear']['r2_test']:
        best_results['linear'] = result

# Evaluate Polynomial Kernel (degree=3)
for C in [1.0, 2., 4.0, 6.0, 8.0]:
    for d in [2, 3, 4, 5]:
        result = evaluate_svm(kernel='poly', degree=d, C=C)
        if result['r2_test'] > best_results['poly']['r2_test']:
            best_results['poly'] = result

# Evaluate RBF Kernel
for C in [1.0, 2., 4.0, 6.0, 8.0]:
    result = evaluate_svm(kernel='rbf', C=C)
    if result['r2_test'] > best_results['rbf']['r2_test']:
        best_results['rbf'] = result

print("\n---------------------------------------")
# Print the best results for each kernel
for kernel, result in best_results.items():
    print(f"\nBest results for {kernel} kernel:")
    print(f"  C: {result['C']}")
    if kernel == 'poly':
        print(f"  Degree: {result['degree']}")
    print(f"  Test Mean Squared Error: {result['mse_test']:.2f}")
    print(f"  Test R^2 Score: {result['r2_test']:.2f}")

    # Plot the graph for the best result
    plt.scatter(y_test, result['y_test_pred'])
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Best Actual vs Predicted G3 ({kernel} kernel, C={result['C']})")
    plt.show()

# Determine the best-performing kernel
best_kernel = max(best_results, key=lambda k: best_results[k]['r2_test'])
best_model = best_results[best_kernel]

# Identify and print 5 samples with largest errors
# Predict on the test set using the best model
y_test_pred = best_model['model'].predict(X_test)
residuals = np.abs(y_test.values - y_test_pred)  # Calculate residuals (use y_test.values for alignment)

# Find indices of the largest errors
largest_errors_idx = np.argsort(-residuals)[:5]  # Indices of top 5 errors

# Print the 5 samples with the largest errors
print("\n---------------------------------------")
print("\nTop 5 samples where the model failed:")
for idx in largest_errors_idx:
    print(f"Sample Index: {idx}")  # Positional index of the sample in the test set
    print(f"  Actual G3: {y_test.iloc[idx]}")
    print(f"  Predicted G3: {y_test_pred[idx]:.2f}")
    print(f"  Residual (Error): {residuals[idx]:.2f}\n")

print("---------------------------------------")
# Print which kernel was selected
print(f"\n{best_kernel.capitalize()} kernel was the best, with C={best_model['C']}")
if best_kernel == 'poly':
    print(f"Degree={best_model['degree']}")
print("Ablation studies will now be performed on this kernel.")
print("Working... This may take a few seconds...")
print("\n---------------------------------------")

# Predict on the test set using the best model
y_pred = best_model['model'].predict(X_test)

# Calculate Residuals
residuals = y_test - y_pred

# Plot Residuals vs Actual G3
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='r', linestyle='--')  # Reference line at y = 0
plt.xlabel('Actual G3')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title(f'Residual Plot for SVM Predictions ({best_kernel} kernel, C={best_model["C"]})')
plt.grid(True)
plt.show()


# Perform Ablation Study
ablations = []  # To store results of ablations
features = X.columns  # List of all features

# Perform ablation: remove one feature at a time
for feature in features:
    # print(f"Performing ablation: Removing feature '{feature}'")

    # Drop the feature
    X_ablated = X.drop(columns=[feature])

    # Normalize features
    scaler = preprocessing.StandardScaler()
    X_scaled_ablated = scaler.fit_transform(X_ablated)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_ablated, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Use the best kernel and hyperparameters
    if best_kernel == 'poly':
        model = svm.SVR(kernel=best_kernel, degree=best_model['degree'], C=best_model['C'])
    else:
        model = svm.SVR(kernel=best_kernel, C=best_model['C'])
    model.fit(X_train, y_train)

    # Evaluate the model
    y_test_pred = model.predict(X_test)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)

    # Store results
    ablations.append({
        'feature_removed': feature,
        'mse_test': mse_test,
        'r2_test': r2_test
    })
    # print(f"  Test Mean Squared Error (after removing '{feature}'): {mse_test:.2f}")
    # print(f"  Test R^2 Score (after removing '{feature}'): {r2_test:.2f}")

# Print summary of ablation results
print("\nAblation Study Results:")
for ablation in ablations:
    print(
        f"  Removed Feature: {ablation['feature_removed']}, Test MSE: {ablation['mse_test']:.2f}, Test R^2: {ablation['r2_test']:.2f}")


# Find the feature removal with the highest R^2 value
best_ablation = max(ablations, key=lambda x: x['r2_test'])

# Print the best ablation result
print("\nFeature removal with the highest R^2 value after ablation studies:")
print(f"  Removed Feature: {best_ablation['feature_removed']}")
print(f"  Test Mean Squared Error: {best_ablation['mse_test']:.2f}")
print(f"  Test R^2 Score: {best_ablation['r2_test']:.2f}")

# Generate Learning Curves for Polynomial and RBF Kernels
print("\n---------------------------------------")
print("\nGenerating learning curves for Polynomial and RBF kernels...")

# Polynomial Kernel
poly_model = svm.SVR(kernel='poly', degree=3, C=8.0)
plot_learning_curve(poly_model, X_train, y_train, title="Learning Curve (Polynomial Kernel)")

# RBF Kernel
rbf_model = svm.SVR(kernel='rbf', C=8.0)
plot_learning_curve(rbf_model, X_train, y_train, title="Learning Curve (RBF Kernel)")

# Generate Convergence curves for Poly and RBF
print("\n---------------------------------------")
print("\nGenerating convergence curve for Polynomial and RBF Kernels...")

# Polynomial Kernel
poly_model = svm.SVR(kernel='poly', degree=3, C=8.0)  # Best parameters
plot_convergence_curve(poly_model, X_train, y_train, title="Convergence Curve (Polynomial Kernel)")

# RBF Kernel
rbf_model = svm.SVR(kernel='rbf', C=8.0)  # Best parameters
plot_convergence_curve(rbf_model, X_train, y_train, title="Convergence Curve (RBF Kernel)")

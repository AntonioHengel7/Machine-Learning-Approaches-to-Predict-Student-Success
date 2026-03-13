# Final Combined Script for Linear Regression Analysis with Ablation Study and Metrics Printing

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
mat_data = pd.read_csv('student-mat.csv', sep=';')
por_data = pd.read_csv('student-por.csv', sep=';')

# Combine datasets
combined_data = pd.concat([mat_data, por_data], ignore_index=True)

# Separate features (X) and target (y)
X = combined_data.drop(columns=["G3"])  # Features
y = combined_data["G3"]  # Target

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(include=["number"]).columns

# Preprocessing steps: OneHotEncoding for categorical, StandardScaling for numerical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ]
)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Models to test
models = {
    "LinearRegression": LinearRegression(),
    "Ridge - a=0.01": Ridge(alpha=0.01),
    "Ridge - a=0.1": Ridge(alpha=0.1),
    "Ridge - a=1": Ridge(alpha=1),
    "Ridge - a=10": Ridge(alpha=10),
    "Lasso - a=0.01": Lasso(alpha=0.01),
    "Lasso - a=0.1": Lasso(alpha=0.1),
    "Lasso - a=1": Lasso(alpha=1),
    "Lasso - a=10": Lasso(alpha=10)
}

# Evaluate models
results = {"LinearRegression": {}, "Ridge": {}, "Lasso": {}}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    if "LinearRegression" in name:
        results["LinearRegression"][name] = {"MSE": mse, "MAE": mae, "R2": r2}
    elif "Ridge" in name:
        results["Ridge"][name] = {"MSE": mse, "MAE": mae, "R2": r2}
    elif "Lasso" in name:
        results["Lasso"][name] = {"MSE": mse, "MAE": mae, "R2": r2}

# Select the best model from each type
best_linear = min(results["LinearRegression"], key=lambda k: results["LinearRegression"][k]["MSE"])
best_ridge = min(results["Ridge"], key=lambda k: results["Ridge"][k]["MSE"])
best_lasso = min(results["Lasso"], key=lambda k: results["Lasso"][k]["MSE"])

best_models = {
    "LinearRegression": models[best_linear],
    "Ridge": models[best_ridge],
    "Lasso": models[best_lasso]
}

# Print Metrics for Best Models
print("\nEvaluation Metrics for Best Models:")
print(f"Best Linear Regression ({best_linear}):")
print(f"  - MSE: {results['LinearRegression'][best_linear]['MSE']:.3f}")
print(f"  - MAE: {results['LinearRegression'][best_linear]['MAE']:.3f}")
print(f"  - R²: {results['LinearRegression'][best_linear]['R2']:.3f}\n")

print(f"Best Ridge Regression ({best_ridge}):")
print(f"  - MSE: {results['Ridge'][best_ridge]['MSE']:.3f}")
print(f"  - MAE: {results['Ridge'][best_ridge]['MAE']:.3f}")
print(f"  - R²: {results['Ridge'][best_ridge]['R2']:.3f}\n")

print(f"Best Lasso Regression ({best_lasso}):")
print(f"  - MSE: {results['Lasso'][best_lasso]['MSE']:.3f}")
print(f"  - MAE: {results['Lasso'][best_lasso]['MAE']:.3f}")
print(f"  - R²: {results['Lasso'][best_lasso]['R2']:.3f}")

# Learning Curves for the best model of each type
train_sizes = np.linspace(0.1, 0.9, 10)

plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(best_models.items(), 1):
    train_errors = []
    val_errors = []
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    for train_size in train_sizes:
        X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
        pipeline.fit(X_partial, y_partial)
        train_pred = pipeline.predict(X_partial)
        val_pred = pipeline.predict(X_val)
        train_errors.append(mean_squared_error(y_partial, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))

    # Plot learning curve
    plt.subplot(1, 3, i)
    plt.plot(train_sizes, train_errors, label="Training Error", marker="o")
    plt.plot(train_sizes, val_errors, label="Validation Error", marker="o")
    plt.title(f"Learning Curve - {name}")
    plt.xlabel("Training Size Proportion")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Ablation Study for the three best models
for model_name, model in best_models.items():
    print(f"\nPerforming Ablation Study for {model_name}...")
    ablation_results = {}
    for feature in X.columns:
        # Drop one feature
        X_ablation = X.drop(columns=[feature])

        # Dynamically recompute categorical and numerical features
        categorical_features_ablation = X_ablation.select_dtypes(include=["object"]).columns
        numerical_features_ablation = X_ablation.select_dtypes(include=["number"]).columns

        # Update preprocessor to use the updated column names
        preprocessor_ablation = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features_ablation),
                ("cat", OneHotEncoder(drop="first"), categorical_features_ablation),
            ]
        )

        # Build pipeline with the updated preprocessor
        pipeline_ablation = Pipeline(steps=[
            ("preprocessor", preprocessor_ablation),
            ("regressor", model)
        ])

        # Split the updated data
        X_train_ablation, X_val_ablation, y_train_ablation, y_val_ablation = train_test_split(
            X_ablation, y, test_size=0.3, random_state=42
        )

        # Fit the model and compute MSE
        pipeline_ablation.fit(X_train_ablation, y_train_ablation)
        y_val_ablation_pred = pipeline_ablation.predict(X_val_ablation)
        mse_ablation = mean_squared_error(y_val_ablation, y_val_ablation_pred)

        # Store results
        ablation_results[feature] = mse_ablation

    # Plot ablation study results
    plt.figure(figsize=(10, 6))
    plt.barh(list(ablation_results.keys()), list(ablation_results.values()))
    plt.title(f"Ablation Study: {model_name} - Impact of Removing Features on MSE")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Feature Removed")
    plt.grid(axis="x")
    plt.show()

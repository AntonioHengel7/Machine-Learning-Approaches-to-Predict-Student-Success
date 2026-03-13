import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score

# Confusion Matrix Function
def func_confusion_matrix(y_true, y_pred):
    # Get the unique classes from the true and predicted values
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_classes)
    
    # Create a confusion matrix of the appropriate size
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Map the unique classes to indices for the confusion matrix
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    # Fill the confusion matrix
    for a, p in zip(y_true, y_pred):
        # Get the indices of the true and predicted labels
        a_idx = class_to_index[a]
        p_idx = class_to_index[p]
        conf_matrix[a_idx][p_idx] += 1
    
    # Calculate accuracy, precision, and recall
    acc = np.trace(conf_matrix) / np.sum(conf_matrix)
    recall = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + 1e-8)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    
    return conf_matrix, acc, recall, precision

warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load and Inspect Data
data_por = pd.read_csv('student-por.csv', sep=';')  # Specify the correct delimiter

# Step 2: Preprocess Data
data_por = data_por.drop(columns=[col for col in ['G1', 'G2'] if col in data_por.columns])  # Drop intermediate grades if they exist

# Encode categorical variables
data_por = pd.get_dummies(data_por, drop_first=True)

# Separate features and target
X = data_por.drop(columns=['G3'])
y = data_por['G3']

# Step 3: Filter out rare grades and check the class distribution
class_counts = y.value_counts()
print(f"Class distribution before filtering: {class_counts}")

# Remove classes with fewer than the desired number of samples (e.g., less than 6)
min_samples = 6  # Set this to your desired number of neighbors
rare_classes = class_counts[class_counts < min_samples].index
X_filtered = X[~y.isin(rare_classes)]
y_filtered = y[~y.isin(rare_classes)]

# Check class distribution after filtering
print(f"Class distribution after filtering: {y_filtered.value_counts()}")

# Normalize features again after filtering (if necessary)
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Step 4: Apply SMOTE to Handle Imbalance
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42, k_neighbors=5)  # You can adjust k_neighbors if needed
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_filtered)

# Split the resampled data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

hyperparams = [
    {'hidden_layers': [128, 64], 'learning_rate': 0.0001, 'activation': 'relu'},
    {'hidden_layers': [256, 128], 'learning_rate': 0.005, 'activation': 'tanh'},
    {'hidden_layers': [256, 128], 'learning_rate': 0.01, 'activation': 'sigmoid'}
]

results = []

for i, params in enumerate(hyperparams):
    print(f"\nTraining Model {i+1} with params: {params}")

    # Build the model
    model = Sequential()
    for units in params['hidden_layers']:
        model.add(Dense(units, activation=params['activation']))
        model.add(BatchNormalization())  # Add batch normalization
        model.add(Dropout(0.2))  # Add dropout
    model.add(Dense(1))  # Regression output layer

    # Compile the model
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    # Add callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate on validation data
    y_val_pred = model.predict(X_val).flatten()
    
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R²: {r2:.4f}")

    results.append({
        'model': model,
        'mse': mse,
        'r2': r2,
        'params': params,
        'history': history
    })

# Step 5: Select and Evaluate the Best Model
best_model = min(results, key=lambda x: x['mse'])  # Select the model with the lowest MSE
print("\nBest Model Parameters:", best_model['params'])

# Evaluate the best model on the validation dataset
y_val_pred = best_model['model'].predict(X_val).flatten()
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print("\nValidation Dataset Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R²: {r2:.4f}")

# Visualize Training History for the Best Model
history = best_model['history']
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 6: Final Model Evaluation
y_test_pred = best_model['model'].predict(X_test).flatten()

# Calculate Mean Squared Error (MSE) and R² for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test Set Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Test Set R²: {r2_test:.4f}")

# Visualize Predictions vs True Values for the Test Set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Predicted vs True')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Ideal Fit')
plt.title('True vs Predicted Grades on Test Set')
plt.xlabel('True Grades (G3)')
plt.ylabel('Predicted Grades')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Final Model Evaluation
y_test_pred = best_model['model'].predict(X_test).flatten()

# Calculate Mean Squared Error (MSE) and R² for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test Set Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Test Set R²: {r2_test:.4f}")

# Display results summary
print("\nResults Summary:")
for i, result in enumerate(results):
    print(f"Model {i+1}: Hyperparameters: {result['params']}")
    print(f"Validation MSE: {result['mse']:.4f}, R²: {result['r2']:.4f}")

for i, result in enumerate(results):
    y_val_pred = result['model'].predict(X_val).flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.6, label=f'Tanh')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
    plt.title(f'Actual vs Predicted G3 for Tanh')
    plt.xlabel('Actual Grades (G3)')
    plt.ylabel('Predicted Grades')
    plt.legend()
    plt.grid(True)
    plt.show()

for i, result in enumerate(results):
    y_val_pred = result['model'].predict(X_val).flatten()
    residuals = y_val - y_val_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, residuals, alpha=0.6, label=f'Residuals for Tanh')
    plt.axhline(0, color='r', linestyle='--', linewidth=2)
    plt.title(f'Residual Plot for Tanh')
    plt.xlabel('Actual Grades (G3)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True)
    plt.show()

for i, result in enumerate(results):  # Iterate over all kernels
    model = result['model']
    train_sizes = [int(len(X_train) * frac) for frac in np.linspace(0.1, 1.0, 10)]
    train_scores, val_scores = [], []
    for size in train_sizes:
        X_partial, y_partial = X_train[:size], y_train[:size]
        model.fit(X_partial, y_partial, epochs=10, verbose=0, batch_size=32)  # Train briefly
        y_partial_pred = model.predict(X_partial).flatten()
        y_val_pred = model.predict(X_val).flatten()
        train_scores.append(r2_score(y_partial, y_partial_pred))
        val_scores.append(r2_score(y_val, y_val_pred))
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, label='Training R²')
    plt.plot(train_sizes, val_scores, label='Validation R²')
    plt.title(f'Learning Curve for Tanh, hidden layers[128, 64], learning rate: 0.005')
    plt.xlabel('Training Set Size')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.show()

ablation_results = {}
for feature in X_filtered.columns:
    X_ablation = X_filtered.drop(columns=[feature])
    X_scaled_ablation = scaler.fit_transform(X_ablation)
    X_resampled_ablation, y_resampled_ablation = smote.fit_resample(X_scaled_ablation, y_filtered)
    X_train_ab, X_temp_ab, y_train_ab, y_temp_ab = train_test_split(
        X_resampled_ablation, y_resampled_ablation, test_size=0.2, random_state=42
    )
    X_val_ab, X_test_ab, y_val_ab, y_test_ab = train_test_split(
        X_temp_ab, y_temp_ab, test_size=0.5, random_state=42
    )

    # Rebuild the model
    input_dim = X_train_ab.shape[1]
    model = Sequential()
    for units in best_model['params']['hidden_layers']:
        model.add(Dense(units, activation=best_model['params']['activation']))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    optimizer = Adam(learning_rate=best_model['params']['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    # Train the new model
    model.fit(X_train_ab, y_train_ab, epochs=10, verbose=0, batch_size=32)
    y_test_pred_ab = model.predict(X_test_ab).flatten()
    r2_ab = r2_score(y_test_ab, y_test_pred_ab)
    ablation_results[feature] = r2_ab
    
# Bar chart of ablation results
plt.figure(figsize=(12, 8))
sorted_ablation = sorted(ablation_results.items(), key=lambda x: x[1])
features, r2_scores = zip(*sorted_ablation)
plt.barh(features, r2_scores, color='skyblue')
plt.xlabel('R²')
plt.ylabel('Features Removed')
plt.title('Ablation Study Results')
plt.grid(True)
plt.show()

residuals = y_test - y_test_pred
worst_cases = np.argsort(np.abs(residuals))[-5:]  # Top 5 largest residuals
for i, idx in enumerate(worst_cases):
    print(f"Failure {i+1}: True G3 = {y_test.iloc[idx]}, Predicted G3 = {y_test_pred[idx]:.2f}, Residual = {residuals.iloc[idx]:.2f}")

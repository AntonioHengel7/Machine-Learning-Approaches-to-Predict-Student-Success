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
from sklearn.metrics import mean_absolute_error, r2_score

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
data_mat = pd.read_csv('student-mat.csv', sep=';')  # Specify the correct delimiter

# Combine the datasets
combined_data = pd.concat([data_por, data_mat])

# Step 2: Preprocess Data
combined_data = combined_data.drop(columns=[col for col in ['G1', 'G2'] if col in combined_data.columns])  # Drop intermediate grades if they exist

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, drop_first=True)

# Separate features and target
X = combined_data.drop(columns=['G3'])
y = combined_data['G3']

# Filter out rare grades
rare_grades = y.value_counts()[y.value_counts() < 2].index
X = X[~y.isin(rare_grades)]
y = y[~y.isin(rare_grades)]

# Normalize features
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply SMOTE to Handle Imbalance
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: Define and Train FNN Models with Different Hyperparameters
hyperparams = [
    {'hidden_layers': [128, 64], 'learning_rate': 0.001, 'activation': 'relu'},
    {'hidden_layers': [256, 128], 'learning_rate': 0.005, 'activation': 'tanh'},
    {'hidden_layers': [64], 'learning_rate': 0.01, 'activation': 'relu'}
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
    conf_matrix, acc, recall, precision = func_confusion_matrix(y_val, np.rint(y_val_pred))

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Per-Class Recall: {recall}")
    print(f"Per-Class Precision: {precision}")

    results.append({
        'model': model,
        'accuracy': acc,
        'conf_matrix': conf_matrix,
        'recall': recall,
        'precision': precision,
        'params': params,
        'history': history
    })

# Step 5: Select and Evaluate the Best Model
best_model = max(results, key=lambda x: x['accuracy'])
print("\nBest Model Parameters:", best_model['params'])

# Evaluate the best model on the validation dataset
y_val_pred = best_model['model'].predict(X_val).flatten()
conf_matrix, acc, recall, precision = func_confusion_matrix(y_val, np.rint(y_val_pred))

print("\nValidation Dataset Evaluation:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {acc:.4f}")
print(f"Per-Class Recall: {recall}")
print(f"Per-Class Precision: {precision}")

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

# Calculate Mean Absolute Error and R²
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"R²: {r2:.4f}")

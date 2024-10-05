import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the cleaned data
cleaned_data = pd.read_csv('combined_cleaned_data.csv')

# Define the target variable (e.g., 'PRECTOTCORR' for Precipitation prediction)
target_variable = 'PRECTOTCORR'

# Define the features (all other relevant columns except the target)
feature_columns = [
    'GWETTOP',      # Surface Soil Wetness
    'GWETROOT',     # Root Zone Soil Wetness
    'T2M',          # Temperature at 2 Meters
    'QV2M',         # Specific Humidity at 2 Meters
    'RH2M',         # Relative Humidity at 2 Meters
    'WS10M'         # Wind Speed at 10 Meters
]

# Split the data into features (X) and target (y)
X = cleaned_data[feature_columns]
y = cleaned_data[target_variable]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (important for neural networks)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression (single value)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=1)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {test_mae}")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Test MSE: {mse}")
print(f"Test RMSE: {rmse}")

# Save the trained model
model.save('irrigation_water_management_model_tf.h5')
print("Model training completed and saved as 'irrigation_water_management_model_tf.h5'.")

# Optional: Plot training history to visualize loss over epochs
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

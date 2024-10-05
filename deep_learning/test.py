import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = tf.keras.models.load_model('irrigation_water_management_model_tf.h5')

# Load your new data (or use test data)
new_data = pd.read_csv('new_data.csv')  # Replace with your actual data path

# Assuming new_data has the same features used for training
# Preprocessing steps must match those used in training
feature_columns = [
    'GWETTOP',      # Surface Soil Wetness
    'GWETROOT',     # Root Zone Soil Wetness
    'T2M',          # Temperature at 2 Meters
    'QV2M',         # Specific Humidity at 2 Meters
    'RH2M',         # Relative Humidity at 2 Meters
    'WS10M'         # Wind Speed at 10 Meters
]

# Prepare input features
X_new = new_data[feature_columns]

# Scale the new data using the same scaler used during training
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions using the model
predictions = model.predict(X_new_scaled)

# If you want to display the predictions:
predictions_df = pd.DataFrame(predictions, columns=['Predicted_PRECTOTCORR'])
print(predictions_df)

# Optionally, save predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

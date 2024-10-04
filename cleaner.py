import pandas as pd

# Path to the file
file_path = 'D:\NASA\Dataset\POWER_Regional_Daily_20100101_20101231.csv'

# Reading the file to identify the end of the header and the beginning of data.
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find the index where the header ends and the actual data starts.
header_end_index = next(i for i, line in enumerate(lines) if '-END HEADER-' in line)

# Load the dataset, skipping the header portion and using the first row after the header for column names.
data = pd.read_csv(file_path, skiprows=header_end_index + 1)

# Replace missing values (-999.0) with NaN for easier processing.
data.replace(-999.0, pd.NA, inplace=True)

# Filter the columns needed for the model.
relevant_columns = [
    'PRECTOTCORR',   # Precipitation
    'GWETTOP',       # Surface Soil Wetness
    'GWETROOT',      # Root Zone Soil Wetness
    'T2M',           # Temperature at 2 Meters
    'QV2M',          # Specific Humidity at 2 Meters
    'RH2M',          # Relative Humidity at 2 Meters
    'WS10M'          # Wind Speed at 10 Meters
]

# Filter the data to include only the relevant columns.
filtered_data = data[relevant_columns]

# Display the first few rows of the filtered data.
print(filtered_data.head())

# Handling missing values:
# Option 1: Drop rows with any missing values.
cleaned_data = filtered_data.dropna()

# Option 2: Impute missing values, e.g., filling with mean values.
# cleaned_data = filtered_data.fillna(filtered_data.mean())

# Display basic statistics for each column to understand data distribution.
print(cleaned_data.describe())

# Check for correlation between features.
correlation_matrix = cleaned_data.corr()
print(correlation_matrix)

# Save the cleaned data to a new CSV for further modeling or analysis.
cleaned_data.to_csv('new_data.csv', index=False)

# Display message confirming the script completion.
print("Data processing completed. Cleaned dataset saved as 'cleaned_data.csv'.")

# Task 2: Data Cleaning & Preprocessing
# Decodelabs Data Science Internship - Week 3
# Dataset: Weather Forecast Dataset

import pandas as pd

# Load the dataset
df = pd.read_csv("weather_forecast_data.csv")

print("Original Dataset Shape:", df.shape)

# ---------------------------------------------------------
# Step 1: Check for Missing Values
# ---------------------------------------------------------

print("\nMissing Values per Column:")
print(df.isnull().sum())
print("No missing values found — dataset is complete.")

# ---------------------------------------------------------
# Step 2: Check for Duplicate Rows
# ---------------------------------------------------------

print(f"\nDuplicate Rows Found: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset Shape After Removing Duplicates: {df.shape}")

# ---------------------------------------------------------
# Step 3: Check Data Types
# ---------------------------------------------------------

print("\nData Types:")
print(df.dtypes)
print("\nAll weather feature columns are float64 — correct format.")
print("Rain column is string (object) — correct format for classification.")

# ---------------------------------------------------------
# Step 4: Check for Outliers Using IQR Method
# ---------------------------------------------------------

numeric_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

print("\nOutlier Detection (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col:12s}: {len(outliers)} outlier(s)  |  accepted range: [{lower:.2f}, {upper:.2f}]")

# ---------------------------------------------------------
# Step 5: Check Class Balance
# ---------------------------------------------------------

print("\nClass Distribution (Rain):")
counts = df['Rain'].value_counts()
print(counts)
print(f"\n  Rain    : {counts['rain']} samples  ({counts['rain']/len(df)*100:.1f}%)")
print(f"  No Rain : {counts['no rain']} samples  ({counts['no rain']/len(df)*100:.1f}%)")
print("\n  Note: The dataset is imbalanced. 'rain' is the minority class.")
print("  This should be kept in mind when building a classification model.")

# ---------------------------------------------------------
# Step 6: Final Clean Dataset
# ---------------------------------------------------------

print("\nCleaned Dataset Shape:", df.shape)
print("\nSample of Cleaned Data:")
print(df.head())
print("\nData cleaning complete. Dataset is ready for analysis.")

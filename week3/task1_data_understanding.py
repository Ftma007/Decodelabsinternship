# Task 1: Data Collection & Dataset Understanding
# Decodelabs Data Science Internship - Week 3
# Dataset: Weather Forecast Dataset

import pandas as pd

# Load the dataset
df = pd.read_csv("weather_forecast_data.csv")

# ---------------------------------------------------------
# Basic Dataset Info
# ---------------------------------------------------------

print("Dataset Shape:")
print(f"  Rows   : {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# ---------------------------------------------------------
# Understanding the Features
# ---------------------------------------------------------

print("\nColumn Descriptions:")
print("  Temperature  - Air temperature in degrees Celsius (float)")
print("  Humidity     - Relative humidity as a percentage (float)")
print("  Wind_Speed   - Wind speed in km/h (float)")
print("  Cloud_Cover  - Percentage of sky covered by clouds (float)")
print("  Pressure     - Atmospheric pressure in hPa (float)")
print("  Rain         - Whether it rained or not (categorical: rain / no rain)")

# ---------------------------------------------------------
# Target Variable
# ---------------------------------------------------------

print("\nTarget Variable - Rain:")
print(df['Rain'].value_counts())
print(f"\n  Total 'rain' cases   : {(df['Rain'] == 'rain').sum()}")
print(f"  Total 'no rain' cases: {(df['Rain'] == 'no rain').sum()}")
print(f"  Rain percentage      : {(df['Rain'] == 'rain').sum() / len(df) * 100:.1f}%")

# ---------------------------------------------------------
# Value Ranges
# ---------------------------------------------------------

print("\nValue Ranges per Column:")
for col in df.columns[:-1]:
    print(f"  {col:12s}: min = {df[col].min():.2f}  |  max = {df[col].max():.2f}")

# ---------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------

print("\nDataset Overview:")
print("""
  This dataset contains 2500 weather observations with 5 meteorological
  features: temperature, humidity, wind speed, cloud cover, and pressure.
  The target variable is 'Rain', which indicates whether it rained or not.
  The dataset is imbalanced — only 12.6% of observations are rain cases.
  It is suitable for binary classification tasks in machine learning.
""")

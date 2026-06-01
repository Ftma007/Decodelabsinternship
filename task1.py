# Task 1: Data Collection & Dataset Understanding

import pandas as pd

# Loading the dataset
df = pd.read_csv("IRIS.csv")

# Basic Dataset Info

print("Dataset Shape:")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())


# Understanding the Features


print("\nColumn Descriptions:")
print("  sepal_length  - Length of the sepal in centimeters (numeric)")
print("  sepal_width   - Width of the sepal in centimeters (numeric)")
print("  petal_length  - Length of the petal in centimeters (numeric)")
print("  petal_width   - Width of the petal in centimeters (numeric)")
print("  species       - Iris flower species (categorical)")

print("\nUnique Species in Dataset:")
for species in df['species'].unique():
    print(f"  - {species}")

print("\nNumber of Samples per Species:")
print(df['species'].value_counts())


# Dataset Summary

print("\nDataset Overview:")
print("""
  The Iris dataset is collected from kaggle and contains measurements of 150 iris flowers
  collected from 3 different species. Each flower has 4 physical
  measurements (sepal length, sepal width, petal length, petal width)
  all recorded in centimeters. The dataset is balanced, with exactly
  50 samples per species. It is commonly used for classification tasks
  in machine learning.
""")

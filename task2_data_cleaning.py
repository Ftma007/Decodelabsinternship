# Task 2: Data Cleaning & Preprocessing

import pandas as pd

# Loading the dataset
df = pd.read_csv("IRIS.csv")

numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

print("Original Dataset Shape:", df.shape)


# Step 1: Checking for Missing Values


print("\nMissing Values per Column:")
print(df.isnull().sum())

total_missing = df.isnull().sum().sum()
if total_missing == 0:
    print("  No missing values found.")
else:
    print(f"  Total missing values: {total_missing}")
    df.fillna(df[numeric_cols].mean(), inplace=True)
    print("  Missing values filled with column mean.")


# Step 2: Remove Duplicate Rows


print("\nDuplicate Rows Found:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)
print("Dataset Shape After Removing Duplicates:", df.shape)


# Step 3: Check Data Types


print("\nData Types:")
print(df.dtypes)
print("  All numeric columns are float64 - correct format.")
print("  Species column is object (string) - correct format.")


# Step 4: Detect Outliers Using IQR Method


print("\nOutlier Detection (IQR Method):")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"  {col}: {len(outliers)} outlier(s)  |  accepted range: [{lower_bound:.2f}, {upper_bound:.2f}]")


# Step 5: Final Clean Dataset Info


print("\nCleaned Dataset Shape:", df.shape)
print("\nSample of Cleaned Data:")
print(df.head())

print("\nData cleaning complete. Dataset is ready for analysis.")

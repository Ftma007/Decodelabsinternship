# Task 3: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np

# Load and clean dataset
df = pd.read_csv("IRIS.csv")
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Step 1: Basic Statistics

print("Basic Descriptive Statistics:")
print(df[numeric_cols].describe().round(2))


# Step 2: Per-Species Statistics


print("\nMean Values per Species:")
print(df.groupby('species')[numeric_cols].mean().round(2))

print("\nStandard Deviation per Species:")
print(df.groupby('species')[numeric_cols].std().round(2))

print("\nMin and Max per Species:")
print("Min:")
print(df.groupby('species')[numeric_cols].min())
print("Max:")
print(df.groupby('species')[numeric_cols].max())

# Step 3: Correlation Analysis

print("\nCorrelation Matrix:")
print(df[numeric_cols].corr().round(2))

# Step 4: Identify Trends and Outliers

print("\nVariance per Column:")
for col in numeric_cols:
    print(f"  {col}: {df[col].var():.4f}")

print("\nOutlier Count per Column (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    count = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    print(f"  {col}: {count} outlier(s)")

# Step 5: Summary of Findings

print("\nSummary of Findings:")
print("""
  1. Iris-setosa has noticeably smaller petal length and petal width
     compared to the other two species, making it easy to separate.

  2. Iris-virginica has the largest average values overall, especially
     in petal length (mean = 5.56 cm) and petal width (mean = 2.03 cm).

  3. Petal length has the highest variance across the dataset (std = 1.76),
     meaning it varies the most and is likely the most useful feature
     for classification.

  4. Sepal width has the lowest variance (std = 0.44), so it contributes
     the least in separating species.

  5. Petal length and petal width are strongly correlated (r = 0.96),
     meaning they carry similar information.

  6. Sepal width is negatively correlated with petal dimensions,
     which is an interesting pattern worth noting.
""")

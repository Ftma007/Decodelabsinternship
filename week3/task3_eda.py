# Task 3: Exploratory Data Analysis (EDA)
# Decodelabs Data Science Internship - Week 3
# Dataset: Weather Forecast Dataset

import pandas as pd
import numpy as np

# Load and clean dataset
df = pd.read_csv("weather_forecast_data.csv")
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

# ---------------------------------------------------------
# Step 1: Basic Statistics
# ---------------------------------------------------------

print("Basic Descriptive Statistics:")
print(df[numeric_cols].describe().round(2))

# ---------------------------------------------------------
# Step 2: Statistics by Rain Category
# ---------------------------------------------------------

print("\nMean Values - Rain vs No Rain:")
print(df.groupby('Rain')[numeric_cols].mean().round(2))

print("\nStandard Deviation - Rain vs No Rain:")
print(df.groupby('Rain')[numeric_cols].std().round(2))

# ---------------------------------------------------------
# Step 3: Correlation Analysis
# ---------------------------------------------------------

print("\nCorrelation Matrix:")
print(df[numeric_cols].corr().round(2))

# ---------------------------------------------------------
# Step 4: Variance per Feature
# ---------------------------------------------------------

print("\nVariance per Column:")
for col in numeric_cols:
    print(f"  {col:12s}: {df[col].var():.4f}")

# ---------------------------------------------------------
# Step 5: Outlier Detection
# ---------------------------------------------------------

print("\nOutlier Count per Column (IQR Method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    count = len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
    print(f"  {col:12s}: {count} outlier(s)")

# ---------------------------------------------------------
# Step 6: Feature Analysis for Rain Prediction
# ---------------------------------------------------------

rain = df[df['Rain'] == 'rain']
no_rain = df[df['Rain'] == 'no rain']

print("\nFeature Comparison - Rain vs No Rain:")
print(f"  {'Feature':12s}  {'Rain Mean':>12}  {'No Rain Mean':>12}  {'Difference':>12}")
print(f"  {'-'*52}")
for col in numeric_cols:
    diff = rain[col].mean() - no_rain[col].mean()
    print(f"  {col:12s}  {rain[col].mean():>12.2f}  {no_rain[col].mean():>12.2f}  {diff:>+12.2f}")

# ---------------------------------------------------------
# Step 7: Summary of Findings
# ---------------------------------------------------------

print("\nSummary of Key Findings:")
print("""
  1. Humidity is the strongest indicator of rain. Rainy days have
     an average humidity of ~83% compared to ~62% on dry days.

  2. Cloud cover is also significantly higher on rainy days
     (~74%) compared to no-rain days (~46%).

  3. Pressure is noticeably lower on rainy days (~998 hPa)
     compared to dry days (~1016 hPa), which aligns with
     real meteorological patterns.

  4. Temperature and Wind Speed show smaller differences
     between rain and no-rain days, making them less useful
     as standalone predictors.

  5. The dataset is imbalanced — only 12.6% of samples are
     rain cases, which needs to be handled carefully when
     building a classification model.

  6. No strong correlations exist between the numeric features,
     meaning each feature contributes independently.
""")

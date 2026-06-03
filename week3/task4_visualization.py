# Task 4: Data Visualization
# Decodelabs Data Science Internship - Week 3
# Dataset: Weather Forecast Dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and clean dataset
df = pd.read_csv("weather_forecast_data.csv")
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
colors_rain = {'rain': 'steelblue', 'no rain': 'darkorange'}

# ---------------------------------------------------------
# Chart 1: Rain vs No Rain Distribution (Bar Chart)
# ---------------------------------------------------------

counts = df['Rain'].value_counts()

plt.figure(figsize=(6, 5))
plt.bar(counts.index, counts.values, color=['steelblue', 'darkorange'], edgecolor='black', width=0.5)
plt.xlabel("Rain")
plt.ylabel("Number of Observations")
plt.title("Rain vs No Rain Distribution")
for i, val in enumerate(counts.values):
    plt.text(i, val + 10, str(val), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("chart1_rain_distribution.png", dpi=150)
plt.close()
print("Saved: chart1_rain_distribution.png")

# ---------------------------------------------------------
# Chart 2: Boxplots - All Features by Rain Category
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Feature Distribution by Rain Category", fontsize=13)

for i, col in enumerate(numeric_cols):
    data = [df[df['Rain'] == cat][col].values for cat in ['rain', 'no rain']]
    bp = axes[i].boxplot(data, patch_artist=True, tick_labels=['Rain', 'No Rain'])
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('darkorange')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    axes[i].set_title(col.replace('_', ' '))

plt.tight_layout()
plt.savefig("chart2_boxplots.png", dpi=150)
plt.close()
print("Saved: chart2_boxplots.png")

# ---------------------------------------------------------
# Chart 3: Scatter Plot - Humidity vs Temperature
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))
for cat, color in colors_rain.items():
    subset = df[df['Rain'] == cat]
    plt.scatter(subset['Temperature'], subset['Humidity'],
                label=cat, color=color, alpha=0.4, s=20)
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Humidity vs Temperature by Rain Category")
plt.legend()
plt.tight_layout()
plt.savefig("chart3_humidity_vs_temperature.png", dpi=150)
plt.close()
print("Saved: chart3_humidity_vs_temperature.png")

# ---------------------------------------------------------
# Chart 4: Scatter Plot - Pressure vs Cloud Cover
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))
for cat, color in colors_rain.items():
    subset = df[df['Rain'] == cat]
    plt.scatter(subset['Cloud_Cover'], subset['Pressure'],
                label=cat, color=color, alpha=0.4, s=20)
plt.xlabel("Cloud Cover (%)")
plt.ylabel("Pressure (hPa)")
plt.title("Pressure vs Cloud Cover by Rain Category")
plt.legend()
plt.tight_layout()
plt.savefig("chart4_pressure_vs_cloud.png", dpi=150)
plt.close()
print("Saved: chart4_pressure_vs_cloud.png")

# ---------------------------------------------------------
# Chart 5: Histogram - Humidity Distribution
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))
for cat, color in colors_rain.items():
    subset = df[df['Rain'] == cat]
    plt.hist(subset['Humidity'], bins=20, alpha=0.6, color=color, label=cat)
plt.xlabel("Humidity (%)")
plt.ylabel("Frequency")
plt.title("Humidity Distribution by Rain Category")
plt.legend()
plt.tight_layout()
plt.savefig("chart5_humidity_histogram.png", dpi=150)
plt.close()
print("Saved: chart5_humidity_histogram.png")

# ---------------------------------------------------------
# Chart 6: Mean Feature Comparison (Grouped Bar Chart)
# ---------------------------------------------------------

rain_means = df[df['Rain'] == 'rain'][numeric_cols].mean()
norain_means = df[df['Rain'] == 'no rain'][numeric_cols].mean()

x = np.arange(len(numeric_cols))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, rain_means, width, label='Rain', color='steelblue', edgecolor='black', alpha=0.85)
plt.bar(x + width/2, norain_means, width, label='No Rain', color='darkorange', edgecolor='black', alpha=0.85)
plt.xticks(x, [c.replace('_', ' ') for c in numeric_cols])
plt.ylabel("Mean Value")
plt.title("Average Feature Values: Rain vs No Rain")
plt.legend()
plt.tight_layout()
plt.savefig("chart6_mean_comparison.png", dpi=150)
plt.close()
print("Saved: chart6_mean_comparison.png")

# ---------------------------------------------------------
# Chart 7: Correlation Heatmap
# ---------------------------------------------------------

corr = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))
im = plt.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(numeric_cols)), [c.replace('_', ' ') for c in numeric_cols], rotation=15)
plt.yticks(range(len(numeric_cols)), [c.replace('_', ' ') for c in numeric_cols])
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                 fontsize=9, fontweight='bold')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("chart7_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: chart7_correlation_heatmap.png")

print("\nAll charts saved successfully.")

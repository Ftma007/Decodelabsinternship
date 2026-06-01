# Task 4: Data Visualization


import pandas as pd
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv("IRIS.csv")
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
species_list = df['species'].unique()
colors = ['steelblue', 'darkorange', 'seagreen']


# Chart 1: Species Distribution (Bar Chart)


counts = df['species'].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
plt.title("Number of Samples per Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart1_species_distribution.png", dpi=150)
plt.close()
print("Saved: chart1_species_distribution.png")


# Chart 2: Boxplots for All Features by Species


fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Feature Distribution by Species", fontsize=13)

for i, col in enumerate(numeric_cols):
    data = [df[df['species'] == s][col].values for s in species_list]
    bp = axes[i].boxplot(data, patch_artist=True,
                         tick_labels=[s.split('-')[1] for s in species_list])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(col.replace('_', ' ').title())
    axes[i].set_ylabel("cm")

plt.tight_layout()
plt.savefig("chart2_boxplots.png", dpi=150)
plt.close()
print("Saved: chart2_boxplots.png")


# Chart 3: Scatter Plot - Petal Length vs Petal Width


plt.figure(figsize=(7, 5))
for species, color in zip(species_list, colors):
    subset = df[df['species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'],
                label=species.split('-')[1], color=color, alpha=0.7, s=50)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("chart3_petal_scatter.png", dpi=150)
plt.close()
print("Saved: chart3_petal_scatter.png")


# Chart 4: Scatter Plot - Sepal Length vs Sepal Width


plt.figure(figsize=(7, 5))
for species, color in zip(species_list, colors):
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'],
                label=species.split('-')[1], color=color, alpha=0.7, s=50)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("chart4_sepal_scatter.png", dpi=150)
plt.close()
print("Saved: chart4_sepal_scatter.png")


# Chart 5: Histogram - Petal Length Distribution


plt.figure(figsize=(7, 5))
for species, color in zip(species_list, colors):
    subset = df[df['species'] == species]
    plt.hist(subset['petal_length'], bins=10, alpha=0.6,
             color=color, label=species.split('-')[1])

plt.title("Petal Length Distribution by Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("chart5_petal_histogram.png", dpi=150)
plt.close()
print("Saved: chart5_petal_histogram.png")


# Chart 6: Mean Feature Comparison (Grouped Bar Chart)


import numpy as np

means = df.groupby('species')[numeric_cols].mean()
x = np.arange(len(numeric_cols))
width = 0.25

plt.figure(figsize=(9, 5))
for i, (species, color) in enumerate(zip(species_list, colors)):
    plt.bar(x + i * width, means.loc[species], width,
            label=species.split('-')[1], color=color, alpha=0.85, edgecolor='black')

plt.xticks(x + width, ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.ylabel("Mean (cm)")
plt.title("Average Feature Values per Species")
plt.legend()
plt.tight_layout()
plt.savefig("chart6_mean_comparison.png", dpi=150)
plt.close()
print("Saved: chart6_mean_comparison.png")

print("\nAll charts saved successfully.")

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"
df = pd.read_csv(file_path)

# Separate by landslides and non-landslides
landslides = df[df["IsLandslide"] == 1]["R30d"]
non_landslides = df[df["IsLandslide"] == 0]["R30d"]

# Plot histograms
plt.figure(figsize=(12, 6))

plt.hist(landslides, bins=30, alpha=0.6, label="Landslides", color="red", edgecolor="black")
plt.hist(non_landslides, bins=30, alpha=0.6, label="Non-Landslides", color="blue", edgecolor="black")

plt.xlabel("R30d (30-day rainfall)")
plt.ylabel("Frequency")
plt.title("Histogram of R30d for Landslides vs Non-Landslides")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

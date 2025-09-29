'''
Dataset	File Path or Name	CRS (EPSG)	Units	Description
/LandslideData/North_Carolina_Landslide_Points.csv	EPSG:32119	Feet (U.S. Survey Feet)	Projected coordinates from NC State Plane

/slope_asheville.tif	EPSG:4269	Degrees (lat/lon)	Geographic (NAD83), unprojected
/cb_2022_us_county_500k/...	EPSG:4269	Degrees (lat/lon)	Geographic NAD83 (matches slope raster)

Transformed Landslide GDF	After .to_crs(4269)	EPSG:4269	Degrees	Reprojected to match slope and counties

WNC County Boundaries (filtered from shapefile)	Same as above	EPSG:4269 → 32119 (in older scripts)	Degrees or Feet, depending on projection	You reprojected to EPSG:32119 in earlier visualizations
Generated Non-Landslide Points	Randomly within Buncombe bounds	EPSG:32119	Feet	Created to match reprojected landslide points
'''


import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rasterio.plot import plotting_extent
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from rasterio.mask import mask
import matplotlib.gridspec as gridspec
import shap

# === FILE PATHS ===
landslide_csv = "LandslideData/Buncombe_Landslides_With_NonEvents.csv"
dem_path = "../../SlopeData/slope_asheville.tif"
rainfall_tif = "RainfallData/PRISM_ppt_30yr_normal_800mM4_01_bil/PRISM_ppt_30yr_normal_800mM4_01_bil.bil"
county_shp = "MaskingData/cb_2022_us_county_500k.shp"


# === LOAD BUNCOMBE COUNTY ===
counties = gpd.read_file(county_shp)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]
buncombe = buncombe.to_crs("EPSG:4269")

# === LOAD LANDSLIDE DATA ===
df = pd.read_csv(landslide_csv)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")
gdf = gdf.to_crs("EPSG:4269")
gdf = gdf[gdf.geometry.within(buncombe.unary_union)]

# === LOAD DEM AND CALCULATE SLOPE (MASKED TO BUNCOMBE) ===
with rasterio.open(dem_path) as src:
    slope_masked, slope_transform = mask(src, buncombe.geometry, crop=True)
    dem = slope_masked[0].astype(float)
    transform = slope_transform
    xres = transform[0]
    yres = -transform[4]
    dy, dx = np.gradient(dem, yres, xres)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)

# === LOAD AND RESAMPLE RAINFALL (MASKED TO BUNCOMBE) ===
with rasterio.open(rainfall_tif) as rain_src:
    rainfall_masked, rain_transform = mask(rain_src, buncombe.geometry, crop=True)
    rainfall = rainfall_masked[0].astype(float)
    rainfall[rainfall == rain_src.nodata] = np.nan
    scale_y = slope.shape[0] / rainfall.shape[0]
    scale_x = slope.shape[1] / rainfall.shape[1]
    rainfall_resampled = zoom(rainfall, (scale_y, scale_x), order=1)

# === PLOT MASKED RAINFALL (large title and larger map vs. colorbar) ===
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1.5])  # Map is ~13x wider than colorbar
ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])  # colorbar axis

# Plot rainfall map
im = ax.imshow(rainfall_resampled, cmap='Blues', extent=plotting_extent(slope, transform), origin='upper')

# Title
ax.set_title("Rainfall Data in Buncombe County", fontsize=26, fontweight="bold", pad=20)

# Axis off
ax.axis("off")

# Colorbar with large label and ticks
cb = fig.colorbar(im, cax=cax)
cb.set_label("Precipitation (mm)", fontsize=20, labelpad=15)
cb.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()

# === PLOT MASKED SLOPE (enhanced layout and fonts) ===
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1.5])  # Map vs. colorbar width ratio

ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])  # Axis for the colorbar

# Plot slope raster
im = ax.imshow(slope, cmap='terrain', extent=plotting_extent(slope, transform), origin='upper')

# Title and axis styling
ax.set_title("Slope in Buncombe County", fontsize=26, fontweight="bold", pad=20)
ax.axis("off")

# Colorbar with label and tick styling
cb = fig.colorbar(im, cax=cax)
cb.set_label("Slope (degrees)", fontsize=20, labelpad=15)
cb.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()

# === PLOT LANDSLIDE VS NON-LANDSLIDE POINTS ===
fig, ax = plt.subplots(figsize=(8, 6))
buncombe.boundary.plot(ax=ax, color="black")
gdf[gdf.IsLandslide == 1].plot(ax=ax, color="red", markersize=10, label="Landslide")
gdf[gdf.IsLandslide == 0].plot(ax=ax, color="blue", markersize=10, label="Non-Landslide")
plt.legend()
plt.title("Landslide and Non-Landslide Points in Buncombe County")
plt.axis("off")
plt.tight_layout()
plt.show()

# === EXTRACT RASTER VALUES AT POINT LOCATIONS ===
coords = [(pt.x, pt.y) for pt in gdf.geometry]
rowcol = [~transform * coord for coord in coords]
rowcol = np.array(rowcol).astype(int)
rows, cols = rowcol[:, 1], rowcol[:, 0]

gdf["slope"] = slope[rows, cols]
gdf["rainfall"] = rainfall_resampled[rows, cols]

# === SAVE PROCESSED DATA WITH SLOPE AND RAINFALL ===
gdf_output = gdf.dropna(subset=["slope", "rainfall"])[["X", "Y", "slope", "rainfall", "IsLandslide"]]
gdf_output.to_csv("LandslideData/Processed_Landslide_Data_Buncombe.csv", index=False)

# === CLEAN AND SPLIT DATA ===
gdf_clean = gdf.dropna(subset=["slope", "rainfall"])
X = gdf_clean[["slope", "rainfall"]].values
y = gdf_clean["IsLandslide"].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === TRAIN MODEL ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# === CONFUSION MATRIX ===
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot(cmap='Blues', text_kw={"fontsize": 18})
plt.title("Confusion Matrix", fontsize=18)
plt.xlabel("Predicted Label", fontsize=14)  # X-axis label font size
plt.ylabel("True Label", fontsize=14)       # Y-axis label font size
plt.tight_layout()
plt.show()

# === ROC CURVE ===
y_prob_test = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob_test)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)

plt.figure(figsize=(7, 6))  # Slightly larger figure
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("ROC Curve", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


# === FEATURE IMPORTANCE ===
importances = model.feature_importances_
feature_names = ["slope", "rainfall"]

# Print and plot
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")

plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances, color="forestgreen")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest Classifier")
plt.tight_layout()
plt.show()

# === GENERATE PREDICTION MAP ===
slope_flat = slope.flatten()
rain_flat = rainfall_resampled.flatten()
valid_mask = ~np.isnan(slope_flat) & ~np.isnan(rain_flat)

slope_std = (slope_flat[valid_mask] - np.mean(X[:,0])) / np.std(X[:,0])
rain_std = (rain_flat[valid_mask] - np.mean(X[:,1])) / np.std(X[:,1])

X_grid = np.column_stack((slope_flat[valid_mask], rain_flat[valid_mask]))
probs = model.predict_proba(X_grid)[:, 1]

# === REBUILD 2D PROBABILITY MAP ===
prob_map = np.full_like(slope_flat, np.nan)
prob_map[valid_mask] = probs
prob_map = prob_map.reshape(slope.shape)

print(f"Predicted probability range: {np.nanmin(prob_map):.3f} to {np.nanmax(prob_map):.3f}")

# === PLOT MAP ===
plt.figure(figsize=(10, 8))
im = plt.imshow(prob_map, cmap="Reds", extent=plotting_extent(slope, transform), origin="upper", vmin=0, vmax=1)
cbar = plt.colorbar(im, shrink=0.5, aspect=20)  # Shorter and thinner colorbar
cbar.set_label("Predicted Landslide Probability", fontsize=10)
plt.title("Landslide Probability Map – Buncombe County", fontsize=20)
plt.axis("off")
plt.tight_layout()
plt.show()

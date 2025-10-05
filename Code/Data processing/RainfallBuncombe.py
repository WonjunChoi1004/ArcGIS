import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point


# === FILE PATHS ===
county_shp = "/Users/wonjunchoi/PycharmProjects/ArcGIS/MaskingData/cb_2022_us_county_500k.shp"
rainfall_bil = "/Users/wonjunchoi/PycharmProjects/ArcGIS/RainfallData/PRISM_ppt_30yr_normal_800mM4_01_bil/PRISM_ppt_30yr_normal_800mM4_01_bil.bil"
landslide_csv = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/North_Carolina_Landslide_Points.csv"
output_rainfall_tif = "/Users/wonjunchoi/PycharmProjects/ArcGIS/RainfallData/rainfall_buncombe.tif"



# === STEP 1: Load Buncombe County ===
counties = gpd.read_file(county_shp)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]
print("✅ Buncombe bounds:", buncombe.total_bounds)

# === STEP 2: Load and mask rainfall raster to Buncombe ===
with rasterio.open(rainfall_bil) as src:
    buncombe = buncombe.to_crs(src.crs)
    rainfall_crs = src.crs
    out_image, out_transform = mask(src, buncombe.geometry, crop=True)
    out_meta = src.meta.copy()
    rainfall_buncombe = out_image[0].astype(float)
    rainfall_buncombe[rainfall_buncombe == src.nodata] = np.nan

# === STEP 3: Load and filter landslide points ===
df = pd.read_csv(landslide_csv)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")  # Confirmed earlier
gdf = gdf.to_crs(rainfall_crs)  # Match rainfall raster
gdf_buncombe = gdf[gdf.geometry.within(buncombe.unary_union)]
print(f"✅ Landslides in Buncombe: {len(gdf_buncombe)}")

# === STEP 4: Save clipped rainfall raster ===
out_meta.update({
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "driver": "GTiff"
})

with rasterio.open(output_rainfall_tif, "w", **out_meta) as dest:
    dest.write(out_image)

# === STEP 5: Plot rainfall + landslides (large fonts, wide map) ===
from rasterio.plot import plotting_extent
import matplotlib.gridspec as gridspec

extent = plotting_extent(out_image[0], out_transform)

# --- create a wide figure with two columns (map | colour‑bar) ---
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[20, 1.2])  # map 20× wider than c‑bar
ax  = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])

# --- draw the raster ---
im = ax.imshow(
    rainfall_buncombe,
    cmap="Blues",
    extent=extent,
    origin="upper"
)

# --- county boundary & landslides ---
buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=2, linestyle="--", label="Buncombe County")
if not gdf_buncombe.empty:
    gdf_buncombe.plot(ax=ax, color="red", markersize=18, label="Landslides")

# --- styling ---
ax.set_title(
    "PRISM 30‑Year Rainfall with Landslides – Buncombe County",
    fontsize=28, fontweight="bold", pad=20
)
ax.axis("off")

# --- colour‑bar (narrow, large label) ---
cb = fig.colorbar(im, cax=cax)
cb.set_label("Precipitation (mm)", fontsize=22, labelpad=15)
cb.ax.tick_params(labelsize=18)    # enlarge tick labels

# --- legend (large text) ---
legend = ax.legend(prop={"size":18}, loc="lower left")
legend.get_frame().set_linewidth(0.0)

plt.tight_layout()
plt.show()
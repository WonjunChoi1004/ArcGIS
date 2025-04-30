import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point


# === FILE PATHS ===
county_shp = "MaskingData/cb_2022_us_county_500k.shp"
rainfall_bil = "RainfallData/PRISM_ppt_30yr_normal_800mM4_01_bil/PRISM_ppt_30yr_normal_800mM4_01_bil.bil"
landslide_csv = "LandslideData/North_Carolina_Landslide_Points.csv"
output_rainfall_tif = "RainfallData/rainfall_buncombe.tif"



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

# === STEP 5: Plot rainfall + landslides ===
from rasterio.plot import plotting_extent

extent = plotting_extent(out_image[0], out_transform)

plt.figure(figsize=(10, 8))
plt.imshow(rainfall_buncombe, cmap='Blues', extent=extent, origin="upper")
buncombe.boundary.plot(ax=plt.gca(), edgecolor="black", linewidth=2, linestyle="--", label="Buncombe County")
if not gdf_buncombe.empty:
    gdf_buncombe.plot(ax=plt.gca(), color="red", markersize=10, label="Landslides")
plt.colorbar(label="Precipitation (mm)")
plt.title("PRISM 30-Year Rainfall with Landslides – Buncombe County")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()
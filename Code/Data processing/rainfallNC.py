import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Load the U.S. states shapefile ===
states = gpd.read_file("/MaskingData/cb_2022_us_state_20m.shp")  # path to the folder where you extracted files

# === 2. Filter for North Carolina ===
nc = states[states['NAME'] == 'North Carolina']

# === 3. Load the PRISM rainfall raster ===
with rasterio.open("/RainfallData/PRISM_ppt_30yr_normal_800mM4_01_bil/PRISM_ppt_30yr_normal_800mM4_01_bil.bil") as src:
    # Make sure the projection matches
    nc = nc.to_crs(src.crs)

    # Clip the raster using the NC boundary
    out_image, out_transform = mask(src, nc.geometry, crop=True)
    out_meta = src.meta.copy()

    # Replace no-data with NaN for visualization
    rainfall_nc = out_image[0].astype(float)
    rainfall_nc[rainfall_nc == src.nodata] = np.nan

# === 4. Update metadata and save the clipped raster as GeoTIFF ===
out_meta.update({
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "driver": "GTiff"
})

with rasterio.open("../../RainfallData/rainfall_north_carolina.tif", "w", **out_meta) as dest:
    dest.write(out_image)

# === 5. Visualize the clipped rainfall raster ===
plt.figure(figsize=(10, 6))
plt.imshow(rainfall_nc, cmap='Blues')
plt.title("PRISM 30-Year Normal Rainfall – North Carolina")
plt.colorbar(label="Precipitation (mm)")
plt.axis('off')
plt.show()




import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# === Load U.S. counties and filter for WNC ===
counties = gpd.read_file("/Users/wonjunchoi/PycharmProjects/ArcGIS/MaskingData/cb_2022_us_county_500k.shp")

print(counties.head())
print(counties['NAME'].unique())

wnc_counties = [
    'Buncombe', 'Haywood', 'Jackson', 'Henderson', 'Transylvania',
    'Macon', 'Swain', 'Madison', 'Yancey', 'Mitchell', 'Avery',
    'Watauga', 'Cherokee', 'Clay', 'Graham'
]



wnc = counties[(counties['STATEFP'] == '37') & (counties['NAME'].isin(wnc_counties))]

print("WNC shape count:", len(wnc))
print("WNC bounds:", wnc.total_bounds)
# === Load and clip PRISM rainfall raster ===
with rasterio.open("/Users/wonjunchoi/PycharmProjects/ArcGIS//RainfallData/PRISM_ppt_30yr_normal_800mM4_01_bil/PRISM_ppt_30yr_normal_800mM4_01_bil.bil") as src:
    wnc = wnc.to_crs(src.crs)  # reproject to match raster
    out_image, out_transform = mask(src, wnc.geometry, crop=True)
    out_meta = src.meta.copy()
    rainfall_wnc = out_image[0].astype(float)
    rainfall_wnc[rainfall_wnc == src.nodata] = np.nan

# === Update metadata and save clipped GeoTIFF ===
out_meta.update({
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "driver": "GTiff"
})

with rasterio.open("../../RainfallData/rainfall_wnc.tif", "w", **out_meta) as dest:
    dest.write(out_image)

# === Visualize ===
plt.figure(figsize=(10, 6))
plt.imshow(rainfall_wnc, cmap='Blues')
plt.title("PRISM 30-Year Rainfall – Western North Carolina")
plt.colorbar(label="Precipitation (mm)")
plt.axis('off')
plt.show()
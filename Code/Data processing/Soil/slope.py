from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# === File Paths ===
input_csv = f"{PROJECT_ROOT}/LandslideData/All_With_SoilDepth.csv"
dem_path = f"{PROJECT_ROOT}/SlopeData/USGS_13_n36w083_20220512.tif"
reprojected_dem = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
output_csv = f"{PROJECT_ROOT}/LandslideData/All_With_SoilDepth_Elevation_Slope.csv"

# === Reproject DEM from EPSG:4269 (Geographic NAD83) to EPSG:32119 (NC State Plane, feet) ===
with rasterio.open(dem_path) as src:
    dst_crs = "EPSG:32119"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    with rasterio.open(reprojected_dem, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

# === Load Landslide Dataset ===
df = pd.read_csv(input_csv)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")

# === Open Reprojected DEM and Extract Elevation and Slope ===
with rasterio.open(reprojected_dem) as src:
    elevation = src.read(1).astype(float)
    transform = src.transform
    nodata = src.nodata

    # Compute slope in degrees
    xres = transform[0]
    yres = -transform[4]
    dy, dx = np.gradient(elevation, yres, xres)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)

    # Function to sample raster at point
    def extract_value(point, raster):
        x, y = point.x, point.y
        row, col = ~transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            val = raster[row, col]
            return np.nan if val == nodata else val
        return np.nan

    # Apply to landslide points
    gdf["Elevation_m"] = gdf.geometry.apply(lambda p: extract_value(p, elevation))
    gdf["Slope_deg"] = gdf.geometry.apply(lambda p: extract_value(p, slope))

# === Insert Elevation and Slope Columns After Soil_Depth_cm ===
cols = list(gdf.columns)
for col in ["Elevation_m", "Slope_deg"]:
    if col in cols:
        cols.remove(col)
insert_at = cols.index("Soil_Depth_cm") + 1
cols.insert(insert_at, "Elevation_m")
cols.insert(insert_at + 1, "Slope_deg")
gdf = gdf[cols]

# === Export Final CSV ===
gdf.drop(columns="geometry").to_csv(output_csv, index=False)
print(f"✅ Saved with elevation and slope to: {output_csv}")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from shapely.geometry import Point
import matplotlib.pyplot as plt

# --- Paths ---
slope_path = "/SlopeData/slope_asheville.tif"
county_shp_path = "/MaskingData/cb_2022_us_county_500k.shp"
landslide_csv_path = "../../LandslideData/North_Carolina_Landslide_Points.csv"

# --- Load slope raster ---
with rasterio.open(slope_path) as src:
    dem = src.read(1).astype(float)
    transform = src.transform
    raster_crs = src.crs

# --- Calculate slope (in degrees) ---
xres = transform[0]
yres = -transform[4]
dy, dx = np.gradient(dem, yres, xres)
slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)

# --- Load counties and extract Buncombe ---
counties = gpd.read_file(county_shp_path)
counties = counties.to_crs(raster_crs)
buncombe = counties[(counties['STATEFP'] == '37') & (counties['NAME'] == 'Buncombe')]

# --- Load landslide points (assumed projected) ---
df = pd.read_csv(landslide_csv_path)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")  # You confirmed EPSG:32119
gdf = gdf.to_crs(raster_crs)

# --- Clip points to Buncombe ---
gdf_buncombe = gdf[gdf.geometry.within(buncombe.unary_union)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 10))

# Plot slope (entire raster extent)
extent = plotting_extent(dem, transform)
slope_masked = np.ma.masked_invalid(slope)
cax = ax.imshow(slope_masked, cmap='terrain', extent=extent, vmin=0, vmax=90, origin="upper", alpha=0.9)

# Overlay Buncombe boundary
buncombe.boundary.plot(ax=ax, edgecolor='black', linewidth=2, label="Buncombe County")

# Overlay landslide points
if not gdf_buncombe.empty:
    gdf_buncombe.plot(ax=ax, color="red", markersize=10, label="Landslides")

# Add colorbar and labels
plt.colorbar(cax, label='Slope (degrees)')
plt.title("Slope and Historical Landslides in Buncombe County", fontsize=15)
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()
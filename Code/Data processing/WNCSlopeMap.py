import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from matplotlib.colors import LightSource
import geopandas as gpd

# Step 1: Open your DEM file
with rasterio.open('/SlopeData/slope_asheville.tif') as src:
    elevation = src.read(1)
    transform = src.transform
    raster_crs = src.crs

# Step 2: Calculate slope (in degrees)
xres = transform[0]
yres = -transform[4]
dy, dx = np.gradient(elevation, yres, xres)
slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
# Step 3: Load and filter county shapefile
counties = gpd.read_file('/MaskingData/cb_2022_us_county_500k.shp')  # <- update path
wnc_list = [
    'Buncombe', 'Haywood', 'Jackson', 'Henderson', 'Transylvania',
    'Macon', 'Swain', 'Madison', 'Yancey', 'Mitchell', 'Avery',
    'Watauga', 'Cherokee', 'Clay', 'Graham'
]
nc_counties = counties[counties['STATEFP'] == '37']
wnc = nc_counties[nc_counties['NAME'].isin(wnc_list)]

# Step 4: Reproject counties to match the raster
wnc = wnc.to_crs(raster_crs)

# Step 5: Plot slope map with WNC boundaries
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(slope, cmap='terrain', vmin=0, vmax=90)
wnc.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
plt.colorbar(cax, label='Slope (degrees)')
plt.title("Slope Map with Western NC County Borders")
plt.axis("off")
plt.tight_layout()
plt.show()

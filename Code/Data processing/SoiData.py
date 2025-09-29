import geopandas as gpd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Folder containing shapefiles
base_path = '../../NC021/spatial/'

# Find all .shp files
shapefiles = [f for f in os.listdir(base_path) if f.endswith('.shp')]
print("Found shapefiles:", shapefiles)

# Initialize a dictionary to hold valid GeoDataFrames
layers = {}

# Load each shapefile safely
for filename in shapefiles:
    layer_name = filename.replace('.shp', '')
    file_path = os.path.join(base_path, filename)

    try:
        print(f"\nLoading {layer_name}...")
        gdf = gpd.read_file(file_path)

        # Skip empty GeoDataFrames
        if gdf.empty:
            print(f"⚠️  {layer_name} is empty. Skipping.")
            continue

        # Set CRS if missing
        if gdf.crs is None:
            print(f"🛠  {layer_name} CRS missing. Setting to EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")

        # Drop invalid geometries
        gdf = gdf[gdf.is_valid]
        gdf = gdf.dropna(subset=['geometry'])

        layers[layer_name] = gdf
        print(f"✅ {layer_name} loaded successfully with {len(gdf)} features.")

    except Exception as e:
        print(f"❌ Failed to load {layer_name}: {e}")

# -----------------------------------------------
# Plotting
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(15, 12))
color_map = {
    'soilmu_a_nc021': 'lightgreen',
    'soilsa_a_nc021': 'orange',
    'soilsf_l_nc021': 'blue',
    'soilsf_p_nc021': 'red',
    'soilmu_l_nc021': 'gray',
    'soilmu_p_nc021': 'yellow'
}

for name, gdf in layers.items():
    try:
        color = color_map.get(name, 'lightblue')
        gdf.plot(ax=ax, alpha=0.5, edgecolor='black', color=color, label=name)
    except Exception as e:
        print(f"⚠️  Failed to plot {name}: {e}")

plt.title("SSURGO Soil Map - Buncombe County (NC021)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(True)
ax.set_aspect('auto')  # Avoid aspect ratio errors
plt.tight_layout()
plt.show()

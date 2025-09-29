import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# --- FILE PATHS ---
landslide_csv = "LandslideData/North_Carolina_Landslide_Points.csv"
county_shp = "MaskingData/cb_2022_us_county_500k.shp"
output_csv = "LandslideData/Buncombe_Landslides_With_NonEvents.csv"

# --- STEP 1: Load landslide data as GeoDataFrame ---
df = pd.read_csv(landslide_csv)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")  # Confirmed earlier

# --- STEP 2: Load and filter for Buncombe County ---
counties = gpd.read_file(county_shp)
counties = counties.to_crs("EPSG:32119")
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]

# --- STEP 3: Filter landslides within Buncombe ---
gdf_buncombe = gdf[gdf.geometry.within(buncombe.unary_union)]
print(f"✅ Landslides in Buncombe County: {len(gdf_buncombe)}")

# --- STEP 4: Generate equal number of random non-landslide points in Buncombe ---

# Create bounding box for Buncombe
minx, miny, maxx, maxy = buncombe.total_bounds

# Create random points within bounds
def generate_random_points(polygon, num_points):
    points = []
    while len(points) < num_points:
        random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
    return points

non_landslide_points = generate_random_points(buncombe.unary_union, len(gdf_buncombe))

# --- STEP 5: Convert non-landslide points to GeoDataFrame with dummy fields ---
non_df = pd.DataFrame({
    "X": [pt.x for pt in non_landslide_points],
    "Y": [pt.y for pt in non_landslide_points],
    "IsLandslide": 0
})
non_gdf = gpd.GeoDataFrame(non_df, geometry=non_landslide_points, crs="EPSG:32119")

# --- STEP 6: Label landslide points ---
gdf_buncombe["IsLandslide"] = 1
gdf_buncombe = gdf_buncombe[["X", "Y", "IsLandslide", "geometry"]]

# --- STEP 7: Combine and export ---
combined = pd.concat([gdf_buncombe, non_gdf], ignore_index=True)
combined.to_csv(output_csv, index=False)
print(f"✅ Output saved to: {output_csv}")
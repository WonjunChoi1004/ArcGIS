import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# --- FILE PATHS ---
landslide_csv = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/North_Carolina_Landslide_Points.csv"
county_shp = "/Users/wonjunchoi/PycharmProjects/ArcGIS/MaskingData/cb_2022_us_county_500k.shp"
output_csv = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/Landslides_NonEvents_dates_Sorted.csv"

# --- STEP 1: Load landslide data as GeoDataFrame ---
df = pd.read_csv(landslide_csv)
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32119")

# --- STEP 2: Load and filter for Buncombe County ---
counties = gpd.read_file(county_shp)
counties = counties.to_crs("EPSG:32119")
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]

# --- STEP 3: Filter landslides within Buncombe ---
gdf_buncombe = gdf[gdf.geometry.within(buncombe.unary_union)]
print(f"✅ Landslides in Buncombe County: {len(gdf_buncombe)}")

# --- STEP 4: Create a date field using Mvmnt_Date or Col_Date ---
gdf_buncombe["Mvmnt_Date"] = pd.to_datetime(gdf_buncombe["Mvmnt_Date"], errors="coerce")
gdf_buncombe["Col_Date"] = pd.to_datetime(gdf_buncombe["Col_Date"], errors="coerce")
gdf_buncombe["Sort_Date"] = gdf_buncombe["Mvmnt_Date"].combine_first(gdf_buncombe["Col_Date"])

# --- STEP 5: Label as landslide points ---
gdf_buncombe["IsLandslide"] = 1

# --- STEP 6: Generate equal number of non-landslide points ---
minx, miny, maxx, maxy = buncombe.total_bounds

def generate_random_points(polygon, num_points):
    points = []
    while len(points) < num_points:
        random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
    return points

non_landslide_points = generate_random_points(buncombe.unary_union, len(gdf_buncombe))
non_df = pd.DataFrame({
    "X": [pt.x for pt in non_landslide_points],
    "Y": [pt.y for pt in non_landslide_points],
    "IsLandslide": 0,
    "Sort_Date": pd.NaT
})
non_gdf = gpd.GeoDataFrame(non_df, geometry=non_landslide_points, crs="EPSG:32119")

# --- STEP 7: Combine datasets ---
combined = pd.concat([gdf_buncombe, non_gdf], ignore_index=True)

# --- STEP 8: Sort by date (NaTs go to bottom) ---
combined_sorted = combined.sort_values(by="Sort_Date", na_position="last")

# --- STEP 9: Export ---
combined_sorted.to_csv(output_csv, index=False)
print(f"✅ Output saved to: {output_csv}")
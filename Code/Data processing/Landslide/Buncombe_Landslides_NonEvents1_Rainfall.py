import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import os
from shapely.geometry import Point
from datetime import datetime, timedelta

# === CUSTOMIZE THIS BLOCK ===
INTERVAL_START = "2017-01-01"
INTERVAL_END = "2021-12-31"
INPUT_CSV = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/Landslides_NonEvents_dates_Sorted.csv"
RAINFALL_DIR = "/Users/wonjunchoi/PycharmProjects/ArcGIS/RainfallData/Rainfall"
OUTPUT_CSV = f"/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/Processed_NonLandslides_{INTERVAL_START[:4]}_{INTERVAL_END[:4]}.csv"

# === LOAD AND FILTER ===
df = pd.read_csv(INPUT_CSV)
df = df[df["IsLandslide"] == 0]
print(f"✅ Non-landslide records to process: {len(df)}")

# === ASSIGN RANDOM DATES IN RANGE ===
start = pd.to_datetime(INTERVAL_START)
end = pd.to_datetime(INTERVAL_END)
random_dates = pd.to_datetime(np.random.randint(start.value // 10**9, end.value // 10**9, size=len(df)), unit='s')
df["Random_Date"] = random_dates

# === CONVERT TO GEODATAFRAME ===
geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:32119")
gdf = gdf.to_crs("EPSG:4269")  # Reproject to lat/lon for PRISM

# === RAINFALL EXTRACTION FUNCTION ===
def extract_rainfall_stats(date, x, y):
    vals = []
    for i in range(1, 31):
        target_date = date - timedelta(days=i)
        d_str = target_date.strftime("%Y%m%d")
        folder = f"{RAINFALL_DIR}/PRISM_ppt_stable_4kmD2_{d_str[:4]}0101_{d_str[:4]}1231_bil"
        tif = f"{folder}/PRISM_ppt_stable_4kmD2_{d_str}_bil.bil"
        if not os.path.exists(tif):
            continue
        try:
            with rasterio.open(tif) as src:
                row, col = src.index(x, y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1)[row, col]
                    if val != src.nodata:
                        vals.append(val)
        except Exception as e:
            print(f"⚠️ Skipped {d_str}: {e}")
            continue
    if len(vals) >= 10:
        return np.mean(vals), np.max(vals)
    else:
        return np.nan, np.nan

# === PROCESS EACH NON-LANDSLIDE POINT ===
avg_rainfall = []
max_rainfall = []

for idx, row in gdf.iterrows():
    x, y = row.geometry.x, row.geometry.y
    date = row["Random_Date"]
    avg, max_ = extract_rainfall_stats(date, x, y)
    avg_rainfall.append(avg)
    max_rainfall.append(max_)

    print(f"📅 {date.date()} | 📍 ({x:.3f}, {y:.3f}) → "
          f"30-day Avg: {avg:.2f} mm, Max: {max_:.2f} mm")

# === SAVE OUTPUT ===
gdf["Avg_Rainfall_30day"] = avg_rainfall
gdf["Max_Rainfall_30day"] = max_rainfall

gdf_out = gdf.dropna(subset=["Avg_Rainfall_30day"])
gdf_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(gdf_out)} non-landslide rows with rainfall data to {OUTPUT_CSV}")

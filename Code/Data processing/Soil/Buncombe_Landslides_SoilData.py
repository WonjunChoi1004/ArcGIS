# Robust soil join + soil depth merge + DEM elevation/slope sampling
# (Python 3.9-compatible; no union types)

import os
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ===== PATHS =====
INPUT_CSV   = f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts.csv"
SOIL_SHP    = f"{PROJECT_ROOT}/NC021/spatial/soilmu_a_nc021.shp"
DEPTH_CSV   = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"  # has "Map unit symbol", "Rating (centimeters)"
DEM_SRC     = f"{PROJECT_ROOT}/SlopeData/USGS_13_n36w083_20220512.tif"
DEM_32119   = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
OUTPUT_CSV  = f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv"

NEAREST_MAX_DIST = 300  # feet

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===== LOAD POINTS =====
df = pd.read_csv(INPUT_CSV)
gdf = gpd.GeoDataFrame(
    df.copy(),
    geometry=[Point(xy) for xy in zip(df["X"], df["Y"])],
    crs="EPSG:32119"
)

for c in ("MUKEY", "MUSYM"):
    if c in gdf.columns:
        gdf = gdf.drop(columns=[c])

# ===== LOAD SOIL POLYGONS & SPATIAL JOIN =====
soil = gpd.read_file(SOIL_SHP)
soil.columns = [c.upper() for c in soil.columns]
soil = soil.rename(columns={"GEOMETRY": "geometry"})
soil = soil[["MUKEY", "MUSYM", "geometry"]]

if soil.crs != gdf.crs:
    soil = soil.to_crs(gdf.crs)

j = gpd.sjoin(
    gdf,
    soil,
    how="left",
    predicate="intersects",
    lsuffix="_pt",
    rsuffix="_soil"
).drop(columns=["index_right"], errors="ignore")

def pick(src_df, base):
    for name in (f"{base}_soil", base, f"{base}_right"):
        if name in src_df.columns:
            return src_df[name]
    return pd.Series(pd.NA, index=src_df.index)

j["MUKEY"] = pick(j, "MUKEY")
j["MUSYM"] = pick(j, "MUSYM")

need = j["MUKEY"].isna() | j["MUSYM"].isna()
if need.any():
    try:
        near = gpd.sjoin_nearest(
            j.loc[need, ["geometry"]],
            soil,
            how="left",
            max_distance=NEAREST_MAX_DIST,
            distance_col="__dist__"
        )
        for base in ("MUKEY", "MUSYM"):
            if base in near.columns:
                j.loc[need, base] = j.loc[need, base].fillna(near[base].values)
    except Exception:
        pass

drop_cols = [c for c in j.columns if c.endswith("_pt") or c.endswith("_soil") or c == "__dist__"]
j = j.drop(columns=drop_cols, errors="ignore")

# ===== MERGE SOIL DEPTH (from DEPTH_CSV on MUSYM) =====
depth_df = pd.read_csv(DEPTH_CSV)
depth_df.columns = depth_df.columns.str.strip()

if "Map unit symbol" not in depth_df.columns or "Rating (centimeters)" not in depth_df.columns:
    raise ValueError("Depth CSV must have columns 'Map unit symbol' and 'Rating (centimeters)'.")

depth_df["Map unit symbol"] = depth_df["Map unit symbol"].astype(str).str.strip().str.upper()
depth_df["Rating (centimeters)"] = depth_df["Rating (centimeters)"].astype(object)

j["MUSYM"] = j["MUSYM"].astype(str).str.strip().str.upper()

j = j.merge(
    depth_df[["Map unit symbol", "Rating (centimeters)"]],
    left_on="MUSYM",
    right_on="Map unit symbol",
    how="left"
).drop(columns=["Map unit symbol"])

def to_depth_cm_numeric(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        v = float(s)
        return np.nan if np.isnan(v) else min(200.0, v)
    s = str(s).strip().lower().replace("cm", "")
    if not s or s in {"null", "na", "n/a", "none", "not rated"}:
        return np.nan
    if s.startswith(">"):
        return 200.0
    if s.endswith("+"):
        s = s[:-1].strip()
    try:
        v = float(s)
        return min(200.0, v)
    except Exception:
        return np.nan

j["Soil_Depth_cm_raw"] = j["Rating (centimeters)"]
j["Soil_Depth_cm"] = j["Soil_Depth_cm_raw"].apply(to_depth_cm_numeric)
j = j.drop(columns=["Rating (centimeters)"])

# ===== DEM REPROJECTION (to EPSG:32119) =====
if (not os.path.exists(DEM_32119)) or (os.path.getmtime(DEM_32119) < os.path.getmtime(DEM_SRC)):
    with rasterio.open(DEM_SRC) as src:
        dst_crs = "EPSG:32119"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "lzw"
        })

        with rasterio.open(DEM_32119, "w", **kwargs) as dst:
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

# ===== SAMPLE ELEVATION & COMPUTE SLOPE (degrees) =====
with rasterio.open(DEM_32119) as src:
    elev = src.read(1).astype("float64")
    tfm = src.transform
    nodata = src.nodata

    xres = tfm.a
    yres = -tfm.e

    dy, dx = np.gradient(elev, yres, xres)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    def sample_arr(pt, arr):
        x, y = pt.x, pt.y
        col, row = ~tfm * (x, y)
        row = int(round(row))
        col = int(round(col))
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            v = arr[row, col]
            if nodata is not None and v == nodata:
                return np.nan
            return float(v)
        return np.nan

    j["Elevation_m"] = j.geometry.apply(lambda p: sample_arr(p, elev))
    j["Slope_deg"]   = j.geometry.apply(lambda p: sample_arr(p, slope_deg))

# ===== ORDER COLUMNS =====
cols = list(j.columns)
for c in ["Elevation_m", "Slope_deg", "Soil_Depth_cm_raw"]:
    if c in cols:
        cols.remove(c)
if "Soil_Depth_cm" in cols:
    idx = cols.index("MUSYM") + 1 if "MUSYM" in cols else len(cols)
    cols.insert(idx, "Soil_Depth_cm")
    cols.insert(idx + 1, "Elevation_m")
    cols.insert(idx + 2, "Slope_deg")
    cols.insert(idx + 3, "Soil_Depth_cm_raw")
j = j[cols]

# ===== SAVE =====
# Dedup columns and enforce numeric depth
if j.columns.duplicated().any():
    j = j.loc[:, ~j.columns.duplicated()]

j["Soil_Depth_cm"] = pd.to_numeric(j["Soil_Depth_cm"], errors="coerce")

# Category + (optional) binary flag
j["Soil_Depth_Category"] = np.where(
    j["Soil_Depth_cm"].isna(), "Unknown",
    np.where(j["Soil_Depth_cm"] >= 200, "Deeper_200cm", "Shallower_200cm")
)
# If you want a model-ready flag too, keep this; otherwise delete:
j["Soil_Depth_Deep200_Flag"] = np.where(j["Soil_Depth_cm"].isna(), np.nan, (j["Soil_Depth_cm"] >= 200).astype(int))

# Reorder near Soil_Depth_cm
cols = list(j.columns)
for c in ["Soil_Depth_Category", "Soil_Depth_Deep200_Flag"]:
    if c in cols:
        cols.remove(c)
idx = cols.index("Soil_Depth_cm") + 1 if "Soil_Depth_cm" in cols else len(cols)
cols[idx:idx] = [c for c in ["Soil_Depth_Category", "Soil_Depth_Deep200_Flag"] if c in j.columns]
j = j[cols]

j.drop(columns=["geometry"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved with Soil_Depth_cm, Soil_Depth_Category, Elevation_m, Slope_deg → {OUTPUT_CSV}")
print(f"Matched MUKEY for {(j['MUKEY'].notna()).sum()}/{len(j)} rows ({(j['MUKEY'].notna()).mean():.1%}).")

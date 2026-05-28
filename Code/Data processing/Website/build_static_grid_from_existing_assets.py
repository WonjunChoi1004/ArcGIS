import os
import time
import math
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from affine import Affine
from rasterio.features import rasterize

# ===== INPUTS =====
DEM_32119 = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
SOIL_SHP  = f"{PROJECT_ROOT}/NC021/spatial/soilmu_a_nc021.shp"
DEPTH_CSV = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"

# ===== OUTPUTS =====
OUT_TIF      = f"{PROJECT_ROOT}/Code/Website/buncombe_elev_slope_soildepth.tif"
OUT_PARQUET  = f"{PROJECT_ROOT}/Code/Website/buncombe_elev_slope_soildepth.parquet"
# Set to None to skip CSV (very large)
OUT_CSV      = None  # e.g., f"{PROJECT_ROOT}/Outputs/buncombe_elev_slope_soildepth.csv"

os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)

def pct(n, d):
    return 0.0 if d == 0 else 100.0 * float(n) / float(d)

def to_depth_cm_numeric(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower().replace("centimeters", "").replace("cm", "").strip()
    if s in {"", "null", "na", "n/a", "none", "not rated"}: return np.nan
    if s.startswith(">"): return 200.0
    try: return min(200.0, float(s))
    except: return np.nan

t0 = time.time()

# ===== 1) DEM load & stats =====
with rasterio.open(DEM_32119) as src:
    dem = src.read(1).astype("float32")
    tfm: Affine = src.transform
    crs = src.crs
    nodata = src.nodata

rx, ry = tfm.a, -tfm.e
H, W = dem.shape
pix_total = H * W
dem_mask = np.ones_like(dem, dtype=bool)
if nodata is not None and not np.isnan(nodata):
    dem_mask &= (dem != nodata)
dem_mask &= np.isfinite(dem)
valid_dem = int(dem_mask.sum())
print(f"[1/6 DEM] shape={dem.shape}, pixel_size=({rx:.2f} m, {ry:.2f} m), CRS={crs}")
print(f"[1/6 DEM] nodata={nodata}, valid={valid_dem}/{pix_total} ({pct(valid_dem, pix_total):.2f}%)")

# ===== 2) Slope (deg) =====
t = time.time()
dzdy, dzdx = np.gradient(dem, ry, rx)
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")
slope[~dem_mask] = np.nan
valid_slope = int(np.isfinite(slope).sum())
print(f"[2/6 SLOPE] time={time.time()-t:.2f}s, min/med/max={np.nanmin(slope):.2f}/{np.nanmedian(slope):.2f}/{np.nanmax(slope):.2f} deg")
print(f"[2/6 SLOPE] valid={valid_slope}/{pix_total} ({pct(valid_slope, pix_total):.2f}%)")

# ===== 3) Soil depth lookup (MUSYM -> cm) =====
t = time.time()
depth_df = pd.read_csv(DEPTH_CSV)
depth_df.columns = depth_df.columns.str.strip()
assert "Map unit symbol" in depth_df.columns, "Missing 'Map unit symbol' in depth CSV"
assert any(c.startswith("Rating") for c in depth_df.columns), "Missing 'Rating (centimeters)' column"
rating_col = [c for c in depth_df.columns if c.lower().startswith("rating")][0]

depth_df["musym_norm"] = depth_df["Map unit symbol"].astype(str).str.strip().str.upper()
depth_df["Soil_Depth_cm"] = depth_df[rating_col].map(to_depth_cm_numeric)
rows_total = len(depth_df)
rows_valid = int(depth_df["Soil_Depth_cm"].notna().sum())
musym_unique = depth_df["musym_norm"].nunique()
print(f"[3/6 DEPTH] rows={rows_total}, unique_MUSYM={musym_unique}, valid_depth_rows={rows_valid}, missing={rows_total-rows_valid}")
print(f"[3/6 DEPTH] parse_time={time.time()-t:.2f}s")

# ===== 4) Rasterize soil depth to DEM grid =====
t = time.time()
soil = gpd.read_file(SOIL_SHP)
if soil.crs != crs:
    soil = soil.to_crs(crs)

# Ensure MUSYM exists
musym_field = None
for c in soil.columns:
    if c.upper() == "MUSYM" or c.lower() == "musym":
        musym_field = c
        break
assert musym_field is not None, "MUSYM field not found in soil shapefile."

soil["musym_norm"] = soil[musym_field].astype(str).str.strip().str.upper()
soil = soil.merge(
    depth_df[["musym_norm", "Soil_Depth_cm"]],
    on="musym_norm",
    how="left"
)

poly_total = len(soil)
poly_with_depth = int(soil["Soil_Depth_cm"].notna().sum())
poly_no_depth = poly_total - poly_with_depth
print(f"[4/6 RASTERIZE] polygons={poly_total}, with_depth={poly_with_depth}, missing_depth={poly_no_depth}")

shapes = (
    (geom, float(val))
    for geom, val in zip(soil.geometry, soil["Soil_Depth_cm"])
    if geom is not None and not pd.isna(val)
)

depth_ras = rasterize(
    shapes=shapes,
    out_shape=(H, W),
    transform=tfm,
    fill=np.nan,
    dtype="float32",
    all_touched=True
)

valid_depth = int(np.isfinite(depth_ras).sum())
print(f"[4/6 RASTERIZE] depth_pixels={valid_depth}/{pix_total} ({pct(valid_depth, pix_total):.2f}%), time={time.time()-t:.2f}s")

# ===== 5) Align masks (intersection) =====
valid_mask = dem_mask & np.isfinite(slope) & np.isfinite(depth_ras)
valid_all = int(valid_mask.sum())
dropped = pix_total - valid_all
print(f"[5/6 ALIGN] final_valid_all_bands={valid_all}/{pix_total} ({pct(valid_all, pix_total):.2f}%), dropped={dropped}")

# ===== 6) Write outputs =====
# GeoTIFF 3-band: elev, slope, soil_depth
profile = {
    "driver": "GTiff",
    "height": H,
    "width": W,
    "count": 3,
    "dtype": "float32",
    "crs": crs,
    "transform": tfm,
    "compress": "DEFLATE",
    "predictor": 3,
    "zlevel": 6,
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}

# Prepare bands with consistent nodata
elev_out = dem.copy()
if nodata is not None and not np.isnan(nodata):
    elev_out[~dem_mask] = np.float32(np.nan)
else:
    # dem may contain real NaNs already; keep as is
    pass

slope_out = slope.copy()
slope_out[~np.isfinite(slope_out)] = np.nan

depth_out = depth_ras.copy()
depth_out[~np.isfinite(depth_out)] = np.nan

with rasterio.open(OUT_TIF, "w", **profile) as dst:
    dst.write(elev_out, 1)
    dst.write(slope_out, 2)
    dst.write(depth_out, 3)
    dst.update_tags(1, name="elevation_m")
    dst.update_tags(2, name="slope_deg")
    dst.update_tags(3, name="soil_depth_cm")
print(f"[6/6 WRITE] GeoTIFF: {OUT_TIF}  ({os.path.getsize(OUT_TIF)/1e6:.1f} MB)")

# Tabular export
rows = valid_all
yy, xx = np.where(valid_mask)
xs = tfm * (xx + 0.5, yy + 0.5)
df_out = pd.DataFrame({
    "x": xs[0].astype("float64"),
    "y": xs[1].astype("float64"),
    "elevation_m": elev_out[yy, xx].astype("float64"),
    "slope_deg":    slope_out[yy, xx].astype("float64"),
    "soil_depth_cm":depth_out[yy, xx].astype("float64"),
})
df_out.to_parquet(OUT_PARQUET, index=False)
print(f"[6/6 WRITE] Parquet rows={rows}, cols={df_out.shape[1]}, path={OUT_PARQUET}, size={(os.path.getsize(OUT_PARQUET)/1e6):.1f} MB")

if OUT_CSV:
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[6/6 WRITE] CSV rows={rows}, cols={df_out.shape[1]}, path={OUT_CSV}, size={(os.path.getsize(OUT_CSV)/1e6):.1f} MB")

print(f"✅ Done in {time.time()-t0:.2f}s")

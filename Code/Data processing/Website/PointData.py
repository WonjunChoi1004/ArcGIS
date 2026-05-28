from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from affine import Affine

# ===== FILE PATHS (update as needed) =====
DEM_PATH = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
SOIL_SHP = f"{PROJECT_ROOT}/NC021/spatial/soilmu_a_nc021.shp"
DEPTH_CSV = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"

# ===== HELPER =====
def to_depth_cm_numeric(s):
    """Convert Rating (centimeters) text into float cm (max 200)."""
    if pd.isna(s): return np.nan
    s = str(s).strip().lower().replace("cm", "")
    if not s or s in {"null","na","n/a","none","not rated"}:
        return np.nan
    if s.startswith(">"): return 200.0
    try: return min(200.0, float(s))
    except: return np.nan

# ===== 1. Read DEM and pick one pixel =====
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1, out_dtype="float32")
    tfm: Affine = src.transform
    crs = src.crs

# pick one valid pixel near center
H, W = dem.shape
r, c = H//2, W//2
elev = float(dem[r, c])
x, y = tfm * (c + 0.5, r + 0.5)
print(f"Pixel (row={r}, col={c}) → center X={x:.2f}, Y={y:.2f}, elev={elev:.2f}")

# ===== 2. Compute slope from DEM around that pixel =====
rx, ry = tfm.a, -tfm.e
dzdy, dzdx = np.gradient(dem, ry, rx)
slope_deg = np.degrees(np.arctan(np.hypot(dzdx[r, c], dzdy[r, c])))
print(f"Slope at that pixel: {slope_deg:.2f}°")

# ===== 3. Find which soil polygon covers that point =====
pt = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=crs)
soil = gpd.read_file(SOIL_SHP)
if soil.crs != crs:
    soil = soil.to_crs(crs)
hit = gpd.sjoin(pt, soil, how="left", predicate="within")

if hit.empty:
    print("No soil polygon found for this location.")
else:
    musym = hit.iloc[0].get("MUSYM") or hit.iloc[0].get("MUKEY")
    print(f"Soil polygon MUSYM: {musym}")

    # ===== 4. Look up soil depth from CSV =====
    df = pd.read_csv(DEPTH_CSV)
    df.columns = df.columns.str.strip()
    df["Map unit symbol"] = df["Map unit symbol"].astype(str).str.strip().str.upper()
    df["Soil_Depth_cm"] = df["Rating (centimeters)"].map(to_depth_cm_numeric)

    row = df.loc[df["Map unit symbol"].eq(str(musym).upper())]
    if row.empty:
        print("No soil depth found for that MUSYM in CSV.")
        depth = np.nan
    else:
        depth = float(row["Soil_Depth_cm"].max())
        print(f"Soil depth (cm): {depth:.1f}")

    print("\n✅ Final single-point result:")
    print({
        "X": x,
        "Y": y,
        "elev_40m": elev,
        "slope_40m": slope_deg,
        "soil_depth_cm": depth,
        "MUSYM": musym,
    })
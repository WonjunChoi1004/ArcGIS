# scripts/make_soil_depth_flag.py
import sys, os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import rasterio
from rasterio.features import rasterize
from typing import Optional

SOIL_PATH    = f"{PROJECT_ROOT}/NC021/spatial/soilmu_a_nc021.shp"
SOIL_LAYER   = None  # set if using .gdb/.gpkg
DEPTH_CSV    = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"
TEMPLATE_TIF = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
OUT_TIF      = f"{PROJECT_ROOT}/LandslideData/static/Soil_Depth_Deep200_Flag.tif"

if len(sys.argv) >= 5:
    SOIL_PATH, DEPTH_CSV, TEMPLATE_TIF, OUT_TIF = sys.argv[1:5]
if len(sys.argv) == 6:
    SOIL_LAYER = sys.argv[5]

NODATA = 255
OUT_DTYPE = "uint8"

def read_vector(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".zip":
        zurl = "zip://%s" % p
        try:
            return gpd.read_file(zurl, layer=layer)
        except Exception:
            return gpd.read_file(zurl, layer=layer, engine="fiona")
    if ext == ".gdb" or (p.exists() and p.is_dir() and p.suffix.lower() == ".gdb"):
        lyr = layer
        if lyr is None:
            import fiona
            layers = fiona.listlayers(path)
            if not layers:
                raise ValueError("No layers found in GDB.")
            lyr = layers[0]
        try:
            return gpd.read_file(path, layer=lyr)
        except Exception:
            return gpd.read_file(path, layer=lyr, engine="fiona")
    if ext == ".gpkg":
        try:
            return gpd.read_file(path, layer=layer)
        except Exception:
            return gpd.read_file(path, layer=layer, engine="fiona")
    if ext == ".shp":
        missing = [s for s in (".dbf", ".shx") if not Path(path.replace(".shp", s)).exists()]
        if missing:
            raise ValueError("Shapefile sidecar(s) missing: %s" % missing)
        try:
            return gpd.read_file(path)
        except Exception:
            return gpd.read_file(path, engine="fiona")
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        return gpd.read_file(path, layer=layer, engine="fiona")

def to_depth_cm_numeric(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float)):
        v = float(s)
        return np.nan if np.isnan(v) else min(200.0, v)
    s = str(s).strip().lower().replace("cm", "")
    if s in {"", "null", "na", "n/a", "none", "not rated"}: return np.nan
    if s.startswith(">"): return 200.0
    if s.endswith("+"): s = s[:-1].strip()
    try:
        return min(200.0, float(s))
    except Exception:
        return np.nan

def main():
    soil = read_vector(SOIL_PATH, SOIL_LAYER)
    soil.columns = [c.upper() for c in soil.columns]
    if "MUSYM" not in soil.columns:
        raise ValueError("Soil polygons must contain MUSYM.")
    soil = soil[["MUSYM", "geometry"]].copy()
    soil["MUSYM"] = soil["MUSYM"].astype(str).str.strip().str.upper()

    depth = pd.read_csv(DEPTH_CSV)
    depth.columns = depth.columns.str.strip()
    if "Map unit symbol" not in depth.columns or "Rating (centimeters)" not in depth.columns:
        raise ValueError("Depth CSV must have 'Map unit symbol' and 'Rating (centimeters)'.")
    depth["Map unit symbol"] = depth["Map unit symbol"].astype(str).str.strip().str.upper()
    depth["Rating (centimeters)"] = depth["Rating (centimeters)"].apply(to_depth_cm_numeric)
    depth["Soil_Depth_Deep200_Flag"] = np.where(depth["Rating (centimeters)"].ge(200.0), 1, 0)
    depth = depth[["Map unit symbol", "Soil_Depth_Deep200_Flag"]]

    gdf = soil.merge(depth, left_on="MUSYM", right_on="Map unit symbol", how="left").drop(columns=["Map unit symbol"])

    with rasterio.open(TEMPLATE_TIF) as tpl:
        if gdf.crs is None:
            raise ValueError("Soil layer has no CRS.")
        if gdf.crs != tpl.crs:
            gdf = gdf.to_crs(tpl.crs)

        shapes = (
            (geom, int(val))
            for geom, val in zip(gdf.geometry, gdf["Soil_Depth_Deep200_Flag"])
            if geom is not None and not pd.isna(val)
        )

        arr = rasterize(
            shapes=shapes,
            out_shape=(tpl.height, tpl.width),
            transform=tpl.transform,
            fill=NODATA,
            all_touched=True,
            dtype=OUT_DTYPE,
        )

        meta = tpl.meta.copy()
        meta.update(driver="GTiff", dtype=OUT_DTYPE, count=1, nodata=NODATA, compress="lzw", tiled=True)

    Path(OUT_TIF).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_TIF, "w", **meta) as dst:
        dst.write(arr, 1)

    print("✅ Wrote %s" % OUT_TIF)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Static grid builder: DEM elev/slope + SSURGO soil depth (streamed, Py3.9)

import os, time, json
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pathlib import Path
from rasterio.features import rasterize
import geopandas as gpd

# ===== PATHS =====
DEM_32119   = "/Users/wonjunchoi/PycharmProjects/ArcGIS/SlopeData/dem_buncombe_32119.tif"
SOIL_SHP    = "/Users/wonjunchoi/PycharmProjects/ArcGIS/NC021/spatial/soilmu_a_nc021.shp"
DEPTH_CSV   = "/Users/wonjunchoi/PycharmProjects/ArcGIS/Buncombe_Soil_Depth_Summary.csv"

OUT_PARQUET = "/Users/wonjunchoi/PycharmProjects/realtime-buncombe/data/static/static_grid.parquet"
OUT_GEOJSON = "/Users/wonjunchoi/PycharmProjects/realtime-buncombe/data/static/static_grid_preview.geojson"
OUT_LOOKUP  = "/Users/wonjunchoi/PycharmProjects/realtime-buncombe/data/static/static_grid_lookups.json"

Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_GEOJSON).parent.mkdir(parents=True, exist_ok=True)

# ===== CONTROLS =====
WRITE_MUSYM_STRING = False
ROWGROUP_SIZE      = 2_000_000
BATCH_ROWS         = 2000
PREVIEW_STRIDE     = 60
PARQUET_COMPRESSION= "zstd"

def p(x): print(x, flush=True)
def tic(): return time.time()
def toc(t0): return f"{time.time()-t0:.2f}s"

def slope_from_dem(dem: np.ndarray, tfm: Affine) -> np.ndarray:
    rx, ry = tfm.a, -tfm.e
    dzdy, dzdx = np.gradient(dem, ry, rx)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")

def to_depth_cm_numeric(s):
    if pd.isna(s): return np.nan
    if isinstance(s,(int,float)):
        v=float(s); return np.nan if np.isnan(v) else min(200.0, v)
    s=str(s).strip().lower().replace("cm","")
    if not s or s in {"null","na","n/a","none","not rated"}: return np.nan
    if s.startswith(">"): return 200.0
    if s.endswith("+"): s=s[:-1].strip()
    try: return min(200.0, float(s))
    except: return np.nan

def read_soils_any(path: str, target_crs):
    try:
        soil = gpd.read_file(path)  # pyogrio path
        src_engine = "pyogrio"
    except Exception as e1:
        try:
            soil = gpd.read_file(path, engine="fiona")
            src_engine = "fiona"
        except Exception as e2:
            raise RuntimeError(f"Could not open soils with pyogrio ({e1}) or fiona ({e2}). Path: {path}")
    print(f"  opened soils with {src_engine}", flush=True)
    soil.columns = [c.upper() for c in soil.columns]
    if "GEOMETRY" in soil.columns: soil = soil.rename(columns={"GEOMETRY":"geometry"})
    keep = [c for c in ("MUSYM","MUKEY","geometry") if c in soil.columns]
    if not keep: raise ValueError("Soil layer missing MUSYM/MUKEY.")
    if "geometry" not in keep: keep.append("geometry")
    soil = soil[keep]
    if "MUSYM" not in soil.columns: soil["MUSYM"] = soil["MUKEY"].astype(str)
    soil["MUSYM"] = soil["MUSYM"].astype(str).str.strip().str.upper()
    if soil.crs != target_crs: soil = soil.to_crs(target_crs)
    return soil

def load_soil_and_rasterize(soil_path, depth_csv, target_crs, out_shape, out_transform):
    p("• Loading soils…")
    soil = read_soils_any(soil_path, target_crs)

    p("• MUSYM coding…")
    unique_musym = pd.Index(soil["MUSYM"].unique())
    code_map = {s:i for i,s in enumerate(unique_musym, start=1)}  # 0 = nodata
    inv_code_map = {int(v):k for k,v in code_map.items()}

    p("• Rasterizing MUSYM → grid…")
    shapes = ((geom, code_map[sym]) for geom, sym in zip(soil.geometry, soil["MUSYM"]))
    musym_code = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype="int32",
        all_touched=False
    )

    p("• Loading soil depth CSV…")
    ddf = pd.read_csv(depth_csv)
    ddf.columns = ddf.columns.str.strip()
    if "Map unit symbol" not in ddf.columns or "Rating (centimeters)" not in ddf.columns:
        raise ValueError("Depth CSV must have 'Map unit symbol' and 'Rating (centimeters)'.")
    ddf["Map unit symbol"] = ddf["Map unit symbol"].astype(str).str.strip().str.upper()
    ddf["Soil_Depth_cm"] = ddf["Rating (centimeters)"].map(to_depth_cm_numeric)

    p("• MUSYM code → depth LUT…")
    agg = ddf.groupby("Map unit symbol", as_index=True)["Soil_Depth_cm"].max()
    depth_map = {code: (float(agg.get(sym)) if (sym in agg.index and pd.notna(agg.get(sym))) else np.nan)
                 for code, sym in inv_code_map.items()}
    max_code = int(musym_code.max())
    lut = np.zeros(max_code+1, dtype="float32")
    for c in range(max_code+1):
        v = depth_map.get(c, np.nan)
        lut[c] = np.nan if v!=v else float(v)
    soil_depth = lut[musym_code].astype("float32")

    return musym_code.astype("int32"), soil_depth, inv_code_map

def write_parquet_stream(dem, slope, musym_code, soil_depth, tfm: Affine,
                         out_parquet: str, include_musym_text: bool, inv_code_map: dict):
    import pyarrow as pa, pyarrow.parquet as pq
    H, W = dem.shape
    valid = ~np.isnan(dem)
    total = int(valid.sum())
    if total == 0: raise RuntimeError("No valid DEM pixels found.")
    p("• Writing Parquet (stream)…")
    fields = [
        pa.field("row", pa.int32()),
        pa.field("col", pa.int32()),
        pa.field("X", pa.float64()),
        pa.field("Y", pa.float64()),
        pa.field("elev_40m", pa.float32()),
        pa.field("slope_40m", pa.float32()),
        pa.field("soil_depth_cm", pa.float32()),
        pa.field("musym_code", pa.int32()),
    ]
    if include_musym_text: fields.append(pa.field("MUSYM", pa.string()))
    schema = pa.schema(fields)
    writer = pq.ParquetWriter(out_parquet, schema, compression=PARQUET_COMPRESSION, version="2.6")

    a,b,c,d,e,f = tfm.a, tfm.b, tfm.c, tfm.d, tfm.e, tfm.f
    written = 0
    for r0 in range(0, H, BATCH_ROWS):
        r1 = min(H, r0 + BATCH_ROWS)
        vblock = valid[r0:r1, :]
        if not vblock.any(): continue
        rows_idx, cols_idx = np.where(vblock)
        rows = rows_idx.astype(np.int32, copy=False) + r0
        cols = cols_idx.astype(np.int32, copy=False)
        colp = cols + 0.5; rowp = rows + 0.5
        X = (c + colp*a + rowp*b).astype("float64", copy=False)
        Y = (f + colp*d + rowp*e).astype("float64", copy=False)
        elev = dem[rows, cols].astype("float32", copy=False)
        slp  = slope[rows, cols].astype("float32", copy=False)
        sdc  = soil_depth[rows, cols].astype("float32", copy=False)
        mcd  = musym_code[rows, cols].astype("int32",  copy=False)

        data = {
            "row": rows, "col": cols, "X": X, "Y": Y,
            "elev_40m": elev, "slope_40m": slp,
            "soil_depth_cm": sdc, "musym_code": mcd,
        }
        if include_musym_text:
            musym_txt = np.array([inv_code_map.get(int(k), "") for k in mcd], dtype=object)
            data["MUSYM"] = musym_txt

        table = pa.Table.from_pydict(data, schema=schema)
        start, n = 0, table.num_rows
        while start < n:
            end = min(n, start + ROWGROUP_SIZE)
            writer.write_table(table.slice(start, end - start))
            start = end
        written += n
        if written % 5_000_000 < len(rows): p(f"  wrote {written:,}/{total:,} rows…")
    writer.close()
    p(f"  Parquet done: {out_parquet} ({written:,} rows)")

def write_preview_geojson(dem, slope, soil_depth, tfm: Affine, out_geojson: str, stride: int):
    from pyproj import Transformer
    H, W = dem.shape
    transformer = Transformer.from_crs("EPSG:32119", "EPSG:4326", always_xy=True)
    a,b,c,d,e,f = tfm.a, tfm.b, tfm.c, tfm.d, tfm.e, tfm.f
    feats = []
    for r in range(0, H, stride):
        rowp = (r + 0.5)
        cols = np.arange(0, W, stride, dtype=np.int32)
        colp = cols + 0.5
        X = c + colp*a + rowp*b
        Y = f + colp*d + rowp*e
        ev = dem[r, cols]; sp = slope[r, cols]; sd = soil_depth[r, cols]
        mask = ~np.isnan(ev)
        if not mask.any(): continue
        lons, lats = transformer.transform(X[mask], Y[mask])
        for lon, lat, e0, s0, d0 in zip(lons, lats, ev[mask], sp[mask], sd[mask]):
            feats.append({
                "type":"Feature",
                "geometry":{"type":"Point","coordinates":[float(lon), float(lat)]},
                "properties":{
                    "elev_40m":float(e0),
                    "slope_40m":float(s0),
                    "soil_depth_cm": (None if np.isnan(d0) else float(d0))
                }
            })
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":feats}, f)
    p(f"  Preview GeoJSON wrote {len(feats):,} points (stride={stride}) → {out_geojson}")

def main():
    t0 = tic()
    p("▶ Building static grid (full DEM)")
    p("• Reading DEM…")
    with rasterio.open(DEM_32119) as src:
        dem = src.read(1, out_dtype="float32")
        tfm, crs, nd = src.transform, src.crs, src.nodata
    if nd is not None: dem[dem == nd] = np.nan
    H, W = dem.shape
    p(f"  DEM shape: {H}x{W}")

    p("• Computing slope…")
    slope = slope_from_dem(dem, tfm)

    p("• Soil rasterization & depth mapping…")
    musym_code, soil_depth, inv_code_map = load_soil_and_rasterize(
        SOIL_SHP, DEPTH_CSV, crs, dem.shape, tfm
    )

    with open(OUT_LOOKUP, "w", encoding="utf-8") as f:
        json.dump({"musym_code_to_text": inv_code_map}, f)

    p("• Writing Parquet…")
    write_parquet_stream(
        dem, slope, musym_code, soil_depth, tfm, OUT_PARQUET,
        include_musym_text=WRITE_MUSYM_STRING,
        inv_code_map=inv_code_map
    )

    p("• Writing decimated GeoJSON preview…")
    write_preview_geojson(dem, slope, soil_depth, tfm, OUT_GEOJSON, PREVIEW_STRIDE)

    p(f"✅ Done in {toc(t0)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        raise

import os
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from datetime import timedelta

INPUT_CSV    = f"{PROJECT_ROOT}/LandslideData/Landslides_NonEvents_dates_Sorted.csv"
RAINFALL_DIR = f"{PROJECT_ROOT}/RainfallData/Rainfall"
OUT_DIR      = f"{PROJECT_ROOT}/LandslideData/Rainfall_Processing"

# ---- CHANGE THESE EACH RUN ----
MODE         = "non_events"        # "events" or "non_events"
PERIOD_START = "2017-01-01"    # then 1980-01-01, 2005-01-01, 2011-01-01, 2017-01-01
PERIOD_END   = "2021-12-31"    # then 2004-12-31, 2010-12-31, 2016-12-31, 2021-12-31

def find_year_folder(year):
    prefix = f"PRISM_ppt_stable_4kmD2_{year}0101_"
    for name in os.listdir(RAINFALL_DIR):
        if name.startswith(prefix) and name.endswith("_bil"):
            p = os.path.join(RAINFALL_DIR, name)
            if os.path.isdir(p):
                return p
    return None

def bil_path_for_date(d_str):
    folder = find_year_folder(d_str[:4])
    if not folder:
        return None
    tif = os.path.join(folder, f"PRISM_ppt_stable_4kmD2_{d_str}_bil.bil")
    return tif if os.path.exists(tif) else None

def compute_windows(date, x, y):
    vals = []
    for i in range(30, 0, -1):  # prior 30 days, oldest→newest
        d_str = (date - timedelta(days=i)).strftime("%Y%m%d")
        tif = bil_path_for_date(d_str)
        if not tif:
            continue
        try:
            with rasterio.open(tif) as src:
                r, c = src.index(x, y)
                if 0 <= r < src.height and 0 <= c < src.width:
                    v = src.read(1)[r, c]
                    if src.nodata is None or v != src.nodata:
                        vals.append(float(v))
        except Exception:
            continue

    if not vals:
        return {"Avg_Rainfall_30day": np.nan, "Max_Rainfall_30day": np.nan,
                "Max_Rainfall_3day": np.nan, "R1d": np.nan, "R3d": np.nan,
                "R7d": np.nan, "R30d": np.nan, "API_14": np.nan}

    r1  = vals[-1]
    r3  = sum(vals[-3:])
    r7  = sum(vals[-7:])
    r30 = sum(vals)
    max3 = float(np.nanmax(vals[-3:]))
    avg30 = float(np.mean(vals))
    max30 = float(np.max(vals))

    lam = np.exp(-1.0 / 14.0)
    api = 0.0
    for v in vals:
        api = v + lam * api

    return {"Avg_Rainfall_30day": avg30,
            "Max_Rainfall_30day": max30,
            "Max_Rainfall_3day":  max3,
            "R1d": r1, "R3d": r3, "R7d": r7, "R30d": r30,
            "API_14": api}

def process_events(df_all, start, end):
    df = df_all.copy()
    df["Sort_Date"] = pd.to_datetime(df["Sort_Date"], errors="coerce")
    df = df[(df["IsLandslide"] == 1) & df["Sort_Date"].between(start, end, inclusive="both")]
    df = df.dropna(subset=["Sort_Date", "X", "Y"])
    if df.empty:
        return pd.DataFrame()

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["X"], df["Y"])], crs="EPSG:32119").to_crs("EPSG:4269")
    feats = [compute_windows(row["Sort_Date"], row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
    wdf = pd.DataFrame(feats)
    out = pd.concat([gdf.reset_index(drop=True), wdf], axis=1).drop(columns=["geometry"])
    return out.dropna(subset=["Avg_Rainfall_30day"], how="all")

def process_nonevents(df_all, start, end):
    df = df_all.copy()
    df = df[df["IsLandslide"] == 0].dropna(subset=["X", "Y"])
    if df.empty:
        return pd.DataFrame()

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    rng = np.random.default_rng(42)
    sec = rng.integers(low=int(s.value/1e9), high=int(e.value/1e9), size=len(df))
    df["Random_Date"] = pd.to_datetime(sec, unit="s")

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["X"], df["Y"])], crs="EPSG:32119").to_crs("EPSG:4269")
    feats = [compute_windows(row["Random_Date"], row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
    wdf = pd.DataFrame(feats)
    out = pd.concat([gdf.reset_index(drop=True), wdf], axis=1).drop(columns=["geometry"])
    return out.dropna(subset=["Avg_Rainfall_30day"], how="all")

# ---- run ----
df_all = pd.read_csv(INPUT_CSV)

if MODE == "events":
    ev = process_events(df_all, PERIOD_START, PERIOD_END)
    if len(ev):
        p = os.path.join(OUT_DIR, f"Processed_Landslides_{PERIOD_START[:4]}_{PERIOD_END[:4]}.csv")
        ev.to_csv(p, index=False)
        print(f"✅ Events {PERIOD_START}–{PERIOD_END}: {len(ev)} → {p}")
    else:
        print("ℹ️ No events in this period.")
elif MODE == "non_events":
    ne = process_nonevents(df_all, PERIOD_START, PERIOD_END)
    if len(ne):
        p = os.path.join(OUT_DIR, f"Processed_NonLandslides_{PERIOD_START[:4]}_{PERIOD_END[:4]}.csv")
        ne.to_csv(p, index=False)
        print(f"✅ Non-events {PERIOD_START}–{PERIOD_END}: {len(ne)} → {p}")
    else:
        print("ℹ️ No non-events in this period.")
else:
    raise ValueError("MODE must be 'events' or 'non_events'")

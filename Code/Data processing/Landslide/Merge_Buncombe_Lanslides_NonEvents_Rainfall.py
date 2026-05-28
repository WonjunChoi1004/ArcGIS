# Combine your per-period outputs exactly as saved into master files.

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

DATA_DIR = f"{PROJECT_ROOT}/LandslideData/"

# Expected patterns produced earlier
LAND_GLOB = "Rainfall_Processing/Processed_Landslides_*.csv"
NON_GLOB  = "Rainfall_Processing/Processed_NonLandslides_*.csv"

# Columns to keep (add here if you computed more)
RAIN_COLS = [
    "Avg_Rainfall_30day","Max_Rainfall_30day","Max_Rainfall_3day",
    "R1d","R3d","R7d","R30d","API_14"
]

ID_COLS = [
    "IsLandslide","Sort_Date","Random_Date","X","Y","County","GlobalID","OBJECTID",
    "MUKEY","MUSYM","Soil_Depth_cm","Elevation_m","Slope_deg"
]

ALL_COLS = list(dict.fromkeys(ID_COLS + RAIN_COLS))  # preserve order, unique

def period_from_name(path):
    m = re.search(r"(\d{4}_\d{4})", Path(path).name)
    return m.group(1) if m else "unknown"

def load_std(path, is_landslide):
    df = pd.read_csv(path)
    # ensure label
    if "IsLandslide" not in df.columns:
        df["IsLandslide"] = 1 if is_landslide else 0
    else:
        df["IsLandslide"] = df["IsLandslide"].fillna(1 if is_landslide else 0).astype(int)
    # event date
    if "Sort_Date" in df.columns:
        df["Event_Date"] = pd.to_datetime(df["Sort_Date"], errors="coerce")
    if "Random_Date" in df.columns:
        # prefer Random_Date for non-events; keep Sort_Date if present
        r = pd.to_datetime(df["Random_Date"], errors="coerce") if "Random_Date" in df.columns else pd.NaT
        df["Event_Date"] = r.combine_first(df.get("Event_Date", pd.Series([pd.NaT]*len(df))))
    if "Event_Date" not in df.columns:
        df["Event_Date"] = pd.NaT
    # add missing expected cols
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # metadata
    df["Source_Period"] = period_from_name(path)
    df["Data_Type"] = "Landslide" if is_landslide else "NonLandslide"
    return df

# Load per-period files exactly as saved
land_files = sorted(glob.glob(os.path.join(DATA_DIR, LAND_GLOB)))
non_files  = sorted(glob.glob(os.path.join(DATA_DIR, NON_GLOB)))

lands = pd.concat([load_std(p, True)  for p in land_files], ignore_index=True) if land_files else pd.DataFrame()
nones = pd.concat([load_std(p, False) for p in non_files],  ignore_index=True) if non_files  else pd.DataFrame()

# Clean + order
def clean(df):
    if df.empty: return df
    # canonical column order
    base_order = ["IsLandslide","Event_Date","Sort_Date","Random_Date","X","Y","Elevation_m","Slope_deg","Soil_Depth_cm"] + RAIN_COLS + ["County","MUKEY","MUSYM","GlobalID","OBJECTID","Data_Type","Source_Period"]
    for c in base_order:
        if c not in df.columns:
            df[c] = np.nan
    df = df[base_order]
    # drop dups
    df = df.drop_duplicates(subset=["IsLandslide","X","Y","Event_Date"], keep="last")
    return df.sort_values(["Event_Date","IsLandslide","X","Y"]).reset_index(drop=True)

lands = clean(lands)
nones = clean(nones)

# Save masters
land_out = Path(DATA_DIR) / "All_Landslides_1980_2021.csv"
non_out  = Path(DATA_DIR) / "All_NonLandslides_2017_2021.csv"
comb_out = Path(DATA_DIR) / "All_Combined.csv"

if not lands.empty:
    lands.to_csv(land_out, index=False)
    print(f"✅ Landslides: {len(lands)} → {land_out}")
else:
    print("ℹ️ No landslide files matched.")

if not nones.empty:
    nones.to_csv(non_out, index=False)
    print(f"✅ Non-landslides: {len(nones)} → {non_out}")
else:
    print("ℹ️ No non-landslide files matched.")

both = pd.concat([lands, nones], ignore_index=True) if (not lands.empty or not nones.empty) else pd.DataFrame()
if not both.empty:
    both = both.sort_values(["IsLandslide","Event_Date","X","Y"]).reset_index(drop=True)
    both.to_csv(comb_out, index=False)
    print(f"✅ Combined: {len(both)} → {comb_out}")

print("Done.")

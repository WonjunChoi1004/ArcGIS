# Remove non-landslide anomalies where R30d is ~400–500
import pandas as pd
from pathlib import Path

# ---- paths ----
INPUT_CSV = Path("/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/FinalData/All_Combined_Balanced_EqualCounts_with_soil_depth_elev_slope.csv")
OUTPUT_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_filtered_R30d_400_500_controls_removed.csv")

# ---- params ----
TARGET_COL = "R30d"
LOW, HIGH = 400.0, 600.0  # adjust if needed

# ---- load ----
df = pd.read_csv(INPUT_CSV)

# ---- coerce dtypes (safe) ----
if TARGET_COL not in df.columns:
    raise KeyError(f"Column '{TARGET_COL}' not found.")
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# If IsLandslide is not numeric, try to coerce
if "IsLandslide" not in df.columns:
    raise KeyError("Column 'IsLandslide' not found.")
try:
    is_slide = df["IsLandslide"].astype(int)
except Exception:
    is_slide = pd.to_numeric(df["IsLandslide"], errors="coerce").fillna(-1).astype(int)

# ---- define anomaly mask (controls only) ----
mask_controls = (is_slide == 0)
mask_r30d = df[TARGET_COL].between(LOW, HIGH, inclusive="both")
mask_anomaly = mask_controls & mask_r30d

# ---- report + filter ----
n_before = len(df)
n_remove = int(mask_anomaly.sum())
df_filtered = df.loc[~mask_anomaly].copy()
n_after = len(df_filtered)

print(f"Total rows before: {n_before}")
print(f"Removed control anomalies where {TARGET_COL} in [{LOW}, {HIGH}]: {n_remove}")
print(f"Total rows after:  {n_after}")

# ---- save ----
df_filtered.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

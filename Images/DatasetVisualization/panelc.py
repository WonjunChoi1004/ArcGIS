#!/usr/bin/env python3
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# ---- Paths ----
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
SOIL_SHP   = f"{PROJECT_ROOT}/NC021/spatial/soilmu_a_nc021.shp"
DEPTH_CSV  = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"
OUT_PATH   = f"{PROJECT_ROOT}/Images/DatasetVisualization/panel_c.png"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ---- Load Buncombe ----
counties = gpd.read_file(COUNTY_SHP)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")].to_crs(epsg=32119)

# ---- Load soils + depth table, join by MUKEY (fallback to MUSYM) ----
soils = gpd.read_file(SOIL_SHP)
if soils.crs is None:
    soils.set_crs(epsg=4326, inplace=True)
soils = soils.to_crs(buncombe.crs)
if "MUSYM" in soils.columns:
    soils["MUSYM"] = soils["MUSYM"].astype(str).str.strip()

depth = pd.read_csv(DEPTH_CSV, encoding="utf-8-sig")
depth.columns = depth.columns.str.strip()
depth = depth.rename(columns={
    "Map unit symbol": "MUSYM",
    "map unit symbol": "MUSYM",
    "Map unit name": "MUNAME",
    "Rating (centimeters)": "Soil_Depth_Rating"
})

if "MUSYM" in depth.columns:
    depth["MUSYM"] = depth["MUSYM"].astype(str).str.strip()

if "Soil_Depth_Rating" in depth.columns and "Soil_Depth_cm" not in depth.columns:
    rating_str = depth["Soil_Depth_Rating"].astype(str).str.strip()
    numeric_depth = pd.to_numeric(
        rating_str.str.replace(">", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )
    depth["Soil_Depth_cm"] = numeric_depth
    depth["Soil_Depth_Deep200_Flag"] = np.where(
        rating_str.str.contains(">", na=False) | (numeric_depth > 200),
        1,
        0
    )

join_key = None
for k in ["MUKEY", "mukey", "Mukey"]:
    if k in soils.columns and k in depth.columns:
        join_key = k
        break
if join_key is None:
    for k in ["MUSYM", "musym", "Musym"]:
        if k in soils.columns and k in depth.columns:
            join_key = k
            break
if join_key is None:
    raise ValueError("Could not find a common join key (MUKEY/MUSYM) between soils and depth CSV.")

soils = soils.merge(depth, how="left", on=join_key)

# ---- Depth classification (<=200 cm vs >200 cm). Try flag first, else numeric depth column ----
if "Soil_Depth_Deep200_Flag" in soils.columns:
    # Expect 1 = >200cm (deep), 0 = <=200cm (shallow). Adjust if your flag semantics differ.
    soils["DepthClass"] = np.where(soils["Soil_Depth_Deep200_Flag"].astype(float) >= 1, "> 200 cm", "≤ 200 cm")
else:
    depth_col = None
    for c in ["Soil_Depth_cm", "soil_depth_cm", "Depth_cm", "depth_cm"]:
        if c in soils.columns:
            depth_col = c
            break
    if depth_col is None:
        raise ValueError("No depth column (e.g., Soil_Depth_cm) or Deep200 flag found in the CSV.")
    soils["DepthClass"] = np.where(pd.to_numeric(soils[depth_col], errors="coerce") > 200, "> 200 cm", "≤ 200 cm")

# ---- Clip to Buncombe ----
soils_bn = gpd.clip(soils, buncombe)

# ---- Map extent (fit Buncombe) ----
xmin, ymin, xmax, ymax = buncombe.total_bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7.2, 6.3))

# Categorical colors (balanced, high-contrast palette)
colors = { "≤ 200 cm": "#d73027", "> 200 cm": "#4575b4" }

for cls, dfc in soils_bn.groupby("DepthClass"):
    dfc.plot(ax=ax, facecolor=colors.get(cls, "#cccccc"), edgecolor="none", linewidth=0, label=cls)

buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_axis_off(); ax.set_aspect("equal")

ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                       linewidth=1.2, edgecolor="black", facecolor="none", zorder=10))

# Legend inside map (white box, black border) clarifies depth classes
legend_handles = [
    Patch(facecolor=colors["≤ 200 cm"], edgecolor="black", label="≤ 200 cm"),
    Patch(facecolor=colors["> 200 cm"], edgecolor="black", label="> 200 cm"),
]
leg = ax.legend(legend_handles, [h.get_label() for h in legend_handles],
                loc="lower left", frameon=True, fontsize=9, title="Soil depth")
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.95)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

#!/usr/bin/env python3
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
OUT_PATH   = f"{PROJECT_ROOT}/Images/DatasetVisualization/panel_a.png"

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# --- Load and reproject counties ---
counties = gpd.read_file(COUNTY_SHP)
counties = counties.to_crs(5070)

# --- Include NC and neighboring states (VA, SC, TN, GA, KY, WV) ---
neighbor_fps = ["37", "51", "45", "47", "13", "21", "54"]
subset = counties[counties["STATEFP"].isin(neighbor_fps)]

# --- Extract NC and Buncombe ---
nc_counties = subset[subset["STATEFP"] == "37"]
buncombe = nc_counties[nc_counties["NAME"] == "Buncombe"]

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(4.2, 4.2))  # nearly square layout for 4x4 grid

# Plot all counties in neighboring region
subset.plot(ax=ax, facecolor="white", edgecolor="#D0D0D0", linewidth=0.35)
# Plot NC outline
nc_counties.plot(ax=ax, facecolor="white", edgecolor="#808080", linewidth=0.6)
# Highlight Buncombe
buncombe.boundary.plot(ax=ax, edgecolor="red", linewidth=2.0)

# --- Focus exactly on NC's four corners but include slight padding ---
xmin, ymin, xmax, ymax = nc_counties.total_bounds
pad_x = (xmax - xmin) * 0.08
pad_y = (ymax - ymin) * 0.08
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")
ax.set_axis_off()

# --- Black border (scientific style panel frame) ---
rect = Rectangle(
    (xlim[0], ylim[0]),
    xlim[1] - xlim[0],
    ylim[1] - ylim[0],
    linewidth=1.2,
    edgecolor="black",
    facecolor="none"
)
ax.add_patch(rect)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

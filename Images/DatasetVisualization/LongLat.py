#!/usr/bin/env python3
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# --- Paths ---
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
OUT_PATH   = f"{PROJECT_ROOT}/Figures/buncombe_boundary_graticule.png"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# --- Load Buncombe in WGS84 (lon/lat) ---
counties = gpd.read_file(COUNTY_SHP)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]
buncombe = buncombe.to_crs(epsg=4326)  # lon/lat

# --- Extent + padding (5%) ---
xmin, ymin, xmax, ymax = buncombe.total_bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

# --- Nice graticule spacing (aim ~6 lines) ---
def nice_step(span_deg):
    for s in [0.25, 0.2, 0.1, 0.05, 0.02]:
        if span_deg / s <= 8: return s
    return 0.5

step_lon = nice_step(xlim[1] - xlim[0])
step_lat = nice_step(ylim[1] - ylim[0])

# --- Degree label formatters ---
def fmt_lon(x, pos=None):
    hemi = "E" if x >= 0 else "W"
    deg  = abs(x)
    return f"{deg:.2f}° {hemi}"

def fmt_lat(y, pos=None):
    hemi = "N" if y >= 0 else "S"
    deg  = abs(y)
    return f"{deg:.2f}° {hemi}"

# --- Plot ---
fig, ax = plt.subplots(figsize=(7.2, 6.3))
buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect("equal")  # true lon/lat aspect

# Grid (graticule)
ax.set_xticks(np.arange(np.floor(xlim[0]/step_lon)*step_lon,
                        np.ceil(xlim[1]/step_lon)*step_lon + 1e-9, step_lon))
ax.set_yticks(np.arange(np.floor(ylim[0]/step_lat)*step_lat,
                        np.ceil(ylim[1]/step_lat)*step_lat + 1e-9, step_lat))
ax.xaxis.set_minor_locator(MultipleLocator(step_lon/2))
ax.yaxis.set_minor_locator(MultipleLocator(step_lat/2))
ax.grid(which="major", linestyle="--", linewidth=0.6, color="0.4", alpha=0.6)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, color="0.6", alpha=0.5)

# Axis labels/tick formatting
ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
ax.tick_params(axis="both", which="both", labelsize=9)

ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
ax.set_title("Buncombe County Boundary with Graticule", fontsize=12, pad=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()

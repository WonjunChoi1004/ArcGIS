#!/usr/bin/env python3
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# ---- Paths ----
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
RAIN_PATH  = f"{PROJECT_ROOT}/RainfallData/rainfall_buncombe.tif"
OUT_PATH   = f"{PROJECT_ROOT}/Images/DatasetVisualization/panel_d.png"

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# ---- Load Buncombe County ----
counties = gpd.read_file(COUNTY_SHP)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]

# ---- Clip rainfall raster ----
with rasterio.open(RAIN_PATH) as src:
    rain_crs = src.crs
    buncombe = buncombe.to_crs(rain_crs)
    out_img, out_aff = mask(src, buncombe.geometry, crop=True)
    rain = out_img[0].astype(float)
    nodata = src.nodata
    if nodata is not None:
        rain[rain == nodata] = np.nan
    extent = plotting_extent(rain, out_aff)

# ---- Percentile stretch for contrast ----
valid = np.isfinite(rain)
if valid.any():
    vmin, vmax = np.percentile(rain[valid], [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(rain)), float(np.nanmax(rain))
else:
    vmin, vmax = 0.0, 1.0

# ---- Match coordinate frame with panel (b) ----
xmin, ymin, xmax, ymax = buncombe.total_bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7.2, 6.3))

# Rainfall raster
im = ax.imshow(
    rain,
    cmap="Blues",
    extent=extent,
    origin="upper",
    vmin=vmin,
    vmax=vmax,
    interpolation="bilinear"
)

# County outline
buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

# Axis settings
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_axis_off()
ax.set_aspect("equal")

# Black frame
rect = Rectangle(
    (xlim[0], ylim[0]),
    xlim[1] - xlim[0],
    ylim[1] - ylim[0],
    linewidth=1.2,
    edgecolor="black",
    facecolor="none",
    zorder=10
)
ax.add_patch(rect)

# ---- Colorbar outside map (same width/padding as panel b) ----
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3.8%", pad=0.15)
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=8, length=3, width=0.6)
cb.set_label("Rainfall (mm)", fontsize=9)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

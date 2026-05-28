#!/usr/bin/env python3
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Paths
DEM_PATH = f"{PROJECT_ROOT}/SlopeData/dem_buncombe_32119.tif"
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
LANDSLIDE_CSV = f"{PROJECT_ROOT}/LandslideData/North_Carolina_Landslide_Points.csv"
OUT_PATH = f"{PROJECT_ROOT}/Images/DatasetVisualization/panel_b.png"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# Counties → Buncombe
counties = gpd.read_file(COUNTY_SHP)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]

# DEM (full)
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1).astype(float)
    if src.nodata is not None:
        dem[dem == src.nodata] = np.nan
    dem_crs = src.crs
    dem_extent = plotting_extent(dem, src.transform)

buncombe = buncombe.to_crs(dem_crs)

# Percentile stretch from local area
with rasterio.open(DEM_PATH) as src:
    out_img, _ = mask(src, buncombe.geometry.buffer(5000), crop=True)
    dem_clip = out_img[0].astype(float)
    if src.nodata is not None:
        dem_clip[dem_clip == src.nodata] = np.nan
valid = np.isfinite(dem_clip)
if valid.any():
    p2, p98 = np.percentile(dem_clip[valid], [2, 98])
    if not np.isfinite(p2) or not np.isfinite(p98) or p2 == p98:
        p2, p98 = float(np.nanmin(dem_clip)), float(np.nanmax(dem_clip))
else:
    finite = np.isfinite(dem)
    p2, p98 = np.percentile(dem[finite], [2, 98]) if finite.any() else (0.0, 1.0)

# Landslides (border-inclusive)
df = pd.read_csv(LANDSLIDE_CSV, encoding="utf-8-sig")
gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["X"], df["Y"])], crs="EPSG:32119").to_crs(dem_crs)
gdf_bun = gdf[gdf.intersects(buncombe.unary_union)]

# Map extent: fit Buncombe (small symmetric pad)
xmin, ymin, xmax, ymax = buncombe.total_bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

# Plot
fig, ax = plt.subplots(figsize=(7.2, 6.3))
im = ax.imshow(dem, cmap="gray", extent=dem_extent, origin="upper", vmin=p2, vmax=p98, interpolation="bilinear")
buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

ms = 14
if not gdf_bun.empty:
    gdf_bun.plot(ax=ax, facecolor="#e31a1c", edgecolor="black", markersize=ms, linewidth=0.4, label="Landslides")

ax.set_xlim(*xlim); ax.set_ylim(*ylim)
ax.set_axis_off(); ax.set_aspect("equal")

# Frame around map only
ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                       linewidth=1.2, edgecolor="black", facecolor="none", zorder=10))

# Landslide legend (inside map)
hdl = ax.scatter([], [], s=ms, facecolor="#e31a1c", edgecolors="black", linewidths=0.4, label="Landslides")
leg = ax.legend(handles=[hdl], loc="lower left", frameon=True, fontsize=9)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.95)

# Colorbar OUTSIDE the map (right), balanced size
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3.8%", pad=0.15)
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=8, length=3, width=0.6)
cb.set_label("Elevation (m)", fontsize=9)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

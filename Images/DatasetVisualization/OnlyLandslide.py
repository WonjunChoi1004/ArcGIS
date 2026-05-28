#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# Paths
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
LANDSLIDE_CSV = f"{PROJECT_ROOT}/LandslideData/North_Carolina_Landslide_Points.csv"
OUT_PATH = f"{PROJECT_ROOT}/Images/DatasetVisualization/panel_b.png"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# Counties → Buncombe
counties = gpd.read_file(COUNTY_SHP)
buncombe = counties[(counties["STATEFP"] == "37") & (counties["NAME"] == "Buncombe")]

# Landslides (source CRS is EPSG:32119 per your current script)
df = pd.read_csv(LANDSLIDE_CSV, encoding="utf-8-sig")
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["X"], df["Y"])],
    crs="EPSG:32119"
)

# Make sure both layers match CRS (use county CRS to drive the plot CRS)
buncombe = buncombe.to_crs(gdf.crs)

# Border-inclusive filter
gdf_bun = gdf[gdf.intersects(buncombe.unary_union)]

# Map extent: fit Buncombe (small symmetric pad)
xmin, ymin, xmax, ymax = buncombe.total_bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)

# Plot
fig, ax = plt.subplots(figsize=(7.2, 6.3))

buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

ms = 14
if not gdf_bun.empty:
    gdf_bun.plot(
        ax=ax,
        facecolor="#e31a1c",
        edgecolor="black",
        markersize=ms,
        linewidth=0.4,
        label="Landslides"
    )

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_axis_off()
ax.set_aspect("equal")

# Frame around map only
ax.add_patch(
    Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        linewidth=1.2,
        edgecolor="black",
        facecolor="none",
        zorder=10
    )
)

# Landslide legend (inside map)
hdl = ax.scatter([], [], s=ms, facecolor="#e31a1c", edgecolors="black", linewidths=0.4, label="Landslides")
leg = ax.legend(handles=[hdl], loc="lower left", frameon=True, fontsize=9)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.8)
leg.get_frame().set_alpha(0.95)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

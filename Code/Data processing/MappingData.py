#!/usr/bin/env python
"""
Plot SSURGO rasters for Buncombe Co. – one map per soil property
-----------------------------------------------------------------
Assumes you have already un‑zipped the Web Soil Survey download

    Buncombe_SSURGO/
      └── NC021/            # survey folder
          ├── spatial/      # GeoTIFFs live here
          └── tabular/

If your survey folder has a different ID, just change SURVEY = "NC021"
"""

from pathlib import Path
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
SURVEY   = "NC021"                     # <-- Change if yours differs
ROOT     = Path("../..")                   # run script from project root
SPATIAL  = ROOT / SURVEY / "spatial"   # where .tif live
COUNTIES = ROOT / "MaskingData" / "cb_2022_us_county_500k.shp"
BUNCOMBE = "Buncombe"
# ------------------------------------------------------------------

# **1.  Find Buncombe polygon and get its bbox in raster CRS ----------
counties = gpd.read_file(COUNTIES)
buncombe = counties.query('NAME == @BUNCOMBE').to_crs("EPSG:4269")  # NAD83

# **2.  Dictionary of short‑name ↔ (human label, cmap) ---------------
TARGETS = {
    "DEPTHR" : ("Depth to Restrictive Layer (cm)",       "YlGn"),
    "KSAT"   : ("Saturated Hydraulic Conductivity (µm/s)","PuBuGn"),
    "CLAY"   : ("% Clay (%)",                             "OrRd"),
    "SAND"   : ("% Sand (%)",                             "YlOrBr"),
    "SILT"   : ("% Silt (%)",                             "PuRd"),
    "BD"     : ("Bulk Density, 1/3 bar (g cm⁻³)",          "Greens"),
    "AWS"    : ("Available Water Storage (cm/100 cm)",    "Blues"),
    "HYDGRP" : ("Hydrologic Soil Group",                  "Set3")
}

# **3.  Build filename → meta dictionary ------------------------------
layer_files = {}
for tif in SPATIAL.glob("*.tif"):
    upper = tif.name.upper()
    for short, meta in TARGETS.items():
        if short in upper:
            layer_files[short] = (tif, *meta)

print("Discovered layers:\n", "\n".join(f" • {k}: {v[0].name}" for k,v in layer_files.items()))
missing = [s for s in TARGETS if s not in layer_files]
if missing:
    print("\nWARNING – these layers not found:", ", ".join(missing))

# **4.  Iterate and plot each raster clipped to Buncombe --------------
for short, (tif, title, cmap) in layer_files.items():
    with rasterio.open(tif) as src:
        # Re‑project polygon to raster CRS just in case
        poly = buncombe.to_crs(src.crs)
        out, xform = rasterio.mask.mask(src, poly.geometry, crop=True)
        data = out[0]
        # nodata handling
        data = data.astype(float)
        data[data == src.nodata] = float("nan")

    plt.figure(figsize=(6,6))
    show(data, transform=xform, cmap=cmap)
    plt.title(title); plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"maps/{short.lower()}_buncombe.png", dpi=200)
    plt.close()
    print(f"✓ saved maps/{short.lower()}_buncombe.png")

print("\nDone.")

# prism_sep_2024_buncombe_top10pct_shaded.py

import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import rasterio
from rasterio.mask import mask

try:
    import contextily as cx
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False


# -----------------------
# Settings
# -----------------------
OUT_DIR = Path("prism_sep_2024_buncombe_800m")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRISM_REGION = "us"
PRISM_RES = "800m"
PRISM_ELEMENT = "ppt"

START_DATE = date(2024, 9, 1)
END_DATE = date(2024, 9, 30)

COUNTY_NAME = "Buncombe"

# Regime definition: top 10% of daily rainfall (within Sep 2024)
TOP_PCT = 0.10  # 10%

SLEEP_SECONDS_BETWEEN_REQUESTS = 1.5

OUT_CSV = OUT_DIR / "buncombe_prism_daily_ppt_sep_2024.csv"
OUT_PNG = OUT_DIR / "buncombe_prism_daily_ppt_sep_2024_top10pct_shaded.png"
OUT_MAP_PNG = OUT_DIR / "buncombe_boundary_quicklook.png"


def prism_grid_url(d: date) -> str:
    ymd = d.strftime("%Y%m%d")
    return f"https://services.nacse.org/prism/data/get/{PRISM_REGION}/{PRISM_RES}/{PRISM_ELEMENT}/{ymd}"


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def load_buncombe_boundary() -> gpd.GeoDataFrame:
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
    counties = gpd.read_file(tiger_url)
    nc = counties[counties["STATEFP"] == "37"].copy()
    bun = nc[nc["NAME"] == COUNTY_NAME].copy()
    if bun.empty:
        raise RuntimeError(f"{COUNTY_NAME} not found in TIGER counties.")
    return bun.to_crs("EPSG:4326")


def download_prism_zip(d: date, out_zip: Path) -> None:
    url = prism_grid_url(d)
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    out_zip.write_bytes(r.content)


def extract_tif_from_zip(zip_path: Path, out_tif: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        if not tif_names:
            raise RuntimeError(f"No .tif found in {zip_path.name}")
        with z.open(tif_names[0]) as f:
            out_tif.write_bytes(f.read())


def county_mean_from_tif(tif_path: Path, boundary_gdf: gpd.GeoDataFrame) -> float:
    with rasterio.open(tif_path) as src:
        boundary_proj = boundary_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in boundary_proj.geometry]
        data, _ = mask(src, geoms, crop=True)
        arr = data[0].astype("float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        return float(np.nanmean(arr))


def quicklook_boundary(boundary: gpd.GeoDataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0), dpi=200)
    b3857 = boundary.to_crs(3857)
    b3857.boundary.plot(ax=ax, linewidth=2.2)
    if HAS_BASEMAP:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=b3857.crs.to_string())
    ax.set_axis_off()
    ax.set_title(f"{COUNTY_NAME} boundary")
    plt.tight_layout()
    plt.savefig(OUT_MAP_PNG, bbox_inches="tight")
    plt.close(fig)


def main():
    boundary = load_buncombe_boundary()
    quicklook_boundary(boundary)

    rows = []
    for d in daterange(START_DATE, END_DATE):
        ymd = d.strftime("%Y%m%d")
        zip_path = OUT_DIR / f"prism_{PRISM_ELEMENT}_{PRISM_REGION}_{PRISM_RES}_{ymd}.zip"
        tif_path = OUT_DIR / f"prism_{PRISM_ELEMENT}_{PRISM_REGION}_{PRISM_RES}_{ymd}.tif"

        if not tif_path.exists():
            if not zip_path.exists():
                download_prism_zip(d, zip_path)
                time.sleep(SLEEP_SECONDS_BETWEEN_REQUESTS)
            extract_tif_from_zip(zip_path, tif_path)

        mean_mm = county_mean_from_tif(tif_path, boundary)
        rows.append({"date": d.isoformat(), "ppt_mm": mean_mm})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    # Regime = top 10% rainfall days within Sep 2024
    threshold = float(np.nanquantile(df["ppt_mm"].values, 1.0 - TOP_PCT))
    df["regime_top10pct"] = (df["ppt_mm"] >= threshold).astype(int)

    # Plot + shade regime days
    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=250)
    ax.plot(df["date"], df["ppt_mm"], linewidth=1.2)

    # Shade each regime day as a 1-day span
    for d in df.loc[df["regime_top10pct"] == 1, "date"]:
        ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.22)

    ax.set_ylabel("Daily precipitation (mm)")
    ax.set_xlabel("")
    ax.set_title(
        f"{COUNTY_NAME} County Mean Daily Rainfall (PRISM {PRISM_RES}) — Sep 2024\n"
        f"Shaded = top 10% rainfall days (≥ {threshold:.2f} mm/day)"
    )
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"90th percentile threshold: {threshold:.4f} mm/day")
    print(f"Regime days shaded: {int(df['regime_top10pct'].sum())} / {len(df)}")


if __name__ == "__main__":
    main()

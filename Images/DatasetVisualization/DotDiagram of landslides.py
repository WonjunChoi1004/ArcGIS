#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# -----------------------
# Paths
# -----------------------
COUNTY_SHP = f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp"
LANDSLIDE_CSV = f"{PROJECT_ROOT}/LandslideData/North_Carolina_Landslide_Points.csv"
OUT_PATH = f"{PROJECT_ROOT}/Images/DatasetVisualization/buncombe_landslides_2023_by_day_histogram.png"
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# -----------------------
# Settings
# -----------------------
TARGET_YEAR = 2013
COUNTY_NAME = "Buncombe"
STATEFP_NC = "37"
POINTS_CRS = "EPSG:32119"


def parse_date_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def main():
    # -----------------------
    # Load Buncombe County
    # -----------------------
    counties = gpd.read_file(COUNTY_SHP)
    buncombe = counties[
        (counties["STATEFP"] == STATEFP_NC) &
        (counties["NAME"] == COUNTY_NAME)
    ].copy()

    if buncombe.empty:
        raise RuntimeError("Buncombe County not found in county shapefile.")

    # -----------------------
    # Load landslide points
    # -----------------------
    df = pd.read_csv(LANDSLIDE_CSV, encoding="utf-8-sig")

    if not {"X", "Y"}.issubset(df.columns):
        raise RuntimeError("CSV must contain X and Y columns.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["X"], df["Y"])],
        crs=POINTS_CRS
    )

    # Match CRS
    buncombe = buncombe.to_crs(gdf.crs)

    # -----------------------
    # Spatial filter: Buncombe only
    # -----------------------
    bun_geom = buncombe.union_all()
    gdf = gdf[gdf.intersects(bun_geom)].copy()

    if gdf.empty:
        raise RuntimeError("No landslide points found inside Buncombe County.")

    # -----------------------
    # Date handling (row-by-row fallback)
    # -----------------------
    mv = parse_date_series(gdf["Mvmnt_Date"]) if "Mvmnt_Date" in gdf.columns else pd.Series(pd.NaT, index=gdf.index)
    cd = parse_date_series(gdf["Col_Date"]) if "Col_Date" in gdf.columns else pd.Series(pd.NaT, index=gdf.index)

    gdf["__date"] = mv.where(mv.notna(), cd)
    gdf = gdf.dropna(subset=["__date"]).copy()

    # -----------------------
    # Filter to target year
    # -----------------------
    gdf["Year"] = gdf["__date"].dt.year
    gdf_yr = gdf[gdf["Year"] == TARGET_YEAR].copy()

    if gdf_yr.empty:
        raise RuntimeError(f"No Buncombe landslides found for year {TARGET_YEAR}.")

    # Day of year (1–365)
    gdf_yr["DayOfYear"] = gdf_yr["__date"].dt.dayofyear

    # -----------------------
    # Histogram: burst behavior
    # -----------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    ax.hist(
        gdf_yr["DayOfYear"],
        bins=range(1, 367),
        edgecolor="black"
    )

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Number of landslides")
    ax.set_title(f"Landslides in {TARGET_YEAR} — Buncombe County")

    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.show()

    print(f"Saved: {OUT_PATH}")
    print(f"Landslides in {TARGET_YEAR}: {len(gdf_yr)}")
    print(f"Date range: {gdf_yr['__date'].min().date()} → {gdf_yr['__date'].max().date()}")


if __name__ == "__main__":
    main()

# map_nonlandslides_buncombe_boundary.py

from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import contextily as cx
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False


CSV_PATH = Path(f"{PROJECT_ROOT}/LandslideData/FinalData/All_Combined_Balanced_EqualCounts.csv")
COUNTY_SHP_PATH = Path(f"{PROJECT_ROOT}/MaskingData/cb_2022_us_county_500k.shp")

OUT_PNG = Path("nonlandslides_buncombe_boundary.png")

COUNTY_NAME = "Buncombe"
STATEFP_NC = "37"

INPUT_CRS = "EPSG:32119"  # your X,Y CRS

USE_BASEMAP = True        # requires contextily
POINT_SIZE = 10
POINT_ALPHA = 0.75
BOUNDARY_LW = 2.8
PAD_METERS = 2000         # padding around county extent (meters)


def load_buncombe_boundary(target_crs: str) -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTY_SHP_PATH)
    buncombe = counties[(counties["STATEFP"] == STATEFP_NC) & (counties["NAME"] == COUNTY_NAME)].copy()
    if buncombe.empty:
        raise RuntimeError("Buncombe County not found in county shapefile.")
    buncombe = buncombe.to_crs(target_crs)
    return buncombe


def main():
    df = pd.read_csv(CSV_PATH)

    df_nl = df[df["IsLandslide"] == 0].copy()
    if df_nl.empty:
        raise RuntimeError("No non-landslide rows found (IsLandslide == 0).")

    gdf = gpd.GeoDataFrame(
        df_nl,
        geometry=gpd.points_from_xy(df_nl["X"], df_nl["Y"]),
        crs=INPUT_CRS,
    )

    if USE_BASEMAP and HAS_BASEMAP:
        gdf_3857 = gdf.to_crs(3857)
        buncombe_3857 = load_buncombe_boundary("EPSG:3857")

        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=250)

        buncombe_3857.boundary.plot(ax=ax, edgecolor="black", linewidth=BOUNDARY_LW)
        gdf_3857.plot(ax=ax, markersize=POINT_SIZE, alpha=POINT_ALPHA)

        minx, miny, maxx, maxy = buncombe_3857.total_bounds
        ax.set_xlim(minx - PAD_METERS, maxx + PAD_METERS)
        ax.set_ylim(miny - PAD_METERS, maxy + PAD_METERS)

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857")

        ax.set_title(f"Non-landslide samples (IsLandslide = 0) — {COUNTY_NAME} County")
        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(OUT_PNG, bbox_inches="tight")
        plt.close(fig)

    else:
        buncombe = load_buncombe_boundary(gdf.crs)

        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=250)

        buncombe.boundary.plot(ax=ax, edgecolor="black", linewidth=BOUNDARY_LW)
        gdf.plot(ax=ax, markersize=POINT_SIZE, alpha=POINT_ALPHA)

        minx, miny, maxx, maxy = buncombe.total_bounds
        ax.set_xlim(minx - PAD_METERS, maxx + PAD_METERS)
        ax.set_ylim(miny - PAD_METERS, maxy + PAD_METERS)

        ax.set_title(f"Non-landslide samples (IsLandslide = 0) — {COUNTY_NAME} County")
        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(OUT_PNG, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Non-landslide points plotted: {len(gdf)}")
    if USE_BASEMAP and not HAS_BASEMAP:
        print("Note: contextily not installed. Install with: pip install contextily")


if __name__ == "__main__":
    main()

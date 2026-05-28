# make_helene_spatial_panels.py

import pathlib
import requests
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import contextily as cx
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False


SCIENCEBASE_ITEM_ID = "674634a1d34e6d1dac3abddc"  # Helene inventory item

# Outputs (two separate images)
OUT_PNG_PANEL_A = "helene_landslides_panelA_nc_statewide.png"
OUT_PNG_PANEL_B = "helene_landslides_panelB_buncombe_zoom.png"

# Optional: set to your boundary file (shp/geojson/gpkg). Leave None to skip.
STUDY_BOUNDARY_PATH = None  # e.g., "data/buncombe_study_boundary.geojson"

# Counties to label/outline in Panel A
HIGHLIGHT_COUNTIES = {"Buncombe", "Henderson", "Rutherford", "Yancey"}


def sciencebase_file_url(item_id: str, filename: str) -> str:
    api = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
    data = requests.get(api, timeout=60).json()
    for f in data.get("files", []):
        if f.get("name") == filename:
            return f.get("downloadUri")
    raise RuntimeError(f"Could not find {filename} in ScienceBase item {item_id}")


def load_helene_inventory() -> gpd.GeoDataFrame:
    url = sciencebase_file_url(SCIENCEBASE_ITEM_ID, "HurricaneHelene_LS_Inventory.geojson")
    gdf = gpd.read_file(url)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def load_nc_counties() -> gpd.GeoDataFrame:
    tiger_url = "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
    counties = gpd.read_file(tiger_url)
    nc = counties[counties["STATEFP"] == "37"].copy()
    nc = nc.to_crs("EPSG:4326")
    return nc


def maybe_add_basemap(ax, gdf_3857):
    if not HAS_BASEMAP:
        return
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf_3857.crs.to_string())


def save_panel_a_statewide(nc_3857: gpd.GeoDataFrame, pts_3857: gpd.GeoDataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8), dpi=250)

    # NC statewide frame
    minx, miny, maxx, maxy = nc_3857.total_bounds

    nc_3857.boundary.plot(ax=ax, linewidth=0.5)
    pts_3857.plot(ax=ax, markersize=1.8, alpha=0.65)

    highlight = nc_3857[nc_3857["NAME"].isin(HIGHLIGHT_COUNTIES)]
    if not highlight.empty:
        highlight.boundary.plot(ax=ax, linewidth=2.0)
        for _, row in highlight.iterrows():
            c = row.geometry.representative_point()
            ax.text(c.x, c.y, row["NAME"], fontsize=8, ha="center", va="center")

    maybe_add_basemap(ax, nc_3857)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title("Helene Landslide Observations — North Carolina (Statewide)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(OUT_PNG_PANEL_A, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG_PANEL_A}")


def save_panel_b_buncombe(nc_3857: gpd.GeoDataFrame, pts_3857: gpd.GeoDataFrame, buncombe_3857: gpd.GeoDataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(9, 8), dpi=250)

    buff_m = 30000  # ~30 km
    bun_geom = buncombe_3857.geometry.iloc[0].buffer(buff_m)
    bxmin, bymin, bxmax, bymax = bun_geom.bounds

    bun_nc = nc_3857.cx[bxmin:bxmax, bymin:bymax]
    bun_pts = pts_3857.cx[bxmin:bxmax, bymin:bymax]

    bun_nc.boundary.plot(ax=ax, linewidth=0.6)
    buncombe_3857.boundary.plot(ax=ax, linewidth=2.3)
    bun_pts.plot(ax=ax, markersize=7, alpha=0.85)

    if STUDY_BOUNDARY_PATH:
        p = pathlib.Path(STUDY_BOUNDARY_PATH)
        if not p.exists():
            raise FileNotFoundError(f"STUDY_BOUNDARY_PATH not found: {p}")
        study_boundary = gpd.read_file(p).to_crs(3857)
        study_boundary.boundary.plot(ax=ax, linewidth=2.0, linestyle="--")

    maybe_add_basemap(ax, bun_nc)
    ax.set_xlim(bxmin, bxmax)
    ax.set_ylim(bymin, bymax)
    ax.set_title("Helene Landslide Observations — Buncombe Vicinity (Zoom)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(OUT_PNG_PANEL_B, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG_PANEL_B}")


def main():
    pts = load_helene_inventory()
    nc = load_nc_counties()

    pts_3857 = pts.to_crs(3857)
    nc_3857 = nc.to_crs(3857)

    buncombe_3857 = nc[nc["NAME"] == "Buncombe"].to_crs(3857)
    if buncombe_3857.empty:
        raise RuntimeError("Buncombe not found in NC counties layer.")

    save_panel_a_statewide(nc_3857, pts_3857)
    save_panel_b_buncombe(nc_3857, pts_3857, buncombe_3857)

    if not HAS_BASEMAP:
        print("Note: Install contextily for basemaps: pip install contextily")


if __name__ == "__main__":
    main()

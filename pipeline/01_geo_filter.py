"""
Step 1: Geographic pre-filter
------------------------------
Downloads Census TIGER/Line ZCTAs, joins with ACS population data,
and filters to two latitude bands with a minimum population density.

Outputs
-------
data/processed/north_candidates_geo.geojson
data/processed/south_candidates_geo.geojson
"""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    NORTH_LAT_MIN, NORTH_LAT_MAX,
    SOUTH_LAT_MIN, SOUTH_LAT_MAX,
    MIN_POP_DENSITY,
)

# Census ACS 5-year table B01003 (total population) for ZCTAs
# We use the Census Data API — no key required for basic queries
ZCTA_SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
)
# ACS 2022 5-year estimates, total population by ZCTA
CENSUS_ACS_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B01003_001E,NAME"
    "&for=zip%20code%20tabulation%20area:*"
)


def download_zcta_shapefile() -> Path:
    """Download and unzip the TIGER ZCTA shapefile if not already cached."""
    dest = DATA_RAW / "zcta_2023"
    shp = dest / "tl_2023_us_zcta520.shp"
    if shp.exists():
        print("ZCTA shapefile already cached.")
        return shp

    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading ZCTA shapefile (~100 MB)…")
    resp = requests.get(ZCTA_SHAPEFILE_URL, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="ZCTA shp") as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buf.write(chunk)
            bar.update(len(chunk))

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest)

    print(f"Extracted to {dest}")
    return shp


def fetch_acs_population() -> pd.DataFrame:
    """Fetch total population for every ZCTA from the Census ACS API."""
    cache = DATA_RAW / "zcta_population.parquet"
    if cache.exists():
        print("ACS population data already cached.")
        return pd.read_parquet(cache)

    print("Fetching ACS population data from Census API…")
    resp = requests.get(CENSUS_ACS_URL, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    cols = data[0]      # first row is headers
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)
    df = df.rename(columns={
        "B01003_001E": "population",
        "zip code tabulation area": "ZCTA5CE20",
    })
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df[["ZCTA5CE20", "population"]]

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached {len(df):,} ZCTAs to {cache}")
    return df


def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Load shapefile
    shp_path = download_zcta_shapefile()
    print("Loading ZCTA shapefile…")
    gdf = gpd.read_file(shp_path)
    print(f"  {len(gdf):,} ZCTAs loaded")

    # 2. Compute centroids in WGS84
    gdf = gdf.to_crs("EPSG:4326")
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y

    # 3. Compute area in sq miles using an equal-area projection
    gdf_aea = gdf.to_crs("EPSG:5070")  # Albers Equal Area (continental US)
    gdf["area_sqmi"] = gdf_aea.geometry.area / 2_589_988  # m² → sq miles

    # 4. Join population data
    pop_df = fetch_acs_population()
    gdf = gdf.merge(pop_df, on="ZCTA5CE20", how="left")
    gdf["pop_density"] = gdf["population"] / gdf["area_sqmi"].replace(0, float("nan"))

    # 5. Filter by latitude bands
    north_mask = (
        (gdf["centroid_lat"] >= NORTH_LAT_MIN) &
        (gdf["centroid_lat"] <= NORTH_LAT_MAX) &
        (gdf["pop_density"] >= MIN_POP_DENSITY)
    )
    south_mask = (
        (gdf["centroid_lat"] >= SOUTH_LAT_MIN) &
        (gdf["centroid_lat"] <= SOUTH_LAT_MAX) &
        (gdf["pop_density"] >= MIN_POP_DENSITY)
    )

    north = gdf[north_mask].copy()
    south = gdf[south_mask].copy()

    print(f"\nNorth ({NORTH_LAT_MIN}°–{NORTH_LAT_MAX}°N, ≥{MIN_POP_DENSITY} ppl/mi²): "
          f"{len(north):,} ZCTAs")
    print(f"South ({SOUTH_LAT_MIN}°–{SOUTH_LAT_MAX}°N, ≥{MIN_POP_DENSITY} ppl/mi²): "
          f"{len(south):,} ZCTAs")

    # 6. Save
    north_out = DATA_PROCESSED / "north_candidates_geo.geojson"
    south_out = DATA_PROCESSED / "south_candidates_geo.geojson"
    north.to_file(north_out, driver="GeoJSON")
    south.to_file(south_out, driver="GeoJSON")
    print(f"\nSaved:\n  {north_out}\n  {south_out}")


if __name__ == "__main__":
    run()

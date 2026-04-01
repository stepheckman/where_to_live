"""
Step 1: Geographic pre-filter (Census Block Groups)
----------------------------------------------------
Downloads Census TIGER/Line block group shapefiles, joins with ACS
population data, and filters to two latitude bands with a minimum
population density.

Uses TIGER INTPTLAT/INTPTLON (population-weighted internal points)
rather than geometric centroids.

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
import logging

logger = logging.getLogger(__name__)
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    NORTH_LAT_MIN, NORTH_LAT_MAX,
    SOUTH_LAT_MIN, SOUTH_LAT_MAX,
    MIN_POP_DENSITY,
)

# TIGER/Line block group shapefiles — one per state
BG_SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_{fips}_bg.zip"
)

# ACS 2022 5-year estimates, total population by block group (per state)
ACS_POP_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B01003_001E"
    "&for=block%20group:*&in=state:{fips}&in=county:*&in=tract:*"
)

# State FIPS codes that could have block groups in our latitude bands.
# North band 38–49°N covers roughly: CT, DC, DE, IA, ID, IL, IN, MA, MD, ME,
# MI, MN, MT, ND, NE, NH, NJ, NY, OH, OR, PA, RI, SD, VA, VT, WA, WI, WV, WY
# South band 25–35°N covers roughly: AL, AR, AZ, FL, GA, LA, MS, NM, NC, OK,
# SC, TN, TX
# We include all continental US states to avoid missing edge cases.
CONTINENTAL_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# Rough latitude ranges per state to skip states entirely outside our bands.
# Only download shapefiles for states that overlap with north (38–49) or south (25–35).
_STATE_LAT_RANGES = {
    "01": (30.2, 35.0), "04": (31.3, 37.0), "05": (33.0, 36.5),
    "06": (32.5, 42.0), "08": (37.0, 41.0), "09": (41.0, 42.1),
    "10": (38.5, 39.8), "11": (38.8, 39.0), "12": (24.5, 31.0),
    "13": (30.4, 35.0), "16": (42.0, 49.0), "17": (37.0, 42.5),
    "18": (37.8, 41.8), "19": (40.4, 43.5), "20": (37.0, 40.0),
    "21": (36.5, 39.1), "22": (29.0, 33.0), "23": (43.1, 47.5),
    "24": (38.0, 39.7), "25": (41.2, 42.9), "26": (41.7, 48.3),
    "27": (43.5, 49.4), "28": (30.2, 35.0), "29": (36.0, 40.6),
    "30": (44.4, 49.0), "31": (40.0, 43.0), "32": (35.0, 42.0),
    "33": (42.7, 45.3), "34": (38.9, 41.4), "35": (31.3, 37.0),
    "36": (40.5, 45.0), "37": (33.8, 36.6), "38": (45.9, 49.0),
    "39": (38.4, 42.0), "40": (33.6, 37.0), "41": (42.0, 46.3),
    "42": (39.7, 42.3), "44": (41.1, 42.0), "45": (32.0, 35.2),
    "46": (42.5, 46.0), "47": (35.0, 36.7), "48": (25.8, 36.5),
    "49": (37.0, 42.0), "50": (42.7, 45.0), "51": (36.5, 39.5),
    "53": (45.5, 49.0), "54": (37.2, 40.6), "55": (42.5, 47.1),
    "56": (41.0, 45.0),
}


def _state_overlaps_band(fips: str, lat_min: float, lat_max: float) -> bool:
    """Check if a state's latitude range overlaps a target band."""
    if fips not in _STATE_LAT_RANGES:
        return True  # include unknown states to be safe
    s_min, s_max = _STATE_LAT_RANGES[fips]
    return s_min <= lat_max and s_max >= lat_min


def _relevant_state_fips() -> list[str]:
    """Return state FIPS codes that overlap either lat band."""
    return [
        f for f in CONTINENTAL_FIPS
        if _state_overlaps_band(f, NORTH_LAT_MIN, NORTH_LAT_MAX)
        or _state_overlaps_band(f, SOUTH_LAT_MIN, SOUTH_LAT_MAX)
    ]


def download_bg_shapefiles(state_fips_list: list[str]) -> gpd.GeoDataFrame:
    """Download and merge TIGER/Line block group shapefiles for given states."""
    cache = DATA_RAW / "bg_2023.parquet"
    if cache.exists():
        logger.debug("Block group shapefile cache found.")
        return gpd.read_parquet(cache)

    dest = DATA_RAW / "bg_2023"
    dest.mkdir(parents=True, exist_ok=True)

    frames = []
    for fips in tqdm(state_fips_list, desc="Downloading BG shapefiles"):
        state_dir = dest / fips
        # Look for the .shp file
        shp_candidates = list(state_dir.glob("*.shp")) if state_dir.exists() else []
        if shp_candidates:
            gdf = gpd.read_file(shp_candidates[0])
        else:
            url = BG_SHAPEFILE_URL.format(fips=fips)
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Failed to download BG shapefile for state {fips}: {e}")
                continue

            state_dir.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(state_dir)
            shp_candidates = list(state_dir.glob("*.shp"))
            if not shp_candidates:
                logger.warning(f"No .shp found for state {fips}")
                continue
            gdf = gpd.read_file(shp_candidates[0])

        frames.append(gdf)

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    combined = combined.to_crs("EPSG:4326")

    # Keep useful columns
    keep = ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE",
            "ALAND", "AWATER", "INTPTLAT", "INTPTLON", "geometry"]
    combined = combined[[c for c in keep if c in combined.columns]]

    combined.to_parquet(cache)
    logger.info(f"Cached {len(combined):,} block groups to {cache}")
    return combined


def fetch_acs_population(state_fips_list: list[str]) -> pd.DataFrame:
    """Fetch total population for block groups from the Census ACS API."""
    cache = DATA_RAW / "bg_population.parquet"
    if cache.exists():
        logger.debug("ACS BG population data already cached.")
        return pd.read_parquet(cache)

    logger.info("Fetching ACS population data for block groups…")
    frames = []
    for fips in tqdm(state_fips_list, desc="ACS population"):
        url = ACS_POP_URL.format(fips=fips)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"ACS population fetch failed for state {fips}: {e}")
            continue

        cols = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)

    pop = pd.concat(frames, ignore_index=True)
    # Build GEOID: state(2) + county(3) + tract(6) + block group(1)
    pop["GEOID"] = (
        pop["state"] + pop["county"] + pop["tract"] + pop["block group"]
    )
    pop["population"] = pd.to_numeric(pop["B01003_001E"], errors="coerce")
    pop = pop[["GEOID", "population"]]

    cache.parent.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(cache, index=False)
    logger.debug(f"Cached population for {len(pop):,} block groups")
    return pop


def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. Determine which states to download
    states = _relevant_state_fips()
    logger.info(f"Downloading block groups for {len(states)} states")

    # 2. Load block group shapefiles
    gdf = download_bg_shapefiles(states)
    logger.info(f"{len(gdf):,} block groups loaded")

    # 3. Use TIGER internal points (population-weighted centroids)
    gdf["centroid_lat"] = pd.to_numeric(gdf["INTPTLAT"], errors="coerce")
    gdf["centroid_lon"] = pd.to_numeric(gdf["INTPTLON"], errors="coerce")

    # 4. Compute area in sq miles using an equal-area projection
    gdf_aea = gdf.to_crs("EPSG:5070")  # Albers Equal Area
    gdf["area_sqmi"] = gdf_aea.geometry.area / 2_589_988  # m² → sq miles

    # 5. Join population data
    pop_df = fetch_acs_population(states)
    gdf = gdf.merge(pop_df, on="GEOID", how="left")
    gdf["pop_density"] = gdf["population"] / gdf["area_sqmi"].replace(0, float("nan"))

    # 6. Filter by latitude bands
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

    logger.info(
        f"North ({NORTH_LAT_MIN}°–{NORTH_LAT_MAX}°N, "
        f"≥{MIN_POP_DENSITY} ppl/mi²): {len(north):,} block groups"
    )
    logger.info(
        f"South ({SOUTH_LAT_MIN}°–{SOUTH_LAT_MAX}°N, "
        f"≥{MIN_POP_DENSITY} ppl/mi²): {len(south):,} block groups"
    )

    # 7. Save
    north_out = DATA_PROCESSED / "north_candidates_geo.geojson"
    south_out = DATA_PROCESSED / "south_candidates_geo.geojson"
    north.to_file(north_out, driver="GeoJSON")
    south.to_file(south_out, driver="GeoJSON")
    logger.info(f"Saved: {north_out}  |  {south_out}")


if __name__ == "__main__":
    run()

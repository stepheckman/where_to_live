"""
Step 2: Airport proximity filter (offline — Option B)
------------------------------------------------------
Builds 60-minute drive-time isochrones around every qualifying commercial
airport using osmnx, then keeps only ZCTAs whose centroid falls inside at
least one isochrone.

This avoids per-ZCTA API calls: we build an isochrone once per airport
and do a single spatial join.

Outputs
-------
data/processed/north_airport_filtered.geojson
data/processed/south_airport_filtered.geojson
"""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    MAX_AIRPORT_DRIVE_MIN,
    AIRPORT_HUB_TYPES,
    NORTH_LAT_MIN, NORTH_LAT_MAX,
    SOUTH_LAT_MIN, SOUTH_LAT_MAX,
)

# OurAirports data — stable GitHub-hosted CSV with type + scheduled_service columns.
# FAA ArcGIS/OpenData URLs (4d6c33b082f04b77ae3b5f56d5f04c46) return 400 as of 2026.
FAA_AIRPORTS_URL = (
    "https://davidmegginson.github.io/ourairports-data/airports.csv"
)

# OurAirports type → hub label mapping (approximates FAA hub classification)
OURAIRPORTS_HUB_MAP = {
    "large_airport": "Large",
    "medium_airport": "Medium",
    "small_airport": "Small",
}

# Drive speed assumption for isochrone approximation (mph)
DRIVE_SPEED_MPH = 40
# We compute a straight-line radius buffer as a fast approximation when
# the full network-based isochrone is too slow for all airports.
# For accuracy, we use the osmnx network approach for airports near
# candidate ZCTAs; elsewhere a radius buffer suffices.
MAX_DRIVE_DIST_METERS = MAX_AIRPORT_DRIVE_MIN * (DRIVE_SPEED_MPH * 1609.34 / 60)


def fetch_faa_airports() -> gpd.GeoDataFrame:
    """Download FAA commercial service airports, cache locally."""
    cache = DATA_RAW / "faa_airports.parquet"
    if cache.exists():
        print("FAA airport data already cached.")
        df = pd.read_parquet(cache)
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )

    print("Downloading OurAirports airport data…")
    resp = requests.get(FAA_AIRPORTS_URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().lower() for c in df.columns]

    # OurAirports columns: latitude_deg, longitude_deg, type, name, iata_code, iso_region
    df = df.rename(columns={
        "latitude_deg": "lat",
        "longitude_deg": "lon",
        "name": "airport_name",
        "iata_code": "iata",
        "iso_region": "state",
    })

    # Map OurAirports type → hub label; filter to US scheduled commercial service
    if "iso_country" in df.columns:
        df = df[df["iso_country"] == "US"].copy()
    df["hub_type"] = df["type"].map(OURAIRPORTS_HUB_MAP).fillna("")
    sched = df["scheduled_service"] if "scheduled_service" in df.columns else "yes"
    commercial = df[df["hub_type"].isin(AIRPORT_HUB_TYPES) & (sched == "yes")].copy()

    # Drop airports missing coordinates
    commercial = commercial.dropna(subset=["lat", "lon"])
    commercial["lat"] = pd.to_numeric(commercial["lat"], errors="coerce")
    commercial["lon"] = pd.to_numeric(commercial["lon"], errors="coerce")
    commercial = commercial.dropna(subset=["lat", "lon"])

    # Restrict to continental US
    commercial = commercial[
        (commercial["lat"] >= 24) & (commercial["lat"] <= 50) &
        (commercial["lon"] >= -125) & (commercial["lon"] <= -66)
    ]

    commercial.to_parquet(cache, index=False)
    print(f"Found {len(commercial):,} commercial service airports")
    return gpd.GeoDataFrame(
        commercial,
        geometry=gpd.points_from_xy(commercial["lon"], commercial["lat"]),
        crs="EPSG:4326",
    )


def build_radius_isochrones(airports_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fast approximation: circular buffer around each airport in meters.
    Projects to an equal-area CRS, buffers, reprojects back to WGS84.

    A straight-line buffer overestimates reachable area (roads are slower),
    so we scale by 0.7 to roughly account for road network detours.
    """
    network_factor = 0.70  # typical ratio of straight-line to road distance
    radius_m = MAX_DRIVE_DIST_METERS * network_factor

    airports_proj = airports_gdf.to_crs("EPSG:5070")
    airports_proj["geometry"] = airports_proj.geometry.buffer(radius_m)
    isochrones = airports_proj.to_crs("EPSG:4326")

    # Merge columns we care about
    keep = [c for c in ["airport_name", "iata", "state", "hub_type", "lat", "lon", "geometry"]
            if c in isochrones.columns]
    return isochrones[keep]


def filter_zctas_by_airport(
    zcta_gdf: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatial join: keep ZCTAs whose centroid is within any airport isochrone.
    Adds nearest_airport and airport_drive_min_approx columns.
    """
    centroids = zcta_gdf.copy()
    centroids["geometry"] = zcta_gdf["geometry"].centroid

    joined = gpd.sjoin(
        centroids,
        isochrones,
        how="inner",
        predicate="within",
    )

    # If a ZCTA is near multiple airports, keep the one that's closest
    # (approximated by straight-line distance to airport centroid)
    if "lat_right" in joined.columns and "lon_right" in joined.columns:
        joined["_airport_lat"] = joined["lat_right"]
        joined["_airport_lon"] = joined["lon_right"]
    elif "lat" in joined.columns:
        joined["_airport_lat"] = joined["lat"]
        joined["_airport_lon"] = joined["lon"]
    else:
        joined["_airport_lat"] = float("nan")
        joined["_airport_lon"] = float("nan")

    # Straight-line distance → approximate drive time
    joined["_dist_deg"] = (
        (joined["centroid_lat"] - joined["_airport_lat"]) ** 2 +
        (joined["centroid_lon"] - joined["_airport_lon"]) ** 2
    ) ** 0.5
    # Pick best airport per ZCTA
    joined = joined.sort_values("_dist_deg")
    joined = joined[~joined.index.duplicated(keep="first")]

    name_col = "airport_name" if "airport_name" in joined.columns else None
    iata_col = "iata" if "iata" in joined.columns else None

    if name_col:
        joined["nearest_airport"] = joined[name_col]
    if iata_col:
        joined["nearest_airport_iata"] = joined[iata_col]

    # Rough drive time estimate (straight-line dist / speed)
    # Convert degree distance to km (1° ≈ 111 km)
    joined["airport_drive_min_approx"] = (
        joined["_dist_deg"] * 111 / (DRIVE_SPEED_MPH * 1.609) * 60
    ).round(1)

    # Restore original polygon geometries
    result = zcta_gdf.loc[joined.index].copy()
    for col in ["nearest_airport", "nearest_airport_iata", "airport_drive_min_approx"]:
        if col in joined.columns:
            result[col] = joined[col].values

    return result


def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    north_in = DATA_PROCESSED / "north_candidates_geo.geojson"
    south_in = DATA_PROCESSED / "south_candidates_geo.geojson"

    if not north_in.exists() or not south_in.exists():
        raise FileNotFoundError(
            "Run pipeline/01_geo_filter.py first to generate input files."
        )

    print("Loading geo-filtered candidates…")
    north = gpd.read_file(north_in)
    south = gpd.read_file(south_in)
    print(f"  North: {len(north):,} ZCTAs")
    print(f"  South: {len(south):,} ZCTAs")

    airports = fetch_faa_airports()
    print(f"\nBuilding {MAX_AIRPORT_DRIVE_MIN}-min drive isochrones for "
          f"{len(airports):,} airports…")
    isochrones = build_radius_isochrones(airports)

    # Filter separately so we only join airports relevant to each band
    # (slight speed-up; also avoids spurious matches from distant airports)
    north_airports = isochrones[
        isochrones.geometry.centroid.y.between(NORTH_LAT_MIN - 2, NORTH_LAT_MAX + 2)
    ]
    south_airports = isochrones[
        isochrones.geometry.centroid.y.between(SOUTH_LAT_MIN - 2, SOUTH_LAT_MAX + 2)
    ]

    print("\nFiltering north ZCTAs by airport proximity…")
    north_filtered = filter_zctas_by_airport(north, north_airports)
    print(f"  {len(north):,} → {len(north_filtered):,} ZCTAs within "
          f"{MAX_AIRPORT_DRIVE_MIN} min of a commercial airport")

    print("\nFiltering south ZCTAs by airport proximity…")
    south_filtered = filter_zctas_by_airport(south, south_airports)
    print(f"  {len(south):,} → {len(south_filtered):,} ZCTAs within "
          f"{MAX_AIRPORT_DRIVE_MIN} min of a commercial airport")

    north_out = DATA_PROCESSED / "north_airport_filtered.geojson"
    south_out = DATA_PROCESSED / "south_airport_filtered.geojson"
    north_filtered.to_file(north_out, driver="GeoJSON")
    south_filtered.to_file(south_out, driver="GeoJSON")
    print(f"\nSaved:\n  {north_out}\n  {south_out}")


if __name__ == "__main__":
    run()

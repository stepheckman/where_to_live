"""
Step 3: Walk Score + Bike Score
--------------------------------
Calls the Walk Score API for each candidate ZCTA centroid.
Results are cached to CSV so re-runs don't re-call the API.

If WALKSCORE_API_KEY is not set, this step is skipped and walk_score /
bike_score columns are left as NaN — step 05 will then weight OSM signals
more heavily.

Outputs (appended columns, written to new files)
-------
data/processed/north_walkscored.parquet
data/processed/south_walkscored.parquet
"""

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import os
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, WALKSCORE_SLEEP_S, MIN_WALK_SCORE, MIN_BIKE_SCORE

load_dotenv()
WALKSCORE_API_KEY = os.getenv("WALKSCORE_API_KEY", "")

WALKSCORE_URL = "https://api.walkscore.com/score"


def fetch_walkscore(lat: float, lon: float, address: str = "") -> dict:
    """Call Walk Score API for a single point. Returns dict with walk/bike scores."""
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "address": address or f"{lat},{lon}",
        "transit": 1,
        "bike": 1,
        "wsapikey": WALKSCORE_API_KEY,
    }
    try:
        resp = requests.get(WALKSCORE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "walk_score": data.get("walkscore"),
            "walk_description": data.get("description"),
            "bike_score": data.get("bike", {}).get("score") if data.get("bike") else None,
            "transit_score": data.get("transit", {}).get("score") if data.get("transit") else None,
        }
    except Exception as exc:
        return {"walk_score": None, "bike_score": None, "transit_score": None,
                "walk_description": None, "_error": str(exc)}


def score_region(gdf: gpd.GeoDataFrame, region: str, cache_path: Path) -> pd.DataFrame:
    """Fetch Walk/Bike scores for all ZCTAs in a GeoDataFrame, with caching."""
    df = gdf[["ZCTA5CE20", "centroid_lat", "centroid_lon"]].copy()
    df = df.rename(columns={"ZCTA5CE20": "zcta"})

    # Load existing cache
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        done_zips = set(cached["zcta"].astype(str))
    else:
        cached = pd.DataFrame()
        done_zips = set()

    todo = df[~df["zcta"].astype(str).isin(done_zips)]
    logger.debug(f"{region}: {len(done_zips)} cached, {len(todo)} to fetch")

    if todo.empty:
        return cached

    results = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"WalkScore {region}"):
        scores = fetch_walkscore(row["centroid_lat"], row["centroid_lon"])
        scores["zcta"] = row["zcta"]
        results.append(scores)
        time.sleep(WALKSCORE_SLEEP_S)

    new_df = pd.DataFrame(results)
    combined = pd.concat([cached, new_df], ignore_index=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def run() -> None:
    north_in = DATA_PROCESSED / "north_airport_filtered.geojson"
    south_in = DATA_PROCESSED / "south_airport_filtered.geojson"

    if not north_in.exists() or not south_in.exists():
        raise FileNotFoundError("Run pipeline/02_airport_proximity.py first.")

    north = gpd.read_file(north_in)
    south = gpd.read_file(south_in)

    if not WALKSCORE_API_KEY:
        logger.warning(
            "WALKSCORE_API_KEY not set in .env — skipping Walk Score API calls. "
            "walk_score and bike_score will be NaN; step 05 will rely on OSM signals only. "
            "To enable: add WALKSCORE_API_KEY=<your key> to .env"
        )
        # Write through with NaN scores so downstream steps work
        for gdf, region in [(north, "north"), (south, "south")]:
            out = gdf.drop(columns=["geometry"])
            out["walk_score"] = float("nan")
            out["bike_score"] = float("nan")
            out["transit_score"] = float("nan")
            out.to_parquet(DATA_PROCESSED / f"{region}_walkscored.parquet", index=False)
        return

    for gdf, region in [(north, "north"), (south, "south")]:
        cache = DATA_PROCESSED / f"{region}_walkscore_cache.parquet"
        scores = score_region(gdf, region, cache)

        # Merge scores back to full candidate table
        base = gdf.drop(columns=["geometry"])
        merged = base.merge(
            scores[["zcta", "walk_score", "bike_score", "transit_score", "walk_description"]],
            left_on="ZCTA5CE20", right_on="zcta", how="left"
        )

        # Apply hard filters
        pre = len(merged)
        merged = merged[
            (merged["walk_score"].isna() | (merged["walk_score"] >= MIN_WALK_SCORE)) &
            (merged["bike_score"].isna() | (merged["bike_score"] >= MIN_BIKE_SCORE))
        ]
        logger.info(f"{region.title()}: {pre:,} → {len(merged):,} after Walk/Bike Score filter")

        out_path = DATA_PROCESSED / f"{region}_walkscored.parquet"
        merged.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    run()

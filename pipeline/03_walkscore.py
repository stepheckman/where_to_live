"""
Step 3: Walk Score + Bike Score (web scraping)
----------------------------------------------
Scrapes walk/bike/transit scores from walkscore.com for candidate block groups.

Two-pass design:
  Pass 1 (no prior amenity data): writes NaN scores so later steps can run.
  Pass 2 (after step 4 has run): scrapes walk/bike scores for ALL candidates
    that passed amenity hard filters, then writes actual scores.

Re-run the pipeline with `--from 3` after an initial full run to trigger
the scraping pass.

Outputs
-------
data/processed/north_walkscored.parquet
data/processed/south_walkscored.parquet
"""

import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
from config import (
    DATA_PROCESSED,
    SCRAPE_SLEEP_S,
    MIN_WALK_SCORE,
    MIN_BIKE_SCORE,
)


WALKSCORE_PAGE_URL = "https://www.walkscore.com/score/loc/lat={lat}/lng={lng}"

# Regex patterns to extract scores from SVG badge URLs on the page
_RE_WALK = re.compile(r"pp\.walk\.sc/badge/walk/score/(\d+)\.svg")
_RE_BIKE = re.compile(r"pp\.walk\.sc/badge/bike/score/(\d+)\.svg")
_RE_TRANSIT = re.compile(r"pp\.walk\.sc/badge/transit/score/(\d+)\.svg")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_walkscore(lat: float, lon: float) -> dict:
    """Scrape walk/bike/transit scores from walkscore.com for a single point."""
    url = WALKSCORE_PAGE_URL.format(lat=round(lat, 6), lng=round(lon, 6))
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        html = resp.text

        walk_m = _RE_WALK.search(html)
        bike_m = _RE_BIKE.search(html)
        transit_m = _RE_TRANSIT.search(html)

        result = {
            "walk_score": int(walk_m.group(1)) if walk_m else None,
            "bike_score": int(bike_m.group(1)) if bike_m else None,
            "transit_score": int(transit_m.group(1)) if transit_m else None,
        }

        if result["walk_score"] is None:
            logger.debug(f"No walk score found in page for ({lat}, {lon})")

        return result
    except Exception as exc:
        logger.warning(f"Scrape failed for ({lat}, {lon}): {exc}")
        return {"walk_score": None, "bike_score": None, "transit_score": None}


def run() -> None:
    north_in = DATA_PROCESSED / "north_airport_filtered.geojson"
    south_in = DATA_PROCESSED / "south_airport_filtered.geojson"

    if not north_in.exists() or not south_in.exists():
        raise FileNotFoundError("Run pipeline/02_airport_proximity.py first.")

    north = gpd.read_file(north_in)
    south = gpd.read_file(south_in)

    for gdf, region in [(north, "north"), (south, "south")]:
        amenity_path = DATA_PROCESSED / f"{region}_amenities.parquet"
        cache_path = DATA_PROCESSED / f"{region}_walkscore_cache.parquet"

        # Load any previously scraped results
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            done_ids = set(cached["geoid"].astype(str))
        else:
            cached = pd.DataFrame()
            done_ids = set()

        if amenity_path.exists():
            # --- SCRAPE PASS: amenity data available, scrape ALL candidates ---
            amenity_df = pd.read_parquet(amenity_path)

            # Scrape every candidate that isn't already cached
            to_scrape = amenity_df[~amenity_df["GEOID"].astype(str).isin(done_ids)]

            if len(to_scrape) == 0:
                logger.info(
                    f"{region.capitalize()}: all {len(done_ids)} candidates already cached"
                )
                combined = cached
            else:
                logger.info(
                    f"{region.capitalize()}: scraping Walk Score for {len(to_scrape)} candidates "
                    f"({len(done_ids)} already cached)"
                )

                results = []
                for _, row in tqdm(to_scrape.iterrows(), total=len(to_scrape),
                                   desc=f"Scrape {region}"):
                    scores = scrape_walkscore(row["centroid_lat"], row["centroid_lon"])
                    scores["geoid"] = str(row["GEOID"])
                    results.append(scores)
                    time.sleep(SCRAPE_SLEEP_S)

                if results:
                    new_df = pd.DataFrame(results)
                    combined = pd.concat([cached, new_df], ignore_index=True)
                    combined.to_parquet(cache_path, index=False)
                    scraped_count = new_df["walk_score"].notna().sum()
                    logger.info(f"Successfully scraped {scraped_count}/{len(new_df)} scores")
                else:
                    combined = cached

            # Merge scores back to full candidate table
            base = gdf.drop(columns=["geometry"])
            if not combined.empty:
                merged = base.merge(
                    combined[["geoid", "walk_score", "bike_score", "transit_score"]],
                    left_on="GEOID", right_on="geoid", how="left"
                ).drop(columns=["geoid"], errors="ignore")
            else:
                merged = base
                merged["walk_score"] = float("nan")
                merged["bike_score"] = float("nan")
                merged["transit_score"] = float("nan")

            # Apply hard filters only to candidates that have scores
            pre = len(merged)
            merged = merged[
                (merged["walk_score"].isna() | (merged["walk_score"] >= MIN_WALK_SCORE)) &
                (merged["bike_score"].isna() | (merged["bike_score"] >= MIN_BIKE_SCORE))
            ]
            dropped = pre - len(merged)
            if dropped:
                logger.info(
                    f"{region.capitalize()}: dropped {dropped} block groups below "
                    f"Walk Score {MIN_WALK_SCORE} / Bike Score {MIN_BIKE_SCORE}"
                )

        elif not cached.empty:
            # --- CACHED PASS: use existing scraped data ---
            logger.info(
                f"{region.capitalize()}: using {len(cached)} cached Walk Score results"
            )
            base = gdf.drop(columns=["geometry"])
            merged = base.merge(
                cached[["geoid", "walk_score", "bike_score", "transit_score"]],
                left_on="GEOID", right_on="geoid", how="left"
            ).drop(columns=["geoid"], errors="ignore")

            pre = len(merged)
            merged = merged[
                (merged["walk_score"].isna() | (merged["walk_score"] >= MIN_WALK_SCORE)) &
                (merged["bike_score"].isna() | (merged["bike_score"] >= MIN_BIKE_SCORE))
            ]
            dropped = pre - len(merged)
            if dropped:
                logger.info(
                    f"{region.capitalize()}: dropped {dropped} block groups below "
                    f"Walk Score {MIN_WALK_SCORE} / Bike Score {MIN_BIKE_SCORE}"
                )

        else:
            # --- PASS-THROUGH: no data yet, write NaN scores ---
            logger.info(
                f"{region.capitalize()}: no amenity data yet — writing NaN walk/bike scores. "
                f"Re-run from step 3 after initial pipeline run to scrape actual scores."
            )
            merged = gdf.drop(columns=["geometry"])
            merged["walk_score"] = float("nan")
            merged["bike_score"] = float("nan")
            merged["transit_score"] = float("nan")

        out_path = DATA_PROCESSED / f"{region}_walkscored.parquet"
        merged.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path} ({len(merged):,} candidates)")


if __name__ == "__main__":
    run()

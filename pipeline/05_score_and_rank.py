"""
Step 5: Composite Scoring and Ranking
---------------------------------------
Normalizes all signals to [0, 1], applies weights from config.py,
and produces the final ranked candidate lists.

For both regions: home value (Census ACS B25077 via Zillow ZHVI) is scored
  inversely — lower value → higher score, since affordability matters.

If Walk Score is missing (not yet scraped), weights are redistributed to OSM signals.

Outputs
-------
outputs/north_candidates.csv
outputs/south_candidates.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
from config import (
    DATA_PROCESSED,
    OUTPUTS,
    TOP_N,
    W_WALK, W_BIKE, W_GROCERY, W_CAFE, W_RESTAURANT, W_PHARMACY, W_TRANSIT, W_HIKING,
    W_HOME_VALUE, W_BARS,
    CAP_GROCERY, CAP_CAFE, CAP_RESTAURANT, CAP_PHARMACY, CAP_TRANSIT, CAP_BARS,
    MIN_WALK_SCORE, MIN_BIKE_SCORE,
)


def normalize_col(series: pd.Series, cap: float = None, invert: bool = False) -> pd.Series:
    """
    Normalize a series to [0, 1].
    - cap: values above this are treated as cap (soft ceiling)
    - invert: lower raw values → higher normalized score (for cost signals)
    """
    s = series.copy().astype(float)
    if cap is not None:
        s = s.clip(upper=cap)

    s_min = s.min()
    s_max = s.max()

    if s_max == s_min:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=series.index)

    normalized = (s - s_min) / (s_max - s_min)
    if invert:
        normalized = 1.0 - normalized

    # Leave NaN as NaN so we can handle missing signals explicitly
    normalized[series.isna()] = np.nan
    return normalized


def compute_scores(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Compute composite score for a region's candidate DataFrame."""
    df = df.copy()

    has_walkscore = df["walk_score"].notna().any() if "walk_score" in df.columns else False

    # Normalize all input signals
    df["n_walk"]       = normalize_col(df.get("walk_score"), cap=100)
    df["n_bike"]       = normalize_col(df.get("bike_score"), cap=100)
    df["n_grocery"]    = normalize_col(df.get("grocery_count"), cap=CAP_GROCERY)
    df["n_cafe"]       = normalize_col(df.get("cafe_count"), cap=CAP_CAFE)
    df["n_restaurant"] = normalize_col(df.get("restaurant_count"), cap=CAP_RESTAURANT)
    df["n_pharmacy"]   = normalize_col(df.get("pharmacy_count"), cap=CAP_PHARMACY)
    df["n_transit"]    = normalize_col(df.get("transit_stops"), cap=CAP_TRANSIT)

    # Hiking: combine distance (inverted) and count signals equally
    has_hiking = (
        "dist_nearest_park_km" in df.columns and
        df["dist_nearest_park_km"].notna().any()
    )
    if has_hiking:
        n_dist = normalize_col(df["dist_nearest_park_km"], invert=True)
        n_count = normalize_col(df.get("parks_within_50km"))
        df["n_hiking"] = (n_dist.fillna(0.5) + n_count.fillna(0.5)) / 2
        hiking_weight = W_HIKING
    else:
        df["n_hiking"] = np.nan
        hiking_weight = 0.0

    if "median_home_value" in df.columns:
        df["n_affordability"] = normalize_col(df["median_home_value"], invert=True)
        affordability_weight = W_HOME_VALUE
    else:
        df["n_affordability"] = np.nan
        affordability_weight = 0.0

    # Bars: fixed inversion — 0 bars → 1.0, CAP_BARS+ bars → 0.0
    if "bar_count" in df.columns:
        df["n_bars"] = 1.0 - (df["bar_count"].clip(upper=CAP_BARS) / CAP_BARS)
        bars_weight = W_BARS
    else:
        df["n_bars"] = np.nan
        bars_weight = 0.0

    has_transit = "transit_stops" in df.columns and df["transit_stops"].notna().any()

    # If Walk Score is missing, redistribute its weight to OSM signals
    if not has_walkscore:
        w_walk = 0.0
        w_bike = 0.0
        # Redistribute walk+bike weight proportionally to OSM signals
        total_osm_base = W_GROCERY + W_CAFE + W_RESTAURANT + W_PHARMACY + W_TRANSIT
        scale = (W_WALK + W_BIKE + total_osm_base) / total_osm_base
        w_grocery    = W_GROCERY    * scale
        w_cafe       = W_CAFE       * scale
        w_restaurant = W_RESTAURANT * scale
        w_pharmacy   = W_PHARMACY   * scale
        w_transit    = W_TRANSIT    * scale
    else:
        w_walk       = W_WALK
        w_bike       = W_BIKE
        w_grocery    = W_GROCERY
        w_cafe       = W_CAFE
        w_restaurant = W_RESTAURANT
        w_pharmacy   = W_PHARMACY
        w_transit    = W_TRANSIT if has_transit else 0.0

    # Scale down base weights to make room for hiking, affordability, and bars
    fixed_weight = hiking_weight + affordability_weight + bars_weight
    if fixed_weight > 0:
        scale_factor = 1.0 - fixed_weight
        w_walk       *= scale_factor
        w_bike       *= scale_factor
        w_grocery    *= scale_factor
        w_cafe       *= scale_factor
        w_restaurant *= scale_factor
        w_pharmacy   *= scale_factor
        w_transit    *= scale_factor
    if not has_hiking:
        hiking_weight = 0.0
    if not (affordability_weight > 0 and df["n_affordability"].notna().any()):
        affordability_weight = 0.0
    if not (bars_weight > 0 and df["n_bars"].notna().any()):
        bars_weight = 0.0

    def safe_fill(col: pd.Series, default: float = 0.5) -> pd.Series:
        """Fill NaN with a neutral value for scoring (doesn't penalize missing)."""
        return col.fillna(default)

    df["composite_score"] = (
        w_walk               * safe_fill(df["n_walk"])           +
        w_bike               * safe_fill(df["n_bike"])           +
        w_grocery            * safe_fill(df["n_grocery"])        +
        w_cafe               * safe_fill(df["n_cafe"])           +
        w_restaurant         * safe_fill(df["n_restaurant"])     +
        w_pharmacy           * safe_fill(df["n_pharmacy"])       +
        w_transit            * safe_fill(df["n_transit"])        +
        hiking_weight        * safe_fill(df["n_hiking"])         +
        affordability_weight * safe_fill(df["n_affordability"])  +
        bars_weight          * safe_fill(df["n_bars"])
    ) * 100  # scale to 0–100

    return df


def build_output_row(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Select and rename columns for the final output CSV."""
    cols = {
        "GEOID": "geoid",
        "centroid_lat": "lat",
        "centroid_lon": "lon",
        "walk_score": "walk_score",
        "bike_score": "bike_score",
        "transit_score": "transit_score",
        "grocery_count": "grocery_count",
        "cafe_count": "cafe_count",
        "restaurant_count": "restaurant_count",
        "pharmacy_count": "pharmacy_count",
        "transit_stops": "transit_stops",
        "bar_count": "bar_count",
        "dist_nearest_park_km": "dist_nearest_park_km",
        "parks_within_50km": "parks_within_50km",
        "nearest_airport": "nearest_airport",
        "airport_drive_min_approx": "airport_drive_min",
        "composite_score": "composite_score",
    }
    cols["median_home_value"] = "median_home_value"

    present = {k: v for k, v in cols.items() if k in df.columns}
    out = df.rename(columns=present)[[v for v in present.values()]]
    return out.sort_values("composite_score", ascending=False).head(TOP_N)


def run() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    for region in ("north", "south"):
        in_path = DATA_PROCESSED / f"{region}_amenities.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"Run pipeline/04_osm_amenities.py first ({in_path})")

        df = pd.read_parquet(in_path)
        logger.info(f"--- {region.upper()} ({len(df):,} candidates after all filters) ---")

        scored = compute_scores(df, region)
        top = build_output_row(scored, region)

        out_path = OUTPUTS / f"{region}_candidates.csv"
        top.to_csv(out_path, index=False)

        logger.success(f"Top {len(top)} {region} candidates saved to {out_path}")
        logger.debug("\n" + top[["geoid", "composite_score"] +
              (["walk_score"] if "walk_score" in top.columns else []) +
              (["median_home_value"] if "median_home_value" in top.columns else [])
              ].head(10).to_string(index=False))


if __name__ == "__main__":
    run()

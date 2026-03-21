"""
FastAPI REST API for the where-to-live pipeline outputs.

Endpoints
---------
GET /                      — API info and available endpoints
GET /candidates/{region}   — top candidates for 'north', 'south', or 'combined'

Query params (all optional)
---------------------------
min_walk_score   int   minimum Walk Score (0–100)          default: config value
min_bike_score   int   minimum Bike Score (0–100)          default: config value
max_airport_min  int   max drive to airport in minutes     default: config value
min_groceries    int   min grocery stores within 1200m     default: config value
min_pharmacies   int   min pharmacies within 1200m         default: config value
top_n            int   number of results to return         default: config value

Weight params (0.0–1.0 each, auto-normalized to sum to 1.0)
------------------------------------------------------------
w_walk, w_bike, w_grocery, w_cafe, w_restaurant,
w_pharmacy, w_hiking, w_afford

Run with:
    uv run uvicorn api:app --reload
"""

from pathlib import Path
import sys
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CAP_CAFE,
    CAP_GROCERY,
    CAP_PHARMACY,
    CAP_RESTAURANT,
    DATA_PROCESSED,
    MAX_AIRPORT_DRIVE_MIN,
    MIN_BIKE_SCORE,
    MIN_GROCERY_COUNT,
    MIN_PHARMACY_COUNT,
    MIN_WALK_SCORE,
    TOP_N,
    W_BIKE,
    W_CAFE,
    W_GROCERY,
    W_HIKING,
    W_HOME_VALUE,
    W_PHARMACY,
    W_RESTAURANT,
    W_WALK,
)

app = FastAPI(
    title="Where to Live API",
    description=(
        "Query and re-score neighborhood candidates from the where-to-live pipeline. "
        "Reads from cached parquet files produced by pipeline step 4."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Data loading (at module startup; parquet reads are fast)
# ---------------------------------------------------------------------------

def _load_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


_north_raw = _load_parquet(DATA_PROCESSED / "north_amenities.parquet")
_south_raw = _load_parquet(DATA_PROCESSED / "south_amenities.parquet")


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors app.py logic)
# ---------------------------------------------------------------------------

def _normalize(s: pd.Series, cap: float | None = None, invert: bool = False) -> pd.Series:
    s = s.copy().astype(float)
    if cap is not None:
        s = s.clip(upper=cap)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    n = (s - mn) / (mx - mn)
    if invert:
        n = 1.0 - n
    n[s.isna()] = np.nan
    return n


def _build_weights(
    w_walk: float,
    w_bike: float,
    w_grocery: float,
    w_cafe: float,
    w_restaurant: float,
    w_pharmacy: float,
    w_hiking: float,
    w_afford: float,
) -> dict:
    raw = dict(
        walk=w_walk, bike=w_bike, grocery=w_grocery, cafe=w_cafe,
        restaurant=w_restaurant, pharmacy=w_pharmacy, hiking=w_hiking, afford=w_afford,
    )
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _score_and_filter(
    df: pd.DataFrame,
    region: str,
    weights: dict,
    min_walk: int,
    min_bike: int,
    max_airport: int,
    min_grocery: int,
    min_pharmacy: int,
    top_n: int,
) -> pd.DataFrame:
    df = df.copy()

    # Rename canonical columns to match app.py display names
    rename = {
        "GEOID": "geoid",
        "centroid_lat": "lat",
        "centroid_lon": "lon",
        "airport_drive_min_approx": "airport_drive_min",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Hard filters
    if "airport_drive_min" in df.columns:
        df = df[df["airport_drive_min"] <= max_airport]
    if "grocery_count" in df.columns:
        df = df[df["grocery_count"] >= min_grocery]
    if "pharmacy_count" in df.columns:
        df = df[df["pharmacy_count"] >= min_pharmacy]

    has_walk = "walk_score" in df.columns and df["walk_score"].notna().any()
    has_bike = "bike_score" in df.columns and df["bike_score"].notna().any()
    if has_walk:
        df = df[df["walk_score"] >= min_walk]
    if has_bike:
        df = df[df["bike_score"] >= min_bike]

    if df.empty:
        return df

    # Normalize signals
    def col_or_nan(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    df["n_walk"] = _normalize(col_or_nan("walk_score"), cap=100)
    df["n_bike"] = _normalize(col_or_nan("bike_score"), cap=100)
    df["n_grocery"] = _normalize(df["grocery_count"], cap=CAP_GROCERY)
    df["n_cafe"] = _normalize(col_or_nan("cafe_count"), cap=CAP_CAFE)
    df["n_restaurant"] = _normalize(col_or_nan("restaurant_count"), cap=CAP_RESTAURANT)
    df["n_pharmacy"] = _normalize(col_or_nan("pharmacy_count"), cap=CAP_PHARMACY)

    has_hiking = (
        "dist_nearest_park_km" in df.columns
        and df["dist_nearest_park_km"].notna().any()
    )
    if has_hiking:
        n_dist = _normalize(df["dist_nearest_park_km"], invert=True)
        n_count = _normalize(col_or_nan("parks_within_50km"))
        df["n_hiking"] = (n_dist.fillna(0.5) + n_count.fillna(0.5)) / 2
    else:
        df["n_hiking"] = pd.Series(np.nan, index=df.index)

    df["n_afford"] = (
        _normalize(df["median_home_value"], invert=True)
        if "median_home_value" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Re-normalize weights where data is missing
    w = weights.copy()
    if df["n_afford"].isna().all():
        w["afford"] = 0.0
    if df["n_hiking"].isna().all():
        w["hiking"] = 0.0
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    def safe(col: pd.Series) -> pd.Series:
        return col.fillna(0.5)

    df["composite_score"] = (
        w["walk"] * safe(df["n_walk"])
        + w["bike"] * safe(df["n_bike"])
        + w["grocery"] * safe(df["n_grocery"])
        + w["cafe"] * safe(df["n_cafe"])
        + w["restaurant"] * safe(df["n_restaurant"])
        + w["pharmacy"] * safe(df["n_pharmacy"])
        + w["hiking"] * safe(df["n_hiking"])
        + w["afford"] * safe(df["n_afford"])
    ) * 100

    # Drop internal normalization columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("n_")])

    return df.sort_values("composite_score", ascending=False).head(top_n)


def _to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records (NaN → None)."""
    return [
        {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


# ---------------------------------------------------------------------------
# Shared query param type aliases
# ---------------------------------------------------------------------------

WeightQ = Annotated[float, Query(ge=0.0, le=1.0, description="Scoring weight (auto-normalized)")]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="API info")
def root() -> dict:
    return {
        "name": "Where to Live API",
        "version": "1.0.0",
        "endpoints": {
            "GET /candidates/north":    "Top north candidates (buy)",
            "GET /candidates/south":    "Top south candidates (rent)",
            "GET /candidates/combined": "North + south merged, sorted by score",
        },
        "docs": "/docs",
        "data_available": {
            "north": _north_raw is not None,
            "south": _south_raw is not None,
        },
    }


@app.get("/candidates/{region}", summary="Top candidates for a region")
def get_candidates(
    region: str,
    # Hard filters
    min_walk_score:  Annotated[int, Query(ge=0, le=100, description="Minimum Walk Score")] = MIN_WALK_SCORE,
    min_bike_score:  Annotated[int, Query(ge=0, le=100, description="Minimum Bike Score")] = MIN_BIKE_SCORE,
    max_airport_min: Annotated[int, Query(ge=0, le=300, description="Max drive to airport (minutes)")] = MAX_AIRPORT_DRIVE_MIN,
    min_groceries:   Annotated[int, Query(ge=0, description="Min grocery stores within 1200m")] = MIN_GROCERY_COUNT,
    min_pharmacies:  Annotated[int, Query(ge=0, description="Min pharmacies within 1200m")] = MIN_PHARMACY_COUNT,
    top_n:           Annotated[int, Query(ge=1, le=500, description="Number of results")] = TOP_N,
    # Score weights
    w_walk:       WeightQ = W_WALK,
    w_bike:       WeightQ = W_BIKE,
    w_grocery:    WeightQ = W_GROCERY,
    w_cafe:       WeightQ = W_CAFE,
    w_restaurant: WeightQ = W_RESTAURANT,
    w_pharmacy:   WeightQ = W_PHARMACY,
    w_hiking:     WeightQ = W_HIKING,
    w_afford:     WeightQ = W_HOME_VALUE,
) -> JSONResponse:
    if region not in ("north", "south", "combined"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown region '{region}'. Valid values: north, south, combined.",
        )

    weights = _build_weights(
        w_walk, w_bike, w_grocery, w_cafe, w_restaurant, w_pharmacy, w_hiking, w_afford,
    )
    filter_kwargs = dict(
        weights=weights,
        min_walk=min_walk_score,
        min_bike=min_bike_score,
        max_airport=max_airport_min,
        min_grocery=min_groceries,
        min_pharmacy=min_pharmacies,
        top_n=top_n,
    )

    if region == "combined":
        parts = []
        if _north_raw is not None:
            n = _score_and_filter(_north_raw, "north", **filter_kwargs)
            n = n.assign(region="north")
            parts.append(n)
        if _south_raw is not None:
            s = _score_and_filter(_south_raw, "south", **filter_kwargs)
            s = s.assign(region="south")
            parts.append(s)
        if not parts:
            raise HTTPException(status_code=404, detail="No pipeline data found. Run the pipeline first.")
        df = (
            pd.concat(parts, ignore_index=True)
            .sort_values("composite_score", ascending=False)
            .head(top_n)
        )
    else:
        raw = _north_raw if region == "north" else _south_raw
        if raw is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data for region '{region}'. Run the pipeline first (uv run python run_pipeline.py --only 4).",
            )
        df = _score_and_filter(raw, region, **filter_kwargs)

    return JSONResponse({
        "region": region,
        "count": len(df),
        "filters": {
            "min_walk_score": min_walk_score,
            "min_bike_score": min_bike_score,
            "max_airport_min": max_airport_min,
            "min_groceries": min_groceries,
            "min_pharmacies": min_pharmacies,
        },
        "weights_normalized": weights,
        "candidates": _to_records(df),
    })

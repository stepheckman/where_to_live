"""
Step 4: OSM Amenity Verification
---------------------------------
Counts key POI types within walking/biking distance of each candidate
centroid using the Overpass API (via osmnx). Results are cached.

Also pulls affordability data:
  - North (buy): Zillow ZHVI median home values by ZCTA (free CSV download)
  - South (rent): HUD Fair Market Rents 2BR by county → mapped to ZCTA

Outputs
-------
data/processed/north_amenities.parquet
data/processed/south_amenities.parquet
"""

import io
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    POI_RADIUS_METERS,
    MIN_GROCERY_COUNT,
    MIN_PHARMACY_COUNT,
    OSM_SLEEP_S,
)

# OSM tags to query
POI_TAGS = {
    "amenity": ["cafe", "restaurant", "fast_food", "pharmacy", "bar", "pub"],
    "shop": ["supermarket", "grocery", "convenience", "greengrocer"],
}

# Zillow ZHVI — ZCTA-level median home values (all homes, mid-tier)
# Update URL from https://www.zillow.com/research/data/ if link changes
ZILLOW_ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# HUD Fair Market Rents (FY2024) — county-level, 2BR
HUD_FMR_URL = (
    "https://www.huduser.gov/portal/datasets/fmr/fmr2024/FY2024_4050_FMR.xlsx"
)


# ---------------------------------------------------------------------------
# OSM amenity fetching
# ---------------------------------------------------------------------------

def count_pois(lat: float, lon: float) -> dict:
    """Count POI categories within POI_RADIUS_METERS of a point via osmnx."""
    try:
        pois = ox.features_from_point((lat, lon), tags=POI_TAGS, dist=POI_RADIUS_METERS)
    except Exception:
        # Overpass may time out or return empty
        pois = gpd.GeoDataFrame()

    if pois.empty:
        return {
            "grocery_count": 0,
            "pharmacy_count": 0,
            "cafe_count": 0,
            "restaurant_count": 0,
        }

    # Grocery: supermarket/grocery shops
    grocery_mask = (
        pois.get("shop", pd.Series(dtype=str)).isin(
            ["supermarket", "grocery", "greengrocer"]
        )
    )
    # Convenience stores don't count as full grocery
    pharmacy_mask = pois.get("amenity", pd.Series(dtype=str)) == "pharmacy"
    cafe_mask = pois.get("amenity", pd.Series(dtype=str)) == "cafe"
    restaurant_mask = pois.get("amenity", pd.Series(dtype=str)).isin(
        ["restaurant", "fast_food", "bar", "pub"]
    )

    return {
        "grocery_count": int(grocery_mask.sum()),
        "pharmacy_count": int(pharmacy_mask.sum()),
        "cafe_count": int(cafe_mask.sum()),
        "restaurant_count": int(restaurant_mask.sum()),
    }


def fetch_amenities_for_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Fetch and cache OSM amenity counts for all candidates."""
    cache = DATA_RAW / f"{region}_osm_cache.parquet"

    if cache.exists():
        cached = pd.read_parquet(cache)
        done_zips = set(cached["zcta"].astype(str))
    else:
        cached = pd.DataFrame()
        done_zips = set()

    todo = df[~df["ZCTA5CE20"].astype(str).isin(done_zips)]
    print(f"  {region}: {len(done_zips)} cached, {len(todo)} to fetch")

    results = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"OSM {region}"):
        counts = count_pois(row["centroid_lat"], row["centroid_lon"])
        counts["zcta"] = row["ZCTA5CE20"]
        results.append(counts)
        time.sleep(OSM_SLEEP_S)

    if results:
        new_df = pd.DataFrame(results)
        combined = pd.concat([cached, new_df], ignore_index=True)
    else:
        combined = cached

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache, index=False)
    return combined


# ---------------------------------------------------------------------------
# Zillow ZHVI (north — buying)
# ---------------------------------------------------------------------------

def fetch_zillow_zhvi() -> pd.DataFrame:
    """
    Download Zillow ZHVI (median home value) by ZIP code.
    Returns DataFrame with columns: zip, zhvi_latest
    """
    cache = DATA_RAW / "zillow_zhvi.parquet"
    if cache.exists():
        print("Zillow ZHVI data already cached.")
        return pd.read_parquet(cache)

    print("Downloading Zillow ZHVI data…")
    try:
        resp = requests.get(ZILLOW_ZHVI_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        print(f"WARNING: Could not download Zillow data ({exc}). Skipping home value signal.")
        return pd.DataFrame(columns=["zip", "zhvi_latest"])

    df = pd.read_csv(io.StringIO(resp.text))

    # The rightmost columns are dates; the last one is the most recent value
    date_cols = [c for c in df.columns if c[:4].isdigit()]
    if not date_cols:
        print("WARNING: Zillow CSV format unexpected. Skipping home value signal.")
        return pd.DataFrame(columns=["zip", "zhvi_latest"])

    latest_col = sorted(date_cols)[-1]
    df = df.rename(columns={"RegionName": "zip"})
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    result = df[["zip", latest_col]].rename(columns={latest_col: "zhvi_latest"})
    result["zhvi_latest"] = pd.to_numeric(result["zhvi_latest"], errors="coerce")

    result.to_parquet(cache, index=False)
    print(f"Cached Zillow ZHVI for {len(result):,} ZIPs")
    return result


# ---------------------------------------------------------------------------
# HUD Fair Market Rents (south — renting)
# ---------------------------------------------------------------------------

def fetch_hud_fmr() -> pd.DataFrame:
    """
    Download HUD Fair Market Rents (2BR) by county.
    Returns DataFrame with columns: fips_county (5-digit), fmr_2br
    """
    cache = DATA_RAW / "hud_fmr.parquet"
    if cache.exists():
        print("HUD FMR data already cached.")
        return pd.read_parquet(cache)

    print("Downloading HUD Fair Market Rents…")
    try:
        resp = requests.get(HUD_FMR_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"WARNING: Could not download HUD FMR data ({exc}). Skipping rent signal.")
        return pd.DataFrame(columns=["fips_county", "fmr_2br"])

    df = pd.read_excel(io.BytesIO(resp.content))
    df.columns = [c.strip().lower() for c in df.columns]

    # Typical HUD FMR columns: fips2010, countyname, fmr_2
    fips_col = next((c for c in df.columns if "fips" in c), None)
    fmr_col = next((c for c in df.columns if "fmr_2" in c or "fmr2" in c), None)

    if fips_col is None or fmr_col is None:
        print(f"WARNING: HUD FMR columns not found. Available: {list(df.columns)}")
        return pd.DataFrame(columns=["fips_county", "fmr_2br"])

    result = df[[fips_col, fmr_col]].rename(columns={
        fips_col: "fips_county",
        fmr_col: "fmr_2br",
    })
    result["fips_county"] = result["fips_county"].astype(str).str[:5].str.zfill(5)
    result["fmr_2br"] = pd.to_numeric(result["fmr_2br"], errors="coerce")

    result.to_parquet(cache, index=False)
    print(f"Cached HUD FMR for {len(result):,} counties")
    return result


def fetch_zcta_county_crosswalk() -> pd.DataFrame:
    """
    Download HUD USPS ZCTA→County crosswalk (free).
    Returns DataFrame with columns: zcta, county_fips
    """
    cache = DATA_RAW / "zcta_county_crosswalk.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # HUD USPS crosswalk — Q4 2023
    url = (
        "https://www.huduser.gov/hudapi/public/usps?type=1&query=All"
    )
    # Fallback: use a Census crosswalk file
    census_url = (
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
        "zcta520/tab20_zcta520_county20_natl.txt"
    )
    print("Downloading ZCTA→County crosswalk…")
    try:
        resp = requests.get(census_url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            sep="|",
            dtype=str,
        )
        # Columns: GEOID_ZCTA5_20, GEOID_COUNTY_20, ...
        df = df.rename(columns={
            "GEOID_ZCTA5_20": "zcta",
            "GEOID_COUNTY_20": "county_fips",
        })
        df = df[["zcta", "county_fips"]].dropna()
        # Keep the county with the largest areal intersection (first row per zcta)
        df = df.drop_duplicates(subset="zcta", keep="first")
    except Exception as exc:
        print(f"WARNING: Could not download crosswalk ({exc}). Skipping county join.")
        return pd.DataFrame(columns=["zcta", "county_fips"])

    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    for region in ("north", "south"):
        in_path = DATA_PROCESSED / f"{region}_walkscored.parquet"
        if not in_path.exists():
            raise FileNotFoundError(f"Run pipeline/03_walkscore.py first ({in_path})")

        df = pd.read_parquet(in_path)
        print(f"\n--- {region.upper()} ({len(df):,} candidates) ---")

        # OSM amenities
        amenity_scores = fetch_amenities_for_region(df, region)
        df = df.merge(
            amenity_scores[["zcta", "grocery_count", "pharmacy_count",
                            "cafe_count", "restaurant_count"]],
            left_on="ZCTA5CE20", right_on="zcta", how="left"
        ).drop(columns=["zcta"], errors="ignore")

        # Hard filter: must have at least one grocery and one pharmacy
        pre = len(df)
        df = df[
            (df["grocery_count"] >= MIN_GROCERY_COUNT) &
            (df["pharmacy_count"] >= MIN_PHARMACY_COUNT)
        ]
        print(f"  Amenity hard filter: {pre:,} → {len(df):,} ZCTAs")

    # Affordability signals
    zillow = fetch_zillow_zhvi()
    hud = fetch_hud_fmr()
    crosswalk = fetch_zcta_county_crosswalk()

    for region in ("north", "south"):
        in_path = DATA_PROCESSED / f"{region}_walkscored.parquet"
        df = pd.read_parquet(in_path)

        # Re-apply amenity merge (in case we restarted)
        amenity_cache = DATA_RAW / f"{region}_osm_cache.parquet"
        if amenity_cache.exists():
            amenity_scores = pd.read_parquet(amenity_cache)
            df = df.merge(
                amenity_scores[["zcta", "grocery_count", "pharmacy_count",
                                "cafe_count", "restaurant_count"]],
                left_on="ZCTA5CE20", right_on="zcta", how="left"
            ).drop(columns=["zcta"], errors="ignore")
            df = df[
                (df["grocery_count"] >= MIN_GROCERY_COUNT) &
                (df["pharmacy_count"] >= MIN_PHARMACY_COUNT)
            ]

        if region == "north" and not zillow.empty:
            df["ZCTA5CE20_str"] = df["ZCTA5CE20"].astype(str).str.zfill(5)
            df = df.merge(
                zillow.rename(columns={"zip": "ZCTA5CE20_str"}),
                on="ZCTA5CE20_str", how="left"
            ).drop(columns=["ZCTA5CE20_str"], errors="ignore")
            print(f"  North: Zillow ZHVI joined for "
                  f"{df['zhvi_latest'].notna().sum():,}/{len(df):,} ZCTAs")

        if region == "south" and not hud.empty and not crosswalk.empty:
            df["ZCTA5CE20_str"] = df["ZCTA5CE20"].astype(str).str.zfill(5)
            df = df.merge(
                crosswalk.rename(columns={"zcta": "ZCTA5CE20_str"}),
                on="ZCTA5CE20_str", how="left"
            )
            df = df.merge(hud, on="county_fips", how="left")
            df = df.drop(columns=["ZCTA5CE20_str", "county_fips"], errors="ignore")
            print(f"  South: HUD FMR joined for "
                  f"{df['fmr_2br'].notna().sum():,}/{len(df):,} ZCTAs")

        out_path = DATA_PROCESSED / f"{region}_amenities.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    run()

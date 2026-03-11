"""
Step 4: OSM Amenity Verification
---------------------------------
Counts key POI types within walking/biking distance of each candidate
centroid using the Overpass API (via osmnx). Results are cached.

Also pulls affordability data:
  - North (buy): Zillow ZHVI median home values by ZCTA (free CSV download)
  - South (rent): HUD Fair Market Rents 2BR by county → mapped to ZCTA

And hiking access data:
  - PAD-US (USGS Protected Areas Database): distance to nearest qualifying
    protected area (≥ HIKING_MIN_ACRES, GAP status 1–3) + count within
    HIKING_RADIUS_KM. Downloaded once as GeoPackage, cached as filtered
    parquet for fast reuse.

Outputs
-------
data/processed/north_amenities.parquet
data/processed/south_amenities.parquet
"""

import io
import time
import zipfile
from pathlib import Path

import fiona
import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    POI_RADIUS_METERS,
    MIN_GROCERY_COUNT,
    MIN_PHARMACY_COUNT,
    OSM_SLEEP_S,
    HIKING_MIN_ACRES,
    HIKING_RADIUS_KM,
    HIKING_GAP_STATUS,
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
    logger.debug(f"{region}: {len(done_zips)} cached, {len(todo)} to fetch via Overpass")

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
        logger.debug("Zillow ZHVI data already cached.")
        return pd.read_parquet(cache)

    logger.info("Downloading Zillow ZHVI data…")
    try:
        resp = requests.get(ZILLOW_ZHVI_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(f"Could not download Zillow data ({exc}) — skipping home value signal.")
        return pd.DataFrame(columns=["zip", "zhvi_latest"])

    df = pd.read_csv(io.StringIO(resp.text))

    # The rightmost columns are dates; the last one is the most recent value
    date_cols = [c for c in df.columns if c[:4].isdigit()]
    if not date_cols:
        logger.warning("Zillow CSV format unexpected — skipping home value signal.")
        return pd.DataFrame(columns=["zip", "zhvi_latest"])

    latest_col = sorted(date_cols)[-1]
    df = df.rename(columns={"RegionName": "zip"})
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    result = df[["zip", latest_col]].rename(columns={latest_col: "zhvi_latest"})
    result["zhvi_latest"] = pd.to_numeric(result["zhvi_latest"], errors="coerce")

    result.to_parquet(cache, index=False)
    logger.debug(f"Cached Zillow ZHVI for {len(result):,} ZIPs")
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
        logger.debug("HUD FMR data already cached.")
        return pd.read_parquet(cache)

    logger.info("Downloading HUD Fair Market Rents…")
    try:
        resp = requests.get(HUD_FMR_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(f"Could not download HUD FMR data ({exc}) — skipping rent signal.")
        return pd.DataFrame(columns=["fips_county", "fmr_2br"])

    df = pd.read_excel(io.BytesIO(resp.content))
    df.columns = [c.strip().lower() for c in df.columns]

    # Typical HUD FMR columns: fips2010, countyname, fmr_2
    fips_col = next((c for c in df.columns if "fips" in c), None)
    fmr_col = next((c for c in df.columns if "fmr_2" in c or "fmr2" in c), None)

    if fips_col is None or fmr_col is None:
        logger.warning(f"HUD FMR columns not found. Available: {list(df.columns)}")
        return pd.DataFrame(columns=["fips_county", "fmr_2br"])

    result = df[[fips_col, fmr_col]].rename(columns={
        fips_col: "fips_county",
        fmr_col: "fmr_2br",
    })
    result["fips_county"] = result["fips_county"].astype(str).str[:5].str.zfill(5)
    result["fmr_2br"] = pd.to_numeric(result["fmr_2br"], errors="coerce")

    result.to_parquet(cache, index=False)
    logger.debug(f"Cached HUD FMR for {len(result):,} counties")
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
    logger.info("Downloading ZCTA→County crosswalk…")
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
        logger.warning(f"Could not download crosswalk ({exc}) — skipping county join.")
        return pd.DataFrame(columns=["zcta", "county_fips"])

    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# PAD-US hiking access (USGS Protected Areas Database)
# ---------------------------------------------------------------------------

# PAD-US 4.0 — Combined GeoPackage via USGS ScienceBase
# If auto-download fails, manually download from:
#   https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview
# and place the extracted .gpkg at DATA_RAW / "padus" / "padus.gpkg"
_PADUS_SCIENCEBASE_ITEM = "652ee185d34e6bef37e0c4d0"
_PADUS_GPKG_DIR = DATA_RAW / "padus"
_PADUS_FILTERED_CACHE = DATA_RAW / "padus_filtered.parquet"


def _download_padus() -> Path:
    """
    Download PAD-US Combined GeoPackage from USGS ScienceBase.
    Returns path to the extracted .gpkg file.
    """
    _PADUS_GPKG_DIR.mkdir(parents=True, exist_ok=True)

    # Query ScienceBase API to discover the download URL dynamically
    api_url = (
        f"https://www.sciencebase.gov/catalog/item/{_PADUS_SCIENCEBASE_ITEM}"
        "?format=json"
    )
    logger.info("Querying ScienceBase for PAD-US download URL…")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])

    # Find the Combined GeoPackage zip
    zip_url = None
    for f in files:
        name = f.get("name", "")
        if "Combined" in name and name.endswith(".zip") and "GeoPackage" in name:
            zip_url = f.get("downloadUri") or f.get("url")
            break

    if not zip_url:
        raise RuntimeError(
            "Could not find PAD-US Combined GeoPackage in ScienceBase listing.\n"
            "Download manually from:\n"
            "  https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview\n"
            f"Extract the .gpkg file to: {_PADUS_GPKG_DIR}"
        )

    zip_path = _PADUS_GPKG_DIR / "padus_combined.zip"
    logger.info(f"Downloading PAD-US GeoPackage (~500 MB) from ScienceBase…")
    with requests.get(zip_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    logger.debug(f"  {pct:.0f}%", end="\r")

    logger.info("Extracting PAD-US zip…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_PADUS_GPKG_DIR)
    zip_path.unlink()  # remove zip to save space

    # Find the extracted .gpkg
    gpkg_files = list(_PADUS_GPKG_DIR.glob("*.gpkg"))
    if not gpkg_files:
        raise RuntimeError(
            f"No .gpkg found after extracting to {_PADUS_GPKG_DIR}.\n"
            "Check the download and extraction manually."
        )
    return gpkg_files[0]


def _find_gpkg() -> Path:
    """Return path to the PAD-US GeoPackage, downloading if necessary."""
    existing = list(_PADUS_GPKG_DIR.glob("*.gpkg"))
    if existing:
        return existing[0]
    return _download_padus()


def _load_padus_filtered() -> gpd.GeoDataFrame:
    """
    Load PAD-US, filter to qualifying protected areas, and cache as parquet.
    Qualifying = GAP status in HIKING_GAP_STATUS and GIS_Acres >= HIKING_MIN_ACRES.
    """
    if _PADUS_FILTERED_CACHE.exists():
        logger.debug("Loading cached filtered PAD-US…")
        return gpd.read_parquet(_PADUS_FILTERED_CACHE)

    gpkg_path = _find_gpkg()
    logger.info(f"Loading PAD-US from {gpkg_path} (this may take a few minutes)…")

    # Discover the Combined layer name (varies by version)
    layers = fiona.listlayers(str(gpkg_path))
    combined_layer = next((l for l in layers if "Combined" in l), layers[0])
    logger.debug(f"Using PAD-US layer: {combined_layer}")

    gdf = gpd.read_file(
        gpkg_path,
        layer=combined_layer,
        columns=["geometry", "GAP_Sts", "GIS_Acres", "Mang_Type", "Mang_Name"],
    )

    # Normalize column types
    gdf["GAP_Sts"] = gdf["GAP_Sts"].astype(str).str.strip()
    gdf["GIS_Acres"] = pd.to_numeric(gdf["GIS_Acres"], errors="coerce")

    # Filter: qualifying GAP status and minimum acreage
    gdf = gdf[
        gdf["GAP_Sts"].isin(HIKING_GAP_STATUS) &
        (gdf["GIS_Acres"] >= HIKING_MIN_ACRES)
    ].copy()

    logger.info(f"PAD-US filtered to {len(gdf):,} qualifying protected areas")
    gdf = gdf.to_crs("EPSG:4326")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(_PADUS_FILTERED_CACHE)
    return gdf


def fetch_hiking_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hiking access metrics for all candidate ZCTAs.

    Returns DataFrame with columns:
      zcta, dist_nearest_park_km, parks_within_50km
    """
    cache = DATA_RAW / "hiking_scores.parquet"
    if cache.exists():
        logger.debug("Hiking scores already cached.")
        return pd.read_parquet(cache)

    padus = _load_padus_filtered()

    candidates = gpd.GeoDataFrame(
        df[["ZCTA5CE20", "centroid_lat", "centroid_lon"]].copy(),
        geometry=gpd.points_from_xy(df["centroid_lon"], df["centroid_lat"]),
        crs="EPSG:4326",
    )

    # Reproject to Albers Equal Area (meters) for accurate distance/buffer
    crs_proj = "EPSG:5070"
    candidates_proj = candidates.to_crs(crs_proj)
    padus_proj = padus[["geometry"]].to_crs(crs_proj)

    # Distance to nearest qualifying protected area
    logger.info("Computing distance to nearest protected area…")
    nearest = gpd.sjoin_nearest(
        candidates_proj[["ZCTA5CE20", "geometry"]],
        padus_proj,
        how="left",
        distance_col="dist_nearest_park_m",
    ).drop_duplicates(subset="ZCTA5CE20")
    nearest["dist_nearest_park_km"] = (nearest["dist_nearest_park_m"] / 1000).round(2)

    # Count protected areas within HIKING_RADIUS_KM
    logger.info(f"Counting protected areas within {HIKING_RADIUS_KM} km…")
    radius_m = HIKING_RADIUS_KM * 1000
    buffers = candidates_proj[["ZCTA5CE20", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)

    joined = gpd.sjoin(buffers, padus_proj, how="left", predicate="intersects")
    counts = (
        joined.groupby("ZCTA5CE20")
        .apply(lambda g: g["index_right"].notna().sum())
        .reset_index(name="parks_within_50km")
    )

    result = nearest[["ZCTA5CE20", "dist_nearest_park_km"]].merge(
        counts, on="ZCTA5CE20", how="left"
    )
    result["parks_within_50km"] = result["parks_within_50km"].fillna(0).astype(int)
    result = result.rename(columns={"ZCTA5CE20": "zcta"})

    result.to_parquet(cache, index=False)
    logger.success(f"Hiking scores cached for {len(result):,} ZCTAs")
    return result


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
        logger.info(f"--- {region.upper()} ({len(df):,} candidates) ---")

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
        logger.info(f"Amenity hard filter: {pre:,} → {len(df):,} ZCTAs")

    # Affordability signals
    zillow = fetch_zillow_zhvi()
    hud = fetch_hud_fmr()
    crosswalk = fetch_zcta_county_crosswalk()

    # Hiking scores — computed once across all candidates combined
    all_candidates = pd.concat([
        pd.read_parquet(DATA_PROCESSED / f"{region}_walkscored.parquet")
        for region in ("north", "south")
        if (DATA_PROCESSED / f"{region}_walkscored.parquet").exists()
    ], ignore_index=True).drop_duplicates(subset="ZCTA5CE20")

    try:
        hiking = fetch_hiking_scores(all_candidates)
    except Exception as exc:
        logger.warning(f"Could not compute hiking scores ({exc}) — skipping hiking signal.")
        hiking = pd.DataFrame(columns=["zcta", "dist_nearest_park_km", "parks_within_50km"])

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

        # Hiking scores
        if not hiking.empty:
            df = df.merge(
                hiking.rename(columns={"zcta": "ZCTA5CE20"}),
                on="ZCTA5CE20", how="left"
            )
            logger.info(
                f"{region.capitalize()}: hiking scores joined for "
                f"{df['dist_nearest_park_km'].notna().sum():,}/{len(df):,} ZCTAs"
            )

        if region == "north" and not zillow.empty:
            df["ZCTA5CE20_str"] = df["ZCTA5CE20"].astype(str).str.zfill(5)
            df = df.merge(
                zillow.rename(columns={"zip": "ZCTA5CE20_str"}),
                on="ZCTA5CE20_str", how="left"
            ).drop(columns=["ZCTA5CE20_str"], errors="ignore")
            logger.info(f"North: Zillow ZHVI joined for {df['zhvi_latest'].notna().sum():,}/{len(df):,} ZCTAs")

        if region == "south" and not hud.empty and not crosswalk.empty:
            df["ZCTA5CE20_str"] = df["ZCTA5CE20"].astype(str).str.zfill(5)
            df = df.merge(
                crosswalk.rename(columns={"zcta": "ZCTA5CE20_str"}),
                on="ZCTA5CE20_str", how="left"
            )
            df = df.merge(hud, left_on="county_fips", right_on="fips_county", how="left")
            df = df.drop(columns=["ZCTA5CE20_str", "county_fips", "fips_county"], errors="ignore")
            logger.info(f"South: HUD FMR joined for {df['fmr_2br'].notna().sum():,}/{len(df):,} ZCTAs")

        out_path = DATA_PROCESSED / f"{region}_amenities.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    run()

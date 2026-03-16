"""
Step 4: OSM Amenity Verification
---------------------------------
Counts key POI types within walking/biking distance of each candidate
centroid using the Overpass API (via osmnx). Results are cached.

Also pulls affordability data:
  - North (buy): Census ACS median home value (B25077) by block group
  - South (rent): HUD Fair Market Rents 2BR by county (extracted from GEOID)

And hiking access data:
  - PAD-US (USGS Protected Areas Database): distance to nearest qualifying
    protected area (≥ HIKING_MIN_ACRES, GAP status 1–3) + count within
    HIKING_RADIUS_KM.

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
import osmium
import osmnx as ox
import pandas as pd
import requests
import shapely.wkb as wkblib
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
    PBF_DIR,
    GEOFABRIK_BASE_URL,
    FIPS_TO_GEOFABRIK,
)

_wkbfab = osmium.geom.WKBFactory()

# OSM tags to query
POI_TAGS = {
    "amenity": ["cafe", "restaurant", "fast_food", "pharmacy", "bar", "pub"],
    "shop": ["supermarket", "grocery", "convenience", "greengrocer"],
}

# Census ACS median home value (B25077_001E) at block group level
ACS_HOME_VALUE_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B25077_001E"
    "&for=block%20group:*&in=state:{fips}&in=county:*&in=tract:*"
)

# HUD Fair Market Rents (FY2024) — county-level, 2BR
HUD_FMR_URL = (
    "https://www.huduser.gov/portal/datasets/fmr/fmr2024/FY2024_4050_FMR.xlsx"
)


# ---------------------------------------------------------------------------
# Geofabrik PBF download + offline POI extraction
# ---------------------------------------------------------------------------

class _POIHandler(osmium.SimpleHandler):
    """Extract POI nodes and area centroids from a PBF file."""

    def __init__(self):
        super().__init__()
        self.pois: list[tuple[float, float, str]] = []

    @staticmethod
    def _classify(tags) -> str | None:
        """Map OSM tags to our POI category."""
        shop = tags.get("shop")
        if shop in ("supermarket", "grocery", "greengrocer"):
            return "grocery"
        amenity = tags.get("amenity")
        if amenity == "pharmacy":
            return "pharmacy"
        if amenity == "cafe":
            return "cafe"
        if amenity in ("restaurant", "fast_food", "bar", "pub"):
            return "restaurant"
        return None

    def node(self, n):
        cat = self._classify(n.tags)
        if cat and n.location.valid():
            self.pois.append((n.location.lon, n.location.lat, cat))

    def area(self, a):
        cat = self._classify(a.tags)
        if cat is None:
            return
        try:
            wkb = _wkbfab.create_multipolygon(a)
            poly = wkblib.loads(wkb, hex=True)
            centroid = poly.representative_point()
            self.pois.append((centroid.x, centroid.y, cat))
        except Exception:
            pass


def _download_state_pbf(state_fips: str) -> Path:
    """Download a state PBF from Geofabrik, skip if cached."""
    state_name = FIPS_TO_GEOFABRIK.get(state_fips)
    if state_name is None:
        raise ValueError(f"Unknown state FIPS: {state_fips}")

    PBF_DIR.mkdir(parents=True, exist_ok=True)
    local_path = PBF_DIR / f"{state_name}-latest.osm.pbf"

    if local_path.exists():
        logger.debug(f"PBF cached: {local_path.name}")
        return local_path

    url = f"{GEOFABRIK_BASE_URL}/{state_name}-latest.osm.pbf"
    logger.info(f"Downloading {state_name}-latest.osm.pbf …")

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(local_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)

    size_mb = local_path.stat().st_size / 1e6
    logger.success(f"Downloaded {local_path.name} ({size_mb:.0f} MB)")
    return local_path


def _extract_pois_from_pbf(pbf_path: Path) -> pd.DataFrame:
    """Extract relevant POIs from a PBF file. Returns [lon, lat, category]."""
    handler = _POIHandler()
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")

    if not handler.pois:
        return pd.DataFrame(columns=["lon", "lat", "category"])

    return pd.DataFrame(handler.pois, columns=["lon", "lat", "category"])


def _get_state_pois(state_fips: str, pbf_path: Path) -> pd.DataFrame:
    """Get POIs for a state, using per-state cache if available."""
    cache_dir = PBF_DIR / "poi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{state_fips}_pois.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info(f"Parsing POIs from {pbf_path.name} …")
    pois = _extract_pois_from_pbf(pbf_path)
    pois.to_parquet(cache_path, index=False)
    logger.debug(f"  State {state_fips}: {len(pois):,} POIs cached")
    return pois


def _count_pois_spatial(
    candidates: pd.DataFrame,
    state_fips_list: list[str],
    pbf_paths: dict[str, Path],
) -> pd.DataFrame:
    """
    Count POIs within POI_RADIUS_METERS of each candidate via spatial join.
    Returns [geoid, grocery_count, pharmacy_count, cafe_count, restaurant_count].
    """
    # Combine POIs from all needed states
    poi_frames = []
    for fips in state_fips_list:
        if fips in pbf_paths:
            poi_frames.append(_get_state_pois(fips, pbf_paths[fips]))

    if not poi_frames:
        logger.warning("No POI data from any PBF file.")
        return pd.DataFrame({
            "geoid": candidates["GEOID"],
            "grocery_count": 0,
            "pharmacy_count": 0,
            "cafe_count": 0,
            "restaurant_count": 0,
        })

    all_pois = pd.concat(poi_frames, ignore_index=True)
    logger.info(f"Total POIs across {len(state_fips_list)} states: {len(all_pois):,}")

    # Build GeoDataFrames
    candidate_gdf = gpd.GeoDataFrame(
        candidates[["GEOID"]].copy(),
        geometry=gpd.points_from_xy(
            candidates["centroid_lon"], candidates["centroid_lat"]
        ),
        crs="EPSG:4326",
    )
    poi_gdf = gpd.GeoDataFrame(
        all_pois[["category"]],
        geometry=gpd.points_from_xy(all_pois["lon"], all_pois["lat"]),
        crs="EPSG:4326",
    )

    # Reproject to Albers Equal Area (meters)
    crs_proj = "EPSG:5070"
    candidate_gdf = candidate_gdf.to_crs(crs_proj)
    poi_gdf = poi_gdf.to_crs(crs_proj)

    # Buffer candidates by search radius
    candidate_gdf["geometry"] = candidate_gdf.geometry.buffer(POI_RADIUS_METERS)

    # Spatial join
    logger.info("Spatial join: counting POIs per candidate …")
    joined = gpd.sjoin(candidate_gdf, poi_gdf, how="left", predicate="contains")

    # Pivot to counts per category
    counts = (
        joined[joined["index_right"].notna()]
        .groupby(["GEOID", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["grocery", "pharmacy", "cafe", "restaurant"], fill_value=0)
        .reset_index()
    )

    # Ensure all candidates present
    result = candidates[["GEOID"]].merge(counts, on="GEOID", how="left").fillna(0)
    result = result.rename(columns={
        "GEOID": "geoid",
        "grocery": "grocery_count",
        "pharmacy": "pharmacy_count",
        "cafe": "cafe_count",
        "restaurant": "restaurant_count",
    })
    for col in ["grocery_count", "pharmacy_count", "cafe_count", "restaurant_count"]:
        result[col] = result[col].astype(int)

    return result


# ---------------------------------------------------------------------------
# Overpass fallback (used only if all PBF downloads fail)
# ---------------------------------------------------------------------------

def _count_pois_overpass(lat: float, lon: float) -> dict:
    """Count POI categories within POI_RADIUS_METERS of a point via osmnx."""
    try:
        pois = ox.features_from_point((lat, lon), tags=POI_TAGS, dist=POI_RADIUS_METERS)
    except Exception:
        pois = gpd.GeoDataFrame()

    if pois.empty:
        return {
            "grocery_count": 0,
            "pharmacy_count": 0,
            "cafe_count": 0,
            "restaurant_count": 0,
        }

    grocery_mask = pois.get("shop", pd.Series(dtype=str)).isin(
        ["supermarket", "grocery", "greengrocer"]
    )
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


def _fetch_amenities_overpass(
    todo: pd.DataFrame, cached: pd.DataFrame, cache: Path, region: str
) -> pd.DataFrame:
    """Fallback: fetch amenities via Overpass API (slow, one request per candidate)."""
    results = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"OSM {region} (Overpass)"):
        counts = _count_pois_overpass(row["centroid_lat"], row["centroid_lon"])
        counts["geoid"] = row["GEOID"]
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
# Main amenity fetch (PBF with Overpass fallback)
# ---------------------------------------------------------------------------

def fetch_amenities_for_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Fetch and cache OSM amenity counts for all candidates."""
    cache = DATA_RAW / f"{region}_osm_cache.parquet"

    if cache.exists():
        cached = pd.read_parquet(cache)
        done_ids = set(cached["geoid"].astype(str))
    else:
        cached = pd.DataFrame()
        done_ids = set()

    todo = df[~df["GEOID"].astype(str).isin(done_ids)]
    logger.debug(f"{region}: {len(done_ids)} cached, {len(todo)} remaining")

    if todo.empty:
        return cached

    # Determine which states need PBF files
    state_fips_list = sorted(todo["GEOID"].astype(str).str[:2].unique())
    logger.info(f"{region}: downloading PBFs for {len(state_fips_list)} states")

    pbf_paths: dict[str, Path] = {}
    for fips in tqdm(state_fips_list, desc="Downloading PBFs"):
        try:
            pbf_paths[fips] = _download_state_pbf(fips)
        except Exception as e:
            logger.warning(f"PBF download failed for state {fips}: {e}")

    if not pbf_paths:
        logger.warning("All PBF downloads failed — falling back to Overpass API")
        return _fetch_amenities_overpass(todo, cached, cache, region)

    # Count POIs via offline spatial join
    new_counts = _count_pois_spatial(todo, state_fips_list, pbf_paths)

    # Merge with existing cache
    if not cached.empty:
        combined = pd.concat([cached, new_counts], ignore_index=True)
    else:
        combined = new_counts

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache, index=False)
    return combined


# ---------------------------------------------------------------------------
# Census ACS median home value (north — buying)
# ---------------------------------------------------------------------------

def fetch_acs_home_value(state_fips_list: list[str]) -> pd.DataFrame:
    """
    Download Census ACS B25077_001E (median home value) by block group.
    Returns DataFrame with columns: GEOID, median_home_value
    """
    cache = DATA_RAW / "bg_home_value.parquet"
    if cache.exists():
        logger.debug("Census ACS home value data already cached.")
        return pd.read_parquet(cache)

    logger.info("Fetching ACS median home value for block groups…")
    frames = []
    for fips in tqdm(state_fips_list, desc="ACS home value"):
        url = ACS_HOME_VALUE_URL.format(fips=fips)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"ACS home value fetch failed for state {fips}: {e}")
            continue

        cols = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)

    if not frames:
        logger.warning("No ACS home value data fetched — skipping.")
        return pd.DataFrame(columns=["GEOID", "median_home_value"])

    hv = pd.concat(frames, ignore_index=True)
    # Build GEOID: state(2) + county(3) + tract(6) + block group(1)
    hv["GEOID"] = hv["state"] + hv["county"] + hv["tract"] + hv["block group"]
    hv["median_home_value"] = pd.to_numeric(hv["B25077_001E"], errors="coerce")
    # Census uses negative sentinel values for suppressed data
    hv.loc[hv["median_home_value"] < 0, "median_home_value"] = float("nan")
    hv = hv[["GEOID", "median_home_value"]]

    hv.to_parquet(cache, index=False)
    logger.debug(f"Cached home values for {len(hv):,} block groups")
    return hv


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
    Compute hiking access metrics for all candidate block groups.

    Returns DataFrame with columns:
      geoid, dist_nearest_park_km, parks_within_50km
    """
    cache = DATA_RAW / "hiking_scores.parquet"
    if cache.exists():
        logger.debug("Hiking scores already cached.")
        return pd.read_parquet(cache)

    padus = _load_padus_filtered()

    candidates = gpd.GeoDataFrame(
        df[["GEOID", "centroid_lat", "centroid_lon"]].copy(),
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
        candidates_proj[["GEOID", "geometry"]],
        padus_proj,
        how="left",
        distance_col="dist_nearest_park_m",
    ).drop_duplicates(subset="GEOID")
    nearest["dist_nearest_park_km"] = (nearest["dist_nearest_park_m"] / 1000).round(2)

    # Count protected areas within HIKING_RADIUS_KM
    logger.info(f"Counting protected areas within {HIKING_RADIUS_KM} km…")
    radius_m = HIKING_RADIUS_KM * 1000
    buffers = candidates_proj[["GEOID", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)

    joined = gpd.sjoin(buffers, padus_proj, how="left", predicate="intersects")
    counts = (
        joined.groupby("GEOID")
        .apply(lambda g: g["index_right"].notna().sum())
        .reset_index(name="parks_within_50km")
    )

    result = nearest[["GEOID", "dist_nearest_park_km"]].merge(
        counts, on="GEOID", how="left"
    )
    result["parks_within_50km"] = result["parks_within_50km"].fillna(0).astype(int)
    result = result.rename(columns={"GEOID": "geoid"})

    result.to_parquet(cache, index=False)
    logger.success(f"Hiking scores cached for {len(result):,} block groups")
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
            amenity_scores[["geoid", "grocery_count", "pharmacy_count",
                            "cafe_count", "restaurant_count"]],
            left_on="GEOID", right_on="geoid", how="left"
        ).drop(columns=["geoid"], errors="ignore")

        # Hard filter: must have at least one grocery and one pharmacy
        pre = len(df)
        df = df[
            (df["grocery_count"] >= MIN_GROCERY_COUNT) &
            (df["pharmacy_count"] >= MIN_PHARMACY_COUNT)
        ]
        logger.info(f"Amenity hard filter: {pre:,} → {len(df):,} block groups")

    # Affordability signals
    # Determine which state FIPS codes are needed from north candidates
    north_ws = pd.read_parquet(DATA_PROCESSED / "north_walkscored.parquet")
    north_states = sorted(north_ws["GEOID"].astype(str).str[:2].unique())

    home_values = fetch_acs_home_value(north_states)
    hud = fetch_hud_fmr()

    # Hiking scores — computed once across all candidates combined
    all_candidates = pd.concat([
        pd.read_parquet(DATA_PROCESSED / f"{region}_walkscored.parquet")
        for region in ("north", "south")
        if (DATA_PROCESSED / f"{region}_walkscored.parquet").exists()
    ], ignore_index=True).drop_duplicates(subset="GEOID")

    try:
        hiking = fetch_hiking_scores(all_candidates)
    except Exception as exc:
        logger.warning(f"Could not compute hiking scores ({exc}) — skipping hiking signal.")
        hiking = pd.DataFrame(columns=["geoid", "dist_nearest_park_km", "parks_within_50km"])

    for region in ("north", "south"):
        in_path = DATA_PROCESSED / f"{region}_walkscored.parquet"
        df = pd.read_parquet(in_path)

        # Re-apply amenity merge (in case we restarted)
        amenity_cache = DATA_RAW / f"{region}_osm_cache.parquet"
        if amenity_cache.exists():
            amenity_scores = pd.read_parquet(amenity_cache)
            df = df.merge(
                amenity_scores[["geoid", "grocery_count", "pharmacy_count",
                                "cafe_count", "restaurant_count"]],
                left_on="GEOID", right_on="geoid", how="left"
            ).drop(columns=["geoid"], errors="ignore")
            df = df[
                (df["grocery_count"] >= MIN_GROCERY_COUNT) &
                (df["pharmacy_count"] >= MIN_PHARMACY_COUNT)
            ]

        # Hiking scores
        if not hiking.empty:
            df = df.merge(
                hiking.rename(columns={"geoid": "GEOID"}),
                on="GEOID", how="left"
            )
            logger.info(
                f"{region.capitalize()}: hiking scores joined for "
                f"{df['dist_nearest_park_km'].notna().sum():,}/{len(df):,} block groups"
            )

        if region == "north" and not home_values.empty:
            df = df.merge(home_values, on="GEOID", how="left")
            logger.info(
                f"North: Census ACS home value joined for "
                f"{df['median_home_value'].notna().sum():,}/{len(df):,} block groups"
            )

        if region == "south" and not hud.empty:
            # Extract county FIPS (first 5 digits of block group GEOID)
            df["fips_county"] = df["GEOID"].astype(str).str[:5]
            df = df.merge(hud, on="fips_county", how="left")
            df = df.drop(columns=["fips_county"], errors="ignore")
            logger.info(
                f"South: HUD FMR joined for "
                f"{df['fmr_2br'].notna().sum():,}/{len(df):,} block groups"
            )

        out_path = DATA_PROCESSED / f"{region}_amenities.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"Saved {out_path}")


if __name__ == "__main__":
    run()

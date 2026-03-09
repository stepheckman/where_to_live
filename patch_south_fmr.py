"""
One-shot script: fetch HUD FMR + ZCTA→county crosswalk and patch
south_amenities.parquet with fmr_2br column.

Safe to re-run — caches both downloads locally.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.log import setup as setup_logging

setup_logging(log_file=Path(__file__).parent / "pipeline.log")

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

HUD_FMR_URL = (
    "https://www.huduser.gov/portal/datasets/fmr/fmr2024/FMR2024_final_revised.xlsx"
)
CENSUS_CROSSWALK_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "zcta520/tab20_zcta520_county20_natl.txt"
)


def fetch_hud_fmr() -> pd.DataFrame:
    cache = DATA_RAW / "hud_fmr.parquet"
    if cache.exists():
        logger.debug("HUD FMR: using cache")
        return pd.read_parquet(cache)

    logger.info(f"Downloading HUD FMR from {HUD_FMR_URL} …")
    # timeout=(connect_secs, read_secs) — read timeout fires if no new bytes arrive
    resp = requests.get(HUD_FMR_URL, timeout=(10, 30), stream=True)
    resp.raise_for_status()
    buf = io.BytesIO()
    total = int(resp.headers.get("content-length", 0))
    received = 0
    for chunk in resp.iter_content(chunk_size=1 << 16):
        buf.write(chunk)
        received += len(chunk)
    logger.debug(f"Downloaded {received:,} bytes")
    buf.seek(0)
    # openpyxl crashes on this HUD file because docProps/core.xml contains a
    # malformed datetime ("2024- 1-24T19: 8: 0Z"). Strip that entry from the zip
    # in-memory — openpyxl skips properties parsing when the file is absent.
    clean = io.BytesIO()
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(clean, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename != "docProps/core.xml":
                zout.writestr(item, zin.read(item.filename))
    clean.seek(0)
    df = pd.read_excel(clean)
    logger.debug(f"HUD FMR columns: {list(df.columns)}")

    fips_col = next((c for c in df.columns if "fips" in c), None)
    fmr_col  = next((c for c in df.columns if "fmr_2" in c or "fmr2" in c), None)
    if not fips_col or not fmr_col:
        logger.error(f"Couldn't find fips/fmr cols. Got: {list(df.columns)}")
        sys.exit(1)

    result = df[[fips_col, fmr_col]].rename(columns={fips_col: "fips_county", fmr_col: "fmr_2br"})
    result["fips_county"] = result["fips_county"].astype(str).str[:5].str.zfill(5)
    result["fmr_2br"] = pd.to_numeric(result["fmr_2br"], errors="coerce")
    result.to_parquet(cache, index=False)
    logger.debug(f"Cached {len(result):,} counties")
    return result


def fetch_crosswalk() -> pd.DataFrame:
    cache = DATA_RAW / "zcta_county_crosswalk.parquet"
    if cache.exists():
        logger.debug("Crosswalk: using cache")
        return pd.read_parquet(cache)

    logger.info("Downloading ZCTA→county crosswalk…")
    resp = requests.get(CENSUS_CROSSWALK_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str)
    df = df.rename(columns={"GEOID_ZCTA5_20": "zcta", "GEOID_COUNTY_20": "county_fips"})
    df = df[["zcta", "county_fips"]].dropna().drop_duplicates(subset="zcta", keep="first")
    df.to_parquet(cache, index=False)
    logger.debug(f"Cached {len(df):,} ZCTAs")
    return df


def patch_south():
    south_path = DATA_PROCESSED / "south_amenities.parquet"
    if not south_path.exists():
        logger.error(f"{south_path} not found — run pipeline step 4 first")
        sys.exit(1)

    south = pd.read_parquet(south_path)
    if "fmr_2br" in south.columns and south["fmr_2br"].notna().any():
        logger.info("south_amenities.parquet already has fmr_2br — nothing to do")
        return

    hud = fetch_hud_fmr()
    crosswalk = fetch_crosswalk()

    south["zcta_str"] = south["ZCTA5CE20"].astype(str).str.zfill(5)
    south = south.merge(
        crosswalk.rename(columns={"zcta": "zcta_str"}),
        on="zcta_str", how="left"
    ).merge(hud, left_on="county_fips", right_on="fips_county", how="left")
    south = south.drop(columns=["zcta_str", "county_fips", "fips_county"], errors="ignore")

    matched = south["fmr_2br"].notna().sum()
    logger.info(f"fmr_2br matched for {matched:,}/{len(south):,} ZCTAs")
    if matched == 0:
        logger.error("Zero matches — check FIPS join logic")
        sys.exit(1)

    south.to_parquet(south_path, index=False)
    logger.success(f"Patched and saved {south_path}")


if __name__ == "__main__":
    patch_south()

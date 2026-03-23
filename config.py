"""
Central configuration for the where-to-live pipeline.
Adjust thresholds and weights here; no code changes needed.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MAPS = OUTPUTS / "maps"

# ---------------------------------------------------------------------------
# Geographic filters
# ---------------------------------------------------------------------------
# North band: warmer months (buying)
NORTH_LAT_MIN = 38.0
NORTH_LAT_MAX = 49.0

# South band: cooler months (renting)
SOUTH_LAT_MIN = 25.0
SOUTH_LAT_MAX = 35.0

# Minimum population density to exclude rural/exurban block groups
MIN_POP_DENSITY = 2000  # people per sq mile

# ---------------------------------------------------------------------------
# Airport proximity
# ---------------------------------------------------------------------------
MAX_AIRPORT_DRIVE_MIN = 60   # minutes
# Airport hub types to include (FAA classification)
AIRPORT_HUB_TYPES = {"Large", "Medium", "Small"}

# ---------------------------------------------------------------------------
# Walkability hard filters (block groups below these are eliminated before scoring)
# ---------------------------------------------------------------------------
MIN_WALK_SCORE = 65
MIN_BIKE_SCORE = 50

# ---------------------------------------------------------------------------
# OSM amenity search
# ---------------------------------------------------------------------------
POI_RADIUS_METERS = 1200   # ~15-min walk
TRANSIT_RADIUS_METERS = 1609  # ~1 mile

# Hard minimums (block group eliminated if not met)
MIN_GROCERY_COUNT = 1
MIN_PHARMACY_COUNT = 1

# Soft scoring caps (counts above cap get same score as cap)
CAP_GROCERY = 5
CAP_CAFE = 10
CAP_RESTAURANT = 30
CAP_PHARMACY = 3
CAP_TRANSIT = 20
CAP_BARS = 15         # 15+ bars/pubs/nightclubs within ¾ mi = worst score

# ---------------------------------------------------------------------------
# Hiking / protected areas (PAD-US)
# ---------------------------------------------------------------------------
# Minimum size of a protected area to count as a "hiking destination"
HIKING_MIN_ACRES = 500
# Radius within which to count qualifying protected areas
HIKING_RADIUS_KM = 50
# GAP status codes to include: 1=permanent protection, 2=permanent/natural,
# 3=multi-use (includes state parks, national forests, BLM)
HIKING_GAP_STATUS = {"1", "2", "3"}

# ---------------------------------------------------------------------------
# Composite score weights — must sum to 1.0
# NOTE: weights are approximate starting points; adjust to taste.
# ---------------------------------------------------------------------------
# Base walkability signals
W_WALK = 0.28        # reduced by 0.02 to make room for bars signal
W_BIKE = 0.13        # reduced by 0.02 to make room for bars signal
W_GROCERY = 0.20
W_CAFE = 0.025      # reduced to make room for transit
W_RESTAURANT = 0.025  # reduced to make room for transit
W_PHARMACY = 0.05
W_TRANSIT = 0.05     # public transit stops within 1 mile
W_HIKING = 0.10      # PAD-US protected area proximity

# Affordability signal (Zillow ZHVI median home value) — lower is better
# Used for both north and south regions
W_HOME_VALUE = 0.09  # reduced by 0.01 to make room for bars signal

# Nightlife density — INVERTED: more bars/pubs/nightclubs = worse score
# 0 bars → score 1.0; CAP_BARS+ bars → score 0.0
W_BARS = 0.05

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TOP_N = 25   # candidates per region in final ranked output

# ---------------------------------------------------------------------------
# Climate verification (optional, step 05 enhancement)
# ---------------------------------------------------------------------------
# North: confirm comfortable summer conditions
NORTH_JULY_HIGH_MIN_F = 70
NORTH_JULY_LOW_MIN_F = 55

# South: confirm comfortable winter conditions
SOUTH_JAN_HIGH_MIN_F = 55
SOUTH_JAN_LOW_MIN_F = 40

# ---------------------------------------------------------------------------
# Walk Score scraping (no API key needed)
# ---------------------------------------------------------------------------
SCRAPE_TOP_N = 75          # legacy — no longer used as cutoff (all candidates scraped)
SCRAPE_SLEEP_S = 3.0       # seconds between scrape requests (be polite)

# ---------------------------------------------------------------------------
# API rate limiting
# ---------------------------------------------------------------------------
WALKSCORE_SLEEP_S = 0.2    # 5 req/sec max on free tier (unused if scraping)
OSM_SLEEP_S = 1.0          # be polite to Overpass

# ---------------------------------------------------------------------------
# Geofabrik PBF downloads (offline OSM — replaces Overpass for step 4)
# ---------------------------------------------------------------------------
PBF_DIR = DATA_RAW / "pbf"
GEOFABRIK_BASE_URL = "https://download.geofabrik.de/north-america/us"

FIPS_TO_GEOFABRIK: dict[str, str] = {
    "01": "alabama",        "02": "alaska",         "04": "arizona",
    "05": "arkansas",       "06": "california",     "08": "colorado",
    "09": "connecticut",    "10": "delaware",       "11": "district-of-columbia",
    "12": "florida",        "13": "georgia",        "15": "hawaii",
    "16": "idaho",          "17": "illinois",       "18": "indiana",
    "19": "iowa",           "20": "kansas",         "21": "kentucky",
    "22": "louisiana",      "23": "maine",          "24": "maryland",
    "25": "massachusetts",  "26": "michigan",       "27": "minnesota",
    "28": "mississippi",    "29": "missouri",       "30": "montana",
    "31": "nebraska",       "32": "nevada",         "33": "new-hampshire",
    "34": "new-jersey",     "35": "new-mexico",     "36": "new-york",
    "37": "north-carolina", "38": "north-dakota",   "39": "ohio",
    "40": "oklahoma",       "41": "oregon",         "42": "pennsylvania",
    "44": "rhode-island",   "45": "south-carolina", "46": "south-dakota",
    "47": "tennessee",      "48": "texas",          "49": "utah",
    "50": "vermont",        "51": "virginia",       "53": "washington",
    "54": "west-virginia",  "55": "wisconsin",      "56": "wyoming",
}

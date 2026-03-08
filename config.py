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
NORTH_LAT_MIN = 40.0
NORTH_LAT_MAX = 49.0

# South band: cooler months (renting)
SOUTH_LAT_MIN = 25.0
SOUTH_LAT_MAX = 35.0

# Minimum population density to exclude rural/exurban ZCTAs
MIN_POP_DENSITY = 2000  # people per sq mile

# ---------------------------------------------------------------------------
# Airport proximity
# ---------------------------------------------------------------------------
MAX_AIRPORT_DRIVE_MIN = 60   # minutes
# Airport hub types to include (FAA classification)
AIRPORT_HUB_TYPES = {"Large", "Medium", "Small"}

# ---------------------------------------------------------------------------
# Walkability hard filters (ZCTAs below these are eliminated before scoring)
# ---------------------------------------------------------------------------
MIN_WALK_SCORE = 65
MIN_BIKE_SCORE = 50

# ---------------------------------------------------------------------------
# OSM amenity search
# ---------------------------------------------------------------------------
POI_RADIUS_METERS = 1200   # ~15-min walk

# Hard minimums (ZCTA eliminated if not met)
MIN_GROCERY_COUNT = 1
MIN_PHARMACY_COUNT = 1

# Soft scoring caps (counts above cap get same score as cap)
CAP_GROCERY = 5
CAP_CAFE = 10
CAP_RESTAURANT = 30
CAP_PHARMACY = 3

# ---------------------------------------------------------------------------
# Composite score weights — must sum to 1.0
# ---------------------------------------------------------------------------
# Base walkability signals
W_WALK = 0.30
W_BIKE = 0.15
W_GROCERY = 0.20
W_CAFE = 0.10
W_RESTAURANT = 0.10
W_PHARMACY = 0.05

# Region-specific affordability signals (remaining 0.10)
# North: home value (Zillow ZHVI) — lower is better
W_HOME_VALUE = 0.10
# South: rent affordability (HUD FMR 2BR) — lower is better
W_RENT = 0.10

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
# API rate limiting
# ---------------------------------------------------------------------------
WALKSCORE_SLEEP_S = 0.2    # 5 req/sec max on free tier
OSM_SLEEP_S = 1.0          # be polite to Overpass

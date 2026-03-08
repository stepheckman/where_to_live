# Retirement Location Analysis: Finding 15-Minute Neighborhoods

## Project Goal

Identify ZIP Code Tabulation Areas (ZCTAs) or census tracts in:
- A **northern U.S. location** (≥40°N) suitable for warmer months
- A **southern U.S. location** (≤35°N) suitable for cooler months

Both must satisfy:
1. Strong walkability/bikeability — groceries, pharmacy, restaurants, coffee within a 15-minute walk or bike ride
2. Within a 60-minute drive of a commercial-service airport
3. Urban or dense urban-adjacent character (not suburban sprawl)

---

## Repository Structure

```
retirement-search/
├── README.md
├── requirements.txt
├── config.py                  # lat bands, thresholds, API keys
├── data/
│   ├── raw/                   # downloaded source files
│   └── processed/             # scored, filtered outputs
├── pipeline/
│   01_geo_filter.py           # ZCTA lat/lon pre-filter
│   02_airport_proximity.py    # drive-time isochrones to airports
│   03_walkscore.py            # Walk Score + Bike Score API calls
│   04_osm_amenities.py        # OSM POI counts via Overpass
│   05_score_and_rank.py       # composite scoring + ranking
│   06_visualize.py            # folium maps of top candidates
└── outputs/
    ├── north_candidates.csv
    ├── south_candidates.csv
    └── maps/
```

---

## Dependencies

```
# requirements.txt
geopandas
shapely
pandas
numpy
requests
overpy           # Overpass API (OSM)
osmnx            # street network + isochrones
folium           # interactive maps
tqdm             # progress bars
python-dotenv    # API key management
```

API keys needed (store in `.env`, never commit):
- `WALKSCORE_API_KEY` — https://www.walkscore.com/professional/api.php (free tier available)
- `ORS_API_KEY` — https://openrouteservice.org (free tier: 2000 req/day) — used for drive-time isochrones

---

## Step 1: Geographic Pre-filter (`01_geo_filter.py`)

**Goal:** Reduce the universe of ~33,000 ZCTAs to plausible candidates by latitude and basic population density.

**Data source:** Census TIGER/Line ZCTAs
- Download URL: https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip
- Alternatively use `censusdatadownloader` or direct Census API

**Steps:**
1. Load ZCTA shapefile with `geopandas`
2. Compute centroid lat/lon for each ZCTA
3. Filter to two latitude bands:
   - **North:** 40°N – 49°N (excludes Alaska; captures upper Midwest, Northeast, Pacific NW)
   - **South:** 25°N – 35°N (Gulf Coast, Southwest, Carolinas, Central Florida)
4. Join with Census population data (ACS 5-year) to filter out low-density ZCTAs
   - Minimum population density threshold: **~2,000 people/sq mile** (eliminates rural/exurban areas unlikely to be walkable)
   - ACS data via `censusdatadownloader` or Census API (`B01003_001E` total population + ZCTA area)
5. Output: `data/processed/north_candidates_geo.geojson`, `south_candidates_geo.geojson`

**Config parameters:**
```python
NORTH_LAT_MIN = 40.0
NORTH_LAT_MAX = 49.0
SOUTH_LAT_MIN = 25.0
SOUTH_LAT_MAX = 35.0
MIN_POP_DENSITY = 2000  # people per sq mile
```

---

## Step 2: Airport Proximity Filter (`02_airport_proximity.py`)

**Goal:** Retain only ZCTAs whose centroid is within a 60-minute drive of a commercial-service airport.

**Data sources:**
- FAA NPIAS (National Plan of Integrated Airport Systems) — identifies commercial service airports
  - Download: https://www.faa.gov/airports/planning_capacity/npias/reports
  - Or use the FAA airport data CSV: https://adds-faa.opendata.arcgis.com
- Filter to `service_level == "Commercial Service"` and `hub_type` in `["Large", "Medium", "Small"]`

**Approach — Option A (API-based, more accurate):**
1. For each candidate ZCTA centroid, find all commercial airports within 100 miles (straight-line pre-filter)
2. Call OpenRouteService Matrix API to get actual drive times
3. Keep ZCTAs where `min(drive_time_to_any_airport) <= 60 minutes`

**Approach — Option B (offline, faster):**
1. Use `osmnx` to build 60-min drive isochrone polygons around each airport
2. Spatial join: keep ZCTAs whose centroid falls within any airport's isochrone
3. This avoids per-ZCTA API calls (better for large batches)

**Recommendation:** Use Option B — build isochrones once per airport, then do a single spatial join. Much cheaper on API quota.

**Output:** Filtered GeoJSON files with airport proximity metadata added.

---

## Step 3: Walk Score + Bike Score (`03_walkscore.py`)

**Goal:** Get Walk Score and Bike Score for each remaining candidate centroid.

**API:** `https://api.walkscore.com/score?format=json&lat={lat}&lon={lon}&wsapikey={key}`

**Steps:**
1. For each candidate ZCTA centroid, call the Walk Score API
2. Parse `walkscore`, `bikescore` from response
3. Apply thresholds:
   - Walk Score ≥ 70 ("Very Walkable" or better) — can relax to 65 if candidates are sparse
   - Bike Score ≥ 60
4. Rate-limit: Walk Score API allows ~5,000 calls/day on free tier; add `time.sleep(0.2)` between calls
5. Cache results to CSV to avoid re-calling

**Score interpretation:**
| Score | Category |
|-------|----------|
| 90–100 | Walker's Paradise |
| 70–89 | Very Walkable |
| 50–69 | Somewhat Walkable |
| 25–49 | Car-Dependent |

**Output:** `data/processed/north_walkscored.csv`, `south_walkscored.csv`

---

## Step 4: OSM Amenity Verification (`04_osm_amenities.py`)

**Goal:** Directly count specific POI types within walking/biking distance of each candidate centroid. Validates and supplements Walk Score.

**Library:** `overpy` (Overpass API) or `osmnx.features_from_point()`

**Search radius:** 1,200 meters (≈15-min walk at average pace)

**POI categories to query:**

| Category | OSM tags |
|----------|----------|
| Grocery store | `shop=supermarket`, `shop=grocery` |
| Pharmacy | `amenity=pharmacy` |
| Coffee | `amenity=cafe` |
| Restaurants | `amenity=restaurant`, `amenity=fast_food` |
| Bike infrastructure | `highway=cycleway` (network length in radius) |

**For each candidate ZCTA:**
```python
import osmnx as ox

tags = {
    "amenity": ["cafe", "restaurant", "pharmacy"],
    "shop": ["supermarket", "grocery"]
}
pois = ox.features_from_point((lat, lon), tags=tags, dist=1200)
```

**Scoring:**
- Minimum hard requirements (eliminate if not met):
  - ≥1 grocery store (full-service supermarket preferred)
  - ≥1 pharmacy
- Soft scoring (higher = better):
  - Café count (target ≥5)
  - Restaurant count (target ≥15)
  - Diversity of food options

**Overpass rate limits:** Add `time.sleep(1)` between queries; cache to avoid repeat calls.

**Output:** POI counts appended to candidate CSVs.

---

## Step 5: Composite Scoring and Ranking (`05_score_and_rank.py`)

**Goal:** Combine all signals into a single ranked list for north and south separately.

**Scoring formula:**

Each ZCTA gets a score from 0–100:

```python
score = (
    0.35 * normalize(walk_score)          # walkability
  + 0.20 * normalize(bike_score)          # bikeability
  + 0.20 * normalize(grocery_count)       # grocery access (cap at 5)
  + 0.10 * normalize(cafe_count)          # coffee/café density (cap at 10)
  + 0.10 * normalize(restaurant_count)    # restaurant density (cap at 30)
  + 0.05 * normalize(pharmacy_count)      # pharmacy access (cap at 3)
)
```

Weights are adjustable in `config.py`. All inputs normalized to [0, 1] before weighting.

**Hard filters (eliminate before scoring):**
- Walk Score < 65
- Bike Score < 50
- No grocery store within 1,200m
- No pharmacy within 1,200m
- Airport drive time > 60 min

**Output:**
- `outputs/north_candidates.csv` — top 25 northern ZCTAs ranked by score
- `outputs/south_candidates.csv` — top 25 southern ZCTAs ranked by score

Include columns: `zcta`, `city`, `state`, `lat`, `lon`, `walk_score`, `bike_score`, `grocery_count`, `cafe_count`, `restaurant_count`, `pharmacy_count`, `nearest_airport`, `airport_drive_min`, `composite_score`

---

## Step 6: Visualization (`06_visualize.py`)

**Goal:** Interactive map of top candidates to enable visual inspection.

**Library:** `folium`

**For each region (north/south):**
1. Plot top 25 ZCTA centroids as circle markers
2. Color-code by composite score (green = high, yellow = medium)
3. Popup on click shows: city/state, Walk Score, Bike Score, grocery/café/restaurant counts, nearest airport + drive time
4. Add layer toggle for airport locations

```python
import folium

m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6)
for _, row in top_candidates.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=8,
        color=score_to_color(row.composite_score),
        popup=build_popup(row)
    ).add_to(m)
m.save("outputs/maps/north_candidates.html")
```

---

## Optional Enhancements

### Climate Verification
- Pull NOAA Climate Normals for each candidate city
- API: https://www.ncei.noaa.gov/cdo-web/api/v2/
- For northern candidates: confirm avg July high ≥ 70°F, avg July low ≥ 55°F
- For southern candidates: confirm avg January high ≥ 55°F, avg January low ≥ 40°F

### Street Network Quality
- Use `osmnx` to compute intersection density and average block length within candidate ZCTAs
- High intersection density + short blocks = more walkable grid (independent of POI count)
- `ox.basic_stats(G)["intersection_density_km"]`

### Elevation / Terrain
- Flat terrain improves practical bikeability beyond what Bike Score captures
- Use `elevation` via `osmnx` or the Open-Elevation API to flag hilly candidates

### Cost of Living (optional)
- Zillow ZHVI (home values by ZCTA) or HUD Fair Market Rents if cost becomes a factor
- MIT Living Wage calculator by county as a reference floor

---

## Config Reference (`config.py`)

```python
# Geographic filters
NORTH_LAT_MIN = 40.0
NORTH_LAT_MAX = 49.0
SOUTH_LAT_MIN = 25.0
SOUTH_LAT_MAX = 35.0
MIN_POP_DENSITY = 2000          # people/sq mile

# Walkability thresholds (hard filters)
MIN_WALK_SCORE = 65
MIN_BIKE_SCORE = 50
MAX_AIRPORT_DRIVE_MIN = 60

# POI search radius
POI_RADIUS_METERS = 1200

# Composite score weights (must sum to 1.0)
W_WALK = 0.35
W_BIKE = 0.20
W_GROCERY = 0.20
W_CAFE = 0.10
W_RESTAURANT = 0.10
W_PHARMACY = 0.05

# Top N results to output
TOP_N = 25
```

---

## Expected Outputs

After running the full pipeline you should have:
- Two ranked CSV files (north and south) with ~25 candidates each
- Two interactive HTML maps
- A clear shortlist of 5–10 ZCTAs per region to investigate further via street-level review (Google Street View, local real estate listings, neighborhood blogs)

The pipeline is designed to be re-run with different config parameters (e.g., loosening Walk Score threshold, adjusting latitude bands) without modifying code.
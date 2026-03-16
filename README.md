# Where to Live

Find walkable, bikeable neighborhoods for a two-location lifestyle: a **northern** home (40–49°N) for warmer months and a **southern** base (25–35°N) for cooler months.

## How it works

A six-step data pipeline scores every US Census block group in the target latitude bands using free, public data:

1. **Geo filter** — Select block groups by latitude band and minimum population density
2. **Airport proximity** — Keep only block groups within 60 min drive of a commercial airport (offline via OSMnx)
3. **Walk/Bike Score** — Scrape walkability and bikeability scores
4. **OSM amenities** — Count groceries, cafés, restaurants, pharmacies within 1200 m; fetch Zillow home values (north) and HUD Fair Market Rents (south); measure proximity to protected areas for hiking
5. **Score & rank** — Composite weighted score with configurable weights
6. **Visualize** — Folium HTML maps with Google Maps/Street View links

## Interactive app

A Streamlit app lets you adjust filters and weights with sliders — the map and table update instantly without re-running the pipeline.

```bash
uv run streamlit run app.py
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Optional: create a `.env` file with API keys:

```
WALKSCORE_API_KEY=your_key_here   # optional, falls back to scraping
```

## Running the pipeline

```bash
uv run python run_pipeline.py           # full run
uv run python run_pipeline.py --from 3  # resume from step 3
uv run python run_pipeline.py --only 4  # run just one step
```

Pipeline outputs are saved to `data/processed/` (parquet files) and `outputs/maps/` (HTML maps).

## Scoring weights

| Signal       | Default weight |
|-------------|---------------|
| Walk Score  | 0.30          |
| Bike Score  | 0.15          |
| Groceries   | 0.20          |
| Cafés       | 0.05          |
| Restaurants | 0.05          |
| Pharmacies  | 0.05          |
| Hiking      | 0.10          |
| Affordability | 0.10        |

All weights are configurable in `config.py` or via the Streamlit app sidebar.

## Data sources

All free, no paid APIs required:

- **Census TIGER** — block group geometries
- **Census ACS** — population density
- **FAA NPIAS** — commercial airports
- **OpenStreetMap** — amenity counts (via offline PBF extracts from Geofabrik)
- **Walk Score** — walkability/bikeability (scraped)
- **Zillow ZHVI** — median home values by ZIP (north scoring)
- **HUD Fair Market Rents** — 2BR rent by county (south scoring)
- **PAD-US** — protected areas for hiking proximity

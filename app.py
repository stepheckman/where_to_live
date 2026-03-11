"""
Interactive explorer for where-to-live pipeline outputs.

Loads step-4 parquet files (north_amenities.parquet, south_amenities.parquet)
and lets you adjust scoring weights and hard filters via sliders, re-scoring
instantly without re-running the slow data-collection pipeline.

Run with:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Where to Live",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Where to Live — Interactive Explorer")
st.caption(
    "Adjust filters and weights in the sidebar; the map and table update instantly. "
    "Data comes from the cached pipeline outputs (steps 1–4)."
)

# ---------------------------------------------------------------------------
# Load cached step-4 data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    north_path = DATA_PROCESSED / "north_amenities.parquet"
    south_path = DATA_PROCESSED / "south_amenities.parquet"
    north = pd.read_parquet(north_path) if north_path.exists() else None
    south = pd.read_parquet(south_path) if south_path.exists() else None
    return north, south


north_raw, south_raw = load_data()

if north_raw is None or south_raw is None:
    st.error(
        "Pipeline data not found. Generate it first:\n\n"
        "```\nuv run python run_pipeline.py\n```\n\n"
        "Or run just through step 4:\n\n"
        "```\nuv run python run_pipeline.py --only 4\n```"
    )
    st.stop()

# Data availability callout
_missing = []
if north_raw["walk_score"].isna().all():
    _missing.append("**Walk/Bike scores** (set `WALKSCORE_API_KEY` and re-run step 3)")
if "fmr_2br" not in south_raw.columns:
    _missing.append("**South rent data** (re-run step 4 to fetch HUD FMR)")
if _missing:
    st.info(
        "Some data is missing — affordability/walkability weights are auto-zeroed "
        "for those signals:\n- " + "\n- ".join(_missing),
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Hard Filters")
    st.caption("ZCTAs that fail these thresholds are dropped before scoring.")

    min_walk = st.slider("Min Walk Score", 0, 100, MIN_WALK_SCORE, step=5)
    min_bike = st.slider("Min Bike Score", 0, 100, MIN_BIKE_SCORE, step=5)
    max_airport = st.slider(
        "Max Airport Drive (min)", 15, 120, MAX_AIRPORT_DRIVE_MIN, step=5
    )
    min_grocery = st.slider("Min Groceries within 1200m", 0, 5, MIN_GROCERY_COUNT)
    min_pharmacy = st.slider("Min Pharmacies within 1200m", 0, 3, MIN_PHARMACY_COUNT)

    st.divider()

    st.header("Score Weights")
    st.caption("Values are auto-normalized to sum to 1.0 — drag freely.")

    w_walk = st.slider("Walk Score", 0.0, 1.0, W_WALK, 0.05)
    w_bike = st.slider("Bike Score", 0.0, 1.0, W_BIKE, 0.05)
    w_grocery = st.slider("Groceries", 0.0, 1.0, W_GROCERY, 0.05)
    w_cafe = st.slider("Cafés", 0.0, 1.0, W_CAFE, 0.05)
    w_restaurant = st.slider("Restaurants", 0.0, 1.0, W_RESTAURANT, 0.05)
    w_pharmacy = st.slider("Pharmacies", 0.0, 1.0, W_PHARMACY, 0.05)
    w_hiking = st.slider("Hiking access", 0.0, 1.0, W_HIKING, 0.05)
    w_afford = st.slider("Affordability", 0.0, 1.0, W_HOME_VALUE, 0.05)

    raw_weights = dict(
        walk=w_walk,
        bike=w_bike,
        grocery=w_grocery,
        cafe=w_cafe,
        restaurant=w_restaurant,
        pharmacy=w_pharmacy,
        hiking=w_hiking,
        afford=w_afford,
    )
    total_w = sum(raw_weights.values()) or 1.0
    norm_weights = {k: v / total_w for k, v in raw_weights.items()}

    labels = {
        "walk": "Walk",
        "bike": "Bike",
        "grocery": "Groc",
        "cafe": "Café",
        "restaurant": "Rest",
        "pharmacy": "Rx",
        "hiking": "Hike",
        "afford": "$$",
    }
    st.caption(
        "Normalized: "
        + "  ".join(f"{labels[k]} {v:.0%}" for k, v in norm_weights.items())
    )

    st.divider()

    st.header("Output")
    top_n = st.slider("Top N results", 5, 50, TOP_N, step=5)

# ---------------------------------------------------------------------------
# Scoring
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


def score_and_filter(
    df: pd.DataFrame,
    region: str,
    weights: dict,
    min_walk: int,
    min_bike: int,
    max_airport: int,
    min_grocery: int,
    min_pharmacy: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    stats: dict = {"start": len(df)}

    # Hard filter: airport proximity
    if "airport_drive_min_approx" in df.columns:
        df = df[df["airport_drive_min_approx"] <= max_airport]
    stats["after_airport"] = len(df)

    # Hard filter: amenity minimums
    df = df[df["grocery_count"] >= min_grocery]
    df = df[df["pharmacy_count"] >= min_pharmacy]
    stats["after_amenities"] = len(df)

    # Hard filter: walkability (only if walk score data exists)
    has_walk = "walk_score" in df.columns and df["walk_score"].notna().any()
    has_bike = "bike_score" in df.columns and df["bike_score"].notna().any()
    if has_walk:
        df = df[df["walk_score"] >= min_walk]
    if has_bike:
        df = df[df["bike_score"] >= min_bike]
    stats["after_walkability"] = len(df)

    if df.empty:
        stats["shown"] = 0
        return df, stats

    # Normalize signals
    def col_or_nan(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    df["n_walk"] = _normalize(col_or_nan("walk_score"), cap=100)
    df["n_bike"] = _normalize(col_or_nan("bike_score"), cap=100)
    df["n_grocery"] = _normalize(df["grocery_count"], cap=CAP_GROCERY)
    df["n_cafe"] = _normalize(df["cafe_count"], cap=CAP_CAFE)
    df["n_restaurant"] = _normalize(df["restaurant_count"], cap=CAP_RESTAURANT)
    df["n_pharmacy"] = _normalize(df["pharmacy_count"], cap=CAP_PHARMACY)

    # Hiking: combine distance (inverted) and count signals equally
    has_hiking = "dist_nearest_park_km" in df.columns and df["dist_nearest_park_km"].notna().any()
    if has_hiking:
        n_dist = _normalize(df["dist_nearest_park_km"], invert=True)
        n_count = _normalize(col_or_nan("parks_within_50km"))
        df["n_hiking"] = (n_dist.fillna(0.5) + n_count.fillna(0.5)) / 2
    else:
        df["n_hiking"] = pd.Series(np.nan, index=df.index)

    afford_col = "zhvi_latest" if region == "north" else "fmr_2br"
    df["n_afford"] = (
        _normalize(df[afford_col], invert=True)
        if afford_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # Re-normalize weights (zero out signals where data is missing)
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

    df = df.sort_values("composite_score", ascending=False).head(top_n)
    stats["shown"] = len(df)
    return df, stats


def _rename_for_display(df: pd.DataFrame, region: str) -> pd.DataFrame:
    renames = {
        "ZCTA5CE20": "zcta",
        "centroid_lat": "lat",
        "centroid_lon": "lon",
        "airport_drive_min_approx": "airport_drive_min",
    }
    if region == "north":
        renames["zhvi_latest"] = "median_home_value"
    else:
        renames["fmr_2br"] = "fmr_2br_rent"
    present = {k: v for k, v in renames.items() if k in df.columns}
    return df.rename(columns=present)


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------
def _score_color(score: float) -> str:
    if score >= 75:
        return "#2ecc71"
    elif score >= 60:
        return "#f1c40f"
    elif score >= 45:
        return "#e67e22"
    return "#e74c3c"


def _build_popup(row: pd.Series, region: str) -> str:
    gmaps = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
    sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={row['lat']},{row['lon']}"

    walk = f"{row['walk_score']:.0f}" if pd.notna(row.get("walk_score")) else "N/A"
    bike = f"{row['bike_score']:.0f}" if pd.notna(row.get("bike_score")) else "N/A"

    if region == "north" and "median_home_value" in row.index:
        v = row.get("median_home_value")
        afford_row = f"<tr><td><b>Median home</b></td><td>{'N/A' if pd.isna(v) else f'${v:,.0f}'}</td></tr>"
    elif region == "south" and "fmr_2br_rent" in row.index:
        v = row.get("fmr_2br_rent")
        afford_row = f"<tr><td><b>2BR rent/mo</b></td><td>{'N/A' if pd.isna(v) else f'${v:,.0f}'}</td></tr>"
    else:
        afford_row = ""

    airport = row.get("nearest_airport", "N/A")
    drive = row.get("airport_drive_min")
    airport_str = f"{airport} (~{drive:.0f} min)" if pd.notna(drive) else str(airport)

    return f"""
    <div style="font-family:sans-serif;width:280px;">
      <h4 style="margin:0 0 6px 0;">ZCTA {row['zcta']}</h4>
      <table style="border-collapse:collapse;width:100%;">
        <tr style="background:#f5f5f5;">
          <td colspan="2" style="padding:4px 6px;font-weight:bold;font-size:15px;">
            Score: {row['composite_score']:.1f} / 100
          </td>
        </tr>
        <tr><td><b>Walk</b></td><td>{walk}</td></tr>
        <tr><td><b>Bike</b></td><td>{bike}</td></tr>
        <tr><td><b>Grocery</b></td><td>{int(row.get('grocery_count', 0))}</td></tr>
        <tr><td><b>Café</b></td><td>{int(row.get('cafe_count', 0))}</td></tr>
        <tr><td><b>Restaurant</b></td><td>{int(row.get('restaurant_count', 0))}</td></tr>
        <tr><td><b>Pharmacy</b></td><td>{int(row.get('pharmacy_count', 0))}</td></tr>
        {afford_row}
        <tr><td><b>Park dist</b></td><td>{'N/A' if pd.isna(row.get('dist_nearest_park_km')) else f"{row['dist_nearest_park_km']:.1f} km"}</td></tr>
        <tr><td><b>Parks &lt;50km</b></td><td>{int(row.get('parks_within_50km', 0)) if pd.notna(row.get('parks_within_50km')) else 'N/A'}</td></tr>
        <tr><td><b>Airport</b></td><td style="font-size:11px;">{airport_str}</td></tr>
      </table>
      <div style="margin-top:8px;">
        <a href="{gmaps}" target="_blank" style="margin-right:8px;color:#3498db;">Google Maps</a>
        <a href="{sv}" target="_blank" style="color:#3498db;">Street View</a>
      </div>
    </div>"""


_LEGEND_HTML = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:10px 14px;border-radius:6px;
            border:1px solid #ccc;font-family:sans-serif;font-size:13px;">
  <b>Score</b><br>
  <span style="color:#2ecc71;">&#9679;</span> ≥75 Excellent<br>
  <span style="color:#f1c40f;">&#9679;</span> 60–74 Good<br>
  <span style="color:#e67e22;">&#9679;</span> 45–59 Fair<br>
  <span style="color:#e74c3c;">&#9679;</span> &lt;45 Poor
</div>"""


def _render_map_html(df: pd.DataFrame, region: str) -> str:
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )
    m.get_root().html.add_child(folium.Element(_LEGEND_HTML))
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=_score_color(row["composite_score"]),
            fill_opacity=0.85,
            popup=folium.Popup(_build_popup(row, region), max_width=300),
            tooltip=f"ZCTA {row['zcta']} — {row['composite_score']:.1f}",
        ).add_to(m)
    return m._repr_html_()


def _render_combined_map_html(north_df: pd.DataFrame, south_df: pd.DataFrame) -> str:
    all_lats = list(north_df["lat"]) + list(south_df["lat"])
    all_lons = list(north_df["lon"]) + list(south_df["lon"])
    m = folium.Map(
        location=[sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)],
        zoom_start=4,
        tiles="CartoDB positron",
    )
    m.get_root().html.add_child(folium.Element(_LEGEND_HTML))

    north_group = folium.FeatureGroup(name="North — buy", show=True)
    south_group = folium.FeatureGroup(name="South — rent", show=True)

    for _, row in north_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9, color="white", weight=1.5,
            fill=True, fill_color=_score_color(row["composite_score"]), fill_opacity=0.85,
            popup=folium.Popup(_build_popup(row, "north"), max_width=300),
            tooltip=f"[N] ZCTA {row['zcta']} — {row['composite_score']:.1f}",
        ).add_to(north_group)

    for _, row in south_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9, color="white", weight=1.5,
            fill=True, fill_color=_score_color(row["composite_score"]), fill_opacity=0.85,
            popup=folium.Popup(_build_popup(row, "south"), max_width=300),
            tooltip=f"[S] ZCTA {row['zcta']} — {row['composite_score']:.1f}",
        ).add_to(south_group)

    north_group.add_to(m)
    south_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


# ---------------------------------------------------------------------------
# Per-region display
# ---------------------------------------------------------------------------
DISPLAY_COLS = {
    "north": [
        "zcta", "composite_score", "walk_score", "bike_score",
        "grocery_count", "cafe_count", "restaurant_count", "pharmacy_count",
        "dist_nearest_park_km", "parks_within_50km",
        "median_home_value", "nearest_airport", "airport_drive_min",
    ],
    "south": [
        "zcta", "composite_score", "walk_score", "bike_score",
        "grocery_count", "cafe_count", "restaurant_count", "pharmacy_count",
        "dist_nearest_park_km", "parks_within_50km",
        "fmr_2br_rent", "nearest_airport", "airport_drive_min",
    ],
}

COLUMN_CONFIG = {
    "composite_score":      st.column_config.NumberColumn("Score",          format="%.1f"),
    "walk_score":           st.column_config.NumberColumn("Walk",           format="%.0f"),
    "bike_score":           st.column_config.NumberColumn("Bike",           format="%.0f"),
    "grocery_count":        st.column_config.NumberColumn("Grocery",        format="%d"),
    "cafe_count":           st.column_config.NumberColumn("Café",           format="%d"),
    "restaurant_count":     st.column_config.NumberColumn("Restaurant",     format="%d"),
    "pharmacy_count":       st.column_config.NumberColumn("Pharmacy",       format="%d"),
    "dist_nearest_park_km": st.column_config.NumberColumn("Park dist (km)", format="%.1f"),
    "parks_within_50km":    st.column_config.NumberColumn("Parks <50km",    format="%d"),
    "median_home_value":    st.column_config.NumberColumn("Home Value",     format="$%,.0f"),
    "fmr_2br_rent":         st.column_config.NumberColumn("2BR Rent/mo",    format="$%,.0f"),
    "airport_drive_min":    st.column_config.NumberColumn("Airport (min)",  format="%.0f"),
}


def show_region(df_raw: pd.DataFrame, region: str) -> None:
    scored, stats = score_and_filter(
        df_raw,
        region,
        weights=norm_weights,
        min_walk=min_walk,
        min_bike=min_bike,
        max_airport=max_airport,
        min_grocery=min_grocery,
        min_pharmacy=min_pharmacy,
        top_n=top_n,
    )

    if scored.empty:
        st.warning("No candidates survive the current filters — try loosening the sliders.")
        return

    df_display = _rename_for_display(scored, region)

    # Map
    map_html = _render_map_html(df_display, region)
    components.html(map_html, height=520, scrolling=False)

    # Table
    cols = [c for c in DISPLAY_COLS[region] if c in df_display.columns]
    st.dataframe(
        df_display[cols].reset_index(drop=True),
        use_container_width=True,
        column_config=COLUMN_CONFIG,
        height=400,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_combined, tab_north, tab_south = st.tabs(["Combined", "North — buy", "South — rent"])

with tab_combined:
    north_scored, _ = score_and_filter(
        north_raw, "north", weights=norm_weights,
        min_walk=min_walk, min_bike=min_bike, max_airport=max_airport,
        min_grocery=min_grocery, min_pharmacy=min_pharmacy, top_n=top_n,
    )
    south_scored, _ = score_and_filter(
        south_raw, "south", weights=norm_weights,
        min_walk=min_walk, min_bike=min_bike, max_airport=max_airport,
        min_grocery=min_grocery, min_pharmacy=min_pharmacy, top_n=top_n,
    )
    north_display = _rename_for_display(north_scored, "north")
    south_display = _rename_for_display(south_scored, "south")
    if north_display.empty and south_display.empty:
        st.warning("No candidates survive the current filters — try loosening the sliders.")
    else:
        combined_html = _render_combined_map_html(north_display, south_display)
        components.html(combined_html, height=580, scrolling=False)

with tab_north:
    show_region(north_raw, "north")

with tab_south:
    show_region(south_raw, "south")

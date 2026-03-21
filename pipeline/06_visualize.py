"""
Step 6: Interactive Maps
-------------------------
Generates folium HTML maps for the top north and south candidates.
Each marker shows a popup with key stats and links to Google Maps
and Google Street View for quick visual inspection.

Outputs
-------
outputs/maps/north_candidates.html
outputs/maps/south_candidates.html
outputs/maps/combined_candidates.html  (both regions on one map)
"""

from pathlib import Path

import folium
import pandas as pd
import requests
from folium.plugins import MarkerCluster

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
from config import OUTPUTS, MAPS


def reverse_geocode_city(lat: float, lon: float) -> str:
    """Return 'City, State' for a lat/lon using OSM Nominatim (free, no key)."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "where-to-live-pipeline/1.0"},
            timeout=10,
        )
        data = resp.json()
        addr = data.get("address", {})
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("county", "")
        )
        state = addr.get("state", "")
        return f"{city}, {state}".strip(", ")
    except Exception:
        return ""


def add_city_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'city' column via reverse geocoding if not already present."""
    if "city" in df.columns:
        return df
    logger.info("Reverse geocoding city names (this may take a moment)…")
    df = df.copy()
    df["city"] = [reverse_geocode_city(r.lat, r.lon) for r in df.itertuples()]
    return df


def score_to_color(score: float) -> str:
    """Map composite score (0–100) to a color."""
    if score >= 75:
        return "#2ecc71"   # green
    elif score >= 60:
        return "#f1c40f"   # yellow
    elif score >= 45:
        return "#e67e22"   # orange
    else:
        return "#e74c3c"   # red


def format_currency(val: float, prefix: str = "$") -> str:
    if pd.isna(val):
        return "N/A"
    return f"{prefix}{val:,.0f}"


def build_popup(row: pd.Series, region: str) -> str:
    """Build an HTML popup string for a candidate block group."""
    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"
    sv_url = (
        f"https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={row['lat']},{row['lon']}"
    )

    walk = f"{row['walk_score']:.0f}" if pd.notna(row.get("walk_score")) else "N/A"
    bike = f"{row['bike_score']:.0f}" if pd.notna(row.get("bike_score")) else "N/A"

    affordability_row = ""
    if region == "north" and "median_home_value" in row.index:
        affordability_row = (
            f"<tr><td><b>Median home value</b></td>"
            f"<td>{format_currency(row.get('median_home_value'))}</td></tr>"
        )
    elif region == "south" and "fmr_2br_rent" in row.index:
        affordability_row = (
            f"<tr><td><b>2-bed rent/month</b></td>"
            f"<td>{format_currency(row.get('fmr_2br_rent'))}/mo</td></tr>"
        )

    airport_info = row.get("nearest_airport", "N/A")
    airport_min = row.get("airport_drive_min")
    airport_str = (
        f"{airport_info} (~{airport_min:.0f} min)"
        if pd.notna(airport_min) else str(airport_info)
    )

    city = row.get("city", "") or ""
    city_line = f'<div style="font-size:13px; color:#555; margin-bottom:4px;">{city}</div>' if city else ""

    geoid = row.get("geoid", "")

    park_dist = row.get("dist_nearest_park_km")
    park_dist_str = "N/A" if pd.isna(park_dist) else f"{park_dist:.1f} km away"

    return f"""
    <div style="font-family: sans-serif; width: 300px;">
      <h4 style="margin:0 0 2px 0;">BG {geoid}</h4>
      {city_line}
      <table style="border-collapse: collapse; width: 100%;">
        <tr style="background:#f5f5f5;">
          <td colspan="2" style="padding:4px 6px; font-weight:bold; font-size:15px;">
            Score: {row['composite_score']:.1f} / 100
          </td>
        </tr>
        <tr><td><b>Walk Score (0–100)</b></td><td>{walk}</td></tr>
        <tr><td><b>Bike Score (0–100)</b></td><td>{bike}</td></tr>
        <tr><td><b>Grocery stores (¾ mi)</b></td><td>{int(row.get('grocery_count', 0))}</td></tr>
        <tr><td><b>Cafés (¾ mi)</b></td><td>{int(row.get('cafe_count', 0))}</td></tr>
        <tr><td><b>Restaurants (¾ mi)</b></td><td>{int(row.get('restaurant_count', 0))}</td></tr>
        <tr><td><b>Pharmacies (¾ mi)</b></td><td>{int(row.get('pharmacy_count', 0))}</td></tr>
        <tr><td><b>Transit stops (1 mi)</b></td><td>{int(row.get('transit_stops', 0)) if pd.notna(row.get('transit_stops')) else 'N/A'}</td></tr>
        {affordability_row}
        <tr><td><b>Nearest park</b></td><td>{park_dist_str}</td></tr>
        <tr><td><b>Drive to airport</b></td><td style="font-size:11px;">{airport_str}</td></tr>
      </table>
      <div style="margin-top:8px;">
        <a href="{gmaps_url}" target="_blank"
           style="margin-right:8px; color:#3498db;">Google Maps</a>
        <a href="{sv_url}" target="_blank"
           style="color:#3498db;">Street View</a>
      </div>
    </div>
    """


def make_map(df: pd.DataFrame, region: str, center_lat: float, center_lon: float) -> folium.Map:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # Score legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #ccc; font-family: sans-serif; font-size: 13px;">
      <b>Composite Score</b><br>
      <span style="color:#2ecc71;">&#9679;</span> ≥ 75 — Excellent<br>
      <span style="color:#f1c40f;">&#9679;</span> 60–74 — Good<br>
      <span style="color:#e67e22;">&#9679;</span> 45–59 — Fair<br>
      <span style="color:#e74c3c;">&#9679;</span> &lt; 45 — Poor
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in df.iterrows():
        color = score_to_color(row["composite_score"])
        popup_html = build_popup(row, region)
        geoid = row.get("geoid", "")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row.get('city') or ('BG ' + str(geoid))} — Score: {row['composite_score']:.1f}",
        ).add_to(m)

    return m


def run() -> None:
    MAPS.mkdir(parents=True, exist_ok=True)

    for region in ("north", "south"):
        in_path = OUTPUTS / f"{region}_candidates.csv"
        if not in_path.exists():
            raise FileNotFoundError(f"Run pipeline/05_score_and_rank.py first ({in_path})")

        df = pd.read_csv(in_path)
        df = add_city_column(df)
        logger.info(f"{region.upper()}: {len(df)} candidates")

        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()

        m = make_map(df, region, center_lat, center_lon)
        out_path = MAPS / f"{region}_candidates.html"
        m.save(str(out_path))
        logger.success(f"Saved {out_path}")

    # Combined map (both regions)
    logger.info("Building combined map…")
    north_df = pd.read_csv(OUTPUTS / "north_candidates.csv")
    south_df = pd.read_csv(OUTPUTS / "south_candidates.csv")
    north_df = add_city_column(north_df)
    south_df = add_city_column(south_df)
    north_df["region"] = "north"
    south_df["region"] = "south"

    center_lat = 38.0   # roughly mid-US
    center_lon = -95.0
    m_combined = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="CartoDB positron",
    )

    north_group = folium.FeatureGroup(name="North candidates (buy)", show=True)
    south_group = folium.FeatureGroup(name="South candidates (rent)", show=True)

    for _, row in north_df.iterrows():
        color = score_to_color(row["composite_score"])
        popup_html = build_popup(row, "north")
        geoid = row.get("geoid", "")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9, color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"[N] {row.get('city') or ('BG ' + str(geoid))} — {row['composite_score']:.1f}",
        ).add_to(north_group)

    for _, row in south_df.iterrows():
        color = score_to_color(row["composite_score"])
        popup_html = build_popup(row, "south")
        geoid = row.get("geoid", "")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9, color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"[S] {row.get('city') or ('BG ' + str(geoid))} — {row['composite_score']:.1f}",
        ).add_to(south_group)

    north_group.add_to(m_combined)
    south_group.add_to(m_combined)
    folium.LayerControl(collapsed=False).add_to(m_combined)

    combined_path = MAPS / "combined_candidates.html"
    m_combined.save(str(combined_path))
    logger.success(f"Saved combined map: {combined_path}")


if __name__ == "__main__":
    run()

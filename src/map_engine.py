"""
ROADGUARD AI - Geographical Hotspot & Spatial Map Engine
Generates interactive dark Folium maps with marker clustering, heatmaps, rich popups,
and cluster-coded color schemes.
"""

import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from src.config import CLUSTER_COLORS, COLOR_RED_SEVERE, COLOR_ACCENT_ORANGE, COLOR_GREEN_MILD

class MapEngine:
    @staticmethod
    def create_accident_map(
        df: pd.DataFrame,
        cluster_metadata: Optional[Dict[int, Dict]] = None,
        enable_heatmap: bool = True,
        max_markers: int = 1200,
    ) -> Tuple[Optional[folium.Map], Optional[str]]:
        """
        Builds an interactive Folium map with cluster-coded markers and a heatmap layer.
        Returns (folium.Map, warning_or_error_message).
        """
        # Validate coordinate availability
        if "Latitude" not in df.columns or "Longitude" not in df.columns:
            return None, (
                "Geographical coordinates (Latitude & Longitude) are not present in this dataset. "
                "Spatial map visualization is disabled. Location analysis is available in the Location Intelligence tab."
            )

        geo_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        geo_df["Latitude"] = pd.to_numeric(geo_df["Latitude"], errors="coerce")
        geo_df["Longitude"] = pd.to_numeric(geo_df["Longitude"], errors="coerce")
        geo_df = geo_df.dropna(subset=["Latitude", "Longitude"])

        # Filter valid lat/lon ranges
        geo_df = geo_df[
            geo_df["Latitude"].between(-90, 90) &
            geo_df["Longitude"].between(-180, 180)
        ]

        if len(geo_df) == 0:
            return None, "No records with valid WGS84 coordinates were found after filtering."

        # Compute map center
        center_lat = float(geo_df["Latitude"].median())
        center_lon = float(geo_df["Longitude"].median())

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # 1. HeatMap Overlay Layer
        if enable_heatmap and len(geo_df) > 0:
            heat_data = [
                [row["Latitude"], row["Longitude"], float(row.get("Severity_Numeric", 1))]
                for _, row in geo_df.iterrows()
            ]
            heat_layer = HeatMap(
                heat_data,
                name="Accident Heatmap (Density & Severity)",
                radius=15,
                blur=18,
                max_zoom=14,
                gradient={0.2: "#0E7490", 0.5: "#22D3EE", 0.8: "#F59E0B", 1.0: "#EF4444"},
            )
            heat_layer.add_to(m)

        # 2. Marker Cluster Layer
        marker_cluster = MarkerCluster(
            name="Collision Hotspot Markers",
            overlay=True,
            control=True,
            options={"maxClusterRadius": 35},
        ).add_to(m)

        # Sample if too many points for browser responsiveness
        if len(geo_df) > max_markers:
            sample_df = geo_df.sample(n=max_markers, random_state=42)
            warning_msg = f"Displaying {max_markers:,} sampled locations out of {len(geo_df):,} total for fluid rendering."
        else:
            sample_df = geo_df
            warning_msg = None

        has_cluster_col = "Cluster" in sample_df.columns

        for _, row in sample_df.iterrows():
            lat = row["Latitude"]
            lon = row["Longitude"]
            acc_id = row.get("Accident_ID", "N/A")
            date_str = str(row.get("Date", "N/A"))
            time_str = str(row.get("Time", "N/A"))
            severity = str(row.get("Accident_Severity", "Unknown"))
            road_type = str(row.get("Road_Type", "General"))
            vehicle = str(row.get("Vehicle_Type", "Vehicle"))
            casualties = int(row.get("Number_of_Casualties", 1))
            fatalities = int(row.get("Number_of_Fatalities", 0))
            weather = str(row.get("Weather", "Clear"))
            speed = row.get("Speed_Limit", "N/A")

            # Determine marker color
            if has_cluster_col and cluster_metadata:
                c_id = int(row["Cluster"])
                color = cluster_metadata.get(c_id, {}).get("color", "#22D3EE")
                cluster_label = cluster_metadata.get(c_id, {}).get("label", f"Cluster {c_id}")
            else:
                c_id = "N/A"
                cluster_label = "Unassigned"
                color = COLOR_RED_SEVERE if severity == "Fatal" else (
                    COLOR_ACCENT_ORANGE if severity == "Serious" else COLOR_GREEN_MILD
                )

            # Rich light HTML popup
            popup_html = f"""
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; background: #FFFFFF; color: #0F172A; 
                        padding: 12px 14px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); min-width: 220px; font-size: 12px;">
                <div style="font-weight: 700; font-size: 13px; color: #0284C7; margin-bottom: 4px; border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;">
                    {acc_id}
                </div>
                <div style="margin: 4px 0;"><strong>Severity:</strong> <span style="color: {'#DC2626' if severity=='Fatal' else ('#D97706' if severity=='Serious' else '#059669')}; font-weight: 700;">{severity}</span></div>
                <div style="margin: 4px 0;"><strong>Date &amp; Time:</strong> {date_str} {time_str}</div>
                <div style="margin: 4px 0;"><strong>Roadway:</strong> {road_type} ({speed} km/h)</div>
                <div style="margin: 4px 0;"><strong>Vehicle:</strong> {vehicle}</div>
                <div style="margin: 4px 0;"><strong>Casualties:</strong> {casualties} (Fatalities: {fatalities})</div>
                <div style="margin: 4px 0;"><strong>Atmosphere:</strong> {weather}</div>
                <div style="margin-top: 6px; padding-top: 4px; border-top: 1px solid #E2E8F0; color: {color}; font-weight: 700;">
                    {f"Cluster {c_id}: {cluster_label}" if has_cluster_col else "Hotspot Point"}
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(marker_cluster)

        # 3. Custom Legend
        if cluster_metadata:
            legend_items = "".join([
                f'<div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">'
                f'<span style="background: {meta["color"]}; width: 12px; height: 12px; border-radius: 50%; display: inline-block;"></span>'
                f'<span style="font-size: 11px; color: #1E293B; font-weight: 600;">C{c_id}: {meta["label"][:22]}</span>'
                f'</div>'
                for c_id, meta in cluster_metadata.items()
            ])
            legend_html = f"""
            <div style="position: fixed; bottom: 25px; right: 25px; z-index: 9999; background: rgba(255, 255, 255, 0.95);
                        padding: 10px 14px; border-radius: 8px; border: 1px solid #E2E8F0; font-family: sans-serif;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.12); max-width: 280px;">
                <div style="font-weight: 700; font-size: 12px; color: #0284C7; margin-bottom: 6px; text-transform: uppercase;">
                    Cluster Legend
                </div>
                {legend_items}
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(position="topright").add_to(m)
        return m, warning_msg

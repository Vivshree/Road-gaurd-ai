"""
ROADGUARD AI - Cluster Profiling & Semantic Labeling Module
Computes data-driven cluster statistics, synthesizes descriptive profiles, and calculates
standardized feature deviations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.config import CLUSTER_COLORS

class ClusterProfiler:
    @staticmethod
    def _safe_mode(series: Optional[pd.Series], fallback: str = "Mixed") -> str:
        """Safely extracts the statistical mode without raising IndexError on empty or all-null series."""
        if series is None or len(series) == 0:
            return fallback
        non_null = series.dropna()
        if len(non_null) == 0:
            return fallback
        modes = non_null.mode()
        return str(modes.iloc[0]) if len(modes) > 0 else fallback

    @staticmethod
    def generate_cluster_profiles(df: pd.DataFrame, cluster_col: str = "Cluster") -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """
        Analyzes records assigned to each cluster and computes comprehensive safety statistics.
        Returns:
            - summary_df: Clean comparison DataFrame
            - cluster_metadata: Dict with metadata, semantic labels, and color mappings
        """
        total_records = len(df)
        cluster_ids = sorted(df[cluster_col].unique())

        summary_rows = []
        cluster_metadata = {}

        for i, c_id in enumerate(cluster_ids):
            c_df = df[df[cluster_col] == c_id]
            c_size = len(c_df)
            c_pct = round((c_size / max(total_records, 1)) * 100, 1)

            # Key safety metrics
            avg_vehicles = round(c_df["Number_of_Vehicles"].mean() if "Number_of_Vehicles" in c_df.columns else 1.0, 2)
            avg_casualties = round(c_df["Number_of_Casualties"].mean() if "Number_of_Casualties" in c_df.columns else 1.0, 2)
            
            fatality_rate = round(
                ((c_df["Number_of_Fatalities"] > 0).sum() / max(c_size, 1)) * 100, 1
            ) if "Number_of_Fatalities" in c_df.columns else 0.0

            injury_rate = round(
                ((c_df["Number_of_Injuries"] > 0).sum() / max(c_size, 1)) * 100, 1
            ) if "Number_of_Injuries" in c_df.columns else 0.0

            avg_severity_score = round(
                c_df["Severity_Numeric"].mean() if "Severity_Numeric" in c_df.columns else 1.5, 2
            )

            night_pct = round(
                (c_df["Is_Night"].sum() / max(c_size, 1)) * 100, 1
            ) if "Is_Night" in c_df.columns else 0.0

            weather_pct = round(
                (c_df["Is_Adverse_Weather"].sum() / max(c_size, 1)) * 100, 1
            ) if "Is_Adverse_Weather" in c_df.columns else 0.0

            avg_speed = round(
                c_df["Speed_Limit"].mean() if "Speed_Limit" in c_df.columns else 50.0, 1
            )

            top_road = ClusterProfiler._safe_mode(c_df.get("Road_Type"), "Mixed")
            top_weather = ClusterProfiler._safe_mode(c_df.get("Weather"), "Clear")
            top_vehicle = ClusterProfiler._safe_mode(c_df.get("Vehicle_Type"), "General")

            # Synthesize descriptive data-driven label
            label, risk_level = ClusterProfiler._synthesize_label(
                fatality_rate=fatality_rate,
                avg_severity=avg_severity_score,
                night_pct=night_pct,
                weather_pct=weather_pct,
                avg_speed=avg_speed,
                avg_vehicles=avg_vehicles,
            )

            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

            summary_rows.append({
                "Cluster ID": f"Cluster {c_id}",
                "Cluster Label": label,
                "Risk Classification": risk_level,
                "Accident Count": c_size,
                "Share (%)": f"{c_pct}%",
                "Fatality Rate (%)": f"{fatality_rate}%",
                "Injury Rate (%)": f"{injury_rate}%",
                "Avg Severity (1-3)": avg_severity_score,
                "Night-Time (%)": f"{night_pct}%",
                "Adverse Weather (%)": f"{weather_pct}%",
                "Avg Speed (km/h)": avg_speed,
                "Avg Vehicles": avg_vehicles,
                "Top Road Type": top_road,
                "Primary Vehicle": top_vehicle,
            })

            cluster_metadata[c_id] = {
                "id": c_id,
                "name": f"Cluster {c_id}",
                "label": label,
                "risk_level": risk_level,
                "size": c_size,
                "pct": c_pct,
                "fatality_rate": fatality_rate,
                "injury_rate": injury_rate,
                "avg_severity": avg_severity_score,
                "night_pct": night_pct,
                "weather_pct": weather_pct,
                "avg_speed": avg_speed,
                "avg_vehicles": avg_vehicles,
                "top_road": top_road,
                "top_weather": top_weather,
                "top_vehicle": top_vehicle,
                "color": color,
            }

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, cluster_metadata

    @staticmethod
    def _synthesize_label(
        fatality_rate: float,
        avg_severity: float,
        night_pct: float,
        weather_pct: float,
        avg_speed: float,
        avg_vehicles: float,
    ) -> Tuple[str, str]:
        """
        Synthesizes a human-readable, data-driven label based on empirical cluster metrics.
        Labels are explicitly noted as data-driven descriptive patterns.
        """
        # Critical / High-Risk combinations
        if fatality_rate >= 30.0 or avg_severity >= 2.4:
            if night_pct >= 50.0:
                return "Severe Night-Time Collision Corridor", "High Risk"
            elif avg_speed >= 80.0:
                return "High-Speed Expressway Fatal Hotspot", "High Risk"
            else:
                return "Critical High-Fatality Impact Zone", "High Risk"

        # Weather dominant
        if weather_pct >= 40.0:
            if night_pct >= 40.0:
                return "Low-Visibility Wet/Adverse Weather Zone", "Moderate Risk"
            return "Precipitation & Wet Road Skid Cluster", "Moderate Risk"

        # Night dominant
        if night_pct >= 60.0:
            return "Unlit Night-Time Arterial Corridor", "Moderate Risk"

        # High-Speed multi-vehicle
        if avg_speed >= 75.0 and avg_vehicles >= 2.0:
            return "Multi-Vehicle Highway Transit Cluster", "Moderate Risk"

        # Congestion / Low severity
        if avg_severity <= 1.4:
            return "Urban Low-Severity Transit Shunts", "Low Risk"

        # Mid-tier default
        return "Mixed Arterial General Hotspots", "Moderate Risk"

    @staticmethod
    def compute_feature_deviations(
        df: pd.DataFrame,
        feature_names: List[str],
        cluster_col: str = "Cluster"
    ) -> pd.DataFrame:
        """
        Calculates standardised z-score deviations of each cluster's mean feature value
        from the global dataset mean: z = (mean_cluster - mean_global) / std_global.
        Note: Descriptive profiling comparison only; does not imply causal feature importance.
        """
        global_means = df[feature_names].mean()
        global_stds = df[feature_names].std().replace(0, 1.0)

        deviation_records = []
        cluster_ids = sorted(df[cluster_col].unique())

        for c_id in cluster_ids:
            c_df = df[df[cluster_col] == c_id]
            c_means = c_df[feature_names].mean()
            z_scores = (c_means - global_means) / global_stds

            for feat in feature_names:
                deviation_records.append({
                    "Cluster": f"Cluster {c_id}",
                    "Cluster_ID": c_id,
                    "Feature": feat,
                    "Cluster_Mean": round(c_means[feat], 3),
                    "Global_Mean": round(global_means[feat], 3),
                    "Z_Score_Deviation": round(z_scores[feat], 3),
                })

        return pd.DataFrame(deviation_records)

"""
ROADGUARD AI - Interactive Plotly Data Visualizer
Builds high-clarity light-themed visualizations for exploratory and academic decision support.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.config import (
    COLOR_BG, COLOR_PANEL, COLOR_ACCENT_CYAN, COLOR_ACCENT_ORANGE,
    COLOR_RED_SEVERE, COLOR_GREEN_MILD, COLOR_TEXT, COLOR_BORDER, CLUSTER_COLORS
)

def apply_light_theme(fig: go.Figure) -> go.Figure:
    """Applies modern clean light aesthetic to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0F172A", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color="#1E293B", size=11),
        ),
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#0F172A", size=12, family="Plus Jakarta Sans, sans-serif"),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#475569"),
            title_font=dict(color="#0F172A", size=12, family="Plus Jakarta Sans, sans-serif"),
        ),
    )
    return fig

# Alias for backward-compatible imports
apply_dark_theme = apply_light_theme


def plot_executive_trends(df: pd.DataFrame) -> go.Figure:
    """Builds an interactive time-series trend of collision occurrences."""
    df_temp = df.copy()
    if "Date" in df_temp.columns:
        df_temp["Date_Parsed"] = pd.to_datetime(df_temp["Date"], errors="coerce")
        df_temp = df_temp.dropna(subset=["Date_Parsed"])
        df_temp["Month_Year"] = df_temp["Date_Parsed"].dt.to_period("M").astype(str)
        trend_df = df_temp.groupby(["Month_Year", "Accident_Severity"]).size().reset_index(name="Accident_Count")
        
        color_map = {
            "Fatal": COLOR_RED_SEVERE,
            "Serious": COLOR_ACCENT_ORANGE,
            "Slight": COLOR_GREEN_MILD,
        }

        fig = px.bar(
            trend_df,
            x="Month_Year",
            y="Accident_Count",
            color="Accident_Severity",
            color_discrete_map=color_map,
            title="Monthly Collision Frequency by Severity Profile",
            barmode="stack",
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Date feature not available for temporal trend", showarrow=False)

    return apply_light_theme(fig)


def plot_severity_distribution(df: pd.DataFrame) -> go.Figure:
    """Builds a polished donut chart for collision severity distribution."""
    if "Accident_Severity" in df.columns:
        sev_counts = df["Accident_Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]

        color_map = {
            "Fatal": COLOR_RED_SEVERE,
            "Serious": COLOR_ACCENT_ORANGE,
            "Slight": COLOR_GREEN_MILD,
        }
        colors = [color_map.get(s, COLOR_ACCENT_CYAN) for s in sev_counts["Severity"]]

        fig = go.Figure(data=[go.Pie(
            labels=sev_counts["Severity"],
            values=sev_counts["Count"],
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )])
        fig.update_layout(title="Accident Severity Breakdown")
    else:
        fig = go.Figure()
        fig.add_annotation(text="Severity attribute not present", showarrow=False)

    return apply_light_theme(fig)


def plot_accidents_by_hour(df: pd.DataFrame) -> go.Figure:
    """Builds an hourly collision frequency distribution with night/day zones."""
    if "Hour" in df.columns:
        hour_counts = df["Hour"].astype(int).value_counts().sort_index().reset_index()
        hour_counts.columns = ["Hour", "Count"]

        # Ensure all 24 hours are represented
        all_hours = pd.DataFrame({"Hour": range(24)})
        hour_counts = all_hours.merge(hour_counts, on="Hour", how="left").fillna(0)

        colors = [
            "rgba(217, 119, 6, 0.9)" if (h >= 20 or h < 6) else "rgba(2, 132, 199, 0.9)"
            for h in hour_counts["Hour"]
        ]

        fig = go.Figure(data=[go.Bar(
            x=hour_counts["Hour"],
            y=hour_counts["Count"],
            marker=dict(color=colors, line=dict(color="#E2E8F0", width=1)),
            hovertemplate="Hour %{x}:00 - %{y} Collisions<extra></extra>",
        )])
        fig.update_layout(
            title="Collision Distribution by Hour of Day (Amber = Night-time 20:00-06:00)",
            xaxis=dict(title="Hour of Day (0-23)", tickmode="linear", tick0=0, dtick=2),
            yaxis=dict(title="Number of Accidents"),
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Hour feature not present", showarrow=False)

    return apply_light_theme(fig)


def plot_weather_lighting(df: pd.DataFrame) -> go.Figure:
    """Cross-tabulates Weather vs Lighting conditions."""
    if "Weather" in df.columns and "Lighting_Condition" in df.columns:
        cross_df = df.groupby(["Weather", "Lighting_Condition"]).size().reset_index(name="Accident_Count")
        fig = px.bar(
            cross_df,
            x="Weather",
            y="Accident_Count",
            color="Lighting_Condition",
            title="Accidents by Atmospheric Weather & Ambient Lighting",
            barmode="group",
            color_discrete_sequence=CLUSTER_COLORS,
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Weather/Lighting attributes missing", showarrow=False)

    return apply_light_theme(fig)


def plot_pca_2d(pca_coords: np.ndarray, labels: np.ndarray, cluster_metadata: Dict) -> go.Figure:
    """Renders interactive 2D Principal Component Projection with cluster hulls and centroids."""
    df_pca = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Cluster": [f"Cluster {l}" for l in labels],
        "Cluster_ID": labels,
    })

    fig = go.Figure()

    for c_id, meta in cluster_metadata.items():
        sub_df = df_pca[df_pca["Cluster_ID"] == c_id]
        fig.add_trace(go.Scatter(
            x=sub_df["PC1"],
            y=sub_df["PC2"],
            mode="markers",
            name=f"Cluster {c_id}: {meta['label']}",
            marker=dict(
                size=8,
                color=meta["color"],
                opacity=0.8,
                line=dict(width=1, color="#FFFFFF"),
            ),
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            text=[f"Cluster {c_id} ({meta['label']})"] * len(sub_df),
        ))

        # Centroid indicator
        c_x = sub_df["PC1"].mean()
        c_y = sub_df["PC2"].mean()
        fig.add_trace(go.Scatter(
            x=[c_x],
            y=[c_y],
            mode="markers+text",
            marker=dict(size=14, color=meta["color"], symbol="cross", line=dict(width=2, color="#0F172A")),
            name=f"C{c_id} Centroid",
            showlegend=False,
            text=[f"C{c_id}"],
            textposition="top right",
            textfont=dict(color="#0F172A", size=11, family="JetBrains Mono"),
        ))

    fig.update_layout(
        title="2D PCA Latent Space Projection (Unsupervised Clustering)",
        xaxis_title="Principal Component 1 (PC1)",
        yaxis_title="Principal Component 2 (PC2)",
    )
    return apply_light_theme(fig)


def plot_pca_3d(pca_coords: np.ndarray, labels: np.ndarray, cluster_metadata: Dict) -> go.Figure:
    """Renders interactive 3D Principal Component Projection."""
    if pca_coords.shape[1] < 3:
        return plot_pca_2d(pca_coords, labels, cluster_metadata)

    df_pca = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "PC3": pca_coords[:, 2],
        "Cluster_ID": labels,
    })

    fig = go.Figure()

    for c_id, meta in cluster_metadata.items():
        sub_df = df_pca[df_pca["Cluster_ID"] == c_id]
        fig.add_trace(go.Scatter3d(
            x=sub_df["PC1"],
            y=sub_df["PC2"],
            z=sub_df["PC3"],
            mode="markers",
            name=f"C{c_id}: {meta['label'][:24]}...",
            marker=dict(size=4, color=meta["color"], opacity=0.85),
            hovertemplate=f"<b>Cluster {c_id}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title="3D PCA Dimensionality Reduction Space",
        scene=dict(
            xaxis=dict(backgroundcolor="#F8FAFC", gridcolor="#E2E8F0", title="PC1"),
            yaxis=dict(backgroundcolor="#F8FAFC", gridcolor="#E2E8F0", title="PC2"),
            zaxis=dict(backgroundcolor="#F8FAFC", gridcolor="#E2E8F0", title="PC3"),
        ),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(font=dict(color="#0F172A", size=10)),
    )
    return fig


def plot_elbow_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Visualizes the WCSS/Inertia Elbow Method across K values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["K"],
        y=curve_df["WCSS_Inertia"],
        mode="lines+markers",
        name="WCSS (Inertia)",
        line=dict(color=COLOR_ACCENT_CYAN, width=3),
        marker=dict(size=9, color="#FFFFFF", line=dict(width=2, color=COLOR_ACCENT_CYAN)),
        hovertemplate="K = %{x}<br>WCSS: %{y:,.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="Elbow Method for Optimal K (Within-Cluster Sum of Squares)",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear", tick0=2, dtick=1),
        yaxis=dict(title="Inertia / WCSS (Squared Euclidean Distance)"),
    )
    return apply_light_theme(fig)


def plot_silhouette_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Visualizes Silhouette Score variation across K values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["K"],
        y=curve_df["Silhouette_Score"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color=COLOR_ACCENT_ORANGE, width=3),
        marker=dict(size=9, color="#FFFFFF", line=dict(width=2, color=COLOR_ACCENT_ORANGE)),
        hovertemplate="K = %{x}<br>Silhouette Score: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Silhouette Score Trajectory across K Values (Higher is Better)",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear", tick0=2, dtick=1),
        yaxis=dict(title="Mean Silhouette Coefficient"),
    )
    return apply_light_theme(fig)


def plot_davies_bouldin_curve(curve_df: pd.DataFrame) -> go.Figure:
    """Visualizes Davies-Bouldin Index across K values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["K"],
        y=curve_df["Davies_Bouldin_Index"],
        mode="lines+markers",
        name="Davies-Bouldin Index",
        line=dict(color="#7C3AED", width=3),
        marker=dict(size=9, color="#FFFFFF", line=dict(width=2, color="#7C3AED")),
        hovertemplate="K = %{x}<br>DB Index: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Davies-Bouldin Index across K (Lower Indicates Better Separation)",
        xaxis=dict(title="Number of Clusters (K)", tickmode="linear", tick0=2, dtick=1),
        yaxis=dict(title="Davies-Bouldin Index"),
    )
    return apply_light_theme(fig)


def plot_sample_silhouettes(sample_sils: np.ndarray, labels: np.ndarray, cluster_metadata: Dict) -> go.Figure:
    """Builds a classic per-cluster silhouette coefficient silhouette diagram."""
    fig = go.Figure()
    y_lower = 10
    cluster_ids = sorted(cluster_metadata.keys())
    mean_sil = float(np.mean(sample_sils))

    for c_id in cluster_ids:
        c_sils = sample_sils[labels == c_id]
        c_sils.sort()
        c_size = len(c_sils)
        y_upper = y_lower + c_size
        color = cluster_metadata[c_id]["color"]

        fig.add_trace(go.Scatter(
            x=c_sils,
            y=np.arange(y_lower, y_upper),
            mode="lines",
            fill="tozerox",
            fillcolor=color,
            line=dict(color=color, width=1),
            name=f"Cluster {c_id}",
            showlegend=True,
            hovertemplate=f"Cluster {c_id}<br>Silhouette: %{{x:.3f}}<extra></extra>",
        ))
        y_lower = y_upper + 10

    # Overall average line
    fig.add_vline(
        x=mean_sil,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_RED_SEVERE,
        annotation_text=f"Mean Sil ({mean_sil:.3f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Per-Cluster Silhouette Coefficient Profile Diagram",
        xaxis=dict(title="Silhouette Coefficient Values", range=[-0.2, 1.0]),
        yaxis=dict(title="Cluster Samples Thickness", showticklabels=False),
    )
    return apply_light_theme(fig)


def plot_hour_weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    """Builds a 2D cross-tabulation heatmap of Day of Week vs Hour of Day."""
    if "Hour" in df.columns and "Day_of_Week" in df.columns:
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        cross_tab = pd.crosstab(df["Day_of_Week"], df["Hour"].astype(int))
        cross_tab = cross_tab.reindex([d for d in days_order if d in cross_tab.index])

        fig = go.Figure(data=go.Heatmap(
            z=cross_tab.values,
            x=[f"{h:02d}:00" for h in cross_tab.columns],
            y=cross_tab.index.tolist(),
            colorscale=[
                [0.0, "#F0F9FF"],
                [0.3, "#BAE6FD"],
                [0.6, "#0284C7"],
                [1.0, "#D97706"],
            ],
            hovertemplate="Day: %{y}<br>Time: %{x}<br>Accidents: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="Temporal Concentration Heatmap: Day of Week vs Hour of Day",
            xaxis_title="Hour of Occurrence",
            yaxis_title="Day of Week",
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Hour or Day_of_Week feature missing", showarrow=False)

    return apply_light_theme(fig)


def plot_cluster_radar(cluster_metadata: Dict) -> go.Figure:
    """Builds a comparative radar / spider chart across key normalized cluster traits."""
    categories = [
        "Fatality Rate", "Injury Rate", "Avg Severity", 
        "Night-time %", "Weather %", "Speed Limit"
    ]

    fig = go.Figure()

    for c_id, meta in cluster_metadata.items():
        vals = [
            meta["fatality_rate"],
            meta["injury_rate"],
            (meta["avg_severity"] / 3.0) * 100,
            meta["night_pct"],
            meta["weather_pct"],
            (meta["avg_speed"] / 120.0) * 100,
        ]
        vals.append(vals[0])  # close loop

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"Cluster {c_id}",
            line=dict(color=meta["color"], width=2),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", tickfont=dict(color="#64748B")),
            angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#0F172A", size=11)),
            bgcolor="#FFFFFF",
        ),
        paper_bgcolor="#FFFFFF",
        title="Multi-Dimensional Cluster Safety Profile (Normalized Traits)",
    )
    return fig


def plot_feature_deviations(deviation_df: pd.DataFrame, cluster_id: int, cluster_label: str) -> go.Figure:
    """Builds a diverging horizontal bar chart of Z-score deviations for a single cluster."""
    sub_df = deviation_df[deviation_df["Cluster_ID"] == cluster_id].sort_values("Z_Score_Deviation")

    colors = [
        COLOR_RED_SEVERE if z > 0.5 else (COLOR_GREEN_MILD if z < -0.5 else COLOR_ACCENT_CYAN)
        for z in sub_df["Z_Score_Deviation"]
    ]

    fig = go.Figure(go.Bar(
        x=sub_df["Z_Score_Deviation"],
        y=sub_df["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#E2E8F0", width=1)),
        hovertemplate="Feature: %{y}<br>Z-Score Deviation: %{x:.2f} std dev<br>Cluster Mean: %{customdata[0]}<br>Global Mean: %{customdata[1]}<extra></extra>",
        customdata=sub_df[["Cluster_Mean", "Global_Mean"]].values,
    ))

    fig.add_vline(x=0, line_color="#94A3B8", line_width=1.5)

    fig.update_layout(
        title=f"Feature Z-Score Deviations: Cluster {cluster_id} ({cluster_label})<br><sub>Deviation from overall dataset mean (Descriptive Profiling, NOT Causal Importance)</sub>",
        xaxis_title="Standard Deviations from Global Dataset Mean (Z-Score)",
        yaxis_title="Safety / Operational Feature",
    )
    return apply_light_theme(fig)

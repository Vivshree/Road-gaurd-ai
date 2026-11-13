"""
ROADGUARD AI - Academic Report Generator & Export Engine
Generates downloadable, presentation-ready PDF evaluation reports and CSV exports.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

class ReportEngine:
    @staticmethod
    def generate_pdf_report(
        dataset_name: str,
        total_records: int,
        clustering_results: Dict,
        cluster_summary_df: pd.DataFrame,
        pca_results: Optional[Dict] = None,
        preprocessing_summary: Optional[Dict] = None,
    ) -> bytes:
        """
        Builds a comprehensive, publication-grade academic PDF report using ReportLab.
        Returns PDF bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        # Custom academic report styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B1220"),
            alignment=TA_LEFT,
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#0284C7"),
            alignment=TA_LEFT,
        )

        h1_style = ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0B1220"),
            spaceBefore=14,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "Body_Custom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            alignment=TA_LEFT,
        )

        callout_style = ParagraphStyle(
            "Callout",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=TA_JUSTIFY,
        )

        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=TA_CENTER,
        )

        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1E293B"),
            alignment=TA_LEFT,
        )

        story = []

        # 1. Document Header
        story.append(Paragraph("ROADGUARD AI &mdash; System Audit &amp; Research Report", title_style))
        story.append(Paragraph("Road Accident Intelligence &amp; Hotspot Segmentation System", subtitle_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} &bull; "
            f"<b>Dataset:</b> {dataset_name} ({total_records:,} records) &bull; "
            f"<b>Status:</b> Academic Decision-Support System",
            body_style
        ))
        story.append(Spacer(1, 10))

        # Divider line
        divider = Table([[""]], colWidths=[530], rowHeights=[2])
        divider.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0284C7")),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(divider)
        story.append(Spacer(1, 12))

        # 2. Executive Overview & Methodology
        story.append(Paragraph("1. Executive Objective &amp; Unsupervised Framework", h1_style))
        story.append(Paragraph(
            "ROADGUARD AI employs unsupervised machine learning algorithms (K-Means and Agglomerative "
            "Hierarchical Clustering) to discover latent multi-attribute collision profiles without reliance "
            "on labeled ground truth. By synthesizing temporal, environmental, speed, and severity patterns, "
            "the system prioritizes high-risk roadway corridors for municipal and traffic authorities. "
            "<b>Crucial Academic Notice:</b> This platform is an exploratory decision-support system, not a real-time "
            "emergency dispatch or causal predictive model.",
            body_style
        ))
        story.append(Spacer(1, 8))

        # 3. Model Configuration & Validation Metrics
        story.append(Paragraph("2. Unsupervised Model Hyperparameters &amp; Validation Scores", h1_style))
        
        alg_name = clustering_results.get("algorithm", "K-Means")
        k_val = clustering_results.get("n_clusters", 4)
        sil_val = clustering_results.get("silhouette", 0.0)
        db_val = clustering_results.get("davies_bouldin", 0.0)
        ch_val = clustering_results.get("calinski_harabasz", 0.0)
        inertia_val = clustering_results.get("inertia", None)
        runtime_ms = clustering_results.get("runtime_ms", 0.0)

        metrics_table_data = [
            [
                Paragraph("<b>Parameter / Metric</b>", table_header_style),
                Paragraph("<b>Measured Value</b>", table_header_style),
                Paragraph("<b>Theoretical Interpretation</b>", table_header_style),
            ],
            [
                Paragraph("Algorithm", table_cell_style),
                Paragraph(str(alg_name), table_cell_style),
                Paragraph("Centroid/Hierarchical unsupervised partitioning", table_cell_style),
            ],
            [
                Paragraph("Clusters (K)", table_cell_style),
                Paragraph(str(k_val), table_cell_style),
                Paragraph("Partition granularity selected by practitioner", table_cell_style),
            ],
            [
                Paragraph("Silhouette Score", table_cell_style),
                Paragraph(f"{sil_val:.4f}", table_cell_style),
                Paragraph("Range [-1, +1]. Measures cohesion vs separation. Higher is better.", table_cell_style),
            ],
            [
                Paragraph("Davies-Bouldin Index", table_cell_style),
                Paragraph(f"{db_val:.4f}", table_cell_style),
                Paragraph("Similarity of clusters. Lower values indicate superior separation.", table_cell_style),
            ],
            [
                Paragraph("Calinski-Harabasz Index", table_cell_style),
                Paragraph(f"{ch_val:.1f}", table_cell_style),
                Paragraph("Variance ratio criterion. Higher values signify well-defined clusters.", table_cell_style),
            ],
        ]

        if inertia_val is not None:
            metrics_table_data.append([
                Paragraph("Inertia / WCSS", table_cell_style),
                Paragraph(f"{inertia_val:,.1f}", table_cell_style),
                Paragraph("Within-cluster sum of squared Euclidean distances to centroid.", table_cell_style),
            ])

        metrics_table_data.append([
            Paragraph("Execution Latency", table_cell_style),
            Paragraph(f"{runtime_ms} ms", table_cell_style),
            Paragraph("End-to-end model convergence time.", table_cell_style),
        ])

        t_metrics = Table(metrics_table_data, colWidths=[120, 90, 320])
        t_metrics.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1220")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 10))

        # 4. Cluster Profile Breakdown
        story.append(Paragraph("3. Cluster Segmentation &amp; Empirical Profiles", h1_style))
        story.append(Paragraph(
            "The table below details the empirical characteristics of each discovered cluster. "
            "Cluster labels are synthesized strictly from observed data frequencies and do not represent "
            "formal statutory legal fault categories.",
            body_style
        ))
        story.append(Spacer(1, 6))

        # Build cluster profiles table
        prof_cols = ["Cluster ID", "Cluster Label", "Accident Count", "Share (%)", "Fatality Rate (%)", "Night-Time (%)", "Avg Speed (km/h)"]
        sub_prof_df = cluster_summary_df[[c for c in prof_cols if c in cluster_summary_df.columns]]

        prof_table_data = [[
            Paragraph(f"<b>{c}</b>", table_header_style) for c in sub_prof_df.columns
        ]]

        for _, row in sub_prof_df.iterrows():
            row_cells = []
            for col in sub_prof_df.columns:
                val = str(row[col])
                row_cells.append(Paragraph(val, table_cell_style))
            prof_table_data.append(row_cells)

        t_prof = Table(prof_table_data, colWidths=[60, 150, 65, 55, 65, 65, 70])
        t_prof.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_prof)
        story.append(Spacer(1, 12))

        # 5. PCA Dimensionality Reduction
        if pca_results:
            story.append(Paragraph("4. Principal Component Analysis (PCA) Dimensionality Summary", h1_style))
            exp_vars = pca_results.get("explained_variance_ratio", [])
            cum_var = pca_results.get("cumulative_variance", [])
            var_text = ", ".join([f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(exp_vars)])
            total_cum = cum_var[-1] * 100 if cum_var else 0.0

            story.append(Paragraph(
                f"PCA projected the standardized multi-feature space onto {len(exp_vars)} orthogonal dimensions. "
                f"Individual variance explained: <b>{var_text}</b>. "
                f"Cumulative variance captured: <b>{total_cum:.1f}%</b>.",
                body_style
            ))
            story.append(Spacer(1, 8))

        # 6. Academic Limitations & Ethics
        story.append(Paragraph("5. Academic Methodology, Limitations &amp; Ethical Guardrails", h1_style))
        story.append(Paragraph(
            "<b>1. Reporting &amp; Geographic Bias:</b> Accident records frequently exhibit reporting biases where minor "
            "collisions go unrecorded while severe/fatal incidents receive thorough forensic documentation. Geographical "
            "clustering can reflect municipal reporting densities rather than inherent roadway danger.<br/>"
            "<b>2. Distance Metric Constraints:</b> Standard K-Means optimizes squared Euclidean distance ($L_2$) "
            "and assumes spherical, equally-sized clusters. Irregular or non-convex corridors may be fragmented. "
            "Agglomerative clustering provides linkage flexibility for non-Euclidean geometries.<br/>"
            "<b>3. Non-Causality Principle:</b> High correlation or cluster membership must <i>never</i> be construed "
            "as proving direct causation. Cluster profiles represent exploratory statistical patterns intended to "
            "guide on-site civil engineering assessments, speed enforcement patrols, and signage interventions.",
            callout_style
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def export_csv(df: pd.DataFrame) -> str:
        """Exports DataFrame to clean CSV string format."""
        return df.to_csv(index=False)

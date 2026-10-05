"""
RFM Explorer visualizations: Metric distributions, Elbow curve, and Silhouette score evaluation charts.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import SEGMENT_COLORS


def rfm_histograms(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Three side-by-side distribution subplots for Recency, Frequency, and Monetary.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recency Distribution (Days)",
            "Frequency Distribution (Orders)",
            "Monetary Distribution ($ Total Spend)",
        ),
    )

    # Recency
    fig.add_trace(
        go.Histogram(
            x=rfm_df["recency"],
            nbinsx=30,
            name="Recency",
            marker_color="#1f77b4",
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Frequency
    fig.add_trace(
        go.Histogram(
            x=rfm_df["frequency"],
            nbinsx=25,
            name="Frequency",
            marker_color="#ff7f0e",
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    # Monetary
    fig.add_trace(
        go.Histogram(
            x=rfm_df["monetary"],
            nbinsx=30,
            name="Monetary",
            marker_color="#2ca02c",
            opacity=0.8,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Orders", row=1, col=2)
    fig.update_xaxes(title_text="Spend ($)", row=1, col=3)
    fig.update_yaxes(title_text="Customer Count", row=1, col=1)

    return fig


def elbow_curve(eval_results: Dict[str, Any]) -> go.Figure:
    """
    Inertia (Elbow Method) chart across k values.
    """
    k_vals = eval_results["k_values"]
    inertias = eval_results["inertias"]
    best_k = eval_results.get("best_k", k_vals[0])

    fig = go.Figure()

    # Inertia line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color="#4A90E2", width=3),
            marker=dict(size=9, color="#4A90E2"),
        )
    )

    # Highlight best k
    if best_k in k_vals:
        best_inertia = inertias[k_vals.index(best_k)]
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[best_inertia],
                mode="markers",
                name=f"Optimal k={best_k}",
                marker=dict(size=14, color="#E74C3C", symbol="star"),
            )
        )

    fig.update_layout(
        title="Elbow Method (Within-Cluster Sum of Squares)",
        xaxis=dict(title="Number of Clusters (k)", tickmode="linear", tick0=min(k_vals), dtick=1),
        yaxis=dict(title="Inertia (Compactness)"),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def silhouette_plot(eval_results: Dict[str, Any]) -> go.Figure:
    """
    Silhouette score curve across k values highlighting the maximum separation.
    """
    k_vals = eval_results["k_values"]
    scores = eval_results["silhouette_scores"]
    best_k = eval_results.get("best_k", k_vals[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#2ECC71", width=3),
            marker=dict(size=9, color="#2ECC71"),
        )
    )

    # Highlight best k
    if best_k in k_vals:
        best_score = scores[k_vals.index(best_k)]
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[best_score],
                mode="markers+text",
                name=f"Best Silhouette (k={best_k})",
                text=[f"k={best_k} ({best_score:.3f})"],
                textposition="top center",
                marker=dict(size=14, color="#E67E22", symbol="diamond"),
            )
        )

    fig.update_layout(
        title="Silhouette Score (Cluster Separation Quality)",
        xaxis=dict(title="Number of Clusters (k)", tickmode="linear", tick0=min(k_vals), dtick=1),
        yaxis=dict(title="Average Silhouette Score (Higher is Better)", range=[0, max(scores) * 1.25 if max(scores) > 0 else 1.0]),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def segment_radar_comparison(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Radar chart comparing the normalized average R, F, M attributes across segments.
    """
    # Normalize medians between 0 and 100 for radar
    grouped = labeled_df.groupby("segment").agg(
        recency=("recency", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
    ).reset_index()

    categories = ["Recency (Freshness)", "Frequency (Orders)", "Monetary (Spend)"]
    
    # Invert recency so outer ring = better (fresher)
    max_r = labeled_df["recency"].max() or 1
    max_f = labeled_df["frequency"].max() or 1
    max_m = labeled_df["monetary"].max() or 1

    fig = go.Figure()

    for _, row in grouped.iterrows():
        seg_name = row["segment"]
        r_score = max(0, 100 - (row["recency"] / max_r * 100))
        f_score = min(100, (row["frequency"] / max_f * 100))
        m_score = min(100, (row["monetary"] / max_m * 100))

        values = [r_score, f_score, m_score, r_score]  # Close loop
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=seg_name,
                line_color=SEGMENT_COLORS.get(seg_name, "#333333"),
                fill="toself",
                opacity=0.3,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        title="Segment DNA Comparison (Normalized Medians)",
        margin=dict(l=40, r=40, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig

"""
3D and multi-dimensional scatter visualizers for customer cluster exploration.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import SEGMENT_COLORS


def scatter_3d_clusters(
    labeled_df: pd.DataFrame,
    use_log_monetary: bool = False,
) -> go.Figure:
    """
    Interactive 3D scatter plot of customers positioned by Recency, Frequency, and Monetary values.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame containing 'recency', 'frequency', 'monetary', and 'segment'.
    use_log_monetary : bool, default False
        Whether to plot log-transformed monetary scale for better outlier visibility.

    Returns
    -------
    go.Figure
        Interactive 3D Plotly Figure.
    """
    plot_df = labeled_df.copy()
    z_col = "monetary"
    z_title = "Monetary ($ Total Spend)"

    if use_log_monetary:
        plot_df["log_monetary"] = np.log10(plot_df["monetary"].clip(lower=1))
        z_col = "log_monetary"
        z_title = "Monetary (Log10 $)"

    fig = px.scatter_3d(
        plot_df,
        x="recency",
        y="frequency",
        z=z_col,
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="customer_id",
        hover_data={
            "recency": True,
            "frequency": True,
            "monetary": ":$,.2f",
            "segment": True,
            z_col: False,
        },
        labels={
            "recency": "Recency (Days)",
            "frequency": "Frequency (Orders)",
            z_col: z_title,
            "segment": "Segment",
        },
        title="3D Customer Cluster Space",
        opacity=0.85,
    )

    fig.update_traces(
        marker=dict(size=5, line=dict(width=0.5, color="white")),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Recency (Days)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Frequency (Orders)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=z_title, backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=620,
    )
    return fig


def scatter_2d_projections(labeled_df: pd.DataFrame) -> go.Figure:
    """
    Pairwise 2D projections: Recency vs Frequency, Recency vs Monetary, Frequency vs Monetary.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recency vs Frequency",
            "Recency vs Monetary (Log)",
            "Frequency vs Monetary (Log)",
        ),
    )

    segments = labeled_df["segment"].unique()

    for seg in segments:
        seg_data = labeled_df[labeled_df["segment"] == seg]
        color = SEGMENT_COLORS.get(seg, "#4A90E2")

        # 1. R vs F
        fig.add_trace(
            go.Scatter(
                x=seg_data["recency"],
                y=seg_data["frequency"],
                mode="markers",
                name=seg,
                legendgroup=seg,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Cust: %{text}<br>R: %{x}d<br>F: %{y}<extra></extra>",
                text=seg_data["customer_id"],
            ),
            row=1,
            col=1,
        )

        # 2. R vs M
        fig.add_trace(
            go.Scatter(
                x=seg_data["recency"],
                y=seg_data["monetary"],
                mode="markers",
                name=seg,
                legendgroup=seg,
                showlegend=False,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Cust: %{text}<br>R: %{x}d<br>M: $%{y:,.2f}<extra></extra>",
                text=seg_data["customer_id"],
            ),
            row=1,
            col=2,
        )

        # 3. F vs M
        fig.add_trace(
            go.Scatter(
                x=seg_data["frequency"],
                y=seg_data["monetary"],
                mode="markers",
                name=seg,
                legendgroup=seg,
                showlegend=False,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Cust: %{text}<br>F: %{x}<br>M: $%{y:,.2f}<extra></extra>",
                text=seg_data["customer_id"],
            ),
            row=1,
            col=3,
        )

    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=3)

    fig.update_xaxes(title_text="Recency (Days)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Orders)", row=1, col=1)

    fig.update_xaxes(title_text="Recency (Days)", row=1, col=2)
    fig.update_yaxes(title_text="Monetary ($ Log)", row=1, col=2)

    fig.update_xaxes(title_text="Frequency (Orders)", row=1, col=3)
    fig.update_yaxes(title_text="Monetary ($ Log)", row=1, col=3)

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
    )
    return fig

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
    currency_symbol: str = "$",
    size_by_clv: bool = False,
    color_by_churn: bool = False,
) -> go.Figure:
    """
    Interactive 3D scatter plot of customers positioned by Recency, Frequency, and Monetary values.
    Supports point sizing scaled by predicted 6-month CLV and color-mapping to churn probability.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame containing 'recency', 'frequency', 'monetary', and 'segment'.
    use_log_monetary : bool, default False
        Whether to plot log-transformed monetary scale for better outlier visibility.
    currency_symbol : str, default "$"
        Currency symbol to display in axis labels and hover data.
    size_by_clv : bool, default False
        Whether to scale 3D marker size by predicted 6-month CLV.
    color_by_churn : bool, default False
        Whether to color points along a continuous churn risk gradient (0% green to 100% red).

    Returns
    -------
    go.Figure
        Interactive 3D Plotly Figure.
    """
    plot_df = labeled_df.copy()
    z_col = "monetary"
    z_title = f"Monetary ({currency_symbol} Total Spend)"

    if use_log_monetary:
        plot_df["log_monetary"] = np.log10(plot_df["monetary"].clip(lower=1))
        z_col = "log_monetary"
        z_title = f"Monetary (Log10 {currency_symbol})"

    size_col = None
    if size_by_clv and "predicted_clv_6m" in plot_df.columns:
        plot_df["clv_marker_size"] = np.clip(np.sqrt(np.maximum(1.0, plot_df["predicted_clv_6m"])), 3, 20)
        size_col = "clv_marker_size"

    hover_dict = {
        "recency": True,
        "frequency": True,
        "monetary": ":,.2f",
        "segment": True,
        z_col: False,
    }
    if "predicted_clv_6m" in plot_df.columns:
        hover_dict["predicted_clv_6m"] = ":,.2f"
    if "churn_probability" in plot_df.columns:
        hover_dict["churn_probability"] = ":.1%"

    if color_by_churn and "churn_probability" in plot_df.columns:
        fig = px.scatter_3d(
            plot_df,
            x="recency",
            y="frequency",
            z=z_col,
            color="churn_probability",
            color_continuous_scale="RdYlGn_r",
            size=size_col,
            hover_name="customer_id",
            hover_data=hover_dict,
            labels={
                "recency": "Recency (Days)",
                "frequency": "Frequency (Orders)",
                z_col: z_title,
                "churn_probability": "Churn Risk",
            },
            title="3D Customer Space (Color: Churn Risk, Size: Predicted CLV)",
            opacity=0.85,
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x="recency",
            y="frequency",
            z=z_col,
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            size=size_col,
            hover_name="customer_id",
            hover_data=hover_dict,
            labels={
                "recency": "Recency (Days)",
                "frequency": "Frequency (Orders)",
                z_col: z_title,
                "segment": "Segment",
            },
            title="3D Customer Cluster Space",
            opacity=0.85,
        )

    if size_col is None:
        fig.update_traces(
            marker=dict(size=5, line=dict(width=0.5, color="white")),
        )
    else:
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color="white")),
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


def scatter_2d_projections(
    labeled_df: pd.DataFrame,
    currency_symbol: str = "$",
) -> go.Figure:
    """
    Pairwise 2D projections: Recency vs Frequency, Recency vs Monetary, Frequency vs Monetary.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recency vs Frequency",
            f"Recency vs Monetary ({currency_symbol} Log)",
            f"Frequency vs Monetary ({currency_symbol} Log)",
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
                hovertemplate=f"Cust: %{{text}}<br>R: %{{x}}d<br>M: {currency_symbol}%{{y:,.2f}}<extra></extra>",
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
                hovertemplate=f"Cust: %{{text}}<br>F: %{{x}}<br>M: {currency_symbol}%{{y:,.2f}}<extra></extra>",
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
    fig.update_yaxes(title_text=f"Monetary ({currency_symbol} Log)", row=1, col=2)

    fig.update_xaxes(title_text="Frequency (Orders)", row=1, col=3)
    fig.update_yaxes(title_text=f"Monetary ({currency_symbol} Log)", row=1, col=3)

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center"),
    )
    return fig

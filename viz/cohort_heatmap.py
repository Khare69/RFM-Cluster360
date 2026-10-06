"""
Cohort retention heatmap visualizer.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go


def cohort_retention_heatmap(
    retention_matrix: pd.DataFrame,
    counts_matrix: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Constructs an interactive, annotated retention heatmap.

    Parameters
    ----------
    retention_matrix : pd.DataFrame
        Matrix of retention percentages (rows = cohort month, cols = period number).
    counts_matrix : pd.DataFrame, optional
        Matrix of raw customer counts for hover tooltips.

    Returns
    -------
    go.Figure
        Plotly Heatmap Figure.
    """
    z_values = retention_matrix.values
    x_labels = [f"Month {c}" if c != 0 else "Cohort Size (M0)" for c in retention_matrix.columns]
    y_labels = list(retention_matrix.index)

    # Format annotations and hover tooltips
    annotations = []
    hover_text = []

    for i, row_cohort in enumerate(y_labels):
        row_hover = []
        for j, col_idx in enumerate(retention_matrix.columns):
            val = z_values[i][j]
            if np.isnan(val):
                row_hover.append(f"Cohort {row_cohort}<br>Month {col_idx}: No Data")
            elif col_idx == 0:
                count_str = f"{int(counts_matrix.iloc[i, 0]):,}" if counts_matrix is not None else "100%"
                # Dynamic font contrast: dark for pale background (< 50%), white for deep blue (>= 50%)
                font_color = "#1A202C" if val < 50.0 else "#FFFFFF"
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=row_cohort,
                        text=f"<b>{count_str}</b>",
                        font=dict(size=12, color=font_color),
                        showarrow=False,
                    )
                )
                row_hover.append(f"Cohort {row_cohort}<br>Acquired: {count_str} customers")
            else:
                pct_str = f"{val:.1f}%"
                font_color = "#1A202C" if val < 50.0 else "#FFFFFF"
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=row_cohort,
                        text=pct_str,
                        font=dict(size=12, color=font_color),
                        showarrow=False,
                    )
                )
                raw_c = f"{int(counts_matrix.iloc[i, j]):,} cust" if counts_matrix is not None and not np.isnan(counts_matrix.iloc[i, j]) else ""
                row_hover.append(f"Cohort {row_cohort}<br>Month {col_idx}: {pct_str} ({raw_c})")
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            hoverinfo="text",
            hovertext=hover_text,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title="Retention %"),
        )
    )

    fig.update_layout(
        title="Monthly Cohort Retention Rate (%)",
        xaxis=dict(title="Months Since Acquisition", side="top"),
        yaxis=dict(title="Acquisition Cohort (Month)", autorange="reversed"),
        annotations=annotations,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(380, len(y_labels) * 38 + 100),
    )
    return fig

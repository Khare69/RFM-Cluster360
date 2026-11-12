"""
Plotly visualizers for Out-of-Time Backtesting and Segment Migration.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go


def render_transition_heatmap(
    prob_matrix: pd.DataFrame,
    counts_matrix: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Constructs an interactive annotated Markov segment transition probability heatmap.

    Parameters
    ----------
    prob_matrix : pd.DataFrame
        Normalized transition probability matrix (rows = T1 segments, cols = T2 segments).
    counts_matrix : pd.DataFrame, optional
        Raw transition counts matrix for hover tooltips.

    Returns
    -------
    go.Figure
        Plotly Heatmap Figure.
    """
    z_probs = prob_matrix.values * 100.0  # Convert to percentages 0-100%
    x_labels = list(prob_matrix.columns)
    y_labels = list(prob_matrix.index)

    annotations = []
    hover_text = []

    for i, t1_seg in enumerate(y_labels):
        row_hover = []
        for j, t2_seg in enumerate(x_labels):
            prob_pct = z_probs[i][j]
            count_val = int(counts_matrix.iloc[i, j]) if counts_matrix is not None else 0

            # Dynamic text contrast: dark for pale cells (< 40%), white for darker cells (>= 40%)
            font_color = "#1A202C" if prob_pct < 40.0 else "#FFFFFF"

            annotations.append(
                dict(
                    x=t2_seg,
                    y=t1_seg,
                    text=f"<b>{prob_pct:.1f}%</b>",
                    font=dict(size=12, color=font_color),
                    showarrow=False,
                )
            )

            row_hover.append(
                f"<b>Calibration (T1):</b> {t1_seg}<br>"
                f"<b>Outcome (T2):</b> {t2_seg}<br>"
                f"<b>Migration Rate:</b> {prob_pct:.1f}%<br>"
                f"<b>Customers:</b> {count_val:,}"
            )
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_probs,
            x=x_labels,
            y=y_labels,
            hoverinfo="text",
            text=hover_text,
            colorscale=[
                [0.0, "#F7FAFC"],
                [0.2, "#EBF8FF"],
                [0.5, "#63B3ED"],
                [0.8, "#3182CE"],
                [1.0, "#1A365D"],
            ],
            zmin=0.0,
            zmax=100.0,
            colorbar=dict(
                title="Transition %",
                ticksuffix="%",
                thickness=16,
                len=0.9,
            ),
        )
    )

    fig.update_layout(
        title="<b>Segment Migration Matrix (Markov Transition Flow)</b>",
        xaxis=dict(
            title="<b>Outcome Window (T2) Segment State</b>",
            tickangle=-25,
            side="bottom",
        ),
        yaxis=dict(
            title="<b>Calibration Window (T1) Segment Baseline</b>",
            autorange="reversed",
        ),
        annotations=annotations,
        margin=dict(l=60, r=40, t=60, b=80),
        height=480,
    )

    return fig


def render_forward_spend_comparison(
    forward_metrics_df: pd.DataFrame,
    currency_symbol: str = "$",
) -> go.Figure:
    """
    Constructs a horizontal bar chart comparing total forward revenue realized
    in Window T2 across calibration personas.

    Parameters
    ----------
    forward_metrics_df : pd.DataFrame
        DataFrame produced by evaluate_forward_monetary_realization().
    currency_symbol : str, default '$'
        Currency prefix.

    Returns
    -------
    go.Figure
        Plotly Bar Figure.
    """
    sorted_df = forward_metrics_df.sort_values(by="total_forward_spend", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["total_forward_spend"],
            y=sorted_df["segment"],
            orientation="h",
            marker=dict(
                color=sorted_df["total_forward_spend"],
                colorscale="Blues",
            ),
            text=sorted_df.apply(
                lambda r: f"{currency_symbol}{r['total_forward_spend']:,.0f} ({r['forward_retention_pct']}% active)",
                axis=1,
            ),
            textposition="auto",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Total Forward Spend: {currency_symbol}%{{x:,.2f}}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="<b>Forward Revenue Realization by T1 Segment (T2 Actuals)</b>",
        xaxis=dict(title=f"Total Realized Revenue ({currency_symbol})"),
        yaxis=dict(title="Calibration Segment (T1)"),
        margin=dict(l=60, r=40, t=50, b=50),
        height=380,
    )

    return fig


def render_segment_retention_rates(
    forward_metrics_df: pd.DataFrame,
) -> go.Figure:
    """
    Constructs a bar chart showing the forward customer retention rate (%)
    across calibration personas.

    Parameters
    ----------
    forward_metrics_df : pd.DataFrame
        DataFrame produced by evaluate_forward_monetary_realization().

    Returns
    -------
    go.Figure
        Plotly Bar Figure.
    """
    sorted_df = forward_metrics_df.sort_values(by="forward_retention_pct", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["segment"],
            y=sorted_df["forward_retention_pct"],
            marker=dict(color="#3182CE"),
            text=sorted_df["forward_retention_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Retention Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Forward Customer Retention Rate by Persona</b>",
        xaxis=dict(title="Calibration Segment (T1)"),
        yaxis=dict(title="Retention Rate (%)", range=[0, 105]),
        margin=dict(l=50, r=40, t=50, b=50),
        height=360,
    )

    return fig

"""
Customer profile visualizations: Individual transaction timeline and percentile comparison metrics.
"""

from typing import Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def customer_purchase_timeline(
    customer_tx_df: pd.DataFrame,
    customer_id: str,
) -> go.Figure:
    """
    Timeline of individual orders placed by a single customer over time.

    Parameters
    ----------
    customer_tx_df : pd.DataFrame
        Filtered transaction rows for the specified customer.
    customer_id : str
        Customer identifier.

    Returns
    -------
    go.Figure
        Interactive Plotly timeline chart.
    """
    if customer_tx_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No transaction records found for Customer {customer_id}")
        return fig

    # Calculate line totals before grouping to avoid index slicing issues
    work_df = customer_tx_df.copy()
    work_df["line_total"] = work_df["quantity"] * work_df["unit_price"]

    # Group by invoice to aggregate order totals
    agg_dict = {
        "order_total": ("line_total", "sum"),
        "item_count": ("quantity", "sum"),
    }
    if "description" in work_df.columns:
        agg_dict["distinct_products"] = ("description", "nunique")
    else:
        agg_dict["distinct_products"] = ("invoice_no", "count")

    orders = (
        work_df.groupby(["invoice_no", "invoice_date"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values(by="invoice_date")
    )
    orders["order_total"] = orders["order_total"].round(2)

    fig = px.scatter(
        orders,
        x="invoice_date",
        y="order_total",
        size="item_count",
        hover_name="invoice_no",
        hover_data={
            "invoice_date": "|%Y-%m-%d %H:%M",
            "order_total": ":$,.2f",
            "item_count": True,
        },
        labels={
            "invoice_date": "Order Date",
            "order_total": "Order Total ($)",
            "item_count": "Total Items",
        },
        title=f"Purchase History Timeline — Customer {customer_id} ({len(orders)} Orders)",
        size_max=22,
    )

    # Add connecting line
    fig.add_trace(
        go.Scatter(
            x=orders["invoice_date"],
            y=orders["order_total"],
            mode="lines",
            line=dict(color="rgba(74, 144, 226, 0.4)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def customer_benchmark_bar(
    cust_rfm: pd.Series,
    all_rfm: pd.DataFrame,
) -> go.Figure:
    """
    Percentile comparison chart showing where this customer ranks across R, F, and M compared to all customers.
    """
    # Calculate percentile rank (0 to 100)
    # For recency, lower days = higher percentile (fresher)
    n_cust = len(all_rfm)
    if n_cust > 0:
        r_pct = round((all_rfm["recency"] >= cust_rfm["recency"]).mean() * 100, 1)
        f_pct = round((all_rfm["frequency"] <= cust_rfm["frequency"]).mean() * 100, 1)
        m_pct = round((all_rfm["monetary"] <= cust_rfm["monetary"]).mean() * 100, 1)
    else:
        r_pct, f_pct, m_pct = 50.0, 50.0, 50.0

    df_pct = pd.DataFrame({
        "Metric": ["Recency (Freshness)", "Frequency (Order Volume)", "Monetary (Total Spend)"],
        "Percentile": [r_pct, f_pct, m_pct],
        "Raw Value": [
            f"{int(cust_rfm['recency'])} days ago",
            f"{int(cust_rfm['frequency'])} orders",
            f"${cust_rfm['monetary']:,.2f}",
        ],
    })

    fig = px.bar(
        df_pct,
        x="Percentile",
        y="Metric",
        orientation="h",
        text=df_pct.apply(lambda r: f"{r['Percentile']}% percentile ({r['Raw Value']})", axis=1),
        range_x=[0, 105],
        color="Percentile",
        color_continuous_scale="Viridis",
        title="Customer Percentile Ranking (Relative to Entire Base)",
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=40, t=50, b=20),
        xaxis_title="Percentile (Higher = Better/More Active)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig

"""
Page 3: 3D Cluster Map — Interactive Multi-Dimensional Customer Space.
"""

import streamlit as st

from components.filters import render_sidebar_filters
from viz.cluster_scatter import scatter_2d_projections, scatter_3d_clusters

st.set_page_config(page_title="3D Cluster Map | RFM Cluster360", page_icon="🗺️", layout="wide")

# Guard clause
if "labeled_df" not in st.session_state or st.session_state.labeled_df is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

st.title("🗺️ 3D Customer Cluster Space")
st.markdown(
    "Rotate, zoom, and inspect your customer base in 3 dimensions (**Recency, Frequency, Monetary**). "
    "Each point is a customer, colored by their assigned behavioral segment."
)

curr_sym = st.session_state.get("currency_symbol", "$")
filtered_df = render_sidebar_filters(st.session_state.labeled_df, key_prefix="map_3d", currency_symbol=curr_sym)

if filtered_df.empty:
    st.warning("No customers match the current filter selection.")
    st.stop()

# Controls row
has_clv = "predicted_clv_6m" in filtered_df.columns
has_churn = "churn_probability" in filtered_df.columns

col_opt1, col_opt2, col_opt3 = st.columns([1.5, 1.5, 3])
with col_opt1:
    use_log = st.checkbox("Log10 Monetary (Z-Axis)", value=True, help="Squashes spending outliers so high spenders don't compress normal customers into a flat plane.")

with col_opt2:
    size_clv = False
    color_churn = False
    if has_clv:
        size_clv = st.checkbox("Size Points by Predicted CLV", value=True, help="Marker volume scales with predicted 6-month Customer Lifetime Value.")
    if has_churn:
        color_churn = st.checkbox("Color Points by Churn Risk", value=False, help="Switches categorical segment colors to a continuous churn gradient (Green = Active, Red = Churned).")

with col_opt3:
    st.caption("💡 **Tip:** Click and drag to rotate the 3D plot. Scroll to zoom in/out. Hover over any node to inspect customer details.")

# 3D Plot
fig_3d = scatter_3d_clusters(
    filtered_df,
    use_log_monetary=use_log,
    currency_symbol=curr_sym,
    size_by_clv=size_clv,
    color_by_churn=color_churn,
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")
st.subheader("📐 2D Cross-Section Projections")
st.markdown("View pairwise projections of the 3D space to isolate relationships between two dimensions at a time.")

fig_2d = scatter_2d_projections(filtered_df, currency_symbol=curr_sym)
st.plotly_chart(fig_2d, use_container_width=True)

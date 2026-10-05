"""
Page 4: Customer Search & Profile Inspection — Individual Accounts, Timelines, and Benchmark Rankings.
"""

import pandas as pd
import streamlit as st

from config.settings import SEGMENT_COLORS
from src.personas import PERSONA_DEFINITIONS
from viz.customer_timeline import customer_benchmark_bar, customer_purchase_timeline

st.set_page_config(page_title="Customer Search | RFM Cluster360", page_icon="👤", layout="wide")

# Guard clause
if "labeled_df" not in st.session_state or st.session_state.labeled_df is None:
    st.warning("⚠️ No dataset loaded. Please go to the **Home** page to upload a CSV or load the sample data.")
    if st.button("⬅️ Return to Home"):
        st.switch_page("app.py")
    st.stop()

st.title("👤 Customer Account Search & Timeline")
st.markdown("Search for any individual customer to inspect their behavioral metrics, segment ranking, and purchase timeline.")

labeled_df = st.session_state.labeled_df
clean_df = st.session_state.clean_df

# Customer Selection Controls
col_s1, col_s2 = st.columns([2, 2])

with col_s1:
    all_customer_ids = list(labeled_df["customer_id"].astype(str).unique())
    selected_id = st.selectbox(
        "Select or Search Customer ID",
        options=all_customer_ids,
        index=0,
        help="Type or select a customer ID from the dropdown",
    )

# Retrieve customer data
matching = labeled_df[labeled_df["customer_id"].astype(str) == str(selected_id)]
if matching.empty:
    st.warning(f"Customer '{selected_id}' not found in the segmented dataset.")
    st.stop()

cust_rfm = matching.iloc[0]
cust_segment = cust_rfm["segment"]
persona_meta = PERSONA_DEFINITIONS.get(cust_segment, {})
seg_color = SEGMENT_COLORS.get(cust_segment, "#4A90E2")

# Summary Header Card
with st.container(border=True):
    h1, h2, h3, h4 = st.columns([1.5, 1, 1, 1])
    with h1:
        st.markdown(f"### Customer `{selected_id}`")
        color_dot = f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background-color:{seg_color};margin-right:6px;'></span>"
        st.markdown(f"**Segment:** {color_dot} **{cust_segment}**", unsafe_allow_html=True)
        if "country" in cust_rfm and pd.notna(cust_rfm["country"]):
            st.caption(f"📍 Location: **{cust_rfm['country']}**")

    with h2:
        st.metric("Total Spend (M)", f"${cust_rfm['monetary']:,.2f}")
    with h3:
        st.metric("Total Orders (F)", f"{int(cust_rfm['frequency'])} orders")
    with h4:
        st.metric("Recency (R)", f"{int(cust_rfm['recency'])} days ago")

    first_seen = pd.to_datetime(cust_rfm['first_purchase']).strftime('%Y-%m-%d') if 'first_purchase' in cust_rfm and pd.notna(cust_rfm['first_purchase']) else "N/A"
    last_seen = pd.to_datetime(cust_rfm['last_purchase']).strftime('%Y-%m-%d') if 'last_purchase' in cust_rfm and pd.notna(cust_rfm['last_purchase']) else "N/A"
    st.caption(f"**First Seen:** {first_seen} | **Last Seen:** {last_seen}")
    if persona_meta.get("action"):
        st.info(f"💡 **Recommended Strategy for this Customer:** {persona_meta['action']}")

st.markdown("---")

# Percentile Benchmark
st.subheader("📊 Customer Benchmark Ranking")
st.markdown("See where this customer ranks across Recency, Frequency, and Monetary relative to your entire customer base.")
fig_bench = customer_benchmark_bar(cust_rfm, labeled_df)
st.plotly_chart(fig_bench, use_container_width=True)

st.markdown("---")

# Order Timeline
st.subheader("🕒 Purchase History Timeline")
cust_transactions = clean_df[clean_df["customer_id"].astype(str) == str(selected_id)].copy()

if not cust_transactions.empty:
    fig_timeline = customer_purchase_timeline(cust_transactions, selected_id)
    st.plotly_chart(fig_timeline, use_container_width=True)

    with st.expander(f"🧾 View Itemized Transaction Records ({len(cust_transactions)} items)"):
        st.dataframe(
            cust_transactions.sort_values(by="invoice_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No raw transaction line items available for this customer.")

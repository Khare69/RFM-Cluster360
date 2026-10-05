"""
Export UI component for one-click downloading of segmented customer data.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from src.export import prepare_export_dataframe, to_csv_bytes


def render_export_section(
    labeled_df: pd.DataFrame,
    title: str = "📥 Export Customer Segments",
):
    """
    Renders an export button and download options.
    """
    st.markdown(f"### {title}")
    st.markdown(
        "Export the complete segmented customer base with RFM scores, cluster IDs, and persona labels "
        "as a CSV file ready for your CRM, email marketing tool (Klaviyo, Mailchimp), or spreadsheet."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        include_dates = st.checkbox("Include first & last transaction dates", value=True)
        export_df = prepare_export_dataframe(labeled_df, include_dates=include_dates)
        csv_bytes = to_csv_bytes(export_df)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"customer_segments_rfm_{timestamp_str}.csv"

        st.download_button(
            label="⬇️ Download Segmented Customers (CSV)",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )

    with col2:
        st.caption(f"**Export Summary:**")
        st.caption(f"• **Rows:** {len(export_df):,} customers")
        st.caption(f"• **Columns:** {len(export_df.columns)} fields")
        st.caption(f"• **Format:** UTF-8 CSV")

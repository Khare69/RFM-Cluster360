"""
Export UI component for one-click downloading of segmented customer data.
"""

from datetime import datetime
import os
import pandas as pd
import streamlit as st

from src.cloud_storage import is_mock_mode, upload_to_s3
from src.export import prepare_export_dataframe, to_csv_bytes, to_parquet_bytes


def render_export_section(
    labeled_df: pd.DataFrame,
    title: str = "📥 Export Customer Segments",
):
    """
    Renders export options for CSV, Apache Parquet, and direct AWS S3 upload.
    """
    st.markdown(f"### {title}")
    st.markdown(
        "Export the complete segmented customer base with RFM scores, cluster IDs, and persona labels "
        "as a **CSV** or **Apache Parquet** file ready for your CRM, data warehouse, or cloud storage."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        c1, c2 = st.columns([1, 1])
        with c1:
            include_dates = st.checkbox("Include first & last transaction dates", value=True)
        with c2:
            export_format = st.radio(
                "Export Format",
                ["CSV (.csv)", "Apache Parquet (.parquet)"],
                horizontal=True,
            )

        export_df = prepare_export_dataframe(labeled_df, include_dates=include_dates)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

        if "Parquet" in export_format:
            file_bytes = to_parquet_bytes(export_df)
            file_name = f"customer_segments_rfm_{timestamp_str}.parquet"
            mime_type = "application/octet-stream"
            format_ext = "parquet"
            download_label = "⬇️ Download Segmented Customers (Parquet)"
            format_desc = "Apache Parquet (Snappy Columnar)"
        else:
            file_bytes = to_csv_bytes(export_df)
            file_name = f"customer_segments_rfm_{timestamp_str}.csv"
            mime_type = "text/csv"
            format_ext = "csv"
            download_label = "⬇️ Download Segmented Customers (CSV)"
            format_desc = "UTF-8 CSV"

        st.download_button(
            label=download_label,
            data=file_bytes,
            file_name=file_name,
            mime=mime_type,
            use_container_width=True,
            type="primary",
        )

    with col2:
        size_kb = len(file_bytes) / 1024
        size_str = f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
        st.caption("**Export Summary:**")
        st.caption(f"• **Rows:** {len(export_df):,} customers")
        st.caption(f"• **Columns:** {len(export_df.columns)} fields")
        st.caption(f"• **Format:** {format_desc}")
        st.caption(f"• **Payload Size:** {size_str}")

    # Cloud Export to S3 / Mock Storage Section
    st.markdown("---")
    with st.expander("☁️ Export Directly to AWS S3 / Cloud Storage", expanded=False):
        st.markdown(
            "Upload customer segments directly into an AWS S3 data lake or local mock storage."
        )

        mock_active = is_mock_mode()
        has_env_creds = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))

        if mock_active:
            st.info("🧪 **Local Mock Mode Active (`USE_LOCAL_MOCK=true`)**: Data will be written locally to `data/mock_s3/{bucket}/{key}`.")
        elif has_env_creds:
            st.success("🔒 **AWS Credentials Detected** from environment (`.env`).")
        else:
            st.caption("ℹ️ Configure AWS credentials in `.env` or input them below for temporary session upload.")

        s3_col1, s3_col2 = st.columns([1, 1])
        with s3_col1:
            dest_bucket = st.text_input(
                "S3 Bucket Name",
                value=os.getenv("AWS_S3_BUCKET", "rfm-customer-segments" if not mock_active else "retail-data"),
                key="export_s3_bucket",
            )
        with s3_col2:
            default_key = f"exports/customer_segments_{timestamp_str}.{format_ext}"
            dest_key = st.text_input(
                "S3 Destination Key",
                value=default_key,
                key="export_s3_key",
            )

        # Optional manual credentials accordion
        with st.expander("🔑 AWS Credentials Override (Optional)"):
            c_id, c_secret, c_region = st.columns(3)
            with c_id:
                cred_key_id = st.text_input("Access Key ID", type="password", key="export_s3_key_id")
            with c_secret:
                cred_secret = st.text_input("Secret Access Key", type="password", key="export_s3_secret")
            with c_region:
                cred_region = st.text_input(
                    "AWS Region",
                    value=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                    key="export_s3_region",
                )

        if st.button("🚀 Upload Segments to S3", use_container_width=True):
            if not dest_bucket.strip() or not dest_key.strip():
                st.error("Please provide both Bucket Name and Destination Key.")
            else:
                with st.spinner("Serializing and uploading to cloud storage..."):
                    success, msg = upload_to_s3(
                        df=export_df,
                        bucket=dest_bucket.strip(),
                        key=dest_key.strip(),
                        format=format_ext,
                        aws_access_key_id=cred_key_id.strip() or None,
                        aws_secret_access_key=cred_secret.strip() or None,
                        region_name=cred_region.strip() or None,
                    )
                if success:
                    st.success(f"✅ Successfully exported **{len(export_df):,} customers** to `{msg}`!")
                else:
                    st.error(f"❌ Upload failed: {msg}")


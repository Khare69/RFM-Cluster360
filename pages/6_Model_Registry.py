"""
Page 6: MLOps Model Artifact Registry.
Provides version control, artifact inspection, cross-version model comparison,
session hot-swapping, and real-time customer segmentation inference.
"""

from datetime import datetime
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import DEFAULT_CLUSTER_COLORS, SEGMENT_COLORS
from src.clustering import build_clustered_df
from src.model_registry import (
    delete_model_version,
    get_default_registry_dir,
    list_model_versions,
    load_model_artifact,
    predict_segment,
    save_model_artifact,
)
from src.personas import PERSONA_DEFINITIONS

st.set_page_config(page_title="Model Registry | RFM Cluster360", page_icon="🤖", layout="wide")

st.title("🤖 MLOps Model Artifact Registry")
st.markdown(
    "Manage, version, compare, and serve trained K-Means segmentation models. "
    "Every model artifact encapsulates the Scikit-Learn estimator, fitted `StandardScaler`, "
    "clustering metrics, and human-readable persona mappings."
)

registry_dir = get_default_registry_dir()
versions = list_model_versions(registry_dir)

# --- Top Status & Quick Actions Banner ---
top_col1, top_col2, top_col3 = st.columns([1.2, 1.2, 1.6])

with top_col1:
    st.metric("Registered Versions", len(versions))

with top_col2:
    if versions:
        best_sil_ver = max(versions, key=lambda v: (v.get("silhouette_score") or 0.0))
        st.metric(
            "Best Silhouette Score",
            f"{best_sil_ver.get('silhouette_score', 0.0):.4f}",
            f"{best_sil_ver.get('version_id')} (k={best_sil_ver.get('k')})",
        )
    else:
        st.metric("Best Silhouette Score", "N/A")

with top_col3:
    active_name = (
        os.path.basename(st.session_state.model_version)
        if ("model_version" in st.session_state and st.session_state.model_version)
        else "None (In-Memory Session)"
    )
    st.info(f"**Active Session Model:** `{active_name}`")

st.markdown("---")

# --- Tab Layout: Registry, Compare, and Live Inference ---
tab_versions, tab_compare, tab_inference = st.tabs([
    "📦 Model Artifacts & Deployment",
    "📊 Cross-Version Evaluation",
    "⚡ Real-Time Customer Classifier",
])

# ==========================================
# TAB 1: MODEL ARTIFACTS & DEPLOYMENT
# ==========================================
with tab_versions:
    st.subheader("Model Artifact Catalog")
    st.caption(f"Persisted artifacts directory: `{registry_dir}`")

    # Form to save current session model if fitted
    has_active_model = (
        "fitted_model" in st.session_state
        and st.session_state.fitted_model is not None
        and "fitted_scaler" in st.session_state
        and st.session_state.fitted_scaler is not None
        and "cluster_map" in st.session_state
        and st.session_state.cluster_map is not None
    )

    with st.expander("💾 Save Current In-Memory Model as Versioned Artifact", expanded=not bool(versions)):
        if has_active_model:
            c_save1, c_save2 = st.columns([2, 1])
            with c_save1:
                custom_note = st.text_input(
                    "Version Notes / Description",
                    value=f"Model k={st.session_state.selected_k} trained on {len(st.session_state.rfm_df) if 'rfm_df' in st.session_state and st.session_state.rfm_df is not None else 'unknown'} customers",
                    placeholder="e.g. Production Candidate, Q4 Retail Baseline",
                    key="save_custom_note",
                )
            with c_save2:
                st.write("")
                st.write("")
                if st.button("Save Artifact", type="primary", use_container_width=True):
                    with st.spinner("Serializing and writing model artifacts..."):
                        new_dir = save_model_artifact(
                            model=st.session_state.fitted_model,
                            scaler=st.session_state.fitted_scaler,
                            cluster_map=st.session_state.cluster_map,
                            eval_metrics=st.session_state.get("clustering_metrics"),
                            customer_count=len(st.session_state.rfm_df) if ("rfm_df" in st.session_state and st.session_state.rfm_df is not None) else None,
                            notes=custom_note.strip() or None,
                        )
                        st.session_state.model_version = new_dir
                        st.success(f"✅ Successfully persisted artifact version: `{os.path.basename(new_dir)}`")
                        st.rerun()
        else:
            st.info(
                "💡 No fitted model currently in session memory. "
                "Go to the **Home** page and run the pipeline or select a sample dataset to train a model."
            )

    if not versions:
        st.warning("No model versions have been saved to the registry yet.")
    else:
        # Display Registry Summary Table
        summary_rows = []
        for v in versions:
            created_dt = v.get("created_at", "")
            if created_dt:
                try:
                    dt = datetime.fromisoformat(created_dt)
                    formatted_dt = dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    formatted_dt = created_dt
            else:
                formatted_dt = "Unknown"

            summary_rows.append({
                "Version ID": v.get("version_id"),
                "Created At": formatted_dt,
                "Clusters (k)": v.get("k"),
                "Silhouette Score": v.get("silhouette_score", 0.0),
                "Inertia": v.get("inertia", 0.0),
                "Training Customers": v.get("customer_count", "N/A"),
                "Notes": v.get("notes", ""),
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Model Lifecycle Management: Inspect & Deploy")

        ver_col1, ver_col2 = st.columns([1, 2])
        version_names = [v["version_id"] for v in versions]

        with ver_col1:
            selected_v_id = st.selectbox("Select Model Version", options=version_names, index=0)
            selected_meta = next(v for v in versions if v["version_id"] == selected_v_id)

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("🚀 Load & Deploy", type="primary", use_container_width=True):
                    with st.spinner(f"Loading {selected_v_id}..."):
                        try:
                            m, s, cmap, meta = load_model_artifact(selected_meta["path"])
                            st.session_state.fitted_model = m
                            st.session_state.fitted_scaler = s
                            st.session_state.cluster_map = cmap
                            st.session_state.selected_k = m.n_clusters
                            st.session_state.model_version = selected_meta["path"]

                            # If RFM data exists, re-cluster with newly loaded model
                            if "rfm_df" in st.session_state and st.session_state.rfm_df is not None:
                                labeled_updated = predict_segment(m, s, cmap, st.session_state.rfm_df)
                                st.session_state.clustered_df = build_clustered_df(
                                    st.session_state.rfm_df,
                                    labeled_updated["cluster"].values,
                                )
                                st.session_state.labeled_df = labeled_updated

                            st.success(f"✅ Loaded `{selected_v_id}` and activated in session state!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Failed to load artifact: {ex}")

            with b_col2:
                if st.button("🗑️ Delete Version", use_container_width=True):
                    if delete_model_version(selected_meta["path"]):
                        st.warning(f"Deleted `{selected_v_id}` from registry.")
                        if st.session_state.get("model_version") == selected_meta["path"]:
                            st.session_state.model_version = None
                        st.rerun()

        with ver_col2:
            st.markdown(f"#### Artifact Details: `{selected_v_id}`")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("k Clusters", selected_meta.get("k"))
            m_col2.metric("Silhouette Score", f"{selected_meta.get('silhouette_score', 0.0):.4f}")
            m_col3.metric("Inertia", f"{selected_meta.get('inertia', 0.0):,.1f}")
            m_col4.metric("Customers", selected_meta.get("customer_count", "N/A"))

            # Display persona mapping pills
            st.markdown("**Assigned Persona Mappings:**")
            cmap = selected_meta.get("cluster_map", {})
            p_cols = st.columns(min(len(cmap), 4) if cmap else 1)
            for i, (c_id, persona_name) in enumerate(cmap.items()):
                target_col = p_cols[i % len(p_cols)]
                color = SEGMENT_COLORS.get(persona_name, "#555")
                target_col.markdown(
                    f"<div style='padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; "
                    f"background-color: {color}20; border-left: 4px solid {color};'>"
                    f"<b>Cluster {c_id}:</b> {persona_name}</div>",
                    unsafe_allow_html=True,
                )

# ==========================================
# TAB 2: CROSS-VERSION EVALUATION
# ==========================================
with tab_compare:
    st.subheader("Cross-Version Model Evaluation")
    st.markdown("Compare separation quality (Silhouette Score) and compactness (Inertia) across registered models.")

    if len(versions) < 2:
        st.info("💡 At least 2 registered model versions are required to display cross-version comparisons. Save another model version to unlock side-by-side benchmarking.")
    else:
        comp_df = pd.DataFrame([
            {
                "version_id": v.get("version_id"),
                "k": v.get("k"),
                "silhouette_score": v.get("silhouette_score", 0.0) or 0.0,
                "inertia": v.get("inertia", 0.0) or 0.0,
                "customer_count": v.get("customer_count") or 0,
            }
            for v in versions
        ])

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            fig_sil = px.bar(
                comp_df,
                x="version_id",
                y="silhouette_score",
                color="k",
                color_continuous_scale="Blues",
                title="Silhouette Score by Version (Higher is Better)",
                labels={"silhouette_score": "Silhouette Score", "version_id": "Model Version", "k": "Clusters (k)"},
            )
            fig_sil.update_layout(xaxis_tickangle=-45, height=380)
            st.plotly_chart(fig_sil, use_container_width=True)

        with col_plot2:
            fig_ine = px.line(
                comp_df,
                x="version_id",
                y="inertia",
                markers=True,
                title="Inertia by Version (Lower within same k is Better)",
                labels={"inertia": "Inertia (SSE)", "version_id": "Model Version"},
            )
            fig_ine.update_traces(line_color="#e377c2", marker=dict(size=8))
            fig_ine.update_layout(xaxis_tickangle=-45, height=380)
            st.plotly_chart(fig_ine, use_container_width=True)

        st.markdown("#### Head-to-Head Version Comparison")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            v_a = st.selectbox("Baseline Model (A)", options=[v["version_id"] for v in versions], index=min(1, len(versions)-1))
        with cmp2:
            v_b = st.selectbox("Candidate Model (B)", options=[v["version_id"] for v in versions], index=0)

        meta_a = next(v for v in versions if v["version_id"] == v_a)
        meta_b = next(v for v in versions if v["version_id"] == v_b)

        sil_a = meta_a.get("silhouette_score") or 0.0
        sil_b = meta_b.get("silhouette_score") or 0.0
        sil_delta = sil_b - sil_a

        ine_a = meta_a.get("inertia") or 0.0
        ine_b = meta_b.get("inertia") or 0.0
        ine_delta = ine_b - ine_a

        k_a = meta_a.get("k", 0)
        k_b = meta_b.get("k", 0)

        delta_col1, delta_col2, delta_col3 = st.columns(3)
        delta_col1.metric("Clusters (k)", f"{k_b}", f"{k_b - k_a:+d} vs Model A")
        delta_col2.metric("Silhouette Score", f"{sil_b:.4f}", f"{sil_delta:+.4f} vs Model A")
        delta_col3.metric("Inertia", f"{ine_b:,.1f}", f"{ine_delta:+,.1f} vs Model A")

# ==========================================
# TAB 3: REAL-TIME INFERENCE CLASSIFIER
# ==========================================
with tab_inference:
    st.subheader("⚡ Real-Time Customer Inference Classifier")
    st.markdown(
        "Serve real-time customer segmentation predictions using the active or loaded model. "
        "Inputs are preprocessed with the persisted `StandardScaler` (including `log1p` squashing) "
        "and assigned to persona clusters dynamically."
    )

    # Determine which model to use for inference
    active_m = st.session_state.get("fitted_model")
    active_s = st.session_state.get("fitted_scaler")
    active_cmap = st.session_state.get("cluster_map")

    if not active_m or not active_s or not active_cmap:
        if versions:
            st.info("Loading latest registered model version for inference...")
            try:
                active_m, active_s, active_cmap, _ = load_model_artifact(versions[0]["path"])
            except Exception as e:
                st.error(f"Could not load fallback model: {e}")
        else:
            st.warning("⚠️ No trained model available for inference. Please run the segmentation pipeline first.")
            st.stop()

    inf_col_left, inf_col_right = st.columns([1, 1.2])

    with inf_col_left:
        st.markdown("#### Input Customer RFM Metrics")
        recency_val = st.number_input(
            "Recency (Days since last purchase)",
            min_value=0.0,
            max_value=1000.0,
            value=25.0,
            step=1.0,
            help="Number of days elapsed since the customer's most recent order.",
        )
        frequency_val = st.number_input(
            "Frequency (Total distinct orders placed)",
            min_value=1.0,
            max_value=500.0,
            value=8.0,
            step=1.0,
            help="Total count of completed transactions.",
        )
        monetary_val = st.number_input(
            "Monetary Spend (Total lifetime revenue in USD)",
            min_value=0.0,
            max_value=100000.0,
            value=1250.0,
            step=10.0,
            help="Total cumulative monetary spend.",
        )

        classify_btn = st.button("🔮 Classify Customer Segment", type="primary", use_container_width=True)

    with inf_col_right:
        st.markdown("#### Prediction Result & Marketing Playbook")
        if classify_btn or "last_prediction" in st.session_state:
            pred_input = {
                "recency": float(recency_val),
                "frequency": float(frequency_val),
                "monetary": float(monetary_val),
            }
            pred_result = predict_segment(active_m, active_s, active_cmap, pred_input)
            st.session_state.last_prediction = pred_result

            res = st.session_state.last_prediction
            assigned_label = res["segment"]
            assigned_cluster = res["cluster"]
            color = SEGMENT_COLORS.get(assigned_label, "#1f77b4")
            meta = PERSONA_DEFINITIONS.get(assigned_label, {
                "description": "Standard segmented customer profile.",
                "action": "Tailor marketing strategy based on behavioral RFM scores.",
            })

            st.markdown(
                f"""
                <div style="background: {color}18; border: 2px solid {color}; border-radius: 12px; padding: 20px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="color: {color}; margin: 0;">{assigned_label}</h2>
                        <span style="background: {color}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold; font-size: 14px;">
                            Cluster #{assigned_cluster}
                        </span>
                    </div>
                    <p style="margin-top: 12px; font-size: 15px; color: inherit; opacity: 0.9;">{meta.get('description', '')}</p>
                    <hr style="border: 0; border-top: 1px solid {color}40; margin: 12px 0;">
                    <h4 style="margin: 0 0 6px 0; color: inherit; opacity: 0.95;">Recommended Marketing Action:</h4>
                    <p style="margin: 0; font-size: 14px; font-style: italic; color: inherit; opacity: 0.85;">🎯 {meta.get('action', '')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Preprocessed Feature Scaling")
            # Show scaled feature values with defensive clipping
            clipped_inputs = [max(0.0, float(res["recency"])), max(1.0, float(res["frequency"])), max(0.0, float(res["monetary"]))]
            log_vals = np.log1p(clipped_inputs).reshape(1, -1)
            scaled_vals = active_s.transform(log_vals)[0]
            sf1, sf2, sf3 = st.columns(3)
            sf1.metric("Scaled Recency (z-score)", f"{scaled_vals[0]:.2f}")
            sf2.metric("Scaled Frequency (z-score)", f"{scaled_vals[1]:.2f}")
            sf3.metric("Scaled Monetary (z-score)", f"{scaled_vals[2]:.2f}")
        else:
            st.info("Click **'🔮 Classify Customer Segment'** to generate a real-time segment assignment and playbook.")

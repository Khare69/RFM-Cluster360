"""
Out-of-Time Backtesting and Segment Migration Module.
Provides temporal transaction window splitting, Markov segment transition matrices,
and forward revenue realization analysis.
Pure pandas/numpy operations with zero UI dependencies.
"""

from datetime import date, datetime
from typing import Any, Dict, Tuple, Union
import numpy as np
import pandas as pd

from src.rfm import calculate_rfm


def split_transaction_windows(
    df: pd.DataFrame,
    cutoff_date: Union[str, date, datetime, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits transaction records into two non-overlapping temporal windows:
    - Calibration Window (T1): transactions with invoice_date <= cutoff_date
    - Outcome Window (T2): transactions with invoice_date > cutoff_date

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe with an 'invoice_date' column.
    cutoff_date : str, date, datetime, or pd.Timestamp
        Temporal boundary separating the calibration and outcome periods.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (t1_transactions, t2_transactions)
    """
    if df.empty or "invoice_date" not in df.columns:
        raise ValueError("DataFrame is empty or missing 'invoice_date' column.")

    invoice_dates = pd.to_datetime(df["invoice_date"])
    min_date = invoice_dates.min()
    max_date = invoice_dates.max()

    cutoff_ts = pd.Timestamp(cutoff_date)

    if cutoff_ts < min_date or cutoff_ts >= max_date:
        raise ValueError(
            f"Cutoff date '{cutoff_ts.date()}' must fall strictly within dataset date range "
            f"({min_date.date()} to {max_date.date()})."
        )

    t1_df = df[invoice_dates <= cutoff_ts].copy()
    t2_df = df[invoice_dates > cutoff_ts].copy()

    if t1_df.empty or t2_df.empty:
        raise ValueError("Cutoff date creates an empty window in T1 or T2.")

    return t1_df, t2_df


def compute_windowed_rfm(
    df: pd.DataFrame,
    cutoff_date: Union[str, date, datetime, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes RFM metrics for both Calibration (T1) and Outcome (T2) windows.
    - T1 snapshot is set exactly to cutoff_date.
    - T2 snapshot is set to the day after the latest transaction in T2.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
    cutoff_date : Union[str, date, datetime, pd.Timestamp]
        Split boundary date.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (t1_rfm_df, t2_rfm_df)
    """
    t1_trans, t2_trans = split_transaction_windows(df, cutoff_date)

    t1_snapshot = pd.Timestamp(cutoff_date)
    t2_snapshot = pd.Timestamp(pd.to_datetime(t2_trans["invoice_date"]).max().date()) + pd.Timedelta(days=1)

    t1_rfm = calculate_rfm(t1_trans, snapshot_date=t1_snapshot)
    t2_rfm = calculate_rfm(t2_trans, snapshot_date=t2_snapshot)

    return t1_rfm, t2_rfm


def compute_segment_transition_matrix(
    t1_labeled_df: pd.DataFrame,
    t2_labeled_df: pd.DataFrame,
    churn_label: str = "Churned (Inactive)",
) -> Dict[str, Any]:
    """
    Computes customer segment transition counts, Markov transition probabilities,
    and retention/churn summary metrics between Calibration (T1) and Outcome (T2).

    Parameters
    ----------
    t1_labeled_df : pd.DataFrame
        T1 customer RFM table with 'customer_id' and 'segment' columns.
    t2_labeled_df : pd.DataFrame
        T2 customer RFM table with 'customer_id' and 'segment' columns.
    churn_label : str, default 'Churned (Inactive)'
        Label assigned to T1 customers who did not transact during T2.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'counts_matrix': pd.DataFrame of raw transition counts (T1 segments as index, T2 segments as columns)
        - 'prob_matrix': pd.DataFrame of transition probabilities normalized row-wise (row sums = 1.0)
        - 'retention_rate': float proportion of T1 customers active in T2
        - 'churn_rate': float proportion of T1 customers inactive in T2
        - 'total_t1_customers': int count of T1 customers
        - 'retained_count': int count of customers active in both T1 and T2
        - 'churned_count': int count of T1 customers inactive in T2
        - 'new_customers_count': int count of T2 customers who were not present in T1
    """
    if "customer_id" not in t1_labeled_df.columns or "segment" not in t1_labeled_df.columns:
        raise ValueError("t1_labeled_df must contain 'customer_id' and 'segment' columns.")
    if "customer_id" not in t2_labeled_df.columns or "segment" not in t2_labeled_df.columns:
        raise ValueError("t2_labeled_df must contain 'customer_id' and 'segment' columns.")

    t1_subset = t1_labeled_df[["customer_id", "segment"]].drop_duplicates("customer_id")
    t2_subset = t2_labeled_df[["customer_id", "segment"]].drop_duplicates("customer_id")

    t1_customers = set(t1_subset["customer_id"])
    t2_customers = set(t2_subset["customer_id"])

    retained_customers = t1_customers & t2_customers
    churned_customers = t1_customers - t2_customers
    new_customers = t2_customers - t1_customers

    # Merge on customer_id: left join from T1 to evaluate forward migration
    merged = pd.merge(
        t1_subset,
        t2_subset,
        on="customer_id",
        how="left",
        suffixes=("_t1", "_t2"),
    )

    merged["segment_t2"] = merged["segment_t2"].fillna(churn_label)

    # All unique T1 segments and T2 segments
    t1_segments = sorted(t1_subset["segment"].unique().tolist())
    t2_segments_active = sorted(t2_subset["segment"].unique().tolist())
    t2_segments_all = t2_segments_active + [churn_label]

    # Build counts matrix
    counts_matrix = (
        pd.crosstab(
            merged["segment_t1"],
            merged["segment_t2"],
            dropna=False,
        )
        .reindex(index=t1_segments, columns=t2_segments_all, fill_value=0)
    )

    # Compute row-normalized transition probabilities
    row_sums = counts_matrix.sum(axis=1)
    prob_matrix = counts_matrix.div(row_sums, axis=0).fillna(0.0)

    total_t1 = len(t1_customers)
    retention_rate = len(retained_customers) / total_t1 if total_t1 > 0 else 0.0
    churn_rate = len(churned_customers) / total_t1 if total_t1 > 0 else 0.0

    return {
        "counts_matrix": counts_matrix,
        "prob_matrix": prob_matrix,
        "retention_rate": round(retention_rate, 4),
        "churn_rate": round(churn_rate, 4),
        "total_t1_customers": total_t1,
        "retained_count": len(retained_customers),
        "churned_count": len(churned_customers),
        "new_customers_count": len(new_customers),
    }


def evaluate_forward_monetary_realization(
    t1_labeled_df: pd.DataFrame,
    t2_rfm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluates forward monetary realization, actual repeat orders, and active rates
    in Window T2 for each customer segment established in Window T1.

    Parameters
    ----------
    t1_labeled_df : pd.DataFrame
        T1 labeled customers containing 'customer_id' and 'segment'.
    t2_rfm_df : pd.DataFrame
        T2 RFM metrics containing 'customer_id', 'frequency', 'monetary'.

    Returns
    -------
    pd.DataFrame
        Segment-aggregated validation summary proving empirical predictive power.
    """
    if "customer_id" not in t1_labeled_df.columns or "segment" not in t1_labeled_df.columns:
        raise ValueError("t1_labeled_df must contain 'customer_id' and 'segment'.")

    t1_sub = t1_labeled_df[["customer_id", "segment"]].drop_duplicates("customer_id")
    t2_sub = t2_rfm_df[["customer_id", "frequency", "monetary"]].drop_duplicates("customer_id")

    merged = pd.merge(t1_sub, t2_sub, on="customer_id", how="left")

    # Customers not transacting in T2 have 0 forward orders and 0 forward spend
    merged["t2_orders"] = merged["frequency"].fillna(0).astype(int)
    merged["t2_spend"] = merged["monetary"].fillna(0.0)
    merged["t2_active"] = merged["frequency"].notna().astype(int)

    summary = (
        merged.groupby("segment")
        .agg(
            customer_count=("customer_id", "count"),
            forward_active_count=("t2_active", "sum"),
            total_forward_spend=("t2_spend", "sum"),
            avg_forward_spend=("t2_spend", "mean"),
            avg_forward_orders=("t2_orders", "mean"),
        )
        .reset_index()
    )

    summary["forward_retention_pct"] = (
        (summary["forward_active_count"] / summary["customer_count"]) * 100
    ).round(2)

    # Active-only spend average
    active_mask = merged["t2_active"] == 1
    active_avg = (
        merged[active_mask]
        .groupby("segment")["t2_spend"]
        .mean()
        .rename("avg_spend_active_buyers")
        .reset_index()
    )

    summary = pd.merge(summary, active_avg, on="segment", how="left")
    summary["avg_spend_active_buyers"] = summary["avg_spend_active_buyers"].fillna(0.0).round(2)
    summary["avg_forward_spend"] = summary["avg_forward_spend"].round(2)
    summary["avg_forward_orders"] = summary["avg_forward_orders"].round(2)
    summary["total_forward_spend"] = summary["total_forward_spend"].round(2)

    # Sort by total forward spend descending
    return summary.sort_values(by="total_forward_spend", ascending=False).reset_index(drop=True)


def get_customer_migration_details(
    t1_labeled_df: pd.DataFrame,
    t2_labeled_df: pd.DataFrame,
    churn_label: str = "Churned (Inactive)",
) -> pd.DataFrame:
    """
    Produces a customer-level migration drilldown table mapping each individual customer
    from their T1 segment to their T2 segment along with monetary movement.

    Parameters
    ----------
    t1_labeled_df : pd.DataFrame
        T1 labeled customers ('customer_id', 'segment', 'monetary').
    t2_labeled_df : pd.DataFrame
        T2 labeled customers ('customer_id', 'segment', 'monetary').
    churn_label : str
        Label for churned customers.

    Returns
    -------
    pd.DataFrame
        Customer migration drilldown table.
    """
    t1_sub = t1_labeled_df[["customer_id", "segment", "monetary"]].rename(
        columns={"segment": "t1_segment", "monetary": "t1_spend"}
    )
    t2_sub = t2_labeled_df[["customer_id", "segment", "monetary"]].rename(
        columns={"segment": "t2_segment", "monetary": "t2_spend"}
    )

    merged = pd.merge(t1_sub, t2_sub, on="customer_id", how="left")
    merged["t2_segment"] = merged["t2_segment"].fillna(churn_label)
    merged["t2_spend"] = merged["t2_spend"].fillna(0.0)

    # Classify transition type
    def classify_transition(row: pd.Series) -> str:
        if row["t2_segment"] == churn_label:
            return "Churned"
        if row["t1_segment"] == row["t2_segment"]:
            return "Retained (Stable)"
        return "Migrated"

    merged["migration_status"] = merged.apply(classify_transition, axis=1)
    return merged


def run_backtesting_pipeline(
    df: pd.DataFrame,
    cutoff_date: Union[str, date, datetime, pd.Timestamp],
    k: int = 4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the end-to-end Out-of-Time Backtesting analysis:
    1. Splits transactions into Calibration (T1) and Outcome (T2) windows.
    2. Calculates RFM metrics for both periods.
    3. Fits K-Means segmentation on T1 and maps human-interpretable personas.
    4. Projects the T1 model onto T2 customer behavior with zero model drift.
    5. Computes the Markov segment transition matrix, churn rates, and forward monetization.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe.
    cutoff_date : Union[str, date, datetime, pd.Timestamp]
        Split boundary date.
    k : int, default 4
        Number of clusters for T1 baseline.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Any]
        Complete backtesting diagnostic payload.
    """
    from src.clustering import build_clustered_df, preprocess_rfm, run_kmeans
    from src.personas import label_segments

    t1_trans, t2_trans = split_transaction_windows(df, cutoff_date)
    t1_rfm, t2_rfm = compute_windowed_rfm(df, cutoff_date)

    # 1. Fit segmentation model on Calibration Window (T1)
    scaled_t1, scaler, features = preprocess_rfm(t1_rfm)
    actual_k = min(k, max(2, len(t1_rfm) - 1))
    labels_t1, centers_t1, kmeans_t1 = run_kmeans(scaled_t1, k=actual_k, random_state=random_state)

    t1_clustered = build_clustered_df(t1_rfm, labels_t1)
    t1_labeled, cluster_map = label_segments(t1_clustered)

    # 2. Project model onto Outcome Window (T2)
    if len(t2_rfm) > 0:
        t2_data = t2_rfm[features].copy()
        t2_log = np.log1p(np.maximum(t2_data.to_numpy(), 0.0))
        t2_scaled = scaler.transform(t2_log)
        labels_t2 = kmeans_t1.predict(t2_scaled)

        t2_clustered = build_clustered_df(t2_rfm, labels_t2)
        t2_clustered["segment"] = t2_clustered["cluster"].map(cluster_map).fillna("Other")
        t2_labeled = t2_clustered
    else:
        t2_labeled = t2_rfm.copy()
        t2_labeled["cluster"] = 0
        t2_labeled["segment"] = "Other"

    # 3. Compute transition matrix and forward realization
    transition_data = compute_segment_transition_matrix(t1_labeled, t2_labeled)
    forward_metrics = evaluate_forward_monetary_realization(t1_labeled, t2_rfm)
    migration_details = get_customer_migration_details(t1_labeled, t2_labeled)

    return {
        "cutoff_date": pd.Timestamp(cutoff_date),
        "t1_trans": t1_trans,
        "t2_trans": t2_trans,
        "t1_rfm": t1_rfm,
        "t2_rfm": t2_rfm,
        "t1_labeled": t1_labeled,
        "t2_labeled": t2_labeled,
        "cluster_map": cluster_map,
        "counts_matrix": transition_data["counts_matrix"],
        "prob_matrix": transition_data["prob_matrix"],
        "retention_rate": transition_data["retention_rate"],
        "churn_rate": transition_data["churn_rate"],
        "total_t1_customers": transition_data["total_t1_customers"],
        "retained_count": transition_data["retained_count"],
        "churned_count": transition_data["churned_count"],
        "new_customers_count": transition_data["new_customers_count"],
        "forward_metrics": forward_metrics,
        "migration_details": migration_details,
    }


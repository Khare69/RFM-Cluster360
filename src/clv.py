"""
Probabilistic Customer Lifetime Value (CLV) and Churn Modeling module.
Implements analytical Beta-Geometric/Negative Binomial Distribution (BG/NBD)
and Gamma-Gamma spend submodels using Scipy optimization.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


@dataclass
class BGNBDParameters:
    r: float
    alpha: float
    a: float
    b: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class GammaGammaParameters:
    p: float
    q: float
    v: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def prepare_clv_summary(
    transactions_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Transforms transaction logs into canonical RFM summary for BG/NBD and Gamma-Gamma:
    - x: Repeat transaction count (total orders - 1, >= 0).
    - t_x: Recency in days (time from first purchase to last purchase).
    - T: Customer age/tenure in days (time from first purchase to snapshot_date).
    - monetary_value: Average order spend for repeat purchases.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Cleaned transaction logs with ['customer_id', 'invoice_no', 'invoice_date', 'quantity', 'unit_price'].
    snapshot_date : pd.Timestamp, optional
        Reference cutoff date. Defaults to 1 day after max(invoice_date).

    Returns
    -------
    pd.DataFrame
        Summary table indexed by customer_id with ['frequency_repeat', 'recency_days', 'tenure_days', 'monetary_avg', 'first_order_value'].
    """
    df = transactions_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["line_total"] = df["quantity"] * df["unit_price"]

    # Order-level aggregation
    order_agg = df.groupby(["customer_id", "invoice_no"]).agg(
        order_date=("invoice_date", "min"),
        order_spend=("line_total", "sum"),
    ).reset_index()

    snap_dt = snapshot_date or (order_agg["order_date"].max() + pd.Timedelta(days=1))

    # Customer-level summary
    summary_list = []
    for cid, group in order_agg.groupby("customer_id"):
        group_sorted = group.sort_values("order_date")
        n_orders = len(group_sorted)
        first_date = group_sorted["order_date"].iloc[0]
        last_date = group_sorted["order_date"].iloc[-1]
        first_val = float(group_sorted["order_spend"].iloc[0])

        x = n_orders - 1
        t_x = (last_date - first_date).days
        T = (snap_dt - first_date).days

        if x > 0:
            repeat_orders = group_sorted.iloc[1:]
            m_avg = float(repeat_orders["order_spend"].mean())
        else:
            m_avg = first_val

        summary_list.append({
            "customer_id": str(cid),
            "frequency_repeat": int(x),
            "recency_days": float(max(0, t_x)),
            "tenure_days": float(max(1, T)),
            "monetary_avg": float(max(0.01, m_avg)),
            "first_order_value": float(max(0.01, first_val)),
            "total_orders": n_orders,
        })

    return pd.DataFrame(summary_list)


def fit_bgnbd(
    summary_df: pd.DataFrame,
    initial_params: Tuple[float, float, float, float] = (1.0, 10.0, 1.0, 1.0),
) -> BGNBDParameters:
    """
    Fits BG/NBD parameters (r, alpha, a, b) via Maximum Likelihood Estimation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with ['frequency_repeat', 'recency_days', 'tenure_days'].

    Returns
    -------
    BGNBDParameters
        Fitted parameters.
    """
    x = summary_df["frequency_repeat"].to_numpy(dtype=float)
    t_x = summary_df["recency_days"].to_numpy(dtype=float)
    T = summary_df["tenure_days"].to_numpy(dtype=float)

    def _neg_log_likelihood(params):
        r, alpha, a, b = params
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e12

        # Log-likelihood terms for BG/NBD
        term1 = gammaln(r + x) - gammaln(r) - gammaln(x + 1)
        term2 = betaln(a, b + x) - betaln(a, b)
        term3 = r * np.log(alpha / (alpha + T)) - x * np.log(alpha + T)

        # Handling x == 0 vs x > 0
        p_alive_ratio = (a / (b + np.maximum(x - 1, 0))) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        term4 = np.where(x > 0, np.log(1.0 + p_alive_ratio), 0.0)

        ll = term1 + term2 + term3 + term4
        return -float(np.sum(np.nan_to_num(ll, nan=-1e6)))

    bounds = [(1e-4, 50.0), (1e-4, 1000.0), (1e-4, 50.0), (1e-4, 50.0)]
    res = minimize(_neg_log_likelihood, initial_params, method="L-BFGS-B", bounds=bounds)

    if res.success:
        r, alpha, a, b = res.x
    else:
        r, alpha, a, b = initial_params

    return BGNBDParameters(r=round(float(r), 4), alpha=round(float(alpha), 4), a=round(float(a), 4), b=round(float(b), 4))


def calculate_p_alive(
    summary_df: pd.DataFrame,
    params: BGNBDParameters,
) -> np.ndarray:
    """
    Calculates posterior probability that a customer remains active P(Alive).

    Formula:
        P(Alive | r, alpha, a, b, x, t_x, T) = 1 / [1 + (a / (b + x)) * ((alpha + T) / (alpha + t_x))^(r + x)]
    """
    x = summary_df["frequency_repeat"].to_numpy(dtype=float)
    t_x = summary_df["recency_days"].to_numpy(dtype=float)
    T = summary_df["tenure_days"].to_numpy(dtype=float)

    ratio = (params.a / (params.b + x)) * ((params.alpha + T) / (params.alpha + t_x)) ** (params.r + x)
    p_alive = 1.0 / (1.0 + ratio)
    return np.clip(np.nan_to_num(p_alive, nan=0.5), 0.0, 1.0)


def predict_future_transactions(
    summary_df: pd.DataFrame,
    params: BGNBDParameters,
    t_days: float = 180.0,
) -> np.ndarray:
    """
    Computes expected number of repeat transactions E[Y(t)] over forward horizon t_days.
    """
    p_alive = calculate_p_alive(summary_df, params)
    x = summary_df["frequency_repeat"].to_numpy(dtype=float)
    T = summary_df["tenure_days"].to_numpy(dtype=float)

    # Expected rate = (r + x) / (alpha + T) * p_alive * t
    expected_rate = (params.r + x) / (params.alpha + T)
    expected_tx = expected_rate * p_alive * t_days
    return np.maximum(0.0, np.nan_to_num(expected_tx, nan=0.0))


def fit_gamma_gamma(
    summary_df: pd.DataFrame,
    initial_params: Tuple[float, float, float] = (1.0, 1.0, 10.0),
) -> GammaGammaParameters:
    """
    Fits Gamma-Gamma monetary submodel parameters (p, q, v) on repeat buyers.
    """
    repeat_buyers = summary_df[summary_df["frequency_repeat"] >= 1].copy()
    if len(repeat_buyers) < 10:
        return GammaGammaParameters(p=1.5, q=3.0, v=summary_df["monetary_avg"].mean())

    x = repeat_buyers["frequency_repeat"].to_numpy(dtype=float)
    m = repeat_buyers["monetary_avg"].to_numpy(dtype=float)

    def _neg_log_likelihood(params):
        p, q, v = params
        if p <= 0 or q <= 0 or v <= 0:
            return 1e12

        ll = (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(v)
            + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(x * m + v)
        )
        return -float(np.sum(np.nan_to_num(ll, nan=-1e6)))

    bounds = [(1e-3, 50.0), (1e-3, 50.0), (1e-3, 5000.0)]
    res = minimize(_neg_log_likelihood, initial_params, method="L-BFGS-B", bounds=bounds)

    if res.success:
        p, q, v = res.x
    else:
        p, q, v = initial_params

    return GammaGammaParameters(p=round(float(p), 4), q=round(float(q), 4), v=round(float(v), 4))


def predict_conditional_spend(
    summary_df: pd.DataFrame,
    params: GammaGammaParameters,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Predicts conditional expected transaction value E[M].
    For repeat buyers: uses fitted Gamma-Gamma formula.
    For zero-repeat buyers (x=0): applies empirical Bayes shrinkage toward population mean.
    """
    x = summary_df["frequency_repeat"].to_numpy(dtype=float)
    m = summary_df["monetary_avg"].to_numpy(dtype=float)
    pop_mean = summary_df["monetary_avg"].median()

    # Repeat buyers Gamma-Gamma formula
    p, q, v = params.p, params.q, params.v
    denom = p * x + q - 1.0
    valid_denom = np.where(denom > 0, denom, 1.0)
    expected_m = ((q - 1.0) / valid_denom) * (p * v / (q - 1.0 if q > 1.0 else 1.0)) + ((p * x) / valid_denom) * m

    # Empirical Bayes shrinkage for zero-repeat customers
    zero_mask = x == 0
    shrunk_m = (1.0 / (1.0 + kappa)) * m + (kappa / (1.0 + kappa)) * pop_mean
    expected_m[zero_mask] = shrunk_m[zero_mask]

    return np.maximum(0.01, np.nan_to_num(expected_m, nan=pop_mean))


def calculate_customer_clv(
    summary_df: pd.DataFrame,
    bgnbd_params: BGNBDParameters,
    gg_params: GammaGammaParameters,
    horizon_days: float = 180.0,
) -> pd.DataFrame:
    """
    Combines BG/NBD and Gamma-Gamma to generate full CLV and churn metrics per customer.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Customer RFM summary.
    bgnbd_params : BGNBDParameters
        Fitted BG/NBD parameters.
    gg_params : GammaGammaParameters
        Fitted Gamma-Gamma parameters.
    horizon_days : float, default 180.0
        Forward forecast window in days (default 6 months).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['customer_id', 'p_alive', 'churn_probability',
        'expected_transactions_6m', 'expected_avg_spend', 'predicted_clv_6m'].
    """
    p_alive = calculate_p_alive(summary_df, bgnbd_params)
    exp_tx = predict_future_transactions(summary_df, bgnbd_params, t_days=horizon_days)
    exp_spend = predict_conditional_spend(summary_df, gg_params)

    clv_df = pd.DataFrame({
        "customer_id": summary_df["customer_id"],
        "p_alive": np.round(p_alive, 4),
        "churn_probability": np.round(1.0 - p_alive, 4),
        "expected_transactions_6m": np.round(exp_tx, 2),
        "expected_avg_spend": np.round(exp_spend, 2),
        "predicted_clv_6m": np.round(exp_tx * exp_spend, 2),
    })

    return clv_df

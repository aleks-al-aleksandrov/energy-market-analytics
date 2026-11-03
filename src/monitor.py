"""Monitoring: rolling error and feature drift (Population Stability Index).

Retraining rule (also in README):
  * retrain monthly on the expanding window (default schedule), and
  * retrain immediately if the rolling 30-day MAE of the model is > 20 % above its
    CV level for 14 consecutive days, or if any key feature has PSI > 0.25.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import mae

KEY_FEATURES = ["load_forecast", "wind_forecast", "solar_forecast", "residual_load_forecast",
                "price_lag24", "price_RO_lag24", "fuel_gas_eur_mwh", "fuel_brent_usd"]


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI with quantile bins from the reference (training) distribution."""
    e, a = expected.dropna().to_numpy(), actual.dropna().to_numpy()
    if len(e) < 50 or len(a) < 50:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(f: pd.DataFrame, ref_mask, cur_mask, features=KEY_FEATURES) -> pd.DataFrame:
    rows = []
    for c in features:
        if c in f:
            v = psi(f.loc[ref_mask, c], f.loc[cur_mask, c])
            rows.append(dict(feature=c, psi=v,
                             status="stable" if v < 0.1 else ("moderate shift" if v < 0.25 else "significant shift")))
    return pd.DataFrame(rows)


def rolling_mae(pred: pd.DataFrame, models: list[str], window_days: int = 30) -> pd.DataFrame:
    daily = pred.groupby("delivery_day").apply(
        lambda g: pd.Series({m: np.mean(np.abs(g["actual"] - g[m])) for m in models}))
    return daily.rolling(window_days, min_periods=window_days // 2).mean()


def retrain_alert(roll: pd.Series, cv_level: float, ratio: float = 1.2, days: int = 14) -> pd.Series:
    above = roll > ratio * cv_level
    streak = above.groupby((~above).cumsum()).cumsum()
    return streak >= days


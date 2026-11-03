"""Walk-forward validation, metrics and significance tests."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.features import TARGET


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_months(f: pd.DataFrame) -> tuple[list[pd.Period], list[pd.Period]]:
    """(cv_months, holdout_months). Only complete months with targets are used."""
    days = f.loc[f[TARGET].notna(), "delivery_day"]
    months = pd.PeriodIndex(days, freq="M").unique().sort_values()
    last_day = days.max()
    if (last_day + pd.Timedelta(days=1)).day != 1:  # drop incomplete last month
        months = months[:-1]
    months = [m for m in months if m >= pd.Period(C.FIRST_TEST_MONTH, "M")]
    return months[:-C.HOLDOUT_MONTHS], months[-C.HOLDOUT_MONTHS:]


def walk_forward(f: pd.DataFrame, make_model: Callable, features: list[str],
                 months: list[pd.Period], target: str = TARGET, predict_fn: str = "predict",
                 min_train_days: int = 180) -> pd.Series:
    """Expanding window: train on all delivery days before month m, predict month m."""
    out = []
    dmonth = pd.PeriodIndex(f["delivery_day"], freq="M")
    valid = f[target].notna()
    for m in months:
        train = valid & (dmonth < m)
        test = (dmonth == m) & valid
        if train.sum() < min_train_days * 24 or test.sum() == 0:
            continue
        model = make_model()
        model.fit(f.loc[train, features], f.loc[train, target])
        pred = getattr(model, predict_fn)(f.loc[test, features])
        if pred.ndim == 2:
            pred = pred[:, 1]
        out.append(pd.Series(pred, index=f.index[test]))
    return pd.concat(out) if out else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def mae(y, p) -> float:
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(y) - np.asarray(p)) ** 2)))


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y) - np.asarray(q_pred)
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


def diebold_mariano(e_model, e_bench, day, loss: str = "abs", max_lag: int = 7) -> tuple[float, float]:
    """DM test on daily-averaged loss differentials (hours within a day are correlated).

    H0: equal accuracy. Negative statistic -> model better. Returns (stat, p-value two-sided).
    Uses Newey-West HAC variance with `max_lag` days and the Harvey et al. small-sample correction.
    """
    lf = np.abs if loss == "abs" else np.square
    d = pd.Series(lf(np.asarray(e_model)) - lf(np.asarray(e_bench))).groupby(np.asarray(day)).mean().dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    gamma0 = (dc ** 2).mean()
    var = gamma0
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        var += 2 * w * (dc.iloc[k:].values * dc.iloc[:-k].values).mean()
    stat = d.mean() / np.sqrt(var / n)
    h = 1
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


def summarise(pred: pd.DataFrame, models: list[str], spike_threshold: float | None = None) -> pd.DataFrame:
    """Overall metrics per model + per-fold spread + skill vs best naive + DM test."""
    y = pred["actual"]
    naives = [m for m in models if m.startswith("naive")]
    best_naive = min(naives, key=lambda m: mae(y, pred[m])) if naives else None
    rows = []
    folds = pd.PeriodIndex(pred["delivery_day"], freq="M")
    for m in models:
        fold_mae = pred.groupby(folds).apply(lambda g: mae(g["actual"], g[m]))
        r = dict(model=m, MAE=mae(y, pred[m]), RMSE=rmse(y, pred[m]),
                 MAE_fold_mean=fold_mae.mean(), MAE_fold_std=fold_mae.std(), n_folds=len(fold_mae))
        if best_naive:
            r["skill_vs_best_naive"] = 1 - r["MAE"] / mae(y, pred[best_naive])
            if m != best_naive:
                s, p = diebold_mariano(y - pred[m], y - pred[best_naive], pred["delivery_day"])
                r["DM_stat"], r["DM_p"] = s, p
                r["significant_5pct"] = bool(p < 0.05 and s < 0)
        if spike_threshold is not None:
            sp = y > spike_threshold
            r["MAE_spike_hours"] = mae(y[sp], pred.loc[sp, m]) if sp.any() else np.nan
            neg = y < 0
            r["MAE_negative_hours"] = mae(y[neg], pred.loc[neg, m]) if neg.any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["best_naive"] = best_naive
    return out


def error_by(pred: pd.DataFrame, models: list[str], key: str) -> pd.DataFrame:
    g = pred.groupby(key)
    return pd.DataFrame({m: g.apply(lambda d: mae(d["actual"], d[m])) for m in models})



def bootstrap_skill(pred: pd.DataFrame, model: str, naive: str, block_days: int = 7, n_boot: int = 2000,
                    seed: int = C.SEED) -> tuple[float, float, float]:
    """Skill = 1 - MAE_model / MAE_naive with a 95 % moving-block bootstrap CI over delivery days.

    Days (not hours) are resampled in blocks of `block_days`, because prices are strongly autocorrelated.
    Returns (skill, ci_low, ci_high).
    """
    d = pred.dropna(subset=["actual", model, naive])
    day = d["delivery_day"].dt.normalize()
    am = (d["actual"] - d[model]).abs().groupby(day).sum()
    an = (d["actual"] - d[naive]).abs().groupby(day).sum()
    n_hours = d.groupby(day).size()
    am, an, nh = am.to_numpy(), an.to_numpy(), n_hours.to_numpy()
    n = len(am)
    point = 1 - am.sum() / an.sum()
    if n < 2 * block_days:
        return float(point), np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n - block_days + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_days)[None, None, :]).reshape(n_boot, -1)[:, :n]
    skills = 1 - am[idx].sum(axis=1) / an[idx].sum(axis=1)
    lo, hi = np.quantile(skills, [0.025, 0.975])
    return float(point), float(lo), float(hi)

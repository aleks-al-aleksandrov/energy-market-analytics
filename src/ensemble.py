"""Production point forecast: an adaptive ensemble of four members (adopted in src/phase1_experiment.py as `ada4`).

Members: LightGBM, Ridge, level-adjusted LightGBM (all refitted weekly) and LEAR (per-hour lasso on asinh prices).
Weights for delivery day D are proportional to 1 / MAE^2 of each member over the previous 14 delivery days whose prices are
published (up to D-1). When the price level leaves the range the trees were trained on, the members that extrapolate (Ridge,
LEAR, level-adjusted) get the weight automatically.

The weights need each member's recent OUT-OF-SAMPLE forecasts. They live in reports/member_log.csv:
  * every weekly retrain rewrites the last 28 days from a small weekly walk-forward (`seed_member_log`),
  * every daily forecast appends the members' live forecasts (`log_members`; the first forecast of an hour is kept).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config as C
from src.features import TARGET

MEMBERS = ["lightgbm", "ridge", "level", "lear"]
WINDOW_DAYS, POWER, MIN_DAYS = 14, 2.0, 3
LEAR_WINDOW, LEAR_REFIT = 728, 7
SEED_DAYS = 28
LOG = "member_log.csv"


# ---------------------------------------------------------------------------------------------- LEAR
def _daily_wide(f: pd.DataFrame, col: str) -> pd.DataFrame:
    """Delivery day x hour-of-day (0..23) matrix; 23-hour days are forward-filled, the 25th hour of a long day is dropped."""
    pos = f.groupby("delivery_day").cumcount()
    w = f.assign(_pos=pos.to_numpy())[lambda d: d["_pos"] < 24].pivot(index="delivery_day", columns="_pos", values=col)
    return w.reindex(columns=range(24)).ffill(axis=1)


def lear_forecast(f: pd.DataFrame, test_days, window_days: int = LEAR_WINDOW, refit_days: int = LEAR_REFIT) -> pd.Series:
    """Lasso-Estimated AutoRegressive forecast for `test_days` (24 lasso regressions, one per hour, refitted every block).

    Regressors for day D (all known at gate closure): BG prices of D-1, D-2, D-3 and D-7 (24 hours each), neighbour prices of
    D-1 when the frame carries them, the residual-load forecast for D and D-1, and weekday dummies. Prices are asinh-transformed
    with a median/MAD scaler estimated on the calibration window; the lasso penalty is chosen by AIC. Training uses only
    complete days strictly before the block; the test day itself does not need a price (that is the live case).
    """
    from sklearn.linear_model import LassoLarsIC

    price = _daily_wide(f, TARGET)
    blocks = {"p1": price.shift(1, freq="D"), "p2": price.shift(2, freq="D"), "p3": price.shift(3, freq="D"),
              "p7": price.shift(7, freq="D")}
    for z in ("RO", "HU"):
        if f"price_{z}" in f:
            blocks[f"{z}1"] = _daily_wide(f, f"price_{z}").shift(1, freq="D")
    price_keys = list(blocks)
    if "residual_load_forecast" in f:
        res = _daily_wide(f, "residual_load_forecast")
        blocks["res0"], blocks["res1"] = res, res.shift(1, freq="D")
    days = price.index
    X = pd.concat({k: v.reindex(days) for k, v in blocks.items()}, axis=1)
    X.columns = [f"{a}_{b}" for a, b in X.columns]
    dow = pd.get_dummies(pd.DatetimeIndex(days).day_of_week, prefix="dow").reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0)
    dow = dow.set_index(days).astype(float)
    is_price = np.array([c.split("_")[0] in price_keys for c in X.columns])
    complete = price.notna().all(axis=1).to_numpy()
    test_days = pd.DatetimeIndex(sorted(pd.DatetimeIndex(test_days).intersection(days)))
    preds = {}
    for i in range(0, len(test_days), refit_days):
        block = test_days[i:i + refit_days]
        tr = (days < block[0]) & (days >= block[0] - pd.Timedelta(days=window_days)) & complete
        if tr.sum() < 180:
            continue
        Ytr = price[tr].to_numpy()
        med = float(np.nanmedian(Ytr))
        scale = float(np.nanmedian(np.abs(Ytr - med))) * 1.4826 or 1.0
        Xa = X.to_numpy(dtype=float).copy()
        Xa[:, is_price] = np.arcsinh((Xa[:, is_price] - med) / scale)
        mu = np.nanmean(Xa[tr], axis=0)
        sd = np.nanstd(Xa[tr], axis=0)
        sd[~(sd > 1e-9)] = 1.0
        Xa = np.hstack([np.nan_to_num((Xa - mu) / sd), dow.to_numpy()])       # missing regressors -> training mean
        te = days.isin(block)
        out = np.empty((te.sum(), 24))
        for h in range(24):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xa[tr], np.arcsinh((Ytr[:, h] - med) / scale))
            out[:, h] = np.sinh(np.clip(m.predict(Xa[te]), -12, 12)) * scale + med
        for d, row in zip(days[te], out):
            preds[d] = row
    pos = f.groupby("delivery_day").cumcount().clip(upper=23).to_numpy()
    dd = pd.DatetimeIndex(f["delivery_day"])
    lut = {(d, h): v for d, row in preds.items() for h, v in enumerate(row)}
    s = pd.Series([lut.get((d, h), np.nan) for d, h in zip(dd, pos)], index=f.index)
    return s.dropna()


# ---------------------------------------------------------------------------------------------- weights
def member_weights(log: pd.DataFrame, actual: pd.Series, day: pd.Timestamp, members: list[str] = MEMBERS,
                   window: int = WINDOW_DAYS, power: float = POWER) -> dict[str, float]:
    """Weights for delivery day `day` from the members' errors on the previous `window` logged days with published prices."""
    members = [m for m in members if m in log]
    eq = {m: 1.0 / len(members) for m in members}
    if log.empty:
        return eq
    d = log.copy()
    d["actual"] = actual.reindex(pd.DatetimeIndex(d["timestamp_utc"])).to_numpy()
    d = d[(d["delivery_day"] < pd.Timestamp(day)) & d["actual"].notna()].dropna(subset=members)
    if d.empty:
        return eq
    daily = pd.DataFrame({m: (d["actual"] - d[m]).abs().groupby(d["delivery_day"]).mean() for m in members}).sort_index().tail(window)
    if len(daily) < MIN_DAYS:
        return eq
    w = 1.0 / np.power(daily.mean().clip(lower=1e-6), power)
    return (w / w.sum()).to_dict()


def combine(preds: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    ms = [m for m in weights if m in preds]
    w = np.array([weights[m] for m in ms], dtype=float)
    return preds[ms].to_numpy(dtype=float) @ (w / w.sum())


# ---------------------------------------------------------------------------------------------- member log
def read_log() -> pd.DataFrame:
    p = C.REPORTS / LOG
    if not p.exists():
        return pd.DataFrame(columns=["timestamp_utc", "delivery_day", "source"] + MEMBERS)
    d = pd.read_csv(p, parse_dates=["delivery_day"])
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    return d


def _write_log(d: pd.DataFrame) -> None:
    d = d.reset_index(drop=True).sort_values("timestamp_utc")      # the index may carry the same name as the column
    d = d[d["delivery_day"] >= d["delivery_day"].max() - pd.Timedelta(days=120)]
    d.round(3).to_csv(C.REPORTS / LOG, index=False)


def log_members(preds: pd.DataFrame, day: pd.Timestamp) -> None:
    """Append today's member forecasts (index = UTC hours). An hour that is already logged is not overwritten."""
    new = preds[[m for m in MEMBERS if m in preds]].copy()
    new.insert(0, "source", "live")
    new.insert(0, "delivery_day", pd.Timestamp(day).normalize())
    new.insert(0, "timestamp_utc", pd.to_datetime(preds.index, utc=True))
    old = read_log()
    _write_log(pd.concat([old, new]).drop_duplicates("timestamp_utc", keep="first") if len(old) else new)


def seed_member_log(f: pd.DataFrame, feats: list[str], params: dict | None, days: int = SEED_DAYS) -> pd.DataFrame:
    """Out-of-sample member forecasts for the last `days` delivery days (weekly refits), written as the log's back-test part.

    Called at every retrain. Live rows already in the log are kept for hours the back-test does not cover.
    """
    from src.models import LevelAdjusted, make_lgbm, make_ridge

    valid = f[TARGET].notna()
    all_days = np.sort(f.loc[valid, "delivery_day"].unique())
    test_days = all_days[-days:]
    out = {m: [] for m in ("lightgbm", "ridge", "level")}
    makers = {"lightgbm": lambda: make_lgbm(params), "ridge": lambda: make_ridge(feats), "level": lambda: LevelAdjusted(make_lgbm(params))}
    for i in range(0, len(test_days), 7):
        block = test_days[i:i + 7]
        train = valid & (f["delivery_day"] < block[0])
        test = valid & f["delivery_day"].isin(block)
        for name, mk in makers.items():
            m = mk().fit(f.loc[train, feats], f.loc[train, TARGET])
            out[name].append(pd.Series(m.predict(f.loc[test, feats]), index=f.index[test]))
    seed = pd.DataFrame({k: pd.concat(v) for k, v in out.items()})
    seed["lear"] = lear_forecast(f, test_days).reindex(seed.index)
    seed.insert(0, "source", "backtest")
    seed.insert(0, "delivery_day", pd.DatetimeIndex(f.loc[seed.index, "delivery_day"]))
    seed.insert(0, "timestamp_utc", seed.index)
    old = read_log()
    live = old[(old["source"] == "live") & ~old["timestamp_utc"].isin(seed["timestamp_utc"])] if len(old) else old
    _write_log(pd.concat([seed, live]) if len(live) else seed)
    return seed

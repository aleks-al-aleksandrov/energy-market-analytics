"""Quarter-hour forecasts: the Bulgarian day-ahead market has cleared in 15-minute products since October 2025.

The hourly forecast stays the backbone. This module adds the WITHIN-HOUR SHAPE: how far each quarter sits from its hour's
mean (`dev`). Quarter forecast = hourly forecast + predicted dev, with the predicted devs of an hour forced to average zero,
so the hourly forecast (and everything validated about it) is unchanged.

Candidates for the shape (fixed before running):
  * flat      dev = 0: every quarter equals the hourly forecast (the behaviour before this module)
  * prof7     mean dev of the same quarter-of-day over the previous 7 delivery days
  * interp    straight-line interpolation between neighbouring hourly forecasts
  * lgbm      LightGBM on calendar, forecast ramps (price, residual load, solar, load, wind), and yesterday's / last week's dev

RULE: the candidate with the lowest quarter-hour MAE on the CV months replaces `flat` only if it is at least 0.1 EUR/MWh
lower and Diebold-Mariano gives p < 0.05. CV months: 2026-01 .. 2026-07 (15-minute history starts 2025-10, so there are
13 months in all). Aug-Oct 2026 is reported only, like everywhere else in the project.

Information set: quarter prices up to D-1 (published on D-2) and forecasts for D. `tests/test_quarter_hour.py` perturbs every
quarter price of D and later and checks that no feature of D moves.

Usage: python -m src.quarter_hour        (experiment; writes reports/quarter_metrics.csv and quarter_choice.json)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import diebold_mariano, mae
from src.features import MARKET_TZ, TARGET, delivery_day

log = logging.getLogger("quarter_hour")
QH_START = pd.Timestamp("2025-10-01", tz="UTC")
FIRST_TEST_MONTH = "2026-01"
MIN_GAIN, ALPHA = 0.1, 0.05
MIN_TRAIN_DAYS = 60
BUNDLE = "quarter_shape.joblib"
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, random_state=C.SEED, verbose=-1, n_jobs=-1)
RAMP_COLS = ["residual_load_forecast", "solar_forecast", "load_forecast", "wind_forecast"]


# ---------------------------------------------------------------------------------------------- data
def quarter_prices(raw: pd.Series | None = None) -> pd.Series:
    """BG day-ahead prices at 15-minute resolution (UTC index), from the first 15-minute month on. Complete hours only."""
    if raw is None:
        from src.ingest import load_raw
        raw = load_raw(f"prices/{C.FOCUS_ZONE}").iloc[:, 0]
    q = raw.dropna()
    q = q[~q.index.duplicated()].sort_index()
    q = q[q.index >= QH_START]
    q = q[q.index.minute % 15 == 0]
    full = q.groupby(q.index.floor("h")).transform("size") == 4
    return q[full].rename("price_q")


def deviations(q: pd.Series) -> pd.Series:
    """Quarter price minus the mean of its hour."""
    return (q - q.groupby(q.index.floor("h")).transform("mean")).rename("dev")


# ---------------------------------------------------------------------------------------------- features
def build_quarter_features(qidx: pd.DatetimeIndex, dev: pd.Series, f: pd.DataFrame, fc: pd.Series) -> pd.DataFrame:
    """Features for the quarters in `qidx` (15-minute UTC timestamps).

    dev: historical quarter deviations (need not cover qidx: that is the live case).
    f:   hourly feature frame (src.features.build_features) for the day-ahead forecasts of load / wind / solar.
    fc:  hourly price forecast (UTC hour starts). In back-tests this is the out-of-sample hourly forecast.
    """
    x = pd.DataFrame(index=qidx)
    hour_ts = qidx.floor("h")
    dday = delivery_day(qidx)
    local = qidx.tz_convert(C.LOCAL_TZ)
    x["qi"] = qidx.minute // 15
    x["qc"] = x["qi"] - 1.5                                         # position inside the hour, centred
    x["hour"] = local.hour
    x["dow"] = local.day_of_week
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    doy = local.day_of_year
    x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    def ramps(s: pd.Series, name: str) -> None:
        """Change into and out of the hour; the change out of the last hour of D would need D+1, so it is left missing."""
        s = s.sort_index()
        prev = s - s.shift(1, freq="h").reindex(s.index)
        nxt = s.shift(-1, freq="h").reindex(s.index) - s
        same_day_next = delivery_day(s.index + pd.Timedelta(hours=1)).to_numpy() == delivery_day(s.index).to_numpy()
        nxt = nxt.where(same_day_next)
        x[f"{name}_ramp_prev"] = prev.reindex(hour_ts).to_numpy()
        x[f"{name}_ramp_next"] = nxt.reindex(hour_ts).to_numpy()
        mid = (prev.reindex(hour_ts).to_numpy() + np.nan_to_num(nxt.reindex(hour_ts).to_numpy(), nan=0.0)) / 2
        x[f"{name}_slope_q"] = x["qc"].to_numpy() * mid / 4          # what straight-line interpolation would give

    ramps(fc.dropna(), "fc")
    for c in RAMP_COLS:
        if c in f:
            ramps(f[c].dropna(), c.replace("_forecast", ""))

    # deviations of earlier days at the same clock time (k days back). k = 1 can point inside D on a 25-hour day: guard it.
    day_start = pd.DatetimeIndex(dday).tz_localize(MARKET_TZ, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")

    def lag_days(k: int) -> np.ndarray:
        src = qidx - pd.Timedelta(days=k)
        v = dev.reindex(src).to_numpy(dtype=float).copy()
        v[np.asarray(src >= day_start)] = np.nan
        return v

    lags = np.column_stack([lag_days(k) for k in range(1, 8)])
    x["dev_lag1d"], x["dev_lag7d"] = lags[:, 0], lags[:, 6]
    with np.errstate(all="ignore"):
        x["dev_prof7"] = np.where(np.isfinite(lags).sum(axis=1) >= 3, np.nanmean(lags, axis=1), np.nan)
    x["delivery_day"] = dday.to_numpy()
    return x


FEATURES_EXCLUDE = {"delivery_day"}


def feature_list(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if c not in FEATURES_EXCLUDE]


def zero_mean_by_hour(pred: pd.Series) -> pd.Series:
    """Make the four predicted deviations of every hour sum to zero, so the hourly forecast is preserved exactly."""
    return pred - pred.groupby(pred.index.floor("h")).transform("mean")


# ---------------------------------------------------------------------------------------------- candidates
def shape_candidates(x: pd.DataFrame, dev: pd.Series, months: list[pd.Period]) -> pd.DataFrame:
    """Out-of-sample predicted deviations for every candidate over `months` (LightGBM refitted monthly)."""
    import lightgbm as lgb

    out = pd.DataFrame(index=x.index)
    out["flat"] = 0.0
    out["prof7"] = x["dev_prof7"].fillna(0.0)
    out["interp"] = x["fc_slope_q"].fillna(0.0)
    feats = feature_list(x)
    dmonth = pd.PeriodIndex(x["delivery_day"], freq="M")
    y = dev.reindex(x.index)
    pred = pd.Series(np.nan, index=x.index)
    for m in months:
        train = np.asarray(dmonth < m) & y.notna().to_numpy()
        test = np.asarray(dmonth == m)
        if x.loc[train, "delivery_day"].nunique() < MIN_TRAIN_DAYS or not test.any():
            continue
        model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(x.loc[train, feats], y[train])
        pred[test] = model.predict(x.loc[test, feats])
    out["lgbm"] = pred
    scope = np.asarray(dmonth.isin(months)) & out["lgbm"].notna().to_numpy()
    out = out[scope]
    return out.apply(zero_mean_by_hour)


# ---------------------------------------------------------------------------------------------- production
def train_final(f: pd.DataFrame, fc: pd.Series, method: str, quarters: pd.Series | None = None) -> dict:
    """Fit the adopted shape model on all history and store it with the per-hour band widening. Called at every retrain."""
    import joblib

    q = quarter_prices() if quarters is None else quarters
    dev = deviations(q)
    x = build_quarter_features(q.index, dev, f, fc)
    bundle: dict = {"method": method, "trained_until": str(pd.Timestamp(x["delivery_day"].max()).date()), "features": feature_list(x)}
    if method == "lgbm":
        import lightgbm as lgb
        ok = dev.notna().to_numpy()
        bundle["model"] = lgb.LGBMRegressor(**LGBM_PARAMS).fit(x.loc[ok, bundle["features"]], dev[ok])
    # Band: the hourly band shifted by the shape already holds about 78 % of quarter prices (target 80 %), while a per-hour
    # widening overshoots to 90 % (reports/quarter_choice.json), so no widening is applied.
    bundle["band_addon_by_hour"] = {}
    joblib.dump(bundle, C.MODELS_DIR / BUNDLE)
    return bundle


def predict_shape(bundle: dict, x: pd.DataFrame) -> pd.Series:
    method = bundle["method"]
    if method == "lgbm":
        p = pd.Series(bundle["model"].predict(x[bundle["features"]]), index=x.index)
    elif method == "prof7":
        p = x["dev_prof7"].fillna(0.0)
    elif method == "interp":
        p = x["fc_slope_q"].fillna(0.0)
    else:
        p = pd.Series(0.0, index=x.index)
    return zero_mean_by_hour(p)


def forecast_quarters(bundle: dict, hourly: pd.DataFrame, f: pd.DataFrame, quarters: pd.Series | None = None) -> pd.DataFrame:
    """96 (or 92 / 100) quarter forecasts for the delivery day in `hourly`.

    hourly: index = UTC hour starts of the day, columns forecast_eur_mwh, p10_eur_mwh, p90_eur_mwh.
    """
    q = quarter_prices() if quarters is None else quarters
    dev = deviations(q[q.index < hourly.index.min()])                 # only days before D
    qidx = pd.DatetimeIndex([t + pd.Timedelta(minutes=15 * k) for t in hourly.index for k in range(4)])
    x = build_quarter_features(qidx, dev, f, hourly["forecast_eur_mwh"])
    d = predict_shape(bundle, x)
    rep = hourly.reindex(qidx.floor("h"))
    addon = x["hour"].map({int(k): v for k, v in bundle.get("band_addon_by_hour", {}).items()}).fillna(0.0).to_numpy()
    out = pd.DataFrame(index=qidx)
    out["forecast_eur_mwh"] = rep["forecast_eur_mwh"].to_numpy() + d.to_numpy()
    out["p10_eur_mwh"] = np.minimum(rep["p10_eur_mwh"].to_numpy() + d.to_numpy() - addon, out["forecast_eur_mwh"])
    out["p90_eur_mwh"] = np.maximum(rep["p90_eur_mwh"].to_numpy() + d.to_numpy() + addon, out["forecast_eur_mwh"])
    out["shape_eur_mwh"] = d.to_numpy()
    out["time_local"] = qidx.tz_convert(C.LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    return out


# ---------------------------------------------------------------------------------------------- experiment
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.features import build_features

    f = build_features(pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet"))
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["delivery_day"])
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p = p.set_index("timestamp_utc")
    fc = p["lightgbm"]                                               # out-of-sample hourly forecast
    q = quarter_prices()
    q = q[q.index.floor("h").isin(fc.dropna().index)]                 # hours that have an out-of-sample hourly forecast
    dev = deviations(q)
    x = build_quarter_features(q.index, dev, f, fc)
    months = sorted(pd.PeriodIndex(x["delivery_day"], freq="M").unique())
    months = [m for m in months if m >= pd.Period(FIRST_TEST_MONTH, "M")]
    hold = [m for m in months if str(m) in set(p.loc[p["split"] == "holdout", "month"].astype(str))]
    cv = [m for m in months if m not in hold]
    log.info("quarters %d | features %d | CV %s..%s | holdout %s..%s", len(x), len(feature_list(x)), cv[0], cv[-1], hold[0], hold[-1])

    cand = shape_candidates(x, dev, cv + hold)
    d = cand.join(dev).join(q)
    d["hourly_fc"] = fc.reindex(d.index.floor("h")).to_numpy()
    d["delivery_day"] = x["delivery_day"].reindex(d.index)
    d["split"] = np.where(pd.PeriodIndex(d["delivery_day"], freq="M").isin(hold), "holdout", "cv")
    d = d.dropna(subset=["dev", "price_q", "hourly_fc"])
    names = ["flat", "prof7", "interp", "lgbm"]
    rows = []
    for split in ("cv", "holdout"):
        g = d[d["split"] == split]
        for c in names:
            r = dict(candidate=c, split=split, n=len(g), shape_mae=mae(g["dev"], g[c]),
                     price_mae=mae(g["price_q"], g["hourly_fc"] + g[c]))
            if c != "flat":
                s, pv = diebold_mariano(g["price_q"] - g["hourly_fc"] - g[c], g["price_q"] - g["hourly_fc"], g["delivery_day"])
                r["delta_vs_flat"], r["DM_stat"], r["DM_p"] = r["price_mae"] - mae(g["price_q"], g["hourly_fc"]), s, pv
            rows.append(r)
    met = pd.DataFrame(rows)
    met.to_csv(C.REPORTS / "quarter_metrics.csv", index=False)
    cvm = met[(met.split == "cv") & (met.candidate != "flat")]
    passed = cvm[(cvm["delta_vs_flat"] <= -MIN_GAIN) & (cvm["DM_p"] < ALPHA) & (cvm["DM_stat"] < 0)]
    choice = passed.sort_values("price_mae").iloc[0]["candidate"] if len(passed) else "flat"

    # does the per-hour band widening restore coverage at quarter level? (hourly band + chosen shape, CV months)
    cov = {}
    if {"q10_cal", "q90_cal"} <= set(p.columns):
        lo = p["q10_cal"].reindex(d.index.floor("h")).to_numpy() + d[choice].to_numpy()
        hi = p["q90_cal"].reindex(d.index.floor("h")).to_numpy() + d[choice].to_numpy()
        hour = x["hour"].reindex(d.index).to_numpy()
        is_cv = (d["split"] == "cv").to_numpy()
        resid = (d["dev"] - d[choice]).abs()
        first_half = is_cv & np.asarray(pd.PeriodIndex(d["delivery_day"], freq="M") <= cv[len(cv) // 2 - 1])
        addon = resid[first_half].groupby(hour[first_half]).quantile(0.8)                 # estimated on the earlier CV months
        add = pd.Series(hour).map(addon).fillna(0.0).to_numpy()
        second = is_cv & ~first_half
        ok = np.isfinite(lo) & np.isfinite(hi)
        for name, mask in (("cv_later_months", second & ok), ("holdout", (~is_cv) & ok)):
            pq = d["price_q"].to_numpy()[mask]
            cov[name] = dict(hourly_band_only=float(((pq >= lo[mask]) & (pq <= hi[mask])).mean()),
                             with_addon=float(((pq >= lo[mask] - add[mask]) & (pq <= hi[mask] + add[mask])).mean()), n=int(mask.sum()))
    (C.REPORTS / "quarter_choice.json").write_text(json.dumps(dict(
        shape=choice, candidates_passing=passed["candidate"].tolist(),
        rule=f"lowest quarter-hour MAE on CV months, at least {MIN_GAIN} below flat with Diebold-Mariano p < {ALPHA}",
        cv_months=f"{cv[0]}..{cv[-1]}", holdout_months=f"{hold[0]}..{hold[-1]}", band_coverage=cov,
        mean_abs_dev=float(d["dev"].abs().mean()),
        note="15-minute history starts 2025-10; Aug-Oct 2026 reported only"), indent=2))
    print(met.round(3).to_string(index=False))
    print("chosen shape:", choice, "| band coverage:", json.dumps(cov))


if __name__ == "__main__":
    main()

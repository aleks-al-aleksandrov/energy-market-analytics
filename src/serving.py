"""Serving layer: turn the files in reports/ into clean JSON payloads.

One place builds every payload, so the FastAPI service (api/main.py), the static JSON export (src/export_api.py) and any
front end all return the same numbers. Nothing here calls ENTSO-E or trains a model: it reads what the pipeline wrote.
Every payload carries `generated_at` / `as_of` fields, so a consumer can tell how fresh it is.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import battery as B
from src import config as C

API_VERSION = "v1"
STALE_AFTER_HOURS = 30          # a daily forecast older than this is flagged as stale


def _r(path: str) -> Path:
    return C.REPORTS / path


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, timestamps -> ISO strings."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if x is pd.NaT:
        return None
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.replace({np.nan: None}).to_dict(orient="records"))


# ---------------------------------------------------------------------------------------------- forecast
def _load_forecast() -> pd.DataFrame | None:
    p = _r("forecast_next_day.csv")
    if not p.exists():
        return None
    f = pd.read_csv(p)
    f["timestamp_utc"] = pd.to_datetime(f["timestamp_utc"], utc=True)
    return f


def _windows(prices: np.ndarray, k: int = 3) -> tuple[int, int]:
    """Start index of the cheapest and the dearest run of k consecutive hours."""
    if len(prices) < k:
        return 0, 0
    s = np.convolve(prices, np.ones(k), mode="valid")
    return int(s.argmin()), int(s.argmax())


def _ensemble_live(delivery_day: str) -> dict | None:
    """Weights of the adaptive ensemble behind this forecast (None when the forecast came from LightGBM alone)."""
    p = _r("ensemble_live.json")
    if not p.exists():
        return None
    e = json.loads(p.read_text())
    return e if e.get("delivery_day") == delivery_day else None


def forecast_latest() -> dict:
    f = _load_forecast()
    if f is None or f.empty:
        return {"available": False, "reason": "no forecast has been published yet"}
    price = f["forecast_eur_mwh"].to_numpy(dtype=float)
    lo, hi = f["p10_eur_mwh"].to_numpy(dtype=float), f["p90_eur_mwh"].to_numpy(dtype=float)
    cheap, dear = _windows(price)
    spike = f.get("spike_probability", pd.Series(np.nan, index=f.index)).to_numpy(dtype=float)
    local = f["time_local"].astype(str)
    hours = [dict(timestamp_utc=t.isoformat(), time_local=l, hour_local=int(l[11:13]), forecast=p, p10=a, p90=b,
                  spike_probability=s, cross_check=c)
             for t, l, p, a, b, s, c in zip(f["timestamp_utc"], local, price, lo, hi, spike,
                                            f.get("forecast_level_eur_mwh", pd.Series(np.nan, index=f.index)))]
    generated = pd.to_datetime(f["generated_at"].iloc[0]).tz_localize(C.LOCAL_TZ) if "generated_at" in f else None
    age_h = (pd.Timestamp.now(tz="UTC") - generated.tz_convert("UTC")).total_seconds() / 3600 if generated is not None else None
    disagreement = None
    if "forecast_level_eur_mwh" in f:
        lvl = f["forecast_level_eur_mwh"].to_numpy(dtype=float)
        disagreement = float(abs(np.nanmean(lvl) - np.nanmean(price)) / max(abs(np.nanmean(price)), 1e-9))
    return _clean(dict(
        available=True, delivery_day=str(f["delivery_day"].iloc[0]), unit="EUR/MWh", timezone=C.LOCAL_TZ,
        generated_at=generated, age_hours=age_h, stale=bool(age_h is not None and age_h > STALE_AFTER_HOURS),
        model_trained_until=str(f["model_trained_until"].iloc[0]) if "model_trained_until" in f else None,
        band=dict(level="10-90 %", calibrated_as_of=str(f["band_calibrated_as_of"].iloc[0]) if "band_calibrated_as_of" in f else None),
        summary=dict(mean=float(np.nanmean(price)), min=float(np.nanmin(price)), max=float(np.nanmax(price)),
                     min_hour_local=hours[int(np.nanargmin(price))]["hour_local"],
                     max_hour_local=hours[int(np.nanargmax(price))]["hour_local"],
                     spread=float(np.nanmax(price) - np.nanmin(price)),
                     mean_band_width=float(np.nanmean(hi - lo)),
                     cheapest_3h=dict(start_local=hours[cheap]["time_local"], mean=float(price[cheap:cheap + 3].mean())),
                     dearest_3h=dict(start_local=hours[dear]["time_local"], mean=float(price[dear:dear + 3].mean())),
                     max_spike_probability=float(np.nanmax(spike)) if np.isfinite(spike).any() else None,
                     models_disagree_by=disagreement,
                     models_disagree=bool(disagreement is not None and disagreement > 0.10)),
        hours=hours,
        ensemble=_ensemble_live(str(f["delivery_day"].iloc[0])),
        disclaimer="Forecast and analysis only. Not trading or investment advice."))


# ---------------------------------------------------------------------------------------------- live track record
def track_record() -> dict:
    p = _r("live_track_record.csv")
    if not p.exists():
        return {"available": False, "days": []}
    t = pd.read_csv(p)
    if t.empty:
        return {"available": False, "days": []}
    w = t["hours"].to_numpy(dtype=float)
    mae_m = float(np.average(t["mae_model"], weights=w))
    mae_n = float(np.average(t["mae_naive"], weights=w))
    return _clean(dict(
        available=True, days=_records(t), n_days=int(len(t)), first_day=str(t["delivery_day"].min()),
        last_day=str(t["delivery_day"].max()),
        totals=dict(mae_model=mae_m, mae_naive=mae_n, skill_vs_naive=1 - mae_m / mae_n if mae_n else None,
                    coverage_10_90=float(np.average(t["coverage_10_90"], weights=w)) if "coverage_10_90" in t else None,
                    days_model_beat_naive=int((t["mae_model"] < t["mae_naive"]).sum())),
        note="Only forecasts generated before gate closure (12:00 CET on D-1) count. Too few days prove nothing: read the totals with the day count."))


# ---------------------------------------------------------------------------------------------- model quality
def model_metrics() -> dict:
    out: dict = {"available": False}
    for split in ("cv", "holdout"):
        p = _r(f"metrics_{split}.csv")
        if p.exists():
            m = pd.read_csv(p)
            keep = [c for c in ("model", "MAE", "RMSE", "skill_vs_best_naive", "DM_p", "significant_5pct", "n_folds") if c in m]
            out[split] = _records(m[keep])
            out["available"] = True
    for name in ("phase1_choice.json", "model_choice.json", "feature_choice.json", "regional_choice.json"):
        p = _r(name)
        if p.exists():
            out[name.replace(".json", "")] = _clean(json.loads(p.read_text()))
    p = _r("quantile_conformal.csv")
    if p.exists():
        out["band_calibration"] = _records(pd.read_csv(p))
    p = _r("phase1_metrics.csv")            # candidates compared with the previous production model, same hours for all
    if p.exists():
        out["phase1"] = _records(pd.read_csv(p))
    p = _r("battery_uncertainty.csv")
    if p.exists():
        out["battery_uncertainty"] = _records(pd.read_csv(p))
    for name in ("battery_uncertainty_choice.json", "ensemble_live.json"):
        q = _r(name)
        if q.exists():
            out[name.replace(".json", "")] = _clean(json.loads(q.read_text()))
    out["caveat"] = ("Cross-validation covers 25 monthly folds. The Aug-Oct 2026 holdout has been inspected and is not an untouched "
                     "test any more; the live track record is the clean out-of-sample evidence.")
    return _clean(out)


# ---------------------------------------------------------------------------------------------- market history
def market_recent(days: int = 30, zones: list[str] | None = None) -> dict:
    p = _r("market_hourly.parquet")
    if not p.exists():
        return {"available": False}
    m = pd.read_parquet(p)
    rp = _r("market_recent.csv")              # written by the daily forecast: fresher than the weekly market file
    if rp.exists():
        rc = pd.read_csv(rp)
        rc.index = pd.to_datetime(rc.pop("timestamp_utc"), utc=True)
        m = rc.combine_first(m)
    zones = zones or ["BG", "RO", "GR", "HU"]
    cols = [f"price_{z}" for z in zones if f"price_{z}" in m]
    days = int(max(1, min(days, 400)))
    m = m.loc[m.index >= m.index.max() - pd.Timedelta(days=days), cols].dropna(how="all")
    local = m.index.tz_convert(C.LOCAL_TZ)
    daily = m.groupby(local.date).agg(["mean", "min", "max"])
    return _clean(dict(
        available=True, unit="EUR/MWh", timezone=C.LOCAL_TZ, zones=[c.replace("price_", "") for c in cols],
        data_through=m.index.max(),
        hourly=[dict(timestamp_utc=t.isoformat(), **{c.replace("price_", ""): v for c, v in zip(cols, row)})
                for t, row in zip(m.index, m.to_numpy(dtype=float))],
        daily=[dict(day=str(d), **{f"{c.replace('price_', '')}_{k}": daily.loc[d, (c, k)] for c in cols for k in ("mean", "min", "max")})
               for d in daily.index]))


# ---------------------------------------------------------------------------------------------- battery plan
def battery_plan(power_mw: float = 1.0, duration_h: float = 2.0, efficiency: float = 0.90, cycles_per_day: float = 1.0,
                 degradation_eur_mwh: float = 0.0) -> dict:
    """Optimal schedule for tomorrow on the forecast, with the revenue it would earn at the forecast and at each band edge."""
    f = _load_forecast()
    if f is None or f.empty:
        return {"available": False}
    if not (0 < power_mw <= 1000 and 0.25 <= duration_h <= 12 and 0.5 <= efficiency <= 1 and 0 < cycles_per_day <= 4
            and 0 <= degradation_eur_mwh <= 200):
        raise ValueError("battery parameters out of range")
    b = B.Battery(power_mw, duration_h, efficiency, cycles_per_day, degradation_eur_mwh)
    price = f["forecast_eur_mwh"].to_numpy(dtype=float)
    net = B.schedule_day(price, b)
    deg = b.degradation_eur_mwh * B.throughput(net)
    soc, level, se = [], 0.0, math.sqrt(efficiency)
    for x in net:
        level += (-x * se) if x < 0 else (-x / se)
        soc.append(level)
    rows = [dict(time_local=l, hour_local=int(str(l)[11:13]), price=p, net_mw=x,
                 action="charge" if x < -1e-9 else "discharge" if x > 1e-9 else "idle", soc_mwh=s)
            for l, p, x, s in zip(f["time_local"], price, net, soc)]
    return _clean(dict(
        available=True, delivery_day=str(f["delivery_day"].iloc[0]),
        battery=dict(power_mw=power_mw, duration_h=duration_h, energy_mwh=b.energy_mwh, efficiency=efficiency,
                     cycles_per_day=cycles_per_day, degradation_eur_mwh=degradation_eur_mwh),
        expected_revenue_eur=float(np.sum(net * price)) - deg,
        revenue_if_prices_at_p10_eur=float(np.sum(net * f["p10_eur_mwh"].to_numpy(dtype=float))) - deg,
        revenue_if_prices_at_p90_eur=float(np.sum(net * f["p90_eur_mwh"].to_numpy(dtype=float))) - deg,
        charged_mwh=float(np.maximum(-net, 0).sum()), discharged_mwh=float(np.maximum(net, 0).sum()),
        schedule=rows,
        assumptions="Day-ahead arbitrage only, price-taker, starts and ends the day empty. Revenue at forecast prices is "
                    "optimistic: realised revenue is on average lower because a schedule sells where the forecast is highest."))


# ---------------------------------------------------------------------------------------------- status
def status() -> dict:
    f = forecast_latest()
    mr = market_recent(3, ["BG"])
    data_through = mr.get("data_through") if mr.get("available") else None
    band = json.loads(_r("conformal_offsets.json").read_text()) if _r("conformal_offsets.json").exists() else {}
    tr = track_record()
    problems = []
    if not f.get("available"):
        problems.append("no forecast published")
    elif f.get("stale"):
        problems.append(f"latest forecast is {f['age_hours']:.0f} h old")
    return _clean(dict(
        ok=not problems, problems=problems, api_version=API_VERSION, checked_at=datetime.now(timezone.utc),
        forecast=dict(delivery_day=f.get("delivery_day"), generated_at=f.get("generated_at"), age_hours=f.get("age_hours"),
                      model_trained_until=f.get("model_trained_until")),
        market_data_through=data_through,
        band=dict(method=band.get("method", "fixed window"), as_of=band.get("as_of"),
                  live_days_applied=band.get("live_days_applied", 0)),
        live_track_record_days=tr.get("n_days", 0)))


PAYLOADS = {"forecast_latest": forecast_latest, "track_record": track_record, "metrics": model_metrics,
            "market_recent": market_recent, "battery_default": battery_plan, "status": status}

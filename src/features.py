"""Feature engineering - only information available before day-ahead gate closure.

Delivery day D is defined in market time (Europe/Brussels, SDAC). Gate closure is
12:00 CET on D-1. A feature is allowed only if it is known at that moment:

* prices up to the end of D-1 (published on D-2 ~13:00 CET)        -> allowed
* ENTSO-E day-ahead load forecast for D                             -> allowed
* ENTSO-E day-ahead wind/solar forecast for D                       -> allowed (*)
* Open-Meteo forecast issued on D-1 (Previous Runs API, "day1")     -> allowed
* actual load / generation / flows of D                             -> NOT allowed
* actual physical flows: only full days up to D-2                   -> allowed

(*) Regulation 543/2013 sets the deadline for the DA wind/solar forecast at 18:00
on D-1, i.e. potentially after gate closure. In practice traders have vendor or
TSO forecasts by gate closure; we treat it as available and say so in the README.
"""
from __future__ import annotations

import json

import holidays
import numpy as np
import pandas as pd

from src import config as C

MARKET_TZ = "Europe/Brussels"
TARGET = "price_BG"


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    local = idx.tz_convert(C.LOCAL_TZ)
    years = range(idx.min().year - 1, idx.max().year + 2)
    bg_hol = holidays.country_holidays("BG", years=years)
    date_local = pd.Series(local.date, index=idx)
    is_hol = date_local.map(lambda d: d in bg_hol).astype(int)
    dow = local.day_of_week
    # bridge day: a Monday after / Friday before a holiday that falls on Tue / Thu
    prev_d = date_local.map(lambda d: (d - pd.Timedelta(days=1)) in bg_hol)
    next_d = date_local.map(lambda d: (d + pd.Timedelta(days=1)) in bg_hol)
    bridge = (((dow == 0) & next_d) | ((dow == 4) & prev_d)) & (is_hol == 0)
    f = pd.DataFrame(index=idx)
    f["hour"] = local.hour
    f["dow"] = dow
    f["month"] = local.month
    f["is_weekend"] = (dow >= 5).astype(int)
    f["is_holiday"] = is_hol.values
    f["is_bridge"] = bridge.astype(int).values
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["month_sin"] = np.sin(2 * np.pi * (f["month"] - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (f["month"] - 1) / 12)
    return f


def delivery_day(idx: pd.DatetimeIndex) -> pd.Series:
    """Market delivery day (tz-naive date at midnight, Europe/Brussels)."""
    return pd.Series(idx.tz_convert(MARKET_TZ).normalize().tz_localize(None), index=idx)


def _day_start_utc(dday: pd.Series) -> pd.Series:
    return dday.dt.tz_localize(MARKET_TZ, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")


def _safe_lag(s: pd.Series, hours: int, day_start: pd.Series) -> pd.Series:
    """s shifted by `hours`, set to NaN if the lagged timestamp is inside day D (DST edge)."""
    lagged = s.shift(hours)
    src_time = s.index - pd.Timedelta(hours=hours)
    ok = src_time < pd.DatetimeIndex(day_start)
    return lagged.where(ok)


def _daily_stat_prev(s: pd.Series, dday: pd.Series, k: int = 1, how: str = "mean") -> pd.Series:
    """Daily statistic of day D-k mapped onto every hour of day D."""
    daily = s.groupby(dday.values).agg(how)
    daily.index = pd.DatetimeIndex(daily.index)
    shifted = daily.shift(k, freq="D")  # tz-naive -> calendar-day shift
    return pd.Series(dday.map(shifted).values, index=s.index)


FEATURE_GROUPS: dict[str, list[str]] = {}


def build_features(h: pd.DataFrame) -> pd.DataFrame:
    idx = h.index
    dday = delivery_day(idx)
    f = _calendar(idx)
    f["delivery_day"] = dday.values
    day_start = _day_start_utc(dday)

    groups: dict[str, list[str]] = {"calendar": list(f.columns.drop("delivery_day"))}

    # --- price history (BG) ---------------------------------------------------
    p = h[TARGET]
    cols = []
    for lag in (24, 48, 168, 336, 504):
        f[f"price_lag{lag}"] = _safe_lag(p, lag, day_start)
        cols.append(f"price_lag{lag}")
    f["price_same_wd_mean3"] = f[["price_lag168", "price_lag336", "price_lag504"]].mean(axis=1)
    for how in ("mean", "std", "min", "max"):
        f[f"price_d1_{how}"] = _daily_stat_prev(p, dday, 1, how)
        cols.append(f"price_d1_{how}")
    daily_mean = p.groupby(dday.values).mean()
    daily_mean.index = pd.DatetimeIndex(daily_mean.index)
    roll7 = daily_mean.rolling(7, min_periods=5).mean().shift(1, freq="D")
    f["price_7d_mean"] = dday.map(roll7).values
    roll7s = daily_mean.rolling(7, min_periods=5).std().shift(1, freq="D")
    f["price_7d_std"] = dday.map(roll7s).values
    cols += ["price_same_wd_mean3", "price_7d_mean", "price_7d_std"]
    groups["price_lags"] = cols

    # --- neighbours (D-1 only) ------------------------------------------------
    cols = []
    for z in C.NEIGHBOUR_ZONES:
        c = f"price_{z}"
        if c in h:
            f[f"{c}_lag24"] = _safe_lag(h[c], 24, day_start)
            f[f"spread_BG_{z}_lag24"] = f["price_lag24"] - f[f"{c}_lag24"]
            f[f"{c}_d1_mean"] = _daily_stat_prev(h[c], dday, 1, "mean")
            cols += [f"{c}_lag24", f"spread_BG_{z}_lag24", f"{c}_d1_mean"]
    groups["neighbours"] = cols

    # --- system forecasts (ENTSO-E day-ahead) ---------------------------------
    cols = []
    for c in ("load_forecast", "wind_forecast", "solar_forecast", "residual_load_forecast"):
        if c in h:
            f[c] = h[c]
            cols.append(c)
    if {"residual_load_forecast", "load_forecast"} <= set(f.columns):
        f["res_load_ratio"] = f["residual_load_forecast"] / f["load_forecast"]
        f["res_load_d_mean"] = f.groupby("delivery_day")["residual_load_forecast"].transform("mean")
        f["res_load_d_ramp"] = f["residual_load_forecast"] - f["res_load_d_mean"]
        cols += ["res_load_ratio", "res_load_d_mean", "res_load_d_ramp"]
    groups["system_forecasts"] = cols

    # --- day-ahead interconnector capacities (NTC, published before the auction) ---------------
    ntc = [c for c in h.columns if c.startswith("ntc_")]
    for c in ntc:
        f[c] = h[c]
    if {"ntc_RO_to_BG", "ntc_GR_to_BG"} <= set(ntc):
        f["ntc_import_total"] = f["ntc_RO_to_BG"] + f["ntc_GR_to_BG"]
        f["ntc_export_total"] = f["ntc_BG_to_RO"] + f["ntc_BG_to_GR"]
        ntc += ["ntc_import_total", "ntc_export_total"]
    groups["capacity"] = ntc

    # --- flows (full days up to D-2) -------------------------------------------
    cols = []
    if "net_import" in h:
        f["net_import_d2_mean"] = _daily_stat_prev(h["net_import"], dday, 2, "mean")
        cols.append("net_import_d2_mean")
    groups["flows"] = cols

    # --- weather forecast (D-1 issue) ------------------------------------------
    wcols = [c for c in h.columns if c.startswith("wfc_")]
    for c in wcols:
        f[c] = h[c]
    groups["weather"] = wcols

    # --- fuel & carbon prices (already lagged to what is known at gate closure) ---------
    fcols = [c for c in h.columns if c.startswith("fuel_")]
    for c in fcols:
        f[c] = h[c]
    groups["fuels"] = fcols

    f["weather_fc_is_proxy"] = h.get("weather_fc_is_proxy", 0)
    f[TARGET] = p
    f["is_negative"] = (p < 0).astype(float).where(p.notna())
    FEATURE_GROUPS.clear()
    FEATURE_GROUPS.update(groups)
    return f


OPTIONAL_GROUPS = ["capacity", "fuels"]     # groups that must earn their place on the CV months (see run_all)


def default_excluded_groups() -> list[str]:
    """Optional groups switched off by the CV-based adoption rule in run_all (reports/feature_choice.json)."""
    p = C.REPORTS / "feature_choice.json"
    if not p.exists():
        return []
    choice = json.loads(p.read_text())
    return [g for g in OPTIONAL_GROUPS if not choice.get(f"use_{g}", True)]


def feature_columns(groups: list[str] | None = None, exclude: list[str] | None = None,
                    respect_choice: bool = True) -> list[str]:
    groups = groups or list(FEATURE_GROUPS)
    exclude = (exclude or []) + (default_excluded_groups() if respect_choice else [])
    return [c for g in groups if g not in exclude for c in FEATURE_GROUPS[g]]


def load_features() -> pd.DataFrame:
    h = pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet")
    return build_features(h)


if __name__ == "__main__":
    f = load_features()
    f.to_parquet(C.DATA_PROCESSED / "features.parquet")
    print(f.shape, {k: len(v) for k, v in FEATURE_GROUPS.items()})

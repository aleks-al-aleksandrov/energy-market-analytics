"""Cleaning: align every series on one hourly UTC index.

Rules (documented in the data quality report):
* Everything is UTC internally -> DST 23/25-hour days need no special casing.
* Sub-hourly data (15-min MTU since the SDAC go-live, 15-min load/RES) -> hourly mean.
* Gaps of <= 3 consecutive hours are linearly interpolated; longer gaps stay NaN
  (rows with a missing target are dropped at modelling time).
* Negative and extreme prices are kept as they are, and flagged.

Usage: python -m src.clean
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C
from src.ingest import load_raw

log = logging.getLogger("clean")
MAX_INTERP_HOURS = 3


def _resolution_by_month(s: pd.Series) -> pd.Series:
    """Median sampling interval (minutes) per calendar month."""
    s = s.dropna()
    d = s.index.to_series().diff().dt.total_seconds().div(60)
    return d.groupby(s.index.tz_localize(None).to_period("M")).median()


def _hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample("1h").mean()


def _pick(df: pd.DataFrame, *needles: str) -> pd.Series | None:
    for col in df.columns:
        if all(n.lower() in col.lower() for n in needles):
            return df[col]
    return None


def build_hourly(end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end = end or C.default_end()
    idx = pd.date_range(C.START.tz_convert("UTC"), end.tz_convert("UTC"), freq="1h", inclusive="left",
                        name="timestamp_utc")
    out = pd.DataFrame(index=idx)
    resolution = {}

    # --- prices ---------------------------------------------------------------
    for z in C.PRICE_ZONES:
        raw = load_raw(f"prices/{z}")
        if raw.empty:
            log.warning("no prices for %s", z)
            continue
        s = raw.iloc[:, 0]
        resolution[f"price_{z}"] = _resolution_by_month(s)
        out[f"price_{z}"] = s.resample("1h").mean()

    # --- load -----------------------------------------------------------------
    bg = C.FOCUS_ZONE
    la = load_raw(f"load/{bg}")
    if not la.empty:
        resolution["load_actual"] = _resolution_by_month(la.iloc[:, 0])
        out["load_actual"] = _hourly(la).iloc[:, 0]
    lf = load_raw(f"load_forecast/{bg}")
    if not lf.empty:
        resolution["load_forecast"] = _resolution_by_month(lf.iloc[:, 0])
        out["load_forecast"] = _hourly(lf).iloc[:, 0]

    # --- wind / solar day-ahead forecast ---------------------------------------
    ws = _hourly(load_raw(f"wind_solar_forecast/{bg}"))
    if not ws.empty:
        out["solar_forecast"] = _pick(ws, "solar")
        wind_cols = [c for c in ws.columns if "wind" in c.lower()]
        out["wind_forecast"] = ws[wind_cols].sum(axis=1, min_count=1) if wind_cols else np.nan

    # --- actual generation per type (analysis only, NOT a feature) ------------
    gen = _hourly(load_raw(f"generation/{bg}"))
    if not gen.empty:
        # ENTSO-E switches between flat columns ("Biomass") and MultiIndex columns
        # ("Biomass__Actual Aggregated") once a type also reports consumption -> merge both
        for col in gen.columns:
            if "consumption" in col.lower():
                name = "gencons_" + col.split("__")[0]
            else:
                name = "gen_" + col.split("__")[0]
            name = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            out[name] = out[name].combine_first(gen[col]) if name in out else gen[col]

    # --- cross-border physical flows ------------------------------------------
    imp, exp = [], []
    for n in C.FLOW_NEIGHBOURS:
        fi = _hourly(load_raw(f"flows/{n}_to_{bg}"))
        fe = _hourly(load_raw(f"flows/{bg}_to_{n}"))
        if not fi.empty:
            out[f"flow_{n}_to_BG"] = fi.iloc[:, 0]
            imp.append(f"flow_{n}_to_BG")
        if not fe.empty:
            out[f"flow_BG_to_{n}"] = fe.iloc[:, 0]
            exp.append(f"flow_BG_to_{n}")
    if imp or exp:
        out["net_import"] = out[imp].sum(axis=1, min_count=1) - out[exp].sum(axis=1, min_count=1)

    # --- day-ahead net transfer capacities (published before the auction -> usable as features) ---
    for n in ("RO", "GR"):
        for a, b_ in ((n, bg), (bg, n)):
            ntc = load_raw(f"ntc/{a}_to_{b_}")
            if not ntc.empty:
                out[f"ntc_{a}_to_{b_}"] = _hourly(ntc).iloc[:, 0]

    # --- weather ----------------------------------------------------------------
    short = {"temperature_2m": "temp", "wind_speed_100m": "wind100",
             "shortwave_radiation": "rad", "cloud_cover": "cloud"}
    proxy_mask = pd.Series(False, index=idx)
    for name in C.WEATHER_POINTS:
        act = load_raw(f"weather_actual/{name}", source="openmeteo")
        d1 = load_raw(f"weather_fc_d1/{name}", source="openmeteo")
        hist = load_raw(f"weather_fc_hist/{name}", source="openmeteo")
        for var, sv in short.items():
            if var in act:
                out[f"wact_{sv}_{name}"] = act[var].reindex(idx)
            a = d1[var].reindex(idx) if var in d1 else pd.Series(np.nan, index=idx)
            b = hist[var].reindex(idx) if var in hist else pd.Series(np.nan, index=idx)
            proxy_mask |= a.isna() & b.notna()
            out[f"wfc_{sv}_{name}"] = a.fillna(b)
    out["weather_fc_is_proxy"] = proxy_mask.astype(int)

    # --- fuel & carbon prices (lag-safe by construction, see src/fuels.py) -------------
    from src.fuels import fuel_frame
    fuel = fuel_frame(idx)
    for c in fuel.columns:
        if fuel[c].notna().any():
            out[c] = fuel[c]

    # --- gap handling ------------------------------------------------------------
    quality_rows = []
    for col in out.columns:
        if col == "weather_fc_is_proxy":
            continue
        s = out[col]
        n_missing = int(s.isna().sum())
        filled = s.interpolate(limit=MAX_INTERP_HOURS, limit_area="inside")
        # only fill gaps that are entirely <= MAX_INTERP_HOURS long
        gap_id = s.notna().cumsum()
        gap_len = s.isna().groupby(gap_id).transform("sum")
        filled = filled.where(s.notna() | (gap_len <= MAX_INTERP_HOURS))
        out[col] = filled
        first, last = s.first_valid_index(), s.last_valid_index()
        quality_rows.append(dict(series=col, first=first, last=last,
                                 coverage_pct=round(100 * s.notna().mean(), 2),
                                 missing_hours=n_missing,
                                 interpolated=int(n_missing - filled.isna().sum()),
                                 still_missing=int(filled.isna().sum()),
                                 longest_gap_h=int(gap_len[s.isna()].max()) if n_missing else 0))

    # --- flags ------------------------------------------------------------------
    if "price_BG" in out:
        out["is_negative"] = (out["price_BG"] < 0).astype(int)
    if {"load_forecast", "wind_forecast", "solar_forecast"} <= set(out.columns):
        out["residual_load_forecast"] = out["load_forecast"] - out["wind_forecast"].fillna(0) - out["solar_forecast"].fillna(0)
    wind_act = [c for c in out.columns if c.startswith("gen_wind")]
    solar_act = [c for c in out.columns if c.startswith("gen_solar")]
    if "load_actual" in out and wind_act and solar_act:
        out["residual_load_actual"] = out["load_actual"] - out[wind_act].sum(axis=1) - out[solar_act].sum(axis=1)

    quality = pd.DataFrame(quality_rows)
    res = pd.DataFrame(resolution)
    res.index = res.index.astype(str)
    res.index.name = "month"
    return out, quality, res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hourly, quality, res = build_hourly()
    hourly.to_parquet(C.DATA_PROCESSED / "hourly.parquet")
    quality.to_csv(C.REPORTS / "data_quality.csv", index=False)
    res.to_csv(C.REPORTS / "resolution_by_month.csv")
    log.info("hourly frame %s, saved to %s", hourly.shape, C.DATA_PROCESSED / "hourly.parquet")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()


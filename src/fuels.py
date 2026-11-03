"""Fuel and carbon price features that are available before day-ahead gate closure.

For a delivery day D (market date, Europe/Brussels) gate closure is 12:00 CET on D-1, so:

* daily closes (Brent, carbon proxy, EUR/USD) settle after gate closure -> only the last close on or before D-2 is usable
* the World Bank Pink Sheet monthly average for month m is published around the 2nd-3rd day of m+1
  -> we assume it is usable from the 5th of m+1, i.e. for D we take the last month that ended
  before (D - 5 days)
* nothing here looks at the delivery day or later; `tests/test_fuels.py` checks this.

All columns are returned on the hourly UTC index passed in (constant within a delivery day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

MARKET_TZ = "Europe/Brussels"
MMBTU_PER_MWH = 3.412           # 1 MWh = 3.412 mmbtu -> $/mmbtu x 3.412 = $/MWh
PUBLICATION_LAG_DAYS = 5        # Pink Sheet month m is treated as known from day 5 of m+1
DAILY_LAG_DAYS = 2              # close of D-2 is the last one known at D-1 12:00
FUEL_COLUMNS = ["fuel_brent_usd", "fuel_brent_chg20", "fuel_carbon_idx", "fuel_carbon_chg20", "fuel_gas_eur_mwh",
                "fuel_gas_rel12", "fuel_gas_chg3", "fuel_gas_cost_ccgt",
                "fuel_ttf_eur_mwh", "fuel_ttf_chg20", "fuel_ttf_vs_gasm"]


def _load() -> dict[str, pd.Series | pd.DataFrame]:
    d = C.DATA_RAW / "fuels"
    out: dict = {}
    for n in ("brent", "carbon", "eurusd"):
        p = d / f"{n}_daily.parquet"
        if p.exists():
            out[n] = pd.read_parquet(p).iloc[:, 0].sort_index()
    p = d / "pinksheet_monthly.parquet"
    if p.exists():
        out["pink"] = pd.read_parquet(p).sort_index()
    p = d / "ttf_daily.parquet"
    if p.exists():
        out["ttf"] = pd.read_parquet(p).iloc[:, 0].sort_index()
    return out


def monthly_gas_eur_mwh(pink: pd.DataFrame, eurusd: pd.Series | None) -> pd.Series:
    """EU gas in EUR/MWh by month (month-start index), converted with that month's mean EUR/USD."""
    usd = pink["gas_eu_usd_mmbtu"].dropna() * MMBTU_PER_MWH
    if eurusd is not None and len(eurusd):
        fx = eurusd.groupby(eurusd.index.to_period("M").to_timestamp()).mean()
        fx = fx.reindex(usd.index).ffill().bfill()
    else:
        fx = pd.Series(1.1, index=usd.index)
    return (usd / fx).rename("gas_eur_mwh")


def _daily_lagged(s: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """Value of the last close on or before (date - lag_days) for every date; flat after the last close."""
    cal = pd.date_range(min(s.index.min(), dates.min() - pd.Timedelta(days=lag_days + 40)),
                        max(s.index.max(), dates.max()), freq="D")
    ff = s.reindex(cal).ffill()
    return pd.Series(ff.reindex(dates - pd.Timedelta(days=lag_days)).to_numpy(), index=dates)


def fuel_frame(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Lag-safe fuel features on an hourly UTC index (NaN columns if the raw data is missing)."""
    raw = _load()
    mdate = idx_utc.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    dates = pd.DatetimeIndex(mdate.unique()).sort_values()
    f = pd.DataFrame(index=dates, columns=FUEL_COLUMNS, dtype="float64")
    for name, col in (("brent", "fuel_brent"), ("carbon", "fuel_carbon")):
        if name in raw:
            s = raw[name]
            now = _daily_lagged(s, dates, DAILY_LAG_DAYS)
            before = _daily_lagged(s, dates, DAILY_LAG_DAYS + 28)
            f[col + ("_usd" if name == "brent" else "_idx")] = now.to_numpy()
            f[col + "_chg20"] = (now / before - 1).to_numpy()
    if "pink" in raw:
        gas = monthly_gas_eur_mwh(raw["pink"], raw.get("eurusd"))
        ref = dates - pd.Timedelta(days=PUBLICATION_LAG_DAYS)
        month = ref.to_period("M") - 1                      # last month fully published at `ref`
        key = month.to_timestamp()
        trailing12 = gas.rolling(12, min_periods=6).mean()
        chg3 = gas / gas.shift(3) - 1
        for col, ser in (("fuel_gas_eur_mwh", gas), ("fuel_gas_rel12", gas / trailing12), ("fuel_gas_chg3", chg3)):
            f[col] = ser.reindex(key).to_numpy()
        # months not yet in the sheet -> carry the newest published value forward
        f[["fuel_gas_eur_mwh", "fuel_gas_rel12", "fuel_gas_chg3"]] = f[["fuel_gas_eur_mwh", "fuel_gas_rel12", "fuel_gas_chg3"]].ffill()
        f["fuel_gas_cost_ccgt"] = f["fuel_gas_eur_mwh"] / 0.5   # EUR per MWh_e at 50 % efficiency, fuel cost only
    # daily TTF (OilPriceAPI): D-2 average. Before the daily history starts the level falls back to the
    # lag-safe monthly gas value, while the change features stay NaN rather than mixing the two definitions.
    if "ttf" in raw:
        s_ttf = raw["ttf"]
        ttf_now = _daily_lagged(s_ttf, dates, DAILY_LAG_DAYS)
        ttf_before = _daily_lagged(s_ttf, dates, DAILY_LAG_DAYS + 28)
        f["fuel_ttf_chg20"] = (ttf_now / ttf_before - 1).to_numpy()
        gas_m = f["fuel_gas_eur_mwh"] if "pink" in raw else pd.Series(np.nan, index=dates)
        f["fuel_ttf_eur_mwh"] = pd.Series(ttf_now.to_numpy(), index=dates).fillna(gas_m).to_numpy()
        f["fuel_ttf_vs_gasm"] = (pd.Series(ttf_now.to_numpy(), index=dates) / gas_m - 1).to_numpy()   # move since last published month
    return f.reindex(mdate).set_axis(idx_utc)


def export_report() -> None:
    """Chart-ready (unlagged, descriptive) fuel series for the dashboard: reports/fuels_*.csv."""
    raw = _load()
    if not raw:
        return
    daily = pd.concat({k: v for k, v in raw.items() if k != "pink"}, axis=1)
    daily.index.name = "date"
    daily.to_csv(C.REPORTS / "fuels_daily.csv")
    if "pink" in raw:
        m = raw["pink"].copy()
        m["gas_eur_mwh"] = monthly_gas_eur_mwh(raw["pink"], raw.get("eurusd"))
        m.index.name = "month"
        m.to_csv(C.REPORTS / "fuels_monthly.csv")
    snap = C.DATA_RAW / "fuels" / "ttf_snapshots.csv"
    if snap.exists():
        pd.read_csv(snap).to_csv(C.REPORTS / "ttf_snapshots.csv", index=False)

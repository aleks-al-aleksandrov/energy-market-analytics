"""Live scores beyond the hourly track record: each ensemble member, and the 15-minute forecast.

* Members: reports/member_log.csv already holds every member's live forecast (rows with source == "live"), so each one is
  scored per delivery day against the published hourly price. This shows which model is carrying the ensemble right now.
* Quarters: reports/forecast_log_15min.csv keeps the first 15-minute forecast of every quarter, with the time it was made.
  Like the hourly track record, only forecasts made before gate closure (12:00 CET on D-1) count. Each live day is scored
  against the published quarter prices, next to the same forecast without the within-hour shape ("flat"), so the live value
  of the shape model is visible.

Writes reports/live_detail.csv (one row per delivery day). Called by predict_next_day; also `python -m src.live_scores`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C

log = logging.getLogger("live_scores")
QLOG, OUT = "forecast_log_15min.csv", "live_detail.csv"
GATE_TZ = "Europe/Brussels"
MEMBERS = ["lightgbm", "ridge", "level", "lear"]


def append_quarters(q15: pd.DataFrame, generated_at: pd.Timestamp) -> None:
    """Log a 15-minute forecast (index = UTC quarter starts). The first live forecast of a quarter is never replaced."""
    path = C.REPORTS / QLOG
    new = pd.DataFrame({"timestamp_utc": pd.to_datetime(q15.index, utc=True), "delivery_day": q15["delivery_day"].astype(str).to_numpy(),
                        "forecast_eur_mwh": q15["forecast_eur_mwh"].to_numpy(),
                        "flat_eur_mwh": (q15["forecast_eur_mwh"] - q15.get("shape_eur_mwh", 0.0)).to_numpy()})
    gate = (pd.to_datetime(new["delivery_day"]) - pd.Timedelta(days=1)).dt.tz_localize(GATE_TZ) + pd.Timedelta(hours=12)
    new["generated_at_utc"] = generated_at.tz_convert("UTC")
    new["is_live"] = new["generated_at_utc"] < gate.dt.tz_convert("UTC")
    if path.exists():
        old = pd.read_csv(path)
        old["timestamp_utc"] = pd.to_datetime(old["timestamp_utc"], utc=True)
        old["generated_at_utc"] = pd.to_datetime(old["generated_at_utc"], utc=True)
        new = pd.concat([old, new])
    new["is_live"] = new["is_live"].astype(str).str.lower() == "true"
    new = new.sort_values(["timestamp_utc", "is_live", "generated_at_utc"], ascending=[True, False, True])
    new.drop_duplicates("timestamp_utc", keep="first").sort_values("timestamp_utc").round(3).to_csv(path, index=False)


def update(quarter_prices: pd.Series | None = None) -> pd.DataFrame | None:
    """Score every live day that has published prices. `quarter_prices` (15-minute, UTC) is read from the raw cache if None."""
    if quarter_prices is None:
        from src import quarter_hour as Q
        quarter_prices = Q.quarter_prices()
    q = quarter_prices.sort_index()
    hourly = q.groupby(q.index.floor("h")).mean()
    parts = []

    mp = C.REPORTS / "member_log.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        m = m[m["source"] == "live"].copy()
        if len(m):
            m["timestamp_utc"] = pd.to_datetime(m["timestamp_utc"], utc=True)
            m["actual"] = hourly.reindex(m["timestamp_utc"]).to_numpy()
            m = m.dropna(subset=["actual"])
            cols = [c for c in MEMBERS if c in m]
            if len(m) and cols:
                err = m[cols].sub(m["actual"], axis=0).abs()
                err["delivery_day"] = m["delivery_day"].astype(str).str[:10].to_numpy()
                parts.append(err.groupby("delivery_day").mean().add_prefix("mae_"))

    qp = C.REPORTS / QLOG
    if qp.exists():
        ql = pd.read_csv(qp)
        ql = ql[ql["is_live"].astype(str).str.lower() == "true"].copy()
        if len(ql):
            ql["timestamp_utc"] = pd.to_datetime(ql["timestamp_utc"], utc=True)
            ql["actual"] = q.reindex(ql["timestamp_utc"]).to_numpy()
            ql = ql.dropna(subset=["actual"])
            if len(ql):
                ql["e_q"] = (ql["actual"] - ql["forecast_eur_mwh"]).abs()
                ql["e_flat"] = (ql["actual"] - ql["flat_eur_mwh"]).abs()
                g = ql.groupby(ql["delivery_day"].astype(str).str[:10])
                parts.append(pd.DataFrame({"quarters": g.size(), "mae_quarter": g["e_q"].mean(), "mae_quarter_flat": g["e_flat"].mean()}))

    if not parts:
        return None
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "delivery_day"
    out.round(3).to_csv(C.REPORTS / OUT)
    log.info("live detail: %d days", len(out))
    return out


def summary(detail: pd.DataFrame | None = None) -> dict:
    """Averages over the live days, weighted equally per day."""
    d = detail
    if d is None:
        p = C.REPORTS / OUT
        if not p.exists():
            return {}
        d = pd.read_csv(p)
    out: dict = {"days": int(len(d))}
    for c in [f"mae_{m}" for m in MEMBERS] + ["mae_quarter", "mae_quarter_flat"]:
        if c in d and d[c].notna().any():
            out[c] = float(d[c].mean())
            out[c.replace("mae_", "days_") + "_scored"] = int(d[c].notna().sum())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(update())

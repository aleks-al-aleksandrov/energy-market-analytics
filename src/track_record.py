"""Live track record of the next-day forecasts.

predict_next_day appends every forecast to reports/forecast_log.csv together with the time it was generated.
A forecast only counts as *live* if it was generated before day-ahead gate closure (12:00 CET on D-1); later
runs (re-runs, back-fills) are logged but excluded. This module scores live forecasts against the published
prices and the naive baseline, once the prices are available in the raw price cache.

Usage: python -m src.track_record          (called automatically by predict_next_day)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C

log = logging.getLogger("track_record")
LOG = C.REPORTS / "forecast_log.csv"
OUT = C.REPORTS / "live_track_record.csv"
GATE_TZ = "Europe/Brussels"


def append_forecast(out: pd.DataFrame, generated_at: pd.Timestamp) -> None:
    """Add one forecast to the log; the first forecast of a delivery hour is never overwritten."""
    rows = out.reset_index().rename(columns={out.index.name or "index": "timestamp_utc"})
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
    day = pd.to_datetime(rows["delivery_day"])
    gate = (day - pd.Timedelta(days=1)).dt.tz_localize(GATE_TZ) + pd.Timedelta(hours=12)
    rows["generated_at_utc"] = generated_at.tz_convert("UTC")
    rows["is_live"] = rows["generated_at_utc"] < gate.dt.tz_convert("UTC")
    keep = ["timestamp_utc", "delivery_day", "forecast_eur_mwh", "p10_eur_mwh", "p90_eur_mwh", "spike_probability",
            "generated_at_utc", "is_live", "model_trained_until"]
    new = rows[[c for c in keep if c in rows]]
    if LOG.exists():
        old = pd.read_csv(LOG, parse_dates=["timestamp_utc", "generated_at_utc"])
        new = pd.concat([old, new])
    # keep the earliest live forecast per hour (a re-run after gate closure must not replace it)
    new = new.sort_values(["timestamp_utc", "is_live", "generated_at_utc"], ascending=[True, False, True])
    new.drop_duplicates(subset=["timestamp_utc"], keep="first").sort_values("timestamp_utc").to_csv(LOG, index=False)


def update() -> pd.DataFrame | None:
    if not LOG.exists():
        return None
    fl = pd.read_csv(LOG, parse_dates=["timestamp_utc", "generated_at_utc"])
    fl = fl[fl["is_live"].astype(str).str.lower() == "true"]
    from src.ingest import load_raw          # freshly refreshed raw price cache (hourly.parquet may be stale in the daily job)
    h = load_raw("prices/BG").iloc[:, 0].resample("1h").mean()
    fl["timestamp_utc"] = pd.to_datetime(fl["timestamp_utc"], utc=True)
    fl["actual"] = h.reindex(fl["timestamp_utc"]).to_numpy()
    fl["naive"] = h.shift(24).reindex(fl["timestamp_utc"]).to_numpy()
    fl = fl.dropna(subset=["actual"])
    if fl.empty:
        pd.DataFrame(columns=["delivery_day", "hours", "mae_model", "mae_naive", "bias", "coverage_10_90",
                              "avg_price"]).to_csv(OUT, index=False)
        return None
    e = fl["actual"] - fl["forecast_eur_mwh"]
    fl["abs_err"], fl["abs_naive"], fl["err"] = e.abs(), (fl["actual"] - fl["naive"]).abs(), e
    fl["inside"] = (fl["actual"] >= fl["p10_eur_mwh"]) & (fl["actual"] <= fl["p90_eur_mwh"])
    day = fl.groupby("delivery_day").agg(hours=("actual", "size"), mae_model=("abs_err", "mean"), mae_naive=("abs_naive", "mean"),
                                         bias=("err", "mean"), coverage_10_90=("inside", "mean"), avg_price=("actual", "mean"))
    day.round(3).to_csv(OUT)
    log.info("live track record: %d days, MAE %.1f vs naive %.1f", len(day), fl["abs_err"].mean(), fl["abs_naive"].mean())
    return day


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    update()

"""Forecast the next delivery day:  python -m src.predict_next_day [--date YYYY-MM-DD]

Loads models/final_models.joblib (written by `run_all --plus`), refreshes the latest
data (only the incomplete month is re-downloaded), deliberately removes every
price at/after the start of the delivery day (information available before gate
closure only), and writes reports/forecast_next_day.csv with the hourly point
forecast, 10/90 % band and spike probability.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import MARKET_TZ, TARGET, build_features

log = logging.getLogger("predict")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="delivery day (default: tomorrow)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached data only")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle_path = C.MODELS_DIR / "final_models.joblib"
    if not bundle_path.exists():
        raise SystemExit("models/final_models.joblib missing - run `python -m src.run_all --plus` first")
    bundle = joblib.load(bundle_path)

    today = pd.Timestamp.now(tz=MARKET_TZ).normalize()
    day = pd.Timestamp(a.date, tz=MARKET_TZ) if a.date else today + pd.Timedelta(days=1)
    d_start = day.tz_convert("UTC")
    d_end = (day + pd.DateOffset(days=1)).tz_convert("UTC")

    if not a.no_refresh:
        from src.ingest import ingest_entsoe, ingest_fuels, ingest_weather
        start = (day - pd.Timedelta(days=40)).tz_convert(C.LOCAL_TZ).normalize().replace(day=1)
        end = (day + pd.DateOffset(days=2)).tz_convert(C.LOCAL_TZ).normalize()
        ingest_entsoe(start, end, only=["prices", "load_forecast", "wind_solar_forecast", "flows"])
        ingest_weather(start, end, datasets=["weather_fc_d1"])
        ingest_fuels()
        from src.fuels import export_report
        export_report()   # publishes reports/ttf_snapshots.csv (daily TTF log)

    from src.clean import build_hourly
    h, _, _ = build_hourly(end=d_end + pd.Timedelta(hours=1))
    from src import validate
    validate.run(h[h.index < d_start], strict=True, max_age_days=3)        # history up to D-1 must be healthy
    h = h[h.index < d_end]
    actual = h.loc[(h.index >= d_start), TARGET].copy() if TARGET in h else None
    # gate-closure information set: no prices of day D (or later) may be visible
    price_cols = [c for c in h.columns if c.startswith("price_")]
    h.loc[h.index >= d_start, price_cols] = np.nan

    f = build_features(h)
    X = f.loc[f["delivery_day"] == day.tz_localize(None), bundle["features"]]
    if X.empty:
        raise SystemExit(f"no feature rows for {day.date()}")
    missing = X[["load_forecast", "wind_forecast", "solar_forecast"]].isna().mean()
    if (missing > 0.5).any():
        log.warning("ENTSO-E day-ahead forecasts for %s not (fully) published yet:\n%s", day.date(), missing)

    m = bundle["models"]
    out = pd.DataFrame(index=X.index)
    out["time_local"] = X.index.tz_convert(C.LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    out["forecast_eur_mwh"] = m["point"].predict(X)
    if "level" in m:      # cross-check: forecast of the deviation from yesterday's mean, and a 50/50 blend (see robust_experiment)
        out["forecast_level_eur_mwh"] = m["level"].predict(X)
        out["forecast_blend_eur_mwh"] = 0.5 * out["forecast_eur_mwh"] + 0.5 * out["forecast_level_eur_mwh"]
    q = np.sort(np.column_stack([m["q10"].predict(X), out["forecast_eur_mwh"], m["q90"].predict(X)]), axis=1)
    out["p10_raw_eur_mwh"], out["p90_raw_eur_mwh"] = q[:, 0], q[:, 2]
    out["p10_eur_mwh"], out["p90_eur_mwh"] = q[:, 0], q[:, 2]
    conf = C.REPORTS / "conformal_offsets.json"
    if conf.exists():   # conformal recalibration (src/calibrate.py): widen each side by its recent out-of-sample miss
        off = json.loads(conf.read_text())
        out["p10_eur_mwh"] = out["p10_raw_eur_mwh"] - off["a_lo"]
        out["p90_eur_mwh"] = out["p90_raw_eur_mwh"] + off["a_hi"]
        out["band_calibrated_as_of"] = off["as_of"]
    out["spike_probability"] = m["spike"].predict_proba(X)[:, 1]
    if "spike_excess_threshold_eur" in bundle:   # level-relative definition: yesterday's daily mean + the P95 excess
        out["spike_threshold_eur"] = X["price_d1_mean"].to_numpy() + bundle["spike_excess_threshold_eur"]
        out["spike_definition"] = f"price > D-1 mean + {bundle['spike_excess_threshold_eur']:.0f}"
    else:
        out["spike_threshold_eur"] = bundle["spike_threshold_eur"]
        out["spike_definition"] = f"price > {bundle['spike_threshold_eur']:.0f}"
    if actual is not None and actual.notna().any():
        out["actual_eur_mwh"] = actual.reindex(out.index)  # only for back-tests / after publication
    out["delivery_day"] = str(day.date())
    out["model_trained_until"] = bundle["trained_until"]
    out["generated_at"] = pd.Timestamp.now(tz=C.LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    out.round(3).to_csv(C.REPORTS / "forecast_next_day.csv", index_label="timestamp_utc")
    from src import track_record
    track_record.append_forecast(out, pd.Timestamp.now(tz="UTC"))     # live track record (only pre-gate-closure runs count)
    track_record.update()
    try:                                  # point-in-time outage snapshot -> a leak-free outage history accumulates from now on
        from src import outages
        outages.record_point_in_time(day)
    except Exception as exc:              # best effort: never fail the forecast because of this
        log.warning("outage snapshot skipped: %s", str(exc)[:80])
    log.info("forecast for %s written (%d hours), mean %.1f EUR/MWh", day.date(), len(out),
             out["forecast_eur_mwh"].mean())
    print(out[["time_local", "forecast_eur_mwh", "p10_eur_mwh", "p90_eur_mwh", "spike_probability"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()


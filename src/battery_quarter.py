"""Battery value with 15-minute products:  python -m src.battery_quarter

Since October 2025 Bulgarian day-ahead prices settle per quarter hour. The headline battery figure (about EUR 100k per MW-year)
was computed on hourly prices. This study asks how much changes when the battery trades quarters, and whether the forecast's
within-hour shape (src/quarter_hour.py) is good enough to capture that extra value.

Strategies, all 1 MW / 2 MWh / 90 % round trip / one cycle per day, every schedule SETTLED AT QUARTER-HOUR ACTUAL PRICES:
  * perfect_hourly     schedule on the actual hourly averages, held flat across each hour (ceiling of hourly trading)
  * perfect_quarter    schedule on the actual quarter prices (ceiling of quarter trading)
  * forecast_hourly    schedule on the hourly forecast, held flat across each hour (what the project did before)
  * forecast_quarter   schedule on hourly forecast + predicted within-hour shape, per quarter
  * naive_quarter      schedule on yesterday's quarter prices

Forecasts are out-of-sample: the hourly forecast is the walk-forward LightGBM prediction (reports/predictions.csv), the shape
comes from the same monthly walk-forward as the quarter-hour experiment.

RULE (fixed before running): the battery figures switch to quarter scheduling if, on the CV months, forecast_quarter earns more
than forecast_hourly and the 95 % moving-block bootstrap interval (7-day blocks) of the daily gain is above zero. Aug-Oct 2026
reported only. Writes reports/battery_quarter.csv, reports/battery_quarter_daily.csv and reports/battery_quarter_choice.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import battery as B
from src import config as C
from src import quarter_hour as Q
from src.battery_uncertainty import _block_ci

log = logging.getLogger("battery_quarter")
STRATS = ["perfect_quarter", "perfect_hourly", "forecast_quarter", "forecast_hourly", "naive_quarter"]


def quarter_frame() -> pd.DataFrame:
    """One row per quarter: actual price, hourly forecast, predicted shape, yesterday's quarter price, split."""
    from src.features import build_features

    f = build_features(pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet"))
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["delivery_day"])
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p = p.set_index("timestamp_utc")
    fc = p["lightgbm"]
    q = Q.quarter_prices()
    q = q[q.index.floor("h").isin(fc.dropna().index)]
    dev = Q.deviations(q)
    x = Q.build_quarter_features(q.index, dev, f, fc)
    months = sorted(pd.PeriodIndex(x["delivery_day"], freq="M").unique())
    months = [m for m in months if m >= pd.Period(Q.FIRST_TEST_MONTH, "M")]
    shape = Q.shape_candidates(x, dev, months)["lgbm"]
    d = pd.DataFrame({"actual_q": q, "shape": shape}).dropna()
    hour = d.index.floor("h")
    d["delivery_day"] = x["delivery_day"].reindex(d.index)
    d["actual_h"] = d["actual_q"].groupby(hour).transform("mean")
    d["forecast_h"] = fc.reindex(hour).to_numpy()
    d["forecast_q"] = d["forecast_h"] + d["shape"]
    d["naive_q"] = q.reindex(d.index - pd.Timedelta(days=1)).to_numpy()
    hold = set(p.loc[p["split"] == "holdout", "month"].astype(str))
    d["split"] = np.where(pd.PeriodIndex(d["delivery_day"], freq="M").astype(str).isin(hold), "holdout", "cv")
    return d.dropna(subset=["forecast_h", "naive_q"])


def _flat(net_hourly: np.ndarray) -> np.ndarray:
    """An hourly schedule (MWh per hour) expressed per quarter: a quarter of the hour's energy in each of its four quarters."""
    return np.repeat(net_hourly, 4) / 4


def simulate_quarters(d: pd.DataFrame, b: B.Battery) -> pd.DataFrame:
    rows = []
    for day, g in d.groupby("delivery_day"):
        if len(g) % 4 or len(g) < 88:                    # need whole hours (92 / 96 / 100 quarters on DST days)
            continue
        act = g["actual_q"].to_numpy()
        hours = g.index.floor("h")
        if not (pd.Series(hours).value_counts() == 4).all():
            continue
        r = {"delivery_day": day, "split": g["split"].iloc[0], "spread_q": float(act.max() - act.min())}
        sched = {
            "perfect_quarter": B.schedule_day(act, b, step_h=0.25),
            "perfect_hourly": _flat(B.schedule_day(g["actual_h"].to_numpy()[::4], b)),
            "forecast_quarter": B.schedule_day(g["forecast_q"].to_numpy(), b, step_h=0.25),
            "forecast_hourly": _flat(B.schedule_day(g["forecast_h"].to_numpy()[::4], b)),
            "naive_quarter": B.schedule_day(g["naive_q"].to_numpy(), b, step_h=0.25),
        }
        for k, net in sched.items():
            r[k] = float(np.sum(net * act))
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    b = B.Battery()
    d = quarter_frame()
    log.info("quarters %d, days %d, %s .. %s", len(d), d["delivery_day"].nunique(), d["delivery_day"].min().date(), d["delivery_day"].max().date())
    daily = simulate_quarters(d, b)
    daily.to_csv(C.REPORTS / "battery_quarter_daily.csv", index=False)
    rows = []
    for split in ("cv", "holdout"):
        g = daily[daily["split"] == split]
        for s in STRATS:
            r = dict(strategy=s, split=split, days=len(g), eur_per_mw_year=float(g[s].mean() * 365),
                     pct_of_perfect_quarter=float(100 * g[s].sum() / g["perfect_quarter"].sum()))
            if s == "forecast_quarter":
                diff = (g["forecast_quarter"] - g["forecast_hourly"]).to_numpy()
                r["gain_vs_forecast_hourly_eur_per_day"] = float(diff.mean())
                r["ci_lo"], r["ci_hi"] = _block_ci(diff)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "battery_quarter.csv", index=False)
    cvq = out[(out.split == "cv") & (out.strategy == "forecast_quarter")].iloc[0]
    adopt = bool(cvq["gain_vs_forecast_hourly_eur_per_day"] > 0 and cvq["ci_lo"] > 0)
    cv = out[out.split == "cv"].set_index("strategy")
    (C.REPORTS / "battery_quarter_choice.json").write_text(json.dumps(dict(
        schedule_on="quarter" if adopt else "hourly",
        cv_days=int(cv.loc["forecast_quarter", "days"]),
        forecast_quarter_eur_per_mw_year=float(cv.loc["forecast_quarter", "eur_per_mw_year"]),
        forecast_hourly_eur_per_mw_year=float(cv.loc["forecast_hourly", "eur_per_mw_year"]),
        perfect_quarter_eur_per_mw_year=float(cv.loc["perfect_quarter", "eur_per_mw_year"]),
        perfect_hourly_eur_per_mw_year=float(cv.loc["perfect_hourly", "eur_per_mw_year"]),
        gain_eur_per_day=float(cvq["gain_vs_forecast_hourly_eur_per_day"]), ci=[float(cvq["ci_lo"]), float(cvq["ci_hi"])],
        rule="quarter scheduling adopted iff it earns more than hourly scheduling on the CV months and the 95 % block-bootstrap "
             "interval of the daily gain is above zero; holdout reported only",
        months_note="15-minute prices exist from 2025-10; CV months 2026-01..2026-07"), indent=2))
    print(out.round(1).to_string(index=False))
    print("schedule on:", "quarter" if adopt else "hourly")


if __name__ == "__main__":
    main()

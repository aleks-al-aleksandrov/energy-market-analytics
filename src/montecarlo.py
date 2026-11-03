"""Monte Carlo scenarios for battery revenue: what range of outcomes does a forecast-driven schedule face?

The battery schedule is fixed by the forecast (that is the decision). What is uncertain is the price it is settled at.
Two scenario generators are compared, and the better one is chosen on the CV months only (the holdout is only reported):

* ``error_vector``: scenario prices = forecast + a whole 24-hour error vector drawn from the model's recent out-of-sample
  misses (actual - forecast). Keeps the hour-to-hour correlation of errors but treats the error as independent of the
  schedule.
* ``slippage``: scenario revenue = expected revenue at forecast prices + a "slippage" drawn from recent days, where
  slippage = realised - expected revenue of the schedule on its own day (= error vector dotted with that day's
  dispatch). This keeps the optimiser's curse: a schedule sells where the forecast is high and buys where it is low, so
  the errors it meets are not neutral, and realised revenue is systematically below the revenue expected at forecast prices.

Outputs: next-day revenue distribution, a calibration back-test (does the P10-P90 range contain ~80 % of realised days,
using only errors known before each day?), and scenario years (blocks of consecutive days, so regimes persist).

This is a scenario analysis conditional on the error pool, not a structural price model: it cannot produce a price shock the
pool has not seen, and it inherits any bias in the pool.

Usage: python -m src.montecarlo   (after run_all; reads reports/predictions.csv and forecast_next_day.csv)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import battery as B
from src import config as C

log = logging.getLogger("montecarlo")
WINDOW_DAYS = 180
N_DRAWS = 4000
N_YEARS = 2000
BLOCK_DAYS = 7
SEED = 42
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
METHODS = ("error_vector", "slippage")
HOLDOUT_START = pd.Timestamp("2026-08-01")      # the final holdout (reported, never used to choose the method)


def daily_vectors(p: pd.DataFrame, col: str = "lightgbm") -> pd.DataFrame:
    """One row per complete delivery day: forecast and actual vectors (24 values each)."""
    rows = []
    for day, g in p.dropna(subset=["actual", col]).groupby("delivery_day"):
        if len(g) == 24:
            rows.append((pd.Timestamp(day), g[col].to_numpy(float), g["actual"].to_numpy(float)))
    return pd.DataFrame(rows, columns=["day", "forecast", "actual"]).set_index("day")


def schedule_all(v: pd.DataFrame, b: B.Battery) -> tuple[np.ndarray, np.ndarray]:
    """Net dispatch (n_days x 24) and degradation cost (n_days) for every forecast day."""
    nets = np.vstack([B.schedule_day(f, b) for f in v["forecast"]])
    deg = np.array([b.degradation_eur_mwh * B.throughput(n) for n in nets])
    return nets, deg


def history(v: pd.DataFrame, nets: np.ndarray, deg: np.ndarray) -> dict[str, np.ndarray]:
    fc, act = np.vstack(v["forecast"].to_numpy()), np.vstack(v["actual"].to_numpy())
    expected = (fc * nets).sum(1) - deg
    realised = (act * nets).sum(1) - deg
    return {"fc": fc, "act": act, "err": act - fc, "expected": expected, "realised": realised, "slip": realised - expected}


def draws(method: str, net: np.ndarray, forecast: np.ndarray, deg: float, h: dict, lo: int, hi: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """EUR revenue scenarios for a fixed schedule, from history days [lo, hi)."""
    if method == "error_vector":
        e = h["err"][rng.integers(lo, hi, n)]
        return (e + forecast) @ net - deg
    return float(forecast @ net - deg) + h["slip"][rng.integers(lo, hi, n)]


def calibration_backtest(v: pd.DataFrame, nets: np.ndarray, deg: np.ndarray, method: str, window: int = WINDOW_DAYS,
                         n: int = 1000, seed: int = SEED, h: dict | None = None) -> pd.DataFrame:
    """For each day with `window` days of history: simulate with only past errors, compare with the realised revenue.

    Day t uses history days t-window .. t-1 only (the previous day's actuals are public before day t's gate closure).
    """
    h = h or history(v, nets, deg)
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(window, len(v)):
        d = draws(method, nets[t], h["fc"][t], deg[t], h, t - window, t, n, rng)
        real = float(h["realised"][t])
        q = np.quantile(d, [0.05, 0.10, 0.50, 0.90, 0.95])
        rows.append({"delivery_day": v.index[t], "realised": real, "expected_at_forecast": float(h["expected"][t]),
                     "p05": q[0], "p10": q[1], "p50": q[2], "p90": q[3], "p95": q[4],
                     "pit": float((d < real).mean() + 0.5 * (d == real).mean()), "prob_loss": float((d < 0).mean()), "method": method})
    out = pd.DataFrame(rows)
    out["in_10_90"] = (out["realised"] >= out["p10"]) & (out["realised"] <= out["p90"])
    out["in_05_95"] = (out["realised"] >= out["p05"]) & (out["realised"] <= out["p95"])
    return out


def coverage_summary(cal: pd.DataFrame) -> dict:
    out = {"days": int(len(cal)), "coverage_10_90": float(cal["in_10_90"].mean()), "coverage_05_95": float(cal["in_05_95"].mean()),
           "pit_mean": float(cal["pit"].mean()), "mean_predicted_prob_loss": float(cal["prob_loss"].mean()),
           "realised_loss_rate": float((cal["realised"] < 0).mean()), "median_bias_eur": float((cal["realised"] - cal["p50"]).median())}
    for name, sub in (("cv", cal[cal["delivery_day"] < HOLDOUT_START]), ("holdout", cal[cal["delivery_day"] >= HOLDOUT_START])):
        if len(sub):
            out[f"coverage_10_90_{name}"] = float(sub["in_10_90"].mean())
            out[f"median_bias_eur_{name}"] = float((sub["realised"] - sub["p50"]).median())
            out[f"days_{name}"] = int(len(sub))
    return out


def choose_method(summaries: dict[str, dict]) -> str:
    """Pre-set rule, CV months only: the generator whose 10-90 coverage is closest to the 80 % target."""
    return min(summaries, key=lambda m: abs(summaries[m]["coverage_10_90_cv"] - 0.80))


def next_day(forecast: np.ndarray, h: dict, b: B.Battery, method: str, window: int = WINDOW_DAYS, n: int = N_DRAWS, seed: int = SEED) -> dict:
    net = B.schedule_day(forecast, b)
    deg = b.degradation_eur_mwh * B.throughput(net)
    hi = len(h["err"])
    d = draws(method, net, forecast, deg, h, max(0, hi - window), hi, n, np.random.default_rng(seed))
    out = {f"p{int(q * 100):02d}": float(np.quantile(d, q)) for q in QUANTILES}
    out.update(expected_at_forecast=float(forecast @ net - deg), mean=float(d.mean()), prob_loss=float((d < 0).mean()),
               pool_days=min(window, hi), draws=n, method=method)
    return out


def scenario_years(h: dict, window: int = WINDOW_DAYS, n_years: int = N_YEARS, block: int = BLOCK_DAYS, seed: int = SEED) -> np.ndarray:
    """Annual revenue (EUR per MW-year) of scenario years.

    Each year strings together `block`-day runs of back-test days sampled with replacement (seasonality and spread clustering
    survive) and adds slippage blocks drawn from the last `window` days. Consecutive days stay together, so a bad regime persists
    across the block instead of averaging out day by day.
    """
    rng = np.random.default_rng(seed)
    n_days = len(h["expected"])
    n_blocks = int(np.ceil(365 / block))
    lo = max(0, n_days - window)
    offs = np.arange(block)[None, :]
    totals = np.empty(n_years)
    for y in range(n_years):
        idx = (rng.integers(0, n_days - block + 1, n_blocks)[:, None] + offs).ravel()[:365]
        sidx = (rng.integers(lo, n_days - block + 1, n_blocks)[:, None] + offs).ravel()[:365]
        totals[y] = h["expected"][idx].sum() + h["slip"][sidx].sum()
    return totals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["timestamp_utc", "delivery_day"])
    b = B.Battery()
    v = daily_vectors(p)
    nets, deg = schedule_all(v, b)
    h = history(v, nets, deg)

    cals, summaries = {}, {}
    for m in METHODS:
        cals[m] = calibration_backtest(v, nets, deg, m, h=h)
        summaries[m] = coverage_summary(cals[m])
        log.info("calibration %-12s %s", m, {k: round(x, 3) if isinstance(x, float) else x for k, x in summaries[m].items()})
    method = choose_method(summaries)
    pd.concat(cals.values()).to_csv(C.REPORTS / "mc_calibration.csv", index=False)

    years = scenario_years(h)
    pd.DataFrame({"eur_per_mw_year": years}).to_csv(C.REPORTS / "mc_annual_draws.csv", index=False)
    cal_c = cals[method]
    realised_year = float(cal_c["realised"].mean() * 365)
    annual = {f"p{int(q * 100):02d}": float(np.quantile(years, q)) for q in QUANTILES}
    annual.update(mean=float(years.mean()), sd=float(years.std()), n_years=len(years), block_days=BLOCK_DAYS,
                  pool=f"last {WINDOW_DAYS} back-test days (slippage blocks)", backtest_realised_eur_per_mw_year=realised_year,
                  backtest_realised_percentile=float((years < realised_year).mean()))
    log.info("scenario years: %s", {k: round(x) if isinstance(x, float) else x for k, x in annual.items() if k.startswith("p")})

    nxt = None
    nd_path = C.REPORTS / "forecast_next_day.csv"
    if nd_path.exists():
        nd = pd.read_csv(nd_path)
        if len(nd) == 24:
            nxt = next_day(nd["forecast_eur_mwh"].to_numpy(float), h, b, method)
            nxt["delivery_day"] = str(nd["delivery_day"].iloc[0])
    (C.REPORTS / "mc_summary.json").write_text(json.dumps(
        {"chosen_method": method, "rule": "generator whose P10-P90 coverage on the CV months is closest to 80 %; holdout only reported",
         "calibration": summaries, "annual": annual, "next_day": nxt, "battery": "1 MW / 2 MWh / 90 %, no degradation"}, indent=2))
    log.info("chosen method: %s", method)


if __name__ == "__main__":
    main()

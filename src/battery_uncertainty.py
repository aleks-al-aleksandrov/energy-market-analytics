"""Battery scheduling under forecast uncertainty:  python -m src.battery_uncertainty

Revenue is linear in prices, so the schedule that maximises EXPECTED revenue over a set of price scenarios is simply the
optimal schedule for the scenario mean. With scenarios = forecast + recent error vectors, that mean is the forecast plus its
recent average error per hour of day. So "stochastic scheduling" here means: correct the forecast's hourly bias, then schedule.

Candidates (fixed before running), all settled at actual prices, 1 MW / 2 MWh / 90 %:
  * forecast            schedule on the raw LightGBM forecast (current behaviour)
  * bias_14 / bias_30   forecast + mean error of the same hour over the previous 14 / 30 delivery days (up to D-1)
  * shape_14            forecast + that correction with its daily mean removed (fixes the intraday shape, not the level;
                        the level cannot change a one-cycle schedule much, the shape can)

RULE: a candidate replaces the raw forecast for scheduling only if, on the CV months, its revenue is higher and the 95 %
moving-block bootstrap interval (7-day blocks) of the daily revenue difference excludes zero. Holdout reported only.
Writes reports/battery_uncertainty.csv and reports/battery_uncertainty_choice.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import battery as B
from src import config as C

log = logging.getLogger("battery_uncertainty")
MODEL = "lightgbm"


def hourly_bias(p: pd.DataFrame, model: str, window: int) -> pd.Series:
    """Mean error (actual - forecast) of the same position-in-day over the previous `window` delivery days, up to D-1."""
    d = p.dropna(subset=["actual", model]).copy()
    d["pos"] = d.groupby("delivery_day").cumcount().clip(upper=23)
    err = (d["actual"] - d[model]).groupby([d["delivery_day"], d["pos"]]).mean().unstack("pos")
    bias = err.rolling(window, min_periods=max(3, window // 3)).mean().shift(1)          # days before D only
    stacked = bias.stack(future_stack=True).rename("bias")
    return d.join(stacked, on=["delivery_day", "pos"])["bias"].fillna(0.0)


def _block_ci(diff: np.ndarray, block: int = 7, n_boot: int = 2000, seed: int = C.SEED) -> tuple[float, float]:
    n = len(diff)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    lo, hi = np.quantile(diff[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["delivery_day"])
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p = p.set_index("timestamp_utc").dropna(subset=["actual", MODEL]).sort_index()
    b14, b30 = hourly_bias(p, MODEL, 14), hourly_bias(p, MODEL, 30)
    p["bias_14"], p["bias_30"] = p[MODEL] + b14, p[MODEL] + b30
    p["shape_14"] = p[MODEL] + (b14 - b14.groupby(p["delivery_day"]).transform("mean"))
    cols = {"perfect_foresight": "actual", "forecast": MODEL, "bias_14": "bias_14", "bias_30": "bias_30", "shape_14": "shape_14"}
    bat = B.Battery()
    daily = B.simulate(p, bat, cols).merge(p.groupby("delivery_day")["split"].first().reset_index(), on="delivery_day")
    rows, passed = [], []
    for split in ("cv", "holdout"):
        g = daily[daily["split"] == split]
        for c in cols:
            r = dict(strategy=c, split=split, days=len(g), eur_per_mw_year=float(g[c].mean() * 365),
                     pct_of_perfect=float(100 * g[c].sum() / g["perfect_foresight"].sum()))
            if c not in ("perfect_foresight", "forecast"):
                diff = (g[c] - g["forecast"]).to_numpy()
                r["gain_eur_per_day"] = float(diff.mean())
                r["ci_lo"], r["ci_hi"] = _block_ci(diff)
                if split == "cv" and r["gain_eur_per_day"] > 0 and r["ci_lo"] > 0:
                    passed.append((r["eur_per_mw_year"], c))
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "battery_uncertainty.csv", index=False)
    choice = max(passed)[1] if passed else "forecast"
    (C.REPORTS / "battery_uncertainty_choice.json").write_text(json.dumps(dict(
        schedule_on=choice, candidates_passing=[c for _, c in passed],
        rule="higher CV revenue and a 95 % block-bootstrap interval of the daily revenue gain above zero; holdout reported only"),
        indent=2))
    print(out.round(2).to_string(index=False))
    print("schedule on:", choice)


if __name__ == "__main__":
    main()

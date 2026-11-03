"""Conformal recalibration of the 10-90 % quantile band (conformalised quantile regression, per side).

The raw quantile-LightGBM band covers far less than 80 % of outcomes. For every delivery day D we widen
each side by the 90 % quantile of the recent *out-of-sample* misses of that side:

    lower miss = q10 - actual   (positive when the price fell below the band)
    upper miss = actual - q90   (positive when the price rose above the band)

Only forecasts whose delivery day is on or before D-2 are used (outcomes of D-1 are not final when the forecast
for D is made at gate closure). The window length (days) is chosen on the CV months only; the holdout is
reported as is. Usage: python -m src.calibrate   (reads and updates reports/predictions.csv)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import config as C

log = logging.getLogger("calibrate")
WINDOWS = [30, 60, 90, 180, None]     # days of history; None = expanding
MIN_HISTORY_DAYS = 60
LEVEL = 0.9                           # each side must be exceeded 10 % of the time


def _conformal_quantile(scores: np.ndarray, level: float = LEVEL) -> float:
    n = len(scores)
    k = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, k))


def offsets_by_day(p: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """Per delivery day: (a_lo, a_hi) estimated from days <= D-2 inside the window."""
    d = p.dropna(subset=["actual", "q10", "q90"])
    day = d["delivery_day"].dt.normalize()
    lo, hi = (d["q10"] - d["actual"]).to_numpy(), (d["actual"] - d["q90"]).to_numpy()
    days = np.sort(day.unique())
    rows = []
    for D in days:
        end = D - np.timedelta64(2, "D")
        sel = (day.to_numpy() <= end)
        if window is not None:
            sel &= day.to_numpy() > end - np.timedelta64(window, "D")
        n_days = len(np.unique(day.to_numpy()[sel]))
        if n_days < (MIN_HISTORY_DAYS if window is None else min(MIN_HISTORY_DAYS, int(0.9 * window))):
            continue
        rows.append((D, _conformal_quantile(lo[sel]), _conformal_quantile(hi[sel]), int(sel.sum())))
    return pd.DataFrame(rows, columns=["delivery_day", "a_lo", "a_hi", "n"]).set_index("delivery_day")


def apply(p: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    out = p.copy()
    o = off.reindex(out["delivery_day"].dt.normalize())
    out["q10_cal"] = out["q10"].to_numpy() - o["a_lo"].to_numpy()
    out["q90_cal"] = out["q90"].to_numpy() + o["a_hi"].to_numpy()
    return out


def coverage(g: pd.DataFrame, lo: str, hi: str) -> dict:
    g = g.dropna(subset=["actual", lo, hi])
    inside = (g["actual"] >= g[lo]) & (g["actual"] <= g[hi])
    return dict(n=len(g), coverage=float(inside.mean()), below=float((g["actual"] < g[lo]).mean()),
                above=float((g["actual"] > g[hi]).mean()), width=float((g[hi] - g[lo]).mean()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["timestamp_utc", "delivery_day"])
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p = p.set_index("timestamp_utc")
    base = p.dropna(subset=["actual", "q10", "q90"])
    offs = {w: offsets_by_day(base, w) for w in WINDOWS}
    first_cal = max(o.index.min() for o in offs.values())        # common evaluation days for a fair comparison
    rows = []
    for w, off in offs.items():
        cal = apply(base, off)
        cal = cal[cal["delivery_day"] >= first_cal]
        name = "expanding" if w is None else f"{w}d"
        for split in ("cv", "holdout"):
            rows.append(dict(variant=f"conformal {name}", split=split, **coverage(cal[cal["split"] == split], "q10_cal", "q90_cal")))
    comp = pd.DataFrame(rows)
    for split in ("cv", "holdout"):
        g = base[(base["split"] == split) & (base["delivery_day"] >= first_cal)]
        comp = pd.concat([comp, pd.DataFrame([dict(variant="raw quantile band", split=split, **coverage(g, "q10", "q90"))])])
    # window chosen on CV only: coverage closest to 80 %
    cv = comp[(comp.split == "cv") & comp.variant.str.startswith("conformal")].copy()
    cv["gap"] = (cv["coverage"] - 0.8).abs()
    best = cv.loc[cv["gap"].idxmin(), "variant"].replace("conformal ", "")
    window = None if best == "expanding" else int(best.rstrip("d"))
    log.info("chosen window (CV only): %s", best)
    comp["chosen"] = comp["variant"].eq(f"conformal {best}")
    comp.to_csv(C.REPORTS / "quantile_conformal.csv", index=False)
    log.info("\n%s", comp.round(3).to_string(index=False))

    off = offs[window]
    cal = apply(p, off)
    p["q10_cal"], p["q90_cal"] = cal["q10_cal"], cal["q90_cal"]
    p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")

    # by-hour coverage after calibration (CV + holdout), for the dashboard
    g = p.dropna(subset=["actual", "q10_cal", "q90_cal"]).assign(hour=lambda d: d["hour"])
    g.groupby("hour").apply(lambda d: pd.Series({
        "coverage_raw": ((d.actual >= d.q10) & (d.actual <= d.q90)).mean(),
        "coverage_cal": ((d.actual >= d.q10_cal) & (d.actual <= d.q90_cal)).mean(),
        "width_raw": (d.q90 - d.q10).mean(), "width_cal": (d.q90_cal - d.q10_cal).mean()})).to_csv(
        C.REPORTS / "quantile_conformal_by_hour.csv")

    # offsets for the next-day forecast: all days up to yesterday-1 are available at run time
    last = base["delivery_day"].dt.normalize().max()
    sel = base["delivery_day"].dt.normalize() <= last
    if window is not None:
        sel &= base["delivery_day"].dt.normalize() > last - pd.Timedelta(days=window)
    b = base[sel]
    json.dump({"window_days": window, "a_lo": _conformal_quantile((b["q10"] - b["actual"]).to_numpy()),
               "a_hi": _conformal_quantile((b["actual"] - b["q90"]).to_numpy()), "as_of": str(last.date()),
               "rule": "90 % conformal quantile of recent one-sided misses; window chosen on CV months only"},
              open(C.REPORTS / "conformal_offsets.json", "w"), indent=2)
    print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

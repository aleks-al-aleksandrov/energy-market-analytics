"""Data validation for the cleaned hourly frame: fail loudly instead of silently training on broken data.

Errors raise DataValidationError (the pipeline stops); warnings are logged and written to
reports/data_validation.csv. Called from run_all and predict_next_day.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config as C

log = logging.getLogger("validate")
REQUIRED = ["price_BG", "price_RO", "price_GR", "price_HU", "load_forecast", "wind_forecast", "solar_forecast"]
PRICE_RANGE = (-500.0, 4000.0)          # SDAC harmonised min/max clearing price, EUR/MWh


class DataValidationError(RuntimeError):
    pass


@dataclass
class Issue:
    severity: str          # "error" | "warning"
    check: str
    message: str


def check_hourly(h: pd.DataFrame, now: pd.Timestamp | None = None, max_age_days: float = 3.0) -> list[Issue]:
    issues: list[Issue] = []
    add = lambda sev, chk, msg: issues.append(Issue(sev, chk, msg))
    now = now or pd.Timestamp.now(tz="UTC")

    missing = [c for c in REQUIRED if c not in h]
    if missing:
        add("error", "schema", f"required columns missing: {missing}")
        return issues
    if not isinstance(h.index, pd.DatetimeIndex) or h.index.tz is None:
        add("error", "index", "index must be a timezone-aware DatetimeIndex (UTC)")
        return issues
    if h.index.duplicated().any():
        add("error", "index", f"{int(h.index.duplicated().sum())} duplicate timestamps")
    steps = h.index.to_series().diff().dropna().unique()
    if len(steps) and any(s != pd.Timedelta(hours=1) for s in steps):
        add("error", "index", "index is not a regular hourly grid")

    p = h["price_BG"]
    cov = p.notna().mean()
    if cov < 0.98:
        add("error", "coverage", f"price_BG coverage {cov:.1%} < 98 %")
    out = ((p < PRICE_RANGE[0]) | (p > PRICE_RANGE[1])).sum()
    if out:
        add("error", "range", f"{int(out)} price_BG values outside the harmonised range {PRICE_RANGE}")
    if (p.dropna() == 0).mean() > 0.05:
        add("warning", "range", "more than 5 % of prices are exactly 0 (possible fill artefact)")
    for z in ("RO", "GR", "HU"):
        c = h[f"price_{z}"].notna().mean()
        if c < 0.95:
            add("warning", "coverage", f"price_{z} coverage {c:.1%}")
    for c in ("load_forecast", "wind_forecast", "solar_forecast"):
        cv_ = h[c].notna().mean()
        if cv_ < 0.95:
            add("warning", "coverage", f"{c} coverage {cv_:.1%}")

    last_price = p.last_valid_index()
    age = (now - last_price).total_seconds() / 86400 if last_price is not None else np.inf
    if age > max_age_days:
        add("warning", "freshness", f"latest BG price is {age:.1f} days old")
    # flat-lined series (a stuck feed) are a classic silent failure
    tail = p.dropna().tail(72)
    if len(tail) >= 72 and tail.nunique() == 1:
        add("error", "stuck", "last 72 hourly BG prices are identical (stuck feed?)")
    for c in [c for c in h if c.startswith("fuel_")]:
        if h[c].notna().any() and h[c].dropna().tail(24 * 30).nunique() == 1:
            add("warning", "stuck", f"{c} has not changed for 30 days")
    return issues


def run(h: pd.DataFrame, strict: bool = True, **kw) -> list[Issue]:
    issues = check_hourly(h, **kw)
    pd.DataFrame([i.__dict__ for i in issues], columns=["severity", "check", "message"]).to_csv(
        C.REPORTS / "data_validation.csv", index=False)
    for i in issues:
        (log.error if i.severity == "error" else log.warning)("%s: %s", i.check, i.message)
    errors = [i for i in issues if i.severity == "error"]
    if errors and strict:
        raise DataValidationError("; ".join(f"{i.check}: {i.message}" for i in errors))
    log.info("data validation: %d errors, %d warnings", len(errors), len(issues) - len(errors))
    return issues

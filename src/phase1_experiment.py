"""Phase 1: can the point forecast and its uncertainty band be made materially better?

Candidates (fixed BEFORE running; nothing is added after looking at results without saying so):

Point forecast, compared with the production baseline `lgbm_monthly` (LightGBM, monthly refit):
  * lgbm_weekly         same model, refit every 7 delivery days (monthly refits go stale inside the month)
  * lgbm_asinh          variance-stabilised target (asinh of the median/MAD-scaled price), monthly refit
  * lgbm_asinh_weekly   both
  * lear_weekly         LEAR benchmark from the literature: one lasso model per hour on asinh prices, 728-day window
  * ada3                adaptive ensemble of weekly LightGBM, weekly Ridge and weekly level-adjusted LightGBM;
                        weights proportional to 1 / MAE^2 of each member over the previous 14 delivery days
  * ada4                ada3 plus LEAR as a fourth member

ADOPTION RULE (the same one the project has used throughout): a candidate replaces the baseline only if, on the CV
months, its MAE is at least 0.1 EUR/MWh lower AND Diebold-Mariano gives p < 0.05. If several pass, the lowest CV MAE wins.
The Aug-Oct 2026 holdout is REPORTED ONLY: it has been looked at many times and no longer counts as an untouched test. The
clean test from here on is the live track record (forecasts logged before gate closure since November 2026).

Uncertainty band, compared with the production band (conformal offsets from a fixed 180-day window):
  * aci_g002 / aci_g005  adaptive conformal inference: each side's miss-rate target is nudged after every day
                         (alpha <- alpha + gamma * (0.10 - observed miss rate)), so the band widens when it starts missing
  RULE: adopt the ACI variant whose CV coverage is within 1 point of 80 % and whose WORST rolling-30-day CV coverage is the
  highest, provided that worst case beats the production band's. Holdout reported only.

Information set: day-ahead prices of D-1 are published on D-2, so at gate closure for D every error up to and including
D-1 is known. Adaptive weights and ACI use errors up to D-1 only.

Usage: python -m src.phase1_experiment      (reads data/processed/features.parquet and reports/predictions.csv)
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import bootstrap_skill, diebold_mariano, mae, test_months, walk_forward, walk_forward_blocks
from src.features import TARGET, build_features, feature_columns
from src.models import AsinhTarget, LevelAdjusted, make_lgbm, make_ridge

log = logging.getLogger("phase1")
BASELINE = "lgbm_monthly"
MIN_GAIN, ALPHA = 0.1, 0.05
ADA_WINDOW = 14
LEAR_WINDOW, LEAR_REFIT = 728, 7
TARGET_MISS = 0.10            # each side of a 10-90 % band should be exceeded 10 % of the time


# ---------------------------------------------------------------------------------------------- adaptive ensemble
def adaptive_ensemble(p: pd.DataFrame, members: list[str], window: int = ADA_WINDOW, power: float = 2.0) -> pd.Series:
    """Weighted mean of `members`, weights proportional to 1 / MAE^power over the previous `window` delivery days (up to D-1)."""
    d = p.dropna(subset=["actual"] + members)
    day = d["delivery_day"].dt.normalize()
    daily = pd.DataFrame({m: (d["actual"] - d[m]).abs().groupby(day).mean() for m in members}).sort_index()
    recent = daily.rolling(window, min_periods=3).mean().shift(1)          # days up to D-1 only
    w = 1.0 / np.power(recent.clip(lower=1e-6), power)
    w = w.div(w.sum(axis=1), axis=0).fillna(1.0 / len(members))           # equal weights until there is history
    wh = w.reindex(day.to_numpy()).to_numpy()
    return pd.Series((d[members].to_numpy() * wh).sum(axis=1), index=d.index)


# ---------------------------------------------------------------------------------------------- LEAR benchmark
def lear_walk_forward(f: pd.DataFrame, months: list[pd.Period], window_days: int = LEAR_WINDOW,
                      refit_days: int = LEAR_REFIT) -> pd.Series:
    """LEAR benchmark over the complete days of `months` (the production implementation in src/ensemble.py)."""
    from src.ensemble import _daily_wide, lear_forecast
    price = _daily_wide(f, TARGET)
    days = price.index[pd.PeriodIndex(price.index, freq="M").isin(months) & price.notna().all(axis=1).to_numpy()]
    s = lear_forecast(f, days, window_days, refit_days)
    return s[f[TARGET].reindex(s.index).notna()]


# ---------------------------------------------------------------------------------------------- adaptive conformal
def aci_band(p: pd.DataFrame, gamma: float, window: int = 180, min_days: int = 60) -> pd.DataFrame:
    """Adaptive conformal band (the production implementation in src/calibrate.py). Returns q10_aci / q90_aci aligned with p."""
    from src.calibrate import aci_offsets
    off, _ = aci_offsets(p, gamma, window, min_days)
    dn = p["delivery_day"].dt.normalize()
    return pd.DataFrame({"q10_aci": p["q10"] - dn.map(off["a_lo"]), "q90_aci": p["q90"] + dn.map(off["a_hi"])}, index=p.index)


def band_stats(g: pd.DataFrame, lo: str, hi: str) -> dict:
    g = g.dropna(subset=["actual", lo, hi])
    if g.empty:
        return dict(n=0, coverage=np.nan, below=np.nan, above=np.nan, width=np.nan, worst_30d=np.nan)
    inside = (g["actual"] >= g[lo]) & (g["actual"] <= g[hi])
    daily = inside.groupby(g["delivery_day"].dt.normalize()).mean()
    worst = float(daily.rolling(30, min_periods=30).mean().min()) if len(daily) >= 30 else np.nan
    return dict(n=len(g), coverage=float(inside.mean()), below=float((g["actual"] < g[lo]).mean()),
                above=float((g["actual"] > g[hi]).mean()), width=float((g[hi] - g[lo]).mean()), worst_30d=worst)


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h = pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet")
    f = build_features(h)
    cv, hold = test_months(f)
    months = cv + hold
    tuned = C.MODELS_DIR / "lgbm_best_params.json"
    params = json.loads(tuned.read_text()) if tuned.exists() else None
    feats = feature_columns()
    log.info("features %d | CV %s..%s | holdout %s..%s", len(feats), cv[0], cv[-1], hold[0], hold[-1])

    preds = {}
    preds["naive_d1"] = f.loc[pd.PeriodIndex(f["delivery_day"], freq="M").isin(months) & f[TARGET].notna(), "price_lag24"]
    preds[BASELINE] = walk_forward(f, lambda: make_lgbm(params), feats, months)
    log.info("baseline done")
    preds["lgbm_asinh"] = walk_forward(f, lambda: AsinhTarget(make_lgbm(params)), feats, months)
    log.info("asinh monthly done")
    preds["lgbm_weekly"] = walk_forward_blocks(f, lambda: make_lgbm(params), feats, months, 7)
    log.info("weekly done")
    preds["lgbm_asinh_weekly"] = walk_forward_blocks(f, lambda: AsinhTarget(make_lgbm(params)), feats, months, 7)
    log.info("asinh weekly done")
    preds["ridge_weekly"] = walk_forward_blocks(f, lambda: make_ridge(feats), feats, months, 7)
    preds["level_weekly"] = walk_forward_blocks(f, lambda: LevelAdjusted(make_lgbm(params)), feats, months, 7)
    log.info("members done")
    preds["lear_weekly"] = lear_walk_forward(f, months)
    log.info("LEAR done")

    p = pd.DataFrame(preds)
    p.insert(0, "actual", f.loc[p.index, TARGET])
    p.insert(0, "delivery_day", f.loc[p.index, "delivery_day"])
    p["split"] = np.where(pd.PeriodIndex(p["delivery_day"], freq="M").isin(hold), "holdout", "cv")
    p["ada3"] = adaptive_ensemble(p, ["lgbm_weekly", "ridge_weekly", "level_weekly"])
    p["ada4"] = adaptive_ensemble(p, ["lgbm_weekly", "ridge_weekly", "level_weekly", "lear_weekly"])
    p.to_csv(C.REPORTS / "phase1_predictions.csv", index_label="timestamp_utc")

    candidates = ["lgbm_weekly", "lgbm_asinh", "lgbm_asinh_weekly", "lear_weekly", "ada3", "ada4"]
    shown = [BASELINE] + candidates + ["ridge_weekly", "level_weekly"]
    common = p.dropna(subset=["actual", "naive_d1"] + shown)                 # identical hours for every model
    rows = []
    for split in ("cv", "holdout"):
        g = common[common["split"] == split]
        for m in shown:
            r = dict(model=m, split=split, n=len(g), MAE=mae(g["actual"], g[m]),
                     skill_vs_naive=1 - mae(g["actual"], g[m]) / mae(g["actual"], g["naive_d1"]))
            _, r["skill_lo"], r["skill_hi"] = bootstrap_skill(g, m, "naive_d1")
            if m != BASELINE:
                s, pv = diebold_mariano(g["actual"] - g[m], g["actual"] - g[BASELINE], g["delivery_day"])
                r["delta_vs_baseline"], r["DM_stat"], r["DM_p"] = r["MAE"] - mae(g["actual"], g[BASELINE]), s, pv
            rows.append(r)
    met = pd.DataFrame(rows)
    met.to_csv(C.REPORTS / "phase1_metrics.csv", index=False)
    log.info("\n%s", met.round(3).to_string(index=False))

    cvm = met[(met.split == "cv") & met.model.isin(candidates)].copy()
    passed = cvm[(cvm["delta_vs_baseline"] <= -MIN_GAIN) & (cvm["DM_p"] < ALPHA) & (cvm["DM_stat"] < 0)]
    winner = passed.sort_values("MAE").iloc[0]["model"] if len(passed) else BASELINE

    # ---- bands -----------------------------------------------------------------------------------------------------
    band_rows, band_choice = [], "production (180-day conformal)"
    pp = C.REPORTS / "predictions.csv"
    if pp.exists():
        b = pd.read_csv(pp, parse_dates=["delivery_day"])
        b["timestamp_utc"] = pd.to_datetime(b["timestamp_utc"], utc=True)
        b = b.set_index("timestamp_utc")
        variants = {"raw quantiles": ("q10", "q90"), "production (180-day conformal)": ("q10_cal", "q90_cal")}
        for g_ in (0.02, 0.05):
            a = aci_band(b, g_)
            b[f"q10_aci{g_}"], b[f"q90_aci{g_}"] = a["q10_aci"], a["q90_aci"]
            variants[f"aci gamma={g_}"] = (f"q10_aci{g_}", f"q90_aci{g_}")
        need = [c for v in variants.values() for c in v]
        bc = b.dropna(subset=["actual"] + need)                              # identical hours for every band
        for name, (lo, hi) in variants.items():
            for split in ("cv", "holdout"):
                band_rows.append(dict(variant=name, split=split, **band_stats(bc[bc["split"] == split], lo, hi)))
        bands = pd.DataFrame(band_rows)
        bands.to_csv(C.REPORTS / "phase1_bands.csv", index=False)
        log.info("\n%s", bands.round(3).to_string(index=False))
        cvb = bands[bands.split == "cv"].set_index("variant")
        prod_worst = cvb.loc["production (180-day conformal)", "worst_30d"]
        aci = cvb[cvb.index.str.startswith("aci") & ((cvb["coverage"] - 0.8).abs() <= 0.01)]
        if len(aci) and aci["worst_30d"].max() > prod_worst:
            band_choice = aci["worst_30d"].idxmax()

    choice = dict(
        point_forecast=winner, baseline=BASELINE,
        rule=f"CV MAE at least {MIN_GAIN} lower and Diebold-Mariano p < {ALPHA} on the CV months; lowest CV MAE among those that pass",
        candidates_passing=passed["model"].tolist(),
        band=band_choice,
        band_rule="ACI variant with CV coverage within 1 point of 80 % and the best worst-30-day CV coverage, if better than production",
        holdout_note="Aug-Oct 2026 is reported only: it has been inspected before and is not an untouched test any more.",
        cv_months=f"{cv[0]}..{cv[-1]}", holdout_months=f"{hold[0]}..{hold[-1]}")
    (C.REPORTS / "phase1_choice.json").write_text(json.dumps(choice, indent=2))
    log.info("choice: %s", json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()

"""Regime-robust point forecasts, with a protocol fixed in advance.

Problem: trees cannot extrapolate. When the price level leaves the training range (autumn 2026), LightGBM
under-forecasts badly. Candidates (specified here, before looking at results):

  lightgbm          baseline
  lightgbm_level    target = price - yesterday's mean (already in the pipeline)
  lightgbm_recency  recent rows weighted more (half-life 90 days)
  lightgbm_linbase  Ridge first, LightGBM on the Ridge residual (the linear part extrapolates)
  lightgbm_levelshape  linear model of the daily level + LightGBM on the hourly shape around it
                    (ADDED AFTER the first results of the other candidates were seen, so it adds to the multiplicity)
  lightgbm_blend    0.5 * lightgbm + 0.5 * lightgbm_level (fixed weights, nothing fitted)
  lightgbm_guarded  level-adjusted forecast only when yesterday's mean is above the fold's training P98
                    (the model is extrapolating), the baseline otherwise

Selection rule (CV months only): a variant replaces the baseline only if its CV MAE is at least 0.1 EUR/MWh lower
AND the Diebold-Mariano test against the baseline on the CV months gives p < 0.05. Additional lens that does not
rely on the (already used) holdout: the months where the price level shifted by 25 % or more versus the previous
three months ("level-shift episodes"). Usage: python -m src.robust_experiment  (after run_all --plus)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import bootstrap_skill, diebold_mariano, mae, test_months, walk_forward
from src.features import TARGET, feature_columns, load_features
from src.models import LevelShape, LinearBase, Recency, make_lgbm, make_ridge

log = logging.getLogger("robust")
BASE, NAIVE = "lightgbm", "naive_d1"
NEW = ["lightgbm_recency", "lightgbm_linbase", "lightgbm_levelshape", "lightgbm_blend", "lightgbm_guarded"]
ALL_VARIANTS = [BASE, "lightgbm_level", *NEW]
SHIFT = 0.25            # episode threshold: |month mean / mean of previous 3 months - 1|
MIN_GAIN, ALPHA = 0.1, 0.05


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    f = load_features()
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["timestamp_utc", "delivery_day"])
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True)
    p = p.set_index("timestamp_utc")
    cv, hold = test_months(f)
    months = cv + hold
    tuned = C.MODELS_DIR / "lgbm_best_params.json"
    params = json.loads(tuned.read_text()) if tuned.exists() else None
    feats = feature_columns()

    for name, mk in (("lightgbm_recency", lambda: Recency(make_lgbm(params), 90.0)),
                     ("lightgbm_linbase", lambda: LinearBase(make_ridge(feats), make_lgbm(params))),
                     ("lightgbm_levelshape", lambda: LevelShape(make_lgbm(params), feats))):
        if name not in p or p[name].isna().all():
            p[name] = walk_forward(f, mk, feats, months).reindex(p.index)
            log.info("%s done", name)
    p["lightgbm_blend"] = 0.5 * p["lightgbm"] + 0.5 * p["lightgbm_level"]
    dmonth = pd.PeriodIndex(f["delivery_day"], freq="M")
    guard = pd.Series(False, index=p.index)
    d1 = f["price_d1_mean"]
    for m in months:
        thr = d1[(dmonth < m) & f[TARGET].notna()].quantile(0.98)
        rows = p.index[pd.PeriodIndex(p["delivery_day"], freq="M") == m]
        guard.loc[rows] = (d1.reindex(rows) > thr).to_numpy()
    p["lightgbm_guarded"] = np.where(guard, p["lightgbm_level"], p["lightgbm"])
    p["guard_active"] = guard.astype(int)
    p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")

    # ---------------------------------------------------------------- metrics with bootstrap CIs and DM vs baseline
    rows = []
    models = ["naive_w1", "ridge", "mlp", "ensemble", *ALL_VARIANTS]
    for split in ("cv", "holdout"):
        g = p[p["split"] == split].dropna(subset=["actual", NAIVE, *[m for m in ALL_VARIANTS if m in p]])
        for m in [m for m in models if m in g]:
            skill, lo, hi = bootstrap_skill(g, m, NAIVE)
            r = dict(variant=m, split=split, MAE=mae(g["actual"], g[m]), skill=skill, skill_lo=lo, skill_hi=hi)
            if m not in (BASE, NAIVE):
                s_, pv = diebold_mariano(g["actual"] - g[m], g["actual"] - g[BASE], g["delivery_day"])
                r["DM_stat_vs_baseline"], r["DM_p_vs_baseline"] = s_, pv
            rows.append(r)
    met = pd.DataFrame(rows)

    # ---------------------------------------------------------------- level-shift episodes
    monthly = f[TARGET].groupby(dmonth).mean()
    shift = (monthly / monthly.rolling(3).mean().shift(1) - 1)
    episodes = [m for m in months if abs(shift.get(m, 0)) >= SHIFT]
    eps = []
    for m in episodes:
        g = p[pd.PeriodIndex(p["delivery_day"], freq="M") == m].dropna(subset=["actual", *ALL_VARIANTS])
        eps.append(dict(month=str(m), split="holdout" if m in hold else "cv", price_change=float(shift[m]),
                        mean_price=float(monthly[m]), naive_d1=mae(g["actual"], g[NAIVE]),
                        **{v: mae(g["actual"], g[v]) for v in ALL_VARIANTS}))
    ep = pd.DataFrame(eps)
    ep.to_csv(C.REPORTS / "robust_episodes.csv", index=False)

    # ---------------------------------------------------------------- selection on CV only
    cvm = met[met.split == "cv"].set_index("variant")
    base_mae = cvm.loc[BASE, "MAE"]
    cands = cvm.loc[[v for v in ALL_VARIANTS if v != BASE]]
    ok = cands[(cands["MAE"] <= base_mae - MIN_GAIN) & (cands["DM_p_vs_baseline"] < ALPHA)]
    chosen = ok["MAE"].idxmin() if len(ok) else BASE
    met["chosen"] = met["variant"].eq(chosen)
    met.to_csv(C.REPORTS / "robust_metrics.csv", index=False)
    (C.REPORTS / "robust_choice.json").write_text(json.dumps({
        "variant": chosen, "baseline_cv_mae": float(base_mae),
        "rule": f"replace the baseline only if CV MAE is >= {MIN_GAIN} lower and Diebold-Mariano p < {ALPHA} on the CV months",
        "episodes": [str(m) for m in episodes]}, indent=2))
    log.info("chosen on CV: %s", chosen)
    print(met.round(3).to_string(index=False))
    print(ep.round(2).to_string(index=False))

    # confidence intervals for every model in the leaderboard, for the dashboard
    met[met.variant.isin(models)][["variant", "split", "MAE", "skill", "skill_lo", "skill_hi"]].to_csv(
        C.REPORTS / "skill_confidence.csv", index=False)


if __name__ == "__main__":
    main()

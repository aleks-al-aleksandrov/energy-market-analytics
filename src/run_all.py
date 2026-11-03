"""One command, end to end:  python -m src.run_all [--skip-ingest] [--plus]

ingest -> clean -> features -> walk-forward (naive, Ridge, LightGBM) -> evaluate
-> battery simulation -> reports/*.csv.  With --plus also runs the ML-Plus stage
(src.ml_plus: quantiles, classifier, NN, ensemble, SHAP, drift, weather study).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config as C
from src import battery as B
from src.evaluate import error_by, mae, summarise, test_months, walk_forward
from src.features import (FEATURE_GROUPS, OPTIONAL_GROUPS, TARGET, build_features, default_excluded_groups,
                          feature_columns)
from src.models import LevelAdjusted, Naive, make_lgbm, make_ridge

log = logging.getLogger("run_all")


ALT_MODELS = {"capacity": ("lightgbm_nocapacity", "lightgbm_capacity"), "fuels": ("lightgbm_nofuel", "lightgbm_fuels")}


def decide_optional_groups(f: pd.DataFrame, cv: list, params: dict | None) -> dict[str, bool]:
    """Greedy adoption of the optional feature groups (capacity, then fuels).

    A group is adopted only if it lowers LightGBM MAE on the last 12 CV months compared with the set adopted so far.
    The holdout is never used. Writes reports/feature_choice.json.
    """
    choice_path = C.REPORTS / "feature_choice.json"
    choice_path.unlink(missing_ok=True)
    months = cv[-12:]
    optional = [g for g in OPTIONAL_GROUPS if FEATURE_GROUPS.get(g)]

    def score(extra: list[str]) -> float:
        fs = feature_columns(exclude=[g for g in OPTIONAL_GROUPS if g not in extra], respect_choice=False)
        pr = walk_forward(f, lambda: make_lgbm(params), fs, months)
        return mae(f.loc[pr.index, TARGET], pr)

    adopted: list[str] = []
    current = score(adopted)
    out: dict = {"months": f"{months[0]}..{months[-1]}", "cv_mae_base": current,
                 "rule": "greedy: adopt a group iff CV MAE (last 12 CV months) improves on the groups adopted so far"}
    for g in OPTIONAL_GROUPS:
        if g not in optional:
            out[f"use_{g}"] = False
            continue
        with_g = score(adopted + [g])
        out[f"use_{g}"] = bool(with_g < current)
        out[f"cv_mae_without_{g}"], out[f"cv_mae_with_{g}"] = current, with_g
        log.info("optional group %-9s: with %.3f vs without %.3f -> %s", g, with_g, current,
                 "ADOPTED" if with_g < current else "not adopted")
        if with_g < current:
            adopted.append(g)
            current = with_g
    choice_path.write_text(json.dumps(out, indent=2))
    return {g: bool(out.get(f"use_{g}")) for g in OPTIONAL_GROUPS}


def run_base_models(f: pd.DataFrame) -> pd.DataFrame:
    cv, hold = test_months(f)
    months = cv + hold
    tuned = C.MODELS_DIR / "lgbm_best_params.json"
    lgbm_params = json.loads(tuned.read_text()) if tuned.exists() else None
    use = decide_optional_groups(f, cv, lgbm_params)
    feats = feature_columns()
    log.info("features: %d | CV months %s..%s | holdout %s..%s", len(feats), cv[0], cv[-1], hold[0], hold[-1])

    preds = {}
    preds["naive_d1"] = walk_forward(f, lambda: Naive("price_lag24"), feats, months)
    preds["naive_w1"] = walk_forward(f, lambda: Naive("price_lag168"), feats, months)
    preds["ridge"] = walk_forward(f, lambda: make_ridge(feats), feats, months)
    log.info("ridge done")
    preds["lightgbm"] = walk_forward(f, lambda: make_lgbm(lgbm_params), feats, months)
    log.info("lightgbm done")
    # the other side of each optional-group decision, so the comparison table always shows both
    for g, (name_without, name_with) in ALT_MODELS.items():
        if not FEATURE_GROUPS.get(g):
            continue
        if use[g]:
            alt_feats, alt_name = feature_columns(exclude=[g]), name_without
        else:
            others_off = [x for x in OPTIONAL_GROUPS if x != g and x in default_excluded_groups()]
            alt_feats, alt_name = feature_columns(exclude=others_off, respect_choice=False), name_with
        preds[alt_name] = walk_forward(f, lambda: make_lgbm(lgbm_params), alt_feats, months)
    # added after diagnosing the Oct-2026 level jump on the holdout -> its holdout score is not a clean test
    preds["lightgbm_level"] = walk_forward(f, lambda: LevelAdjusted(make_lgbm(lgbm_params)), feats, months)
    log.info("lightgbm_level done")

    p = pd.DataFrame(preds)
    p.insert(0, "actual", f.loc[p.index, TARGET])
    p.insert(0, "delivery_day", f.loc[p.index, "delivery_day"])
    p["hour"] = f.loc[p.index, "hour"]
    p["month"] = pd.PeriodIndex(p["delivery_day"], freq="M").astype(str)
    hold_set = {str(m) for m in hold}
    p["split"] = np.where(p["month"].isin(hold_set), "holdout", "cv")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--plus", action="store_true", help="also run ML-Plus stage")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not a.skip_ingest:
        from src.ingest import ingest_entsoe, ingest_fuels, ingest_weather
        end = C.default_end()
        ingest_entsoe(C.START, end)
        ingest_weather(C.START, end)
        ingest_fuels()

    from src.clean import build_hourly
    hourly, quality, res = build_hourly()
    from src import validate
    validate.run(hourly, max_age_days=7)          # stops the pipeline on broken data (schema, range, stuck feeds)
    hourly.to_parquet(C.DATA_PROCESSED / "hourly.parquet")
    quality.to_csv(C.REPORTS / "data_quality.csv", index=False)
    res.to_csv(C.REPORTS / "resolution_by_month.csv")
    # slim market file for the dashboard (the app never calls the API)
    keep = [c for c in hourly.columns if c.startswith(("price_", "load_", "wind_", "solar_", "gen_", "flow_",
                                                       "net_import", "residual_", "is_negative", "ntc_"))]
    hourly[keep].astype("float32").to_parquet(C.REPORTS / "market_hourly.parquet")
    from src.fuels import export_report
    export_report()

    f = build_features(hourly)
    f.to_parquet(C.DATA_PROCESSED / "features.parquet")
    (C.REPORTS / "feature_groups.json").write_text(json.dumps(FEATURE_GROUPS, indent=2))

    p = run_base_models(f)
    models = ["naive_d1", "naive_w1", "ridge", "lightgbm"] + [c for c in tuple(n for pair in ALT_MODELS.values() for n in pair) if c in p]         + ["lightgbm_level"]
    p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")

    # spike threshold from history before the first test month (no peeking)
    first_test = pd.Timestamp(C.FIRST_TEST_MONTH + "-01")
    spike_thr = float(f.loc[f["delivery_day"] < first_test, TARGET].quantile(C.SPIKE_QUANTILE))
    for split in ("cv", "holdout"):
        sub = p[p["split"] == split].dropna(subset=["actual"] + models)
        s = summarise(sub, models, spike_thr)
        s.to_csv(C.REPORTS / f"metrics_{split}.csv", index=False)
        log.info("%s metrics (best naive = %s)\n%s", split, s.attrs["best_naive"], s.round(3).to_string(index=False))
    sub = p.dropna(subset=["actual"] + models)
    error_by(sub, models, "hour").to_csv(C.REPORTS / "error_by_hour.csv")
    error_by(sub, models, "month").to_csv(C.REPORTS / "error_by_month.csv")

    # battery
    bat = B.Battery()
    df = p.dropna(subset=["actual", "lightgbm", "naive_d1"])
    daily = B.simulate(df, bat, {"perfect_foresight": "actual", "forecast_lightgbm": "lightgbm",
                                 "naive_yesterday": "naive_d1"})
    daily.to_csv(C.REPORTS / "battery_daily.csv", index=False)
    summ = B.summarise(daily, bat)
    summ.to_csv(C.REPORTS / "battery_results.csv", index_label="strategy")
    log.info("battery (1 MW / 2 MWh / 90%%)\n%s", summ.round(1).to_string())

    if a.plus:
        from src import ml_plus
        ml_plus.main(f, p)
        from src import battery_study, calibrate, reserve, robust_experiment
        calibrate.main()          # conformal recalibration of the 10-90 % band (adds q10_cal / q90_cal to predictions.csv)
        robust_experiment.main()  # regime-robust variants, bootstrap skill intervals, level-shift episodes
        reserve.main()            # balancing reserve prices and volumes for the dashboard
        battery_study.main()      # degradation sensitivity, confidence filter, reserve stacking
    log.info("done - reports in %s", C.REPORTS)


if __name__ == "__main__":
    main()


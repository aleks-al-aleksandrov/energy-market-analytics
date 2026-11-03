"""ML-Plus stage: NN, ensemble, quantiles, classifier, SHAP, ablation, DM, drift, weather study.

Called by `python -m src.run_all --plus` (or `python -m src.ml_plus` after run_all).
All outputs go to reports/; every model/configuration is logged as an MLflow run.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src import config as C
from src import battery as B
from src import monitor as M
from src import weather_study as W
from src.evaluate import error_by, mae, pinball, summarise, test_months, walk_forward
from src.features import FEATURE_GROUPS, TARGET, default_excluded_groups, feature_columns
from src.models import LevelAdjusted, TorchMLP, make_lgbm, make_lgbm_classifier, make_ridge
from src.tracking import export_comparison_figure, mlflow_run

log = logging.getLogger("ml_plus")
QUANTILES = (0.1, 0.5, 0.9)
# Business cost assumption for the event classifier: a missed spike (unhedged
# exposure / missed battery discharge) costs 5x a false alarm (unnecessary hedge).
COST_FN, COST_FP = 5.0, 1.0


def _lgbm_params():
    p = C.MODELS_DIR / "lgbm_best_params.json"
    return json.loads(p.read_text()) if p.exists() else None


def ensemble_weights(pred: pd.DataFrame, models: list[str], step: float = 0.05) -> dict:
    """Grid search on the simplex minimising MAE (on CV predictions only)."""
    best, best_w = np.inf, None
    grid = np.arange(0, 1 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1 + 1e-9:
                continue
            w = np.array([w1, w2, 1 - w1 - w2])
            e = mae(pred["actual"], pred[models].to_numpy() @ w)
            if e < best:
                best, best_w = e, w
    return dict(zip(models, best_w.round(3)))


def classifier_walk_forward(f: pd.DataFrame, feats: list[str], months, kind: str) -> pd.DataFrame:
    """Event probability per hour; thresholds come from each fold's own training data.

    spike     : price above the P95 of the training prices (a fixed EUR level)
    spike_rel : price more than the training-P95 *excess over yesterday's average price* (level-relative)
    negative  : price below zero
    cheap     : price below 5 EUR/MWh (about 10x more events than negative prices; what a battery cares about)
    """
    dmonth = pd.PeriodIndex(f["delivery_day"], freq="M")
    valid = f[TARGET].notna()
    dev = f[TARGET] - f["price_d1_mean"]                     # excess over the D-1 daily mean (known at gate closure)
    out = []
    for m in months:
        ok = valid & dev.notna() if kind == "spike_rel" else valid
        tr, te = ok & (dmonth < m), ok & (dmonth == m)
        if tr.sum() < 180 * 24 or te.sum() == 0:
            continue
        if kind == "spike":
            thr = f.loc[tr, TARGET].quantile(C.SPIKE_QUANTILE)
            y = (f[TARGET] > thr).astype(int)
        elif kind == "spike_rel":
            thr = dev[tr].quantile(C.SPIKE_QUANTILE)
            y = (dev > thr).astype(int)
        elif kind == "cheap":
            thr = 5.0
            y = (f[TARGET] < thr).astype(int)
        else:
            thr = 0.0
            y = (f[TARGET] < 0).astype(int)
        if y[tr].sum() < 20:
            continue
        clf = make_lgbm_classifier()
        clf.fit(f.loc[tr, feats], y[tr])
        out.append(pd.DataFrame({"prob": clf.predict_proba(f.loc[te, feats])[:, 1],
                                 "y": y[te], "threshold_eur": thr}, index=f.index[te]))
    return pd.concat(out) if out else pd.DataFrame(columns=["prob", "y", "threshold_eur"])


def pick_threshold(y, prob) -> tuple[float, pd.DataFrame]:
    grid = np.linspace(0.02, 0.98, 49)
    rows = []
    for t in grid:
        yhat = prob >= t
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        tp = int(((yhat == 1) & (y == 1)).sum())
        rows.append(dict(threshold=t, cost=COST_FN * fn + COST_FP * fp, tp=tp, fp=fp, fn=fn,
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
    t = pd.DataFrame(rows)
    return float(t.loc[t["cost"].idxmin(), "threshold"]), t


def fit_spike_model(f: pd.DataFrame, feats: list[str]):
    """Level-relative spike model on all data: event = price exceeds yesterday's daily mean by more than the P95 excess."""
    dev = f[TARGET] - f["price_d1_mean"]
    ok = f[TARGET].notna() & dev.notna()
    thr = float(dev[ok].quantile(C.SPIKE_QUANTILE))
    return make_lgbm_classifier().fit(f.loc[ok, feats], (dev[ok] > thr).astype(int)), thr


def run_classifiers(f: pd.DataFrame, p: pd.DataFrame, feats: list[str], months, hold_set: set,
                    kinds: tuple[str, ...] = ("spike", "spike_rel", "cheap", "negative")) -> pd.DataFrame:
    """Walk-forward event classifiers; thresholds chosen on CV months by an explicit cost, writes reports/classifier_*."""
    clf_rows = []
    for kind in kinds:
        cp = classifier_walk_forward(f, feats, months, kind)
        if cp.empty:
            continue
        cp["split"] = np.where(pd.PeriodIndex(f.loc[cp.index, "delivery_day"], freq="M").astype(str).isin(hold_set),
                               "holdout", "cv")
        cvc, hoc = cp[cp.split == "cv"], cp[cp.split == "holdout"]
        thr, cost_table = pick_threshold(cvc["y"], cvc["prob"])  # chosen on CV only
        cost_table.to_csv(C.REPORTS / f"classifier_{kind}_threshold_costs.csv", index=False)
        prec, rec, _ = precision_recall_curve(cvc["y"], cvc["prob"])
        pd.DataFrame({"precision": prec, "recall": rec}).iloc[::max(len(prec) // 400, 1)].to_csv(
            C.REPORTS / f"classifier_{kind}_pr_curve.csv", index=False)
        for split, g in (("cv", cvc), ("holdout", hoc)):
            if g["y"].sum() == 0:
                clf_rows.append(dict(kind=kind, split=split, events=0, base_rate=0.0))
                continue
            yhat = g["prob"] >= thr
            tp = int((yhat & (g.y == 1)).sum())
            clf_rows.append(dict(kind=kind, split=split, events=int(g.y.sum()), base_rate=float(g.y.mean()),
                                 pr_auc=float(average_precision_score(g.y, g.prob)), threshold=thr,
                                 precision=tp / max(int(yhat.sum()), 1), recall=tp / max(int(g.y.sum()), 1)))
        p[f"prob_{kind}"] = cp["prob"].reindex(p.index)
    cm = pd.DataFrame(clf_rows)
    cm.to_csv(C.REPORTS / "classifier_metrics.csv", index=False)
    log.info("classifiers\n%s", cm.round(3).to_string(index=False))
    return cm


def main(f: pd.DataFrame, p: pd.DataFrame) -> None:
    cv, hold = test_months(f)
    months = cv + hold
    feats = feature_columns()
    params = _lgbm_params()
    h = pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet")
    hold_set = {str(m) for m in hold}
    first_test = pd.Timestamp(C.FIRST_TEST_MONTH + "-01")
    spike_thr = float(f.loc[f["delivery_day"] < first_test, TARGET].quantile(C.SPIKE_QUANTILE))

    # level-adjusted LightGBM for quantiles / final models only if it wins on CV months (holdout not used)
    cvb = p[p["split"] == "cv"].dropna(subset=["actual", "lightgbm", "lightgbm_level"])
    use_level = mae(cvb["actual"], cvb["lightgbm_level"]) < mae(cvb["actual"], cvb["lightgbm"])
    wrap = (lambda m: LevelAdjusted(m)) if use_level else (lambda m: m)
    log.info("level-adjusted target for quantile/final models: %s", use_level)
    (C.REPORTS / "model_choice.json").write_text(json.dumps({"level_adjusted_final_model": bool(use_level)}))

    # ------------------------------------------------------------------ NN
    log.info("neural network walk-forward ...")
    p["mlp"] = walk_forward(f, lambda: TorchMLP(feats), feats, months).reindex(p.index)

    # ------------------------------------------------------------------ ensemble (weights from CV only)
    ens_models = ["lightgbm", "ridge", "mlp"]
    cvp = p[p["split"] == "cv"].dropna(subset=["actual"] + ens_models)
    w = ensemble_weights(cvp, ens_models)
    p["ensemble"] = p[ens_models].to_numpy() @ np.array([w[m] for m in ens_models])
    (C.REPORTS / "ensemble_weights.json").write_text(json.dumps(w, indent=2))
    log.info("ensemble weights (fitted on CV months): %s", w)

    # ------------------------------------------------------------------ quantiles
    log.info("quantile LightGBM ...")
    for q in QUANTILES:
        p[f"q{int(q * 100)}"] = walk_forward(
            f, lambda q=q: wrap(make_lgbm(params, objective="quantile", alpha=q)), feats, months).reindex(p.index)
    qp = p.dropna(subset=["actual", "q10", "q50", "q90"])
    qrows = []
    for split, g in [("cv", qp[qp.split == "cv"]), ("holdout", qp[qp.split == "holdout"])]:
        r = {"split": split}
        for q in QUANTILES:
            col = f"q{int(q * 100)}"
            r[f"pinball_{col}"] = pinball(g["actual"], g[col], q)
            r[f"share_below_{col}"] = float((g["actual"] <= g[col]).mean())
        r["coverage_10_90"] = float(((g["actual"] >= g["q10"]) & (g["actual"] <= g["q90"])).mean())
        r["mean_band_width"] = float((g["q90"] - g["q10"]).mean())
        qrows.append(r)
    pd.DataFrame(qrows).to_csv(C.REPORTS / "quantile_metrics.csv", index=False)
    qp.groupby("hour").apply(lambda g: pd.Series({
        "coverage_10_90": ((g.actual >= g.q10) & (g.actual <= g.q90)).mean(),
        "band_width": (g.q90 - g.q10).mean()})).to_csv(C.REPORTS / "quantile_by_hour.csv")

    # ------------------------------------------------------------------ full comparison + DM
    models = ["naive_d1", "naive_w1", "ridge", "lightgbm"] + [c for c in ("lightgbm_nocapacity", "lightgbm_capacity", "lightgbm_nofuel", "lightgbm_fuels") if c in p]         + ["lightgbm_level", "mlp", "ensemble"]
    for split in ("cv", "holdout"):
        sub = p[p["split"] == split].dropna(subset=["actual"] + models)
        s = summarise(sub, models, spike_thr)
        s.to_csv(C.REPORTS / f"metrics_{split}.csv", index=False)
        log.info("%s\n%s", split, s.round(3).to_string(index=False))
    sub = p.dropna(subset=["actual"] + models)
    error_by(sub, models, "hour").to_csv(C.REPORTS / "error_by_hour.csv")
    error_by(sub, models, "month").to_csv(C.REPORTS / "error_by_month.csv")
    p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")

    mc = pd.read_csv(C.REPORTS / "metrics_cv.csv").set_index("model")
    mh = pd.read_csv(C.REPORTS / "metrics_holdout.csv").set_index("model")
    for m in models:
        with mlflow_run(m, {"features": len(feats), "cv_months": f"{cv[0]}..{cv[-1]}",
                            "holdout": f"{hold[0]}..{hold[-1]}",
                            **({"lgbm_params": params} if m == "lightgbm" and params else {}),
                            **({"weights": w} if m == "ensemble" else {})},
                        {"cv_mae": mc.loc[m, "MAE"], "cv_rmse": mc.loc[m, "RMSE"],
                         "cv_mae_fold_std": mc.loc[m, "MAE_fold_std"],
                         "holdout_mae": mh.loc[m, "MAE"], "holdout_rmse": mh.loc[m, "RMSE"],
                         "holdout_skill": mh.loc[m, "skill_vs_best_naive"]}):
            pass

    # ------------------------------------------------------------------ classifiers
    log.info("spike / negative-price classifiers ...")
    cm = run_classifiers(f, p, feats, months, hold_set)
    p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")

    # ------------------------------------------------------------------ final models (train on all pre-holdout + holdout)
    log.info("final models, SHAP, PDP ...")
    dmonth = pd.PeriodIndex(f["delivery_day"], freq="M")
    valid = f[TARGET].notna()
    pre_hold = valid & (dmonth < hold[0])
    lgb_pre = make_lgbm(params).fit(f.loc[pre_hold, feats], f.loc[pre_hold, TARGET])
    hmask = valid & dmonth.astype(str).isin(hold_set)
    from src.explain import partial_dependence, shap_analysis
    shap_analysis(lgb_pre, f.loc[hmask, feats], f.loc[hmask, TARGET])
    if "residual_load_forecast" in feats:
        partial_dependence(lgb_pre, f.loc[pre_hold, feats], "residual_load_forecast").to_csv(
            C.REPORTS / "pdp_residual_load.csv", index=False)

    allv = valid
    final = {"point": wrap(make_lgbm(params)).fit(f.loc[allv, feats], f.loc[allv, TARGET])}
    for q in (0.1, 0.9):
        final[f"q{int(q * 100)}"] = wrap(make_lgbm(params, objective="quantile", alpha=q)).fit(
            f.loc[allv, feats], f.loc[allv, TARGET])
    thr_spike = float(f.loc[allv, TARGET].quantile(C.SPIKE_QUANTILE))          # fixed-EUR level, kept for reference
    final["spike"], thr_excess = fit_spike_model(f, feats)
    final["level"] = LevelAdjusted(make_lgbm(params)).fit(f.loc[allv, feats], f.loc[allv, TARGET])   # cross-check forecast
    joblib.dump({"models": final, "features": feats, "spike_threshold_eur": thr_spike,
                 "spike_excess_threshold_eur": thr_excess,
                 "trained_until": str(f.loc[allv, "delivery_day"].max().date())},
                C.MODELS_DIR / "final_models.joblib")

    # ------------------------------------------------------------------ ablation (last 12 CV months)
    log.info("feature-group ablation ...")
    abl_months = cv[-12:]
    base = walk_forward(f, lambda: make_lgbm(params), feats, abl_months)
    base_mae = mae(f.loc[base.index, TARGET], base)
    rows = [dict(dropped="(none - full model)", n_features=len(feats), MAE=base_mae, delta_MAE=0.0)]
    for g in FEATURE_GROUPS:
        if g in default_excluded_groups():
            continue   # group switched off by the CV adoption rule: nothing to ablate
        fs = feature_columns(exclude=[g])
        if not FEATURE_GROUPS[g] or not fs:
            continue
        pr = walk_forward(f, lambda: make_lgbm(params), fs, abl_months)
        e = mae(f.loc[pr.index, TARGET], pr)
        rows.append(dict(dropped=g, n_features=len(fs), MAE=e, delta_MAE=e - base_mae))
    abl = pd.DataFrame(rows)
    abl.to_csv(C.REPORTS / "ablation.csv", index=False)
    log.info("ablation\n%s", abl.round(3).to_string(index=False))

    # ------------------------------------------------------------------ monitoring
    roll = M.rolling_mae(p.dropna(subset=["actual", "lightgbm", "naive_d1"]), ["lightgbm", "naive_d1", "ensemble"])
    cv_level = float(mc.loc["lightgbm", "MAE"])
    roll["retrain_alert"] = M.retrain_alert(roll["lightgbm"], cv_level).astype(int)
    roll["cv_level"] = cv_level
    roll.to_csv(C.REPORTS / "monitor_rolling_mae.csv")
    train_ref = valid & (dmonth < pd.Period(C.FIRST_TEST_MONTH, "M"))
    M.psi_table(f, train_ref, hmask).to_csv(C.REPORTS / "monitor_psi.csv", index=False)
    # PSI per month vs. reference
    psi_m = []
    for m in months:
        cur = valid & (dmonth == m)
        t = M.psi_table(f, train_ref, cur)
        t["month"] = str(m)
        psi_m.append(t)
    pd.concat(psi_m).to_csv(C.REPORTS / "monitor_psi_by_month.csv", index=False)

    # ------------------------------------------------------------------ battery with all forecasts + sensitivity
    bat = B.Battery()
    bdf = p.dropna(subset=["actual", "lightgbm", "naive_d1", "ensemble"])
    daily = B.simulate(bdf, bat, {"perfect_foresight": "actual", "forecast_lightgbm": "lightgbm",
                                  "forecast_ensemble": "ensemble", "naive_yesterday": "naive_d1"})
    daily.to_csv(C.REPORTS / "battery_daily.csv", index=False)
    B.summarise(daily, bat).to_csv(C.REPORTS / "battery_results.csv", index_label="strategy")
    sens = []
    for dur in (1, 2, 4):
        for eff in (0.8, 0.85, 0.9, 0.95):
            b = B.Battery(duration_h=dur, efficiency=eff)
            s = B.summarise(B.simulate(bdf, b, {"perfect_foresight": "actual", "forecast_lightgbm": "lightgbm",
                                                 "naive_yesterday": "naive_d1"}), b)
            for strat, r in s.iterrows():
                sens.append(dict(duration_h=dur, efficiency=eff, strategy=strat, **r.to_dict()))
    pd.DataFrame(sens).to_csv(C.REPORTS / "battery_sensitivity.csv", index=False)

    # ------------------------------------------------------------------ weather study
    log.info("weather study ...")
    werr, werr_h = W.weather_error_tables(h)
    werr.to_csv(C.REPORTS / "weather_fc_error.csv", index=False)
    werr_h.to_csv(C.REPORTS / "weather_fc_error_by_hour.csv")
    g = h[h["weather_fc_is_proxy"] == 0]
    sample_cols = [c for c in g.columns if c.startswith(("wfc_", "wact_"))]
    g[sample_cols].dropna().sample(min(4000, len(g)), random_state=C.SEED).to_csv(
        C.REPORTS / "weather_fc_vs_actual_sample.csv", index_label="timestamp_utc")
    res_sum, res_tab, res_err = W.res_forecast_error(h)
    res_sum.to_csv(C.REPORTS / "res_forecast_error_summary.csv", index=False)
    res_tab.to_csv(C.REPORTS / "res_forecast_error_by_hour_month.csv")
    # residual-load forecast error vs price forecast error
    link = pd.DataFrame(index=p.index)
    if "residual_load_actual" in h:
        link["res_load_err_mw"] = (h["residual_load_actual"] - h["residual_load_forecast"]).reindex(p.index)
    link["price_err"] = p["actual"] - p["lightgbm"]
    link["delivery_day"] = p["delivery_day"]
    link = link.dropna()
    link.to_csv(C.REPORTS / "res_error_vs_price_error.csv", index_label="timestamp_utc")
    worst = link.assign(abs_res=link.res_load_err_mw.abs()).groupby("delivery_day").agg(
        mean_abs_res_err_mw=("abs_res", "mean"), mean_price_err=("price_err", "mean"),
        mae_price=("price_err", lambda s: s.abs().mean())).sort_values("mean_abs_res_err_mw", ascending=False).head(10)
    worst.to_csv(C.REPORTS / "worst_res_forecast_days.csv")

    # only months where every weather input is a genuine D-1 forecast (no stitched proxy)
    proxy_share = f["weather_fc_is_proxy"].groupby(dmonth).mean()
    genuine_months = [m for m in months if proxy_share.get(m, 1.0) < 0.01]
    summ, noise, wpred, noisy = W.experiments(f, h, genuine_months, params)
    summ.to_csv(C.REPORTS / "weather_experiment.csv", index=False)
    noise.to_csv(C.REPORTS / "weather_noise_injection.csv", index=False)
    W.battery_under_weather(wpred, noisy).to_csv(C.REPORTS / "battery_weather_scenarios.csv", index_label="strategy")
    log.info("weather experiment\n%s\nnoise\n%s", summ.round(3), noise.round(3))
    with mlflow_run("weather_experiment", {"months": f"{genuine_months[0]}..{genuine_months[-1]}"},
                    {f"mae_{r.setup}": r.MAE for r in summ.itertuples()}):
        pass
    log_final_bundle(mh)


def log_final_bundle(mh: pd.DataFrame | None = None) -> None:
    """MLflow run holding the deployable model bundle, figures and metric tables."""
    mh = mh if mh is not None else pd.read_csv(C.REPORTS / "metrics_holdout.csv").set_index("model")
    choice = json.loads((C.REPORTS / "model_choice.json").read_text()) if (C.REPORTS / "model_choice.json").exists() else {}
    fig = export_comparison_figure()
    arts = [C.MODELS_DIR / "final_models.joblib", C.MODELS_DIR / "lgbm_best_params.json", C.FIGURES / "shap_summary.png",
            C.REPORTS / "metrics_cv.csv", C.REPORTS / "metrics_holdout.csv", C.REPORTS / "quantile_metrics.csv",
            C.REPORTS / "classifier_metrics.csv"] + ([fig] if fig else [])
    with mlflow_run("final_model_bundle", {"lgbm_params": _lgbm_params(), **choice},
                    {"holdout_mae_lightgbm": mh.loc["lightgbm", "MAE"]}, artifacts=arts):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys

    from src.features import load_features

    f = load_features()
    p = pd.read_csv(C.REPORTS / "predictions.csv", index_col=0, parse_dates=["timestamp_utc", "delivery_day"])
    p.index = pd.to_datetime(p.index, utc=True)
    if "--add-level-model" in sys.argv:        # add the level-adjusted cross-check model to the saved bundle
        bundle = joblib.load(C.MODELS_DIR / "final_models.joblib")
        ok = f[TARGET].notna()
        params_ = _lgbm_params()
        bundle["models"]["level"] = LevelAdjusted(make_lgbm(params_)).fit(f.loc[ok, bundle["features"]], f.loc[ok, TARGET])
        joblib.dump(bundle, C.MODELS_DIR / "final_models.joblib")
        print("level-adjusted model added to the bundle")
    elif "--refit-spike" in sys.argv:            # swap the spike model inside the saved bundle for the level-relative one
        bundle = joblib.load(C.MODELS_DIR / "final_models.joblib")
        bundle["models"]["spike"], bundle["spike_excess_threshold_eur"] = fit_spike_model(f, bundle["features"])
        joblib.dump(bundle, C.MODELS_DIR / "final_models.joblib")
        print("spike model refit; P95 excess over yesterday's mean =", round(bundle["spike_excess_threshold_eur"], 1), "EUR/MWh")
    elif "--classifiers-only" in sys.argv:       # refresh just the event classifiers (about 3 minutes)
        cv_, hold_ = test_months(f)
        run_classifiers(f, p, feature_columns(), cv_ + hold_, {str(m) for m in hold_})
        p.to_csv(C.REPORTS / "predictions.csv", index_label="timestamp_utc")
    else:
        main(f, p)


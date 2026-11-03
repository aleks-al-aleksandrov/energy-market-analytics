"""Optuna tuning of LightGBM with an expanding-window, time-based CV objective.

Only the CV period is used - the final holdout months are never touched.
Usage: python -m src.tune [--trials 50] [--folds 4]
Writes models/lgbm_best_params.json (picked up automatically by run_all) and logs to MLflow.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import mae, test_months, walk_forward
from src.features import TARGET, feature_columns, load_features
from src.models import make_lgbm
from src.tracking import mlflow_run

log = logging.getLogger("tune")


def tune(f: pd.DataFrame, n_trials: int = 50, n_folds: int = 4) -> dict:
    import optuna

    cv, _ = test_months(f)
    folds = cv[-n_folds:]  # the most recent CV months, all strictly before the holdout
    feats = feature_columns()

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        pred = walk_forward(f, lambda: make_lgbm(params), feats, folds)
        return mae(f.loc[pred.index, TARGET], pred)

    sampler = optuna.samplers.TPESampler(seed=C.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lgbm_bg_da")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    baseline = walk_forward(f, lambda: make_lgbm(), feats, folds)
    base_mae = mae(f.loc[baseline.index, TARGET], baseline)
    log.info("default MAE %.3f -> tuned MAE %.3f", base_mae, study.best_value)

    best = study.best_params
    (C.MODELS_DIR / "lgbm_best_params.json").write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(C.REPORTS / "optuna_trials.csv", index=False)
    with mlflow_run("optuna_lgbm", {"n_trials": n_trials, "folds": [str(m) for m in folds], **best},
                    {"tuned_cv_mae": study.best_value, "default_cv_mae": base_mae}):
        pass
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(tune(load_features(), a.trials, a.folds))


if __name__ == "__main__":
    main()


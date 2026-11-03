"""Do fuel and carbon prices help? Fixed set of encodings, CV months only (the holdout is never used).

Variants: none | monthly gas + Brent + carbon | levels | scale-free changes | daily Brent/carbon | monthly gas | daily TTF variants | all,
each with plain LightGBM and the level-adjusted LightGBM. Writes reports/fuel_experiment.csv.
Usage: python -m src.fuel_experiment
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config as C
from src.evaluate import mae, test_months, walk_forward
from src.features import FEATURE_GROUPS, TARGET, feature_columns, load_features
from src.models import LevelAdjusted, make_lgbm

log = logging.getLogger("fuel_experiment")

OLD_ALL = ["fuel_brent_usd", "fuel_brent_chg20", "fuel_carbon_idx", "fuel_carbon_chg20", "fuel_gas_eur_mwh",
           "fuel_gas_rel12", "fuel_gas_chg3", "fuel_gas_cost_ccgt"]
TTF = ["fuel_ttf_eur_mwh", "fuel_ttf_chg20", "fuel_ttf_vs_gasm"]
VARIANTS = {
    "none": [],
    "monthly gas + Brent + carbon (8)": OLD_ALL,
    "levels only": ["fuel_brent_usd", "fuel_carbon_idx", "fuel_gas_eur_mwh", "fuel_gas_cost_ccgt"],
    "scale-free changes only": ["fuel_brent_chg20", "fuel_carbon_chg20", "fuel_gas_rel12", "fuel_gas_chg3"],
    "daily only (Brent + carbon)": ["fuel_brent_usd", "fuel_brent_chg20", "fuel_carbon_idx", "fuel_carbon_chg20"],
    "monthly gas only": ["fuel_gas_eur_mwh", "fuel_gas_rel12", "fuel_gas_chg3", "fuel_gas_cost_ccgt"],
    "daily TTF (3)": TTF,
    "daily TTF scale-free (2)": ["fuel_ttf_chg20", "fuel_ttf_vs_gasm"],
    "daily TTF + Brent + carbon": TTF + ["fuel_brent_chg20", "fuel_carbon_chg20"],
    "all fuel features (11)": None,   # filled in main()
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    f = load_features()
    cv, _ = test_months(f)
    months = cv[-12:]
    tuned = C.MODELS_DIR / "lgbm_best_params.json"
    params = json.loads(tuned.read_text()) if tuned.exists() else None
    base = feature_columns(exclude=["fuels"], respect_choice=False)
    variants = dict(VARIANTS)
    variants["all fuel features (11)"] = FEATURE_GROUPS["fuels"]
    rows = []
    for name, extra in variants.items():
        feats = base + [c for c in extra if c in f]
        for model, mk in (("LightGBM", lambda: make_lgbm(params)), ("LightGBM · level-adj.", lambda: LevelAdjusted(make_lgbm(params)))):
            p = walk_forward(f, mk, feats, months)
            y = f.loc[p.index, TARGET]
            per_month = (y - p).abs().groupby(f.loc[p.index, "delivery_day"].dt.to_period("M")).mean()
            rows.append(dict(variant=name, model=model, n_features=len(feats), MAE=mae(y, p), fold_sd=per_month.std()))
            log.info("%-30s %-22s MAE %.3f", name, model, rows[-1]["MAE"])
    out = pd.DataFrame(rows)
    ref = out[out.variant == "none"].set_index("model")["MAE"]
    out["delta_vs_none"] = out["MAE"] - out["model"].map(ref)
    out["months"] = f"{months[0]}..{months[-1]}"
    out.to_csv(C.REPORTS / "fuel_experiment.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

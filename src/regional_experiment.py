"""Do regional drivers improve the forecast?  python -m src.regional_experiment

Group "regional" (all known before gate closure): German day-ahead price of D-1 (level of the continental market), and the
day-ahead residual-load forecast and renewable share of Romania, Greece, Hungary and Germany for the delivery day.

RULE (the one used for capacity and fuels, fixed in advance): adopt the group only if it lowers LightGBM MAE on the last 12 CV
months. Diebold-Mariano is reported for information. The Aug-Oct 2026 holdout is reported only (no longer an untouched test).
Writes reports/regional_experiment.csv and reports/regional_choice.json. It does not change the production feature set:
adoption happens in run_all's greedy rule on the next retrain.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config as C
from src.clean import build_hourly
from src.evaluate import diebold_mariano, mae, test_months, walk_forward
from src.features import FEATURE_GROUPS, HELD_GROUPS, OPTIONAL_GROUPS, TARGET, build_features, default_excluded_groups, feature_columns
from src.models import make_lgbm

log = logging.getLogger("regional")


def main(end: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    old = pd.read_parquet(C.DATA_PROCESSED / "hourly.parquet")
    end_ts = pd.Timestamp(end, tz="UTC") if end else old.index.max() + pd.Timedelta(hours=1)     # same period as the last run
    h, _, _ = build_hourly(end=end_ts)
    f = build_features(h)
    reg = FEATURE_GROUPS.get("regional", [])
    if not reg:
        raise SystemExit("no regional data: run the ingest for prices/DE_LU and the neighbours' forecasts first")
    cover = f[reg].notna().mean().round(3)
    log.info("regional features (%d), share of hours with data:\n%s", len(reg), cover.to_string())

    cv, hold = test_months(f)
    params_p = C.MODELS_DIR / "lgbm_best_params.json"
    params = json.loads(params_p.read_text()) if params_p.exists() else None
    off = [g for g in default_excluded_groups() if g != "regional"]             # capacity / fuels stay as already decided
    base = feature_columns(exclude=off + ["regional"], respect_choice=False)
    plus = feature_columns(exclude=off, respect_choice=False, include_held=True)
    months = cv[-12:] + hold
    pb = walk_forward(f, lambda: make_lgbm(params), base, months)
    pr = walk_forward(f, lambda: make_lgbm(params), plus, months)
    d = pd.DataFrame({"actual": f.loc[pb.index, TARGET], "base": pb, "regional": pr, "delivery_day": f.loc[pb.index, "delivery_day"]}).dropna()
    is_hold = pd.PeriodIndex(d["delivery_day"], freq="M").isin(hold)
    rows = []
    for split, g in (("cv_last12", d[~is_hold]), ("holdout", d[is_hold])):
        s, pv = diebold_mariano(g["actual"] - g["regional"], g["actual"] - g["base"], g["delivery_day"])
        rows.append(dict(split=split, n=len(g), mae_without=mae(g["actual"], g["base"]), mae_with=mae(g["actual"], g["regional"]),
                         delta=mae(g["actual"], g["regional"]) - mae(g["actual"], g["base"]), DM_stat=s, DM_p=pv))
    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "regional_experiment.csv", index=False)
    cvr = out.iloc[0]
    passes_rule = bool(cvr["mae_with"] < cvr["mae_without"])
    hor = out.iloc[1]
    # The rule is CV-only, but the group is in features.HELD_GROUPS: a large loss in the surge months overrides an automatic yes.
    worse_in_surge = bool(hor["mae_with"] > hor["mae_without"] * 1.05)
    adopt = passes_rule and not worse_in_surge
    (C.REPORTS / "regional_choice.json").write_text(json.dumps(dict(
        adopt=adopt, passes_cv_rule=passes_rule, held_back=bool(passes_rule and not adopt),
        holdout_mae_without=float(hor["mae_without"]), holdout_mae_with=float(hor["mae_with"]),
        reason=("passes the CV rule but is more than 5 % worse in the Aug-Oct 2026 surge; held back pending a test inside the ensemble"
                if passes_rule and not adopt else ""),
        features=reg, n_features=len(reg), coverage_min=float(cover.min()),
        cv_months=f"{cv[-12]}..{cv[-1]}", cv_mae_without=float(cvr["mae_without"]), cv_mae_with=float(cvr["mae_with"]),
        dm_p=float(cvr["DM_p"]), rule="adopt iff LightGBM MAE on the last 12 CV months is lower with the group",
        note="optional groups: " + ", ".join(OPTIONAL_GROUPS) + "; held groups: " + ", ".join(HELD_GROUPS)), indent=2))
    print(out.round(3).to_string(index=False))
    print("passes CV rule:", passes_rule, "| adopted:", adopt)


if __name__ == "__main__":
    main()

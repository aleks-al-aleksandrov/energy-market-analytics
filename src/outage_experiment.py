"""How much would outage knowledge be worth? CV months only (the holdout is never used).

Variants (all on top of the currently adopted feature set):
  none            baseline
  oracle outages  final document windows -> information published AFTER the day (not usable, an upper bound)
  strict outages  only documents whose latest revision predates gate closure (leak-free but biased: old days are
                  undercounted because their documents have since been revised)
The gap oracle - none says whether collecting point-in-time outage snapshots (src/outages.py) is worth it.
Usage: python -m src.outage_experiment
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config as C
from src import outages as O
from src.evaluate import mae, test_months, walk_forward
from src.features import TARGET, feature_columns, load_features
from src.models import make_lgbm

log = logging.getLogger("outage_experiment")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    f = load_features()
    cv, _ = test_months(f)
    months = cv[-12:]
    tuned = C.MODELS_DIR / "lgbm_best_params.json"
    params = json.loads(tuned.read_text()) if tuned.exists() else None
    base = feature_columns()
    rows = []
    for name, mode in (("none", None), ("oracle outages (NOT usable: uses later revisions)", "oracle"),
                       ("strict outages (leak-free, undercounted)", "strict")):
        g, feats = f, list(base)
        if mode:
            out = O.outage_frame(f.index, mode)
            g = f.join(out)
            feats = base + list(out.columns)
        pr = walk_forward(g, lambda: make_lgbm(params), feats, months)
        y = g.loc[pr.index, TARGET]
        per_month = (y - pr).abs().groupby(g.loc[pr.index, "delivery_day"].dt.to_period("M")).mean()
        rows.append(dict(variant=name, n_features=len(feats), MAE=mae(y, pr), fold_sd=per_month.std()))
        log.info("%-52s MAE %.3f", name, rows[-1]["MAE"])
    out = pd.DataFrame(rows)
    out["delta_vs_none"] = out["MAE"] - out.loc[0, "MAE"]
    out["months"] = f"{months[0]}..{months[-1]}"
    docs = O.load_docs()
    out["documents"] = len(docs)
    out.to_csv(C.REPORTS / "outage_experiment.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

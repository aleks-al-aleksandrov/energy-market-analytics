"""Battery studies beyond the base case (day-ahead arbitrage, 1 MW / 2 MWh / 90 %).

1. Degradation sensitivity: net revenue when each MWh of throughput costs 0..15 EUR.
2. Confidence filter: stay idle on days whose expected profit is below a threshold. The threshold is chosen on the CV
   months only and then reported on the holdout.
3. Reserve stacking (indicative): capacity payments for FCR / aFRR / mFRR at the latest published prices, with the
   market-depth warning from src/reserve.py. Nothing here is a forecast of what a battery would be awarded.

Usage: python -m src.battery_study   (after run_all --plus; reads reports/predictions.csv)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import battery as B
from src import config as C
from src.reserve import PRODUCTS

log = logging.getLogger("battery_study")
FILTER_GRID = [0, 5, 10, 20, 40, 80, 120]          # EUR of expected daily profit per MW
DEGRADATION = [0, 3, 6, 10, 15]                    # EUR per MWh of throughput
HOURS_PER_YEAR = 8760


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = pd.read_csv(C.REPORTS / "predictions.csv", parse_dates=["timestamp_utc", "delivery_day"])
    cols = {"perfect_foresight": "actual", "forecast_lightgbm": "lightgbm", "naive_yesterday": "naive_d1"}
    p = p.dropna(subset=["actual", "lightgbm", "naive_d1"])
    year_scale = lambda n_days: 365 / n_days

    # 1) degradation ------------------------------------------------------------------------------------------
    rows = []
    for deg in DEGRADATION:
        b = B.Battery(degradation_eur_mwh=float(deg))
        daily = B.simulate(p, b, cols)
        s = daily[list(cols)].sum() * year_scale(len(daily))
        for k in cols:
            rows.append(dict(degradation_eur_mwh=deg, strategy=k, eur_per_mw_year=float(s[k]),
                             pct_of_perfect=float(100 * s[k] / s["perfect_foresight"]) if s["perfect_foresight"] else np.nan))
    pd.DataFrame(rows).to_csv(C.REPORTS / "battery_degradation.csv", index=False)

    # 2) confidence filter (chosen on CV, reported on both) -------------------------------------------------
    b = B.Battery()
    split_of = p.groupby("delivery_day")["split"].first()
    frows = []
    for m in FILTER_GRID:
        daily = B.simulate(p, b, {"forecast_lightgbm": "lightgbm"}, min_expected_profit=float(m))
        daily["split"] = daily["delivery_day"].map(split_of)
        for split, g in daily.groupby("split"):
            frows.append(dict(min_expected_profit=m, split=split, days=len(g), days_traded=int((g["forecast_lightgbm"] != 0).sum()),
                              eur_per_mw_year=float(g["forecast_lightgbm"].sum() * year_scale(len(g)))))
    fil = pd.DataFrame(frows)
    cvf = fil[fil.split == "cv"]
    best = float(cvf.loc[cvf["eur_per_mw_year"].idxmax(), "min_expected_profit"])
    fil["chosen"] = fil["min_expected_profit"].eq(best)
    fil.to_csv(C.REPORTS / "battery_filter.csv", index=False)
    log.info("confidence filter chosen on CV: %.0f EUR/MW/day", best)

    # 3) reserve stacking (indicative) -------------------------------------------------------------------------
    res = pd.read_csv(C.REPORTS / "reserve_monthly.csv", parse_dates=["month"]).set_index("month")
    last12 = res.loc[res.index >= res.index.max() - pd.DateOffset(months=11)]
    arb = float(pd.read_csv(C.REPORTS / "battery_results.csv", index_col=0).loc["forecast_lightgbm", "eur_per_mw_year"])
    rrows = []
    for label in PRODUCTS:
        if f"{label} price" not in res:
            continue
        price, mw = float(last12[f"{label} price"].mean()), float(last12[f"{label} MW"].mean())
        rrows.append(dict(product=label, price_eur_mw_h=price, contracted_mw=mw,
                          capacity_eur_per_mw_year_100pct=price * HOURS_PER_YEAR,
                          arbitrage_eur_per_mw_year=arb,
                          fleet_mw_that_would_be_10pct_of_market=0.1 * mw))
    pd.DataFrame(rrows).to_csv(C.REPORTS / "battery_reserve.csv", index=False)
    (C.REPORTS / "battery_filter_choice.json").write_text(json.dumps({"min_expected_profit_eur_per_mw_day": best,
                                                                       "rule": "chosen on CV months only"}, indent=2))
    print(pd.read_csv(C.REPORTS / "battery_degradation.csv").pivot(index="degradation_eur_mwh", columns="strategy",
                                                                    values="eur_per_mw_year").round(0).to_string())
    print(fil.round(1).to_string(index=False))
    print(pd.DataFrame(rrows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()

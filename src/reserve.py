"""Balancing-reserve capacity prices and volumes for Bulgaria (ENTSO-E contracted reserves) for the dashboard.

IMPORTANT interpretation notes (also shown in the dashboard):
* The published prices are piecewise CONSTANT for long periods (for example FCR 10 -> 50 -> 25 EUR/MW/h), which points
  to administered prices rather than competitively cleared ones. Their level can change with regulation.
* The markets are tiny (FCR about 20-45 MW, aFRR up about 70-155 MW contracted). A single battery is a large share of
  them and a fleet saturates them: the price-taker assumption of the arbitrage case does not hold here.
* Revenue = capacity payment only (price x MW x hours awarded); activation energy, SoC management costs and prequalification
  are ignored.
Usage: python -m src.reserve  (writes reports/reserve_monthly.csv)
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.ingest import load_raw

PRODUCTS = {"FCR (symmetric)": ("fcr", "Symmetric"), "aFRR up": ("afrr", "Up"), "aFRR down": ("afrr", "Down"),
            "mFRR up": ("mfrr", "Up")}


def monthly_table() -> pd.DataFrame:
    cols = {}
    for label, (key, col) in PRODUCTS.items():
        price, amount = load_raw(f"reserve_price/{key}"), load_raw(f"reserve_amount/{key}")
        if not price.empty and col in price:
            cols[f"{label} price"] = price[col].resample("MS").mean()
        if not amount.empty and col in amount:
            cols[f"{label} MW"] = amount[col].resample("MS").mean()
    out = pd.DataFrame(cols)
    out.index = out.index.tz_localize(None)
    out.index.name = "month"
    return out


def main() -> None:
    t = monthly_table()
    t.round(2).to_csv(C.REPORTS / "reserve_monthly.csv")
    print(t.tail(6).round(1).to_string())


if __name__ == "__main__":
    main()

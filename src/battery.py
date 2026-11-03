"""Battery arbitrage on the BG day-ahead market.

Per delivery day a small LP decides hourly charge/discharge to maximise revenue at
the *schedule* prices (perfect foresight, forecast, or naive = yesterday's prices),
and the schedule is then *settled* at actual prices. Constraints:
  * power limit P (MW) for charge and discharge
  * energy capacity E = P x duration (MWh), start and end the day empty
  * round-trip efficiency eta (split sqrt(eta) on charge and discharge)
  * at most one full cycle per day (charged energy <= E)
For one cycle this is the "charge in the cheapest hours, discharge in the most
expensive later hours, only if the spread covers losses" rule, but it respects
the order of charging before discharging.
Price-taker assumption; no degradation, network fees, intraday or balancing revenue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class Battery:
    power_mw: float = 1.0
    duration_h: float = 2.0
    efficiency: float = 0.90
    cycles_per_day: float = 1.0
    degradation_eur_mwh: float = 0.0

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h


def schedule_day(prices: np.ndarray, b: Battery) -> np.ndarray:
    """Return net discharge per hour (MW, + = sell, - = buy) optimal for `prices`."""
    n = len(prices)
    if n == 0 or np.isnan(prices).any():
        return np.zeros(n)
    se = np.sqrt(b.efficiency)
    # variables x = [c_0..c_n-1, d_0..d_n-1]
    cost = np.concatenate([prices + b.degradation_eur_mwh / 2, -prices + b.degradation_eur_mwh / 2])
    L = np.tril(np.ones((n, n)))  # cumulative sum
    # SoC_t = se*cumsum(c) - cumsum(d)/se in [0, E]
    A_soc = np.hstack([se * L, -L / se])
    A_ub = [A_soc, -A_soc, np.concatenate([se * np.ones(n), np.zeros(n)])[None, :]]
    b_ub = [np.full(n, b.energy_mwh), np.zeros(n), [b.energy_mwh * b.cycles_per_day]]
    A_eq = np.concatenate([se * np.ones(n), -np.ones(n) / se])[None, :]  # end empty
    res = linprog(cost, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), A_eq=A_eq, b_eq=[0],
                  bounds=[(0, b.power_mw)] * (2 * n), method="highs")
    if not res.success:
        return np.zeros(n)
    c, d = res.x[:n], res.x[n:]
    net = d - c
    net[np.abs(net) < 1e-9] = 0
    return net


def throughput(net: np.ndarray) -> float:
    """MWh of (charge + discharge) / 2, the unit the degradation cost is charged on (same convention as the LP)."""
    return float((np.maximum(-net, 0).sum() + np.maximum(net, 0).sum()) / 2)


def simulate(df: pd.DataFrame, b: Battery, schedule_cols: dict[str, str], actual_col: str = "actual",
             min_expected_profit: float = 0.0) -> pd.DataFrame:
    """Daily NET revenue (EUR) per strategy: settled at actual prices, minus degradation cost.

    The schedule is optimised on the schedule prices including the degradation cost. If its *expected* profit
    (at the schedule prices, after degradation) is below `min_expected_profit` EUR for the day, the battery stays idle:
    a confidence filter that avoids trading days where the forecast spread is too small to be trusted.
    """
    rows = []
    for day, g in df.groupby("delivery_day"):
        actual = g[actual_col].to_numpy()
        if np.isnan(actual).any():
            continue
        r = {"delivery_day": day}
        for name, col in schedule_cols.items():
            sched = g[col].to_numpy(dtype=float)
            net = schedule_day(sched, b)
            deg = b.degradation_eur_mwh * throughput(net)
            if float(np.sum(net * sched)) - deg < min_expected_profit:
                net, deg = np.zeros_like(net), 0.0
            r[name] = float(np.sum(net * actual)) - deg
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(daily: pd.DataFrame, b: Battery) -> pd.DataFrame:
    strategies = [c for c in daily.columns if c != "delivery_day"]
    days = len(daily)
    out = pd.DataFrame({
        "total_eur": daily[strategies].sum(),
        "eur_per_mw_year": daily[strategies].sum() / b.power_mw * 365 / max(days, 1),
    })
    if "perfect_foresight" in strategies:
        out["pct_of_perfect"] = 100 * out["total_eur"] / out.loc["perfect_foresight", "total_eur"]
    out["days"] = days
    return out



def _irr(cashflows: np.ndarray, lo: float = -0.9, hi: float = 5.0) -> float:
    f = lambda r: float(np.sum(cashflows / (1 + r) ** np.arange(len(cashflows))))
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def investment_case(net_eur_per_mw_year: float, power_mw: float, duration_h: float, capex_eur_per_kwh: float,
                    opex_pct_of_capex: float = 2.0, life_years: int = 12, discount: float = 0.08) -> dict:
    """Illustrative project economics from a yearly net revenue. All inputs are assumptions, not market facts.

    Also returns the break-even capex (EUR per kWh) at which NPV = 0, which needs no capex assumption.
    """
    energy_kwh = power_mw * duration_h * 1000
    capex = capex_eur_per_kwh * energy_kwh
    revenue = net_eur_per_mw_year * power_mw
    annual = revenue - opex_pct_of_capex / 100 * capex
    t = np.arange(1, life_years + 1)
    annuity = float(np.sum(1 / (1 + discount) ** t))
    cash = np.array([-capex] + [annual] * life_years)
    be_capex = revenue * annuity / (1 + opex_pct_of_capex / 100 * annuity) / energy_kwh      # NPV = 0
    return dict(capex_eur=capex, annual_net_eur=annual, payback_years=(capex / annual if annual > 0 else float("inf")),
                irr=_irr(cash), npv_eur=float(np.sum(cash / (1 + discount) ** np.arange(len(cash)))),
                breakeven_capex_eur_per_kwh=float(be_capex))

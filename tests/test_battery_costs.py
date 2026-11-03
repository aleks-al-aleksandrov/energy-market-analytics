"""Degradation settlement, confidence filter and investment arithmetic (src/battery.py)."""
import numpy as np
import pandas as pd

from src import battery as B


def _df(n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        base = 100 + 50 * np.sin(np.arange(24) / 24 * 2 * np.pi) + rng.normal(0, 8, 24)
        rows.append(pd.DataFrame({"delivery_day": day, "actual": base, "fc": base + rng.normal(0, 15, 24)}))
    return pd.concat(rows)


def _revenue(b, **kw):
    return B.simulate(_df(), b, {"fc": "fc"}, **kw)["fc"].sum()


def test_degradation_cost_lowers_revenue_and_a_huge_cost_stops_trading():
    assert _revenue(B.Battery()) > _revenue(B.Battery(degradation_eur_mwh=20.0)) > 0
    assert _revenue(B.Battery(degradation_eur_mwh=1000.0)) == 0.0


def test_net_revenue_equals_gross_minus_degradation_on_one_day():
    df = _df(1)
    b = B.Battery(degradation_eur_mwh=10.0)
    net = B.schedule_day(df["fc"].to_numpy(), b)
    expected = float(np.sum(net * df["actual"].to_numpy())) - 10.0 * B.throughput(net)
    assert np.isclose(B.simulate(df, b, {"fc": "fc"})["fc"].iloc[0], expected)


def test_confidence_filter_skips_days_and_zero_threshold_changes_nothing():
    b = B.Battery()
    assert np.isclose(_revenue(b, min_expected_profit=0.0), _revenue(b))
    assert _revenue(b, min_expected_profit=1e9) == 0.0
    days_traded = lambda m: int((B.simulate(_df(), b, {"fc": "fc"}, min_expected_profit=m)["fc"] != 0).sum())
    assert days_traded(200.0) < days_traded(0.0)


def test_investment_case_arithmetic():
    kw = dict(net_eur_per_mw_year=100_000.0, power_mw=1.0, duration_h=2.0, opex_pct_of_capex=2.0, life_years=12, discount=0.08)
    case = B.investment_case(capex_eur_per_kwh=300.0, **kw)
    assert np.isclose(case["capex_eur"], 300 * 2000)
    assert np.isclose(case["payback_years"], case["capex_eur"] / case["annual_net_eur"])
    be = case["breakeven_capex_eur_per_kwh"]
    at_be = B.investment_case(capex_eur_per_kwh=be, **kw)
    assert abs(at_be["npv_eur"]) < 1.0                              # NPV is zero at the break-even capex
    assert abs(at_be["irr"] - 0.08) < 1e-3                          # and the IRR equals the discount rate
    assert B.investment_case(capex_eur_per_kwh=be * 0.5, **kw)["irr"] > 0.08 > B.investment_case(capex_eur_per_kwh=be * 1.5, **kw)["irr"]


def test_negative_cash_flow_has_infinite_payback():
    case = B.investment_case(net_eur_per_mw_year=1_000.0, power_mw=1.0, duration_h=2.0, capex_eur_per_kwh=500.0)
    assert case["annual_net_eur"] < 0 and np.isinf(case["payback_years"])

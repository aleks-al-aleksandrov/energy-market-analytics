"""Monte Carlo battery scenarios: calibration, leakage and the optimiser's-curse behaviour."""
import numpy as np
import pandas as pd

from src import battery as B
from src import montecarlo as M


def _frame(n_days=330, noise=15.0, seed=0, curse=0.0):
    """Synthetic days: daily-shaped forecast, actual = forecast + noise. `curse` makes errors push against the schedule."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    shape = 40 * np.sin((hours - 6) / 24 * 2 * np.pi)
    rows = []
    for d in range(n_days):
        f = 100 + shape + rng.normal(0, 8, 24)
        e = rng.normal(0, noise, 24)
        if curse:
            e = e - curse * (f - f.mean())          # forecast highs are too high, lows too low
        a = f + e
        rows.append((pd.Timestamp("2025-01-01") + pd.Timedelta(days=d), f, a))
    return pd.DataFrame(rows, columns=["day", "forecast", "actual"]).set_index("day")


def _prep(v, b=None):
    b = b or B.Battery()
    nets, deg = M.schedule_all(v, b)
    return nets, deg, M.history(v, nets, deg)


def test_draws_are_reproducible_and_shaped():
    v = _frame(60)
    nets, deg, h = _prep(v)
    r1 = M.draws("slippage", nets[-1], h["fc"][-1], deg[-1], h, 0, 59, 500, np.random.default_rng(1))
    r2 = M.draws("slippage", nets[-1], h["fc"][-1], deg[-1], h, 0, 59, 500, np.random.default_rng(1))
    assert r1.shape == (500,) and np.allclose(r1, r2)


def test_calibration_near_nominal_when_errors_are_neutral():
    v = _frame(400, curse=0.0)
    nets, deg, h = _prep(v)
    cal = M.calibration_backtest(v, nets, deg, "error_vector", window=120, n=800, h=h)
    assert 0.70 < cal["in_10_90"].mean() < 0.90


def test_slippage_beats_error_vector_when_errors_oppose_the_schedule():
    v = _frame(400, curse=0.5)
    nets, deg, h = _prep(v)
    cov = {m: M.coverage_summary(M.calibration_backtest(v, nets, deg, m, window=120, n=800, h=h)) for m in M.METHODS}
    # error-vector draws are neutral, so realised revenue falls below their median; slippage learns the shortfall
    assert cov["error_vector"]["median_bias_eur"] < cov["slippage"]["median_bias_eur"] - 1
    assert abs(cov["slippage"]["coverage_10_90"] - 0.8) < abs(cov["error_vector"]["coverage_10_90"] - 0.8)


def test_no_leakage_from_the_day_being_simulated_or_later():
    v = _frame(260)
    nets, deg, h = _prep(v)
    base = M.calibration_backtest(v, nets, deg, "slippage", window=100, n=300, h=h)
    t = 200
    v2 = v.copy()
    v2["actual"] = [a + (1000.0 if i >= t else 0.0) for i, a in enumerate(v2["actual"])]    # change day t and everything after
    h2 = M.history(v2, nets, deg)
    pert = M.calibration_backtest(v2, nets, deg, "slippage", window=100, n=300, h=h2)
    i = base.index[base["delivery_day"] == v.index[t]][0]
    cols = ["p05", "p10", "p50", "p90", "p95"]
    assert np.allclose(base.loc[i, cols].astype(float), pert.loc[i, cols].astype(float))
    assert not np.isclose(base.loc[i, "realised"], pert.loc[i, "realised"])    # the realised value does react


def test_next_day_quantiles_are_ordered_and_scenario_years_positive_for_a_profitable_market():
    v = _frame(300)
    nets, deg, h = _prep(v)
    nd = M.next_day(v["forecast"].iloc[-1], h, B.Battery(), "slippage", window=100, n=800)
    assert nd["p05"] <= nd["p10"] <= nd["p50"] <= nd["p90"] <= nd["p95"]
    yrs = M.scenario_years(h, window=100, n_years=200)
    assert yrs.shape == (200,) and yrs.mean() > 0 and yrs.std() > 0


def test_choose_method_uses_cv_coverage_only():
    s = {"error_vector": {"coverage_10_90_cv": 0.72, "coverage_10_90_holdout": 0.80},
         "slippage": {"coverage_10_90_cv": 0.83, "coverage_10_90_holdout": 0.55}}
    assert M.choose_method(s) == "slippage"

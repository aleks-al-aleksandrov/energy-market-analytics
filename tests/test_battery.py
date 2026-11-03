"""Physical constraints of the battery arbitrage optimiser (src/battery.py)."""
import numpy as np
import pandas as pd

from src import battery as B


def _soc(net, b):
    se = np.sqrt(b.efficiency)
    c, d = np.maximum(-net, 0), np.maximum(net, 0)
    return np.cumsum(se * c - d / se)


def test_flat_prices_do_not_trade():
    assert np.allclose(B.schedule_day(np.full(24, 80.0), B.Battery()), 0)


def test_losses_make_small_spreads_unprofitable():
    prices = np.array([10.0, 10.0, 10.0, 12.0])
    assert np.allclose(B.schedule_day(prices, B.Battery(efficiency=0.5, duration_h=1)), 0)   # sell 12 < buy 10 / 0.5


def test_schedule_respects_power_energy_and_cycle_limits():
    rng = np.random.default_rng(0)
    b = B.Battery(power_mw=2.0, duration_h=3.0, efficiency=0.9)
    for _ in range(20):
        prices = 100 + 60 * np.sin(np.arange(24) / 24 * 2 * np.pi) + rng.normal(0, 15, 24)
        net = B.schedule_day(prices, b)
        soc = _soc(net, b)
        assert np.all(np.abs(net) <= b.power_mw + 1e-6)
        assert soc.min() >= -1e-6 and soc.max() <= b.energy_mwh + 1e-6        # never below empty or above capacity
        assert abs(soc[-1]) < 1e-6                                              # ends the day empty
        assert np.maximum(-net, 0).sum() * np.sqrt(b.efficiency) <= b.energy_mwh * b.cycles_per_day + 1e-6


def test_battery_cannot_sell_before_it_has_charged():
    prices = np.array([200.0, 30.0, 40.0, 220.0] + [100.0] * 20)
    net = B.schedule_day(prices, B.Battery(duration_h=2.0))
    assert net[0] <= 1e-9                                                       # hour 0: empty battery cannot discharge


def test_perfect_foresight_is_an_upper_bound_for_any_other_schedule():
    rng = np.random.default_rng(3)
    rows = []
    for day in range(30):
        actual = 100 + 50 * np.sin(np.arange(24) / 24 * 2 * np.pi + rng.normal(0, .3)) + rng.normal(0, 10, 24)
        rows.append(pd.DataFrame({"delivery_day": day, "actual": actual, "noisy": actual + rng.normal(0, 25, 24),
                                  "naive": np.roll(actual, 1)}))
    df = pd.concat(rows)
    daily = B.simulate(df, B.Battery(), {"perfect": "actual", "noisy": "noisy", "naive": "naive"})
    assert (daily["perfect"] >= daily["noisy"] - 1e-6).all() and (daily["perfect"] >= daily["naive"] - 1e-6).all()
    assert daily["perfect"].sum() > 0


def test_higher_efficiency_never_earns_less():
    prices = 100 + 50 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    rev = lambda eff: float(np.sum(B.schedule_day(prices, B.Battery(efficiency=eff)) * prices))
    assert rev(0.95) >= rev(0.85) >= rev(0.75) - 1e-9

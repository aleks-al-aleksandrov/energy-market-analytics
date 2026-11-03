"""Evaluation maths and walk-forward splits (src/evaluate.py)."""
import numpy as np
import pandas as pd

from src import evaluate as E
from src.features import TARGET


def test_basic_metrics():
    y, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
    assert np.isclose(E.mae(y, p), (2 + 2 + 3) / 3)
    assert np.isclose(E.rmse(y, p), np.sqrt((4 + 4 + 9) / 3))


def test_pinball_loss_known_values():
    assert np.isclose(E.pinball([10.0], [8.0], 0.9), 1.8)      # under-forecast costs q * miss
    assert np.isclose(E.pinball([8.0], [10.0], 0.9), 0.2)      # over-forecast costs (1-q) * miss
    assert np.isclose(E.pinball([10.0], [10.0], 0.5), 0.0)


def _days(n=200, per=24):
    return np.repeat(np.arange(n), per)


def test_diebold_mariano_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    day = _days()
    bench = rng.normal(0, 20, len(day))
    model = 0.5 * bench                                         # half the error everywhere
    stat, p = E.diebold_mariano(model, bench, day)
    assert stat < 0 and p < 0.01


def test_diebold_mariano_has_roughly_the_nominal_false_positive_rate():
    """Monte Carlo size check: equal-accuracy models must be flagged about 5 % of the time (measured: 5.5-7 %)."""
    day = _days()
    ps = []
    for seed in range(120):
        rng = np.random.default_rng(seed)
        bench = rng.normal(0, 20, len(day))
        ps.append(E.diebold_mariano(rng.normal(0, 20, len(day)), bench, day)[1])
    rate = float(np.mean(np.array(ps) < 0.05))
    assert rate < 0.15, f"DM test is too liberal: {rate:.2f} false positives at the 5 % level"


def test_diebold_mariano_needs_enough_days():
    stat, p = E.diebold_mariano(np.ones(48), np.ones(48), np.repeat([0, 1], 24))
    assert np.isnan(stat) and np.isnan(p)


class _Spy:
    """Records which rows the model was trained on."""
    seen = []

    def fit(self, X, y):
        _Spy.seen.append(X.index)
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_walk_forward_never_trains_on_the_test_month_or_later():
    idx = pd.date_range("2024-01-01", "2025-06-30 23:00", freq="1h", tz="UTC")
    f = pd.DataFrame({"x": 1.0, TARGET: 50.0}, index=idx)
    f["delivery_day"] = idx.tz_convert("Europe/Brussels").normalize().tz_localize(None)
    months = [pd.Period("2025-04", "M"), pd.Period("2025-05", "M")]
    _Spy.seen = []
    pred = E.walk_forward(f, _Spy, ["x"], months)
    assert len(_Spy.seen) == 2 and len(pred) > 0
    for m, train_idx in zip(months, _Spy.seen):
        assert f.loc[train_idx, "delivery_day"].max() < m.start_time        # strictly before the test month
    assert pd.PeriodIndex(f.loc[pred.index, "delivery_day"], freq="M").isin(months).all()


def test_test_months_reserve_the_last_three_months_as_holdout():
    idx = pd.date_range("2024-01-01", "2026-10-31 23:00", freq="1h", tz="UTC")
    f = pd.DataFrame({TARGET: 50.0}, index=idx)
    f["delivery_day"] = idx.tz_convert("Europe/Brussels").normalize().tz_localize(None)
    cv, hold = E.test_months(f)
    assert len(hold) == 3 and hold[-1] == pd.Period("2026-10", "M")
    assert max(cv) < min(hold)

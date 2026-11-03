"""Regime-robust building blocks (src/models.py) and the bootstrap skill interval (src/evaluate.py)."""
import numpy as np
import pandas as pd

from src import evaluate as E
from src import models as M


def _xy(lo, hi, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    x = np.sort(rng.uniform(lo, hi, n))
    return pd.DataFrame({"x": x}, index=idx), pd.Series(2 * x + rng.normal(0, 0.5, n), index=idx)


def test_linear_base_extrapolates_where_a_tree_cannot():
    X, y = _xy(0, 10)
    Xt = pd.DataFrame({"x": [20.0]}, index=pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC"))
    tree = M.make_lgbm({"n_estimators": 60, "min_child_samples": 5, "verbose": -1}).fit(X, y)
    lin = M.LinearBase(M.make_ridge(["x"], alpha=0.01), M.make_lgbm({"n_estimators": 60, "min_child_samples": 5})).fit(X, y)
    assert tree.predict(Xt)[0] < 25                  # stuck near the largest training target (about 20)
    assert 35 < lin.predict(Xt)[0] < 45              # follows the line 2x -> 40


class _Spy:
    def fit(self, X, y, sample_weight=None):
        self.w = sample_weight                       # keep the raw object: LightGBM requires a numpy array
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_recency_weights_halve_every_half_life():
    X, y = _xy(0, 1, n=24 * 200)
    rec = M.Recency(_Spy(), half_life_days=30).fit(X, y)
    w = rec.model.w
    assert isinstance(w, np.ndarray)
    assert np.isclose(w[-1], 1.0) and w[0] < w[-1]
    age_days = (X.index.max() - X.index).total_seconds() / 86400
    assert np.allclose(w, 0.5 ** (age_days / 30))


def _pred(better: bool, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(pd.date_range("2026-01-01", periods=n_days, freq="D"), 24)
    actual = 100 + rng.normal(0, 30, len(day))
    naive = actual + rng.normal(0, 25, len(day))
    model = actual + rng.normal(0, 12 if better else 25, len(day))
    return pd.DataFrame({"delivery_day": day, "actual": actual, "naive": naive, "model": model})


def test_bootstrap_skill_interval_is_above_zero_for_a_better_model_and_straddles_zero_for_an_equal_one():
    skill, lo, hi = E.bootstrap_skill(_pred(True), "model", "naive")
    assert lo > 0 and lo < skill < hi
    skill, lo, hi = E.bootstrap_skill(_pred(False), "model", "naive")
    assert lo < 0 < hi


def test_bootstrap_is_deterministic():
    p = _pred(True)
    assert E.bootstrap_skill(p, "model", "naive") == E.bootstrap_skill(p, "model", "naive")


def test_recency_works_with_a_real_lightgbm():
    X, y = _xy(0, 10, n=400)
    m = M.Recency(M.make_lgbm({"n_estimators": 30, "min_child_samples": 5}), 60.0).fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_level_shape_follows_a_level_shift_outside_the_training_range():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=24 * 240, freq="1h", tz="UTC")
    day = idx.tz_convert("Europe/Brussels").normalize().tz_localize(None)
    days = pd.DatetimeIndex(sorted(set(day)))
    walk = pd.Series(100 + np.cumsum(rng.normal(0, 4, len(days))), index=days)               # persistent level (random walk)
    level = walk.reindex(day).to_numpy()
    hour = idx.tz_convert("Europe/Brussels").hour.to_numpy()
    y = pd.Series(level + 30 * np.sin(hour / 24 * 2 * np.pi), index=idx)
    d1 = walk.shift(1).bfill().reindex(day).to_numpy()                                         # yesterday's mean
    X = pd.DataFrame({"price_d1_mean": d1, "hour": hour, "dow": idx.day_of_week, "month": idx.month}, index=idx)
    model = M.LevelShape(M.make_lgbm({"n_estimators": 60, "min_child_samples": 5}), list(X.columns)).fit(X, y)
    tree = M.make_lgbm({"n_estimators": 60, "min_child_samples": 5}).fit(X, y)
    Xt = X.iloc[:24].copy()
    Xt["price_d1_mean"] = walk.max() + 150.0                                                    # far above anything in training
    assert model.predict(Xt).mean() > walk.max() + 100        # follows the new level
    assert tree.predict(Xt).mean() < walk.max() + 30          # a plain tree stays near the training maximum

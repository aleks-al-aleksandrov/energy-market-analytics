"""Phase 1 building blocks: weekly-refit walk-forward, asinh target, adaptive ensemble, LEAR, adaptive conformal bands."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.evaluate import walk_forward_blocks                                    # noqa: E402
from src.models import AsinhTarget                                              # noqa: E402
from src.phase1_experiment import aci_band, adaptive_ensemble, band_stats, lear_walk_forward   # noqa: E402


def _frame(days=420, seed=0):
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(index=idx)
    f["delivery_day"] = idx.tz_convert("Europe/Brussels").normalize().tz_localize(None)
    f["x"] = rng.normal(size=len(idx))
    f["price_BG"] = 50 + 10 * f["x"] + rng.normal(scale=1.0, size=len(idx))
    return f


class _Recorder:
    """Remembers the last training day it saw, so the test can check that no block trained on itself or later."""
    seen = []

    def fit(self, X, y):
        self.last = X.index.max()
        return self

    def predict(self, X):
        _Recorder.seen.append((self.last, X.index.min()))
        return np.zeros(len(X))


def test_weekly_walk_forward_never_trains_on_the_block_it_predicts():
    f = _frame()
    _Recorder.seen = []
    months = list(pd.period_range("2024-09", "2024-11", freq="M"))
    out = walk_forward_blocks(f, _Recorder, ["x"], months, refit_days=7)
    assert len(_Recorder.seen) >= 12                                             # about 13 weekly refits in three months
    assert all(train_end < test_start for train_end, test_start in _Recorder.seen)
    assert set(pd.PeriodIndex(f.loc[out.index, "delivery_day"], freq="M")) == set(months)


def test_window_limits_the_training_history():
    f = _frame()
    sizes = []

    class Count:
        def fit(self, X, y):
            sizes.append(len(X))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    walk_forward_blocks(f, Count, ["x"], [pd.Period("2024-11", "M")], refit_days=7, window_days=200)
    assert max(sizes) <= 200 * 24 + 24


def test_asinh_target_round_trips_and_uses_training_statistics_only():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=500)})
    y = pd.Series(40 + 25 * X["x"])                                              # includes negative prices
    m = AsinhTarget(LinearRegression()).fit(X, y)
    med, scale = m.med, m.scale
    m.predict(pd.DataFrame({"x": [1000.0]}))                                     # predicting must not change the scaler
    assert (m.med, m.scale) == (med, scale)
    z = np.arcsinh((y - m.med) / m.scale)
    assert np.allclose(np.sinh(z) * m.scale + m.med, y)


def test_adaptive_ensemble_moves_weight_to_the_member_that_is_right_lately():
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    idx = pd.date_range("2025-01-01", periods=60 * 24, freq="1h", tz="UTC")
    p = pd.DataFrame({"delivery_day": np.repeat(days, 24), "actual": 100.0}, index=idx)
    first_half = np.repeat(np.arange(60) < 30, 24)
    p["a"] = np.where(first_half, 100.0, 140.0)                                  # a is right first, then wrong
    p["b"] = np.where(first_half, 140.0, 100.0)                                  # b is the opposite
    e = adaptive_ensemble(p, ["a", "b"], window=7)
    assert abs(e.iloc[20 * 24] - 100) < 1 and abs(e.iloc[-1] - 100) < 1          # follows whichever member is right
    assert abs(e.iloc[0] - 120) < 1e-9                                           # equal weights before any history


def test_adaptive_ensemble_ignores_the_current_day():
    days = pd.date_range("2025-01-01", periods=30, freq="D")
    idx = pd.date_range("2025-01-01", periods=30 * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(2)
    p = pd.DataFrame({"delivery_day": np.repeat(days, 24), "actual": rng.normal(100, 10, len(idx)),
                      "a": rng.normal(100, 10, len(idx)), "b": rng.normal(100, 10, len(idx))}, index=idx)
    before = adaptive_ensemble(p, ["a", "b"])
    q = p.copy()
    last = q["delivery_day"] == days[-1]
    q.loc[last, "actual"] += 500                                                 # the outcome of the last day changes
    after = adaptive_ensemble(q, ["a", "b"])
    assert np.allclose(before[last.to_numpy()], after[last.to_numpy()])          # ... its own forecast must not


def _band_frame(days=400, shift_at=300, seed=3):
    d = pd.date_range("2025-01-01", periods=days, freq="D")
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=10, size=len(idx))
    noise[np.repeat(np.arange(days) >= shift_at, 24)] *= 3                       # volatility triples: a regime shift
    return pd.DataFrame({"delivery_day": np.repeat(d, 24), "actual": 100 + noise, "q10": 100 - 12.8, "q90": 100 + 12.8},
                        index=idx)


def test_aci_recovers_coverage_after_a_regime_shift_better_than_a_fixed_band():
    p = _band_frame()
    a = aci_band(p, gamma=0.05)
    p["lo"], p["hi"] = a["q10_aci"], a["q90_aci"]
    late = p[p["delivery_day"] >= p["delivery_day"].max() - pd.Timedelta(days=60)]
    fixed = band_stats(late, "q10", "q90")["coverage"]
    adaptive = band_stats(late, "lo", "hi")["coverage"]
    assert fixed < 0.45 and adaptive > 0.70


def test_aci_offsets_use_only_earlier_days():
    p = _band_frame()
    a1 = aci_band(p, gamma=0.05)
    q = p.copy()
    last = q["delivery_day"] == q["delivery_day"].max()
    q.loc[last, "actual"] += 1000
    a2 = aci_band(q, gamma=0.05)
    assert np.allclose(a1.loc[last.to_numpy(), "q90_aci"], a2.loc[last.to_numpy(), "q90_aci"])


def test_lear_predicts_a_weekly_pattern_and_never_sees_the_target_day():
    days = 500
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="1h", tz="UTC")
    f = pd.DataFrame(index=idx)
    f["delivery_day"] = idx.normalize().tz_localize(None)
    dow = f["delivery_day"].dt.day_of_week.to_numpy()
    hour = idx.hour.to_numpy()
    rng = np.random.default_rng(4)
    f["price_BG"] = 80 + 20 * np.sin(2 * np.pi * hour / 24) - 25 * (dow >= 5) + rng.normal(scale=2, size=len(idx))
    months = [pd.Period("2025-04", "M")]
    pred = lear_walk_forward(f, months, window_days=300)
    err = (f.loc[pred.index, "price_BG"] - pred).abs().mean()
    naive = (f["price_BG"] - f["price_BG"].shift(24)).abs().loc[pred.index].mean()
    assert err < 0.6 * naive                                                     # learns the weekend drop a lag-24 rule misses
    g = f.copy()
    target = g["delivery_day"] == pd.Timestamp("2025-04-20")
    g.loc[target, "price_BG"] += 300
    pred2 = lear_walk_forward(g, months, window_days=300)
    assert np.allclose(pred[target.reindex(pred.index).to_numpy()], pred2[target.reindex(pred2.index).to_numpy()])


def test_live_update_widens_the_band_after_misses_and_only_uses_published_days(tmp_path, monkeypatch):
    import json
    from src import calibrate, config as C
    monkeypatch.setattr(C, "REPORTS", tmp_path)
    p = _band_frame(days=200, shift_at=10_000)                                   # calm history
    off, state = calibrate.aci_offsets(p)
    state.pop("_scores").to_csv(tmp_path / calibrate.BAND_SCORES, index=False)
    (tmp_path / "conformal_offsets.json").write_text(json.dumps(state))
    start = pd.Timestamp(state["as_of"]) + pd.Timedelta(days=1)
    idx = pd.date_range(start, periods=3 * 24, freq="1h", tz="UTC")
    out = pd.DataFrame({"delivery_day": idx.strftime("%Y-%m-%d"), "p10_raw_eur_mwh": 87.2, "p90_raw_eur_mwh": 112.8}, index=idx)
    calibrate.log_raw_band(out)
    actual = pd.Series(200.0, index=idx[:48])                                    # two days published, far above the band
    st = calibrate.update_live(actual)
    assert st["as_of"] == idx[24].strftime("%Y-%m-%d") and st["live_days_applied"] == 2      # the third day is not out yet
    assert st["alpha_hi"] < state["alpha_hi"] and st["a_hi"] > state["a_hi"]                  # upper side widened
    assert calibrate.update_live(actual)["live_days_applied"] == 2                            # running again changes nothing
    calibrate.log_raw_band(out.assign(p90_raw_eur_mwh=999.0))                                 # the first forecast of an hour is kept
    assert (pd.read_csv(tmp_path / calibrate.BAND_LOG)["p90_raw"] == 112.8).all()


def test_member_weights_follow_recent_errors_and_ignore_the_target_day(tmp_path, monkeypatch):
    from src import config as C, ensemble as E
    monkeypatch.setattr(C, "REPORTS", tmp_path)
    days = pd.date_range("2026-01-01", periods=20, freq="D")
    idx = pd.date_range("2026-01-01", periods=20 * 24, freq="1h", tz="UTC")
    actual = pd.Series(100.0, index=idx)
    preds = pd.DataFrame({"lightgbm": 130.0, "ridge": 102.0, "level": 110.0, "lear": 100.5}, index=idx)
    for d in days:                                                            # the daily job logs one day at a time
        E.log_members(preds[preds.index.normalize().tz_localize(None) == d], d)
    lg = E.read_log()
    assert len(lg) == len(idx) and set(lg["source"]) == {"live"}
    w = E.member_weights(lg, actual, days[-1])
    assert abs(sum(w.values()) - 1) < 1e-9 and w["lear"] > w["ridge"] > w["level"] > w["lightgbm"]
    shocked = actual.copy()
    shocked[idx.normalize().tz_localize(None) == days[-1]] = 9999.0           # the target day's own outcome
    assert E.member_weights(lg, shocked, days[-1]) == w
    assert E.member_weights(lg, actual, days[1]) == {m: 0.25 for m in E.MEMBERS}      # too little history -> equal weights
    blend = E.combine(preds.iloc[:24], w)
    assert 100 < blend[0] < 110                                               # dominated by the members that were right
    E.log_members(preds.iloc[:24] + 50, days[0])
    assert E.read_log()["ridge"].iloc[0] == 102.0                             # the first forecast of an hour is kept


def test_lear_forecasts_a_live_day_without_its_price():
    from src.ensemble import lear_forecast
    days = 400
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="1h", tz="UTC")
    f = pd.DataFrame(index=idx)
    f["delivery_day"] = idx.normalize().tz_localize(None)
    rng = np.random.default_rng(7)
    f["price_BG"] = 90 + 25 * np.sin(2 * np.pi * idx.hour.to_numpy() / 24) + rng.normal(scale=2, size=len(idx))
    last = f["delivery_day"].max()
    truth = f.loc[f["delivery_day"] == last, "price_BG"].copy()
    f.loc[f["delivery_day"] == last, "price_BG"] = np.nan                     # tomorrow: no price yet
    pred = lear_forecast(f, [last], window_days=300)
    assert len(pred) == 24 and (pred - truth).abs().mean() < 6

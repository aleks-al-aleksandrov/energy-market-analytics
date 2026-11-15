"""Quarter-hour layer: no leakage, hourly forecast preserved, band ordered."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src import quarter_hour as Q                       # noqa: E402

DAY = pd.Timestamp("2026-02-10")


def _data(days=60, seed=0):
    rng = np.random.default_rng(seed)
    qidx = pd.date_range("2026-01-01", periods=days * 96, freq="15min", tz="UTC")
    hidx = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
    hour = 100 + 40 * np.sin(2 * np.pi * (hidx.hour.to_numpy() - 6) / 24) + rng.normal(0, 5, len(hidx))
    hourly = pd.Series(hour, index=hidx)
    ramp = (hourly.shift(-1) - hourly.shift(1)).fillna(0) / 2
    qi = (qidx.minute // 15).to_numpy()
    price = hourly.reindex(qidx.floor("h")).to_numpy() + (qi - 1.5) / 4 * ramp.reindex(qidx.floor("h")).to_numpy() + rng.normal(0, 1, len(qidx))
    f = pd.DataFrame({"residual_load_forecast": 3000 + 20 * hour, "solar_forecast": np.clip(400 - 20 * np.abs(hidx.hour - 12), 0, None).astype(float),
                      "load_forecast": 4000 + 10 * hour, "wind_forecast": 300.0}, index=hidx)
    return pd.Series(price, index=qidx, name="price_q"), f, hourly


def _feats(q, f, fc):
    x = Q.build_quarter_features(q.index, Q.deviations(q), f, fc)
    return x[x["delivery_day"] == DAY]


def test_quarter_prices_of_the_delivery_day_and_later_never_change_features():
    q, f, fc = _data()
    base = _feats(q, f, fc)
    assert len(base) == 96
    shocked = q.copy()
    start = DAY.tz_localize("Europe/Brussels").tz_convert("UTC")
    shocked[shocked.index >= start] += 500.0
    pd.testing.assert_frame_equal(base, _feats(shocked, f, fc))
    earlier = q.copy()
    rng = np.random.default_rng(1)
    prev = (earlier.index >= start - pd.Timedelta(days=1)) & (earlier.index < start)
    earlier[prev] += rng.normal(0, 30, prev.sum())                      # the day before does feed the lag
    assert not _feats(earlier, f, fc)["dev_lag1d"].equals(base["dev_lag1d"])


def test_last_hour_of_the_day_does_not_look_into_the_next_day():
    q, f, fc = _data()
    base = _feats(q, f, fc)
    f2 = f.copy()
    nxt = DAY.tz_localize("Europe/Brussels").tz_convert("UTC") + pd.Timedelta(days=1)
    f2.loc[f2.index >= nxt, :] += 10_000                                # tomorrow's forecasts do not exist at gate closure
    fc2 = fc.copy()
    fc2[fc2.index >= nxt] += 10_000
    pd.testing.assert_frame_equal(base, _feats(q, f2, fc2))


def test_shape_predictions_average_to_zero_within_every_hour():
    idx = pd.date_range("2026-03-01", periods=8, freq="15min", tz="UTC")
    z = Q.zero_mean_by_hour(pd.Series([1.0, 2, 3, 10, -5, 0, 0, 1], index=idx))
    assert np.allclose(z.groupby(z.index.floor("h")).sum(), 0)


def test_forecast_quarters_keeps_the_hourly_forecast_and_an_ordered_band(tmp_path, monkeypatch):
    from src import config as C
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    q, f, fc = _data()
    start = DAY.tz_localize("Europe/Brussels").tz_convert("UTC")
    hist = q[q.index < start]
    bundle = Q.train_final(f, fc, "lgbm", quarters=hist)
    hidx = pd.date_range(start, periods=24, freq="1h")
    hourly = pd.DataFrame({"forecast_eur_mwh": fc.reindex(hidx).to_numpy(), "p10_eur_mwh": fc.reindex(hidx).to_numpy() - 20,
                           "p90_eur_mwh": fc.reindex(hidx).to_numpy() + 20}, index=hidx)
    out = Q.forecast_quarters(bundle, hourly, f, quarters=hist)
    assert len(out) == 96
    back = out["forecast_eur_mwh"].groupby(out.index.floor("h")).mean()
    assert np.allclose(back.to_numpy(), hourly["forecast_eur_mwh"].to_numpy())           # hourly forecast unchanged
    assert ((out["p10_eur_mwh"] <= out["forecast_eur_mwh"]) & (out["forecast_eur_mwh"] <= out["p90_eur_mwh"])).all()
    truth = q.reindex(out.index)
    flat = hourly["forecast_eur_mwh"].reindex(out.index.floor("h")).to_numpy()
    assert (truth - out["forecast_eur_mwh"]).abs().mean() < (truth - flat).abs().mean()  # the shape helps on a ramping day


def test_flat_bundle_reproduces_the_hourly_forecast(tmp_path, monkeypatch):
    from src import config as C
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    q, f, fc = _data(days=45)
    bundle = Q.train_final(f, fc, "flat", quarters=q)
    hidx = pd.date_range("2026-02-20", periods=24, freq="1h", tz="UTC")
    hourly = pd.DataFrame({"forecast_eur_mwh": 100.0, "p10_eur_mwh": 80.0, "p90_eur_mwh": 120.0}, index=hidx)
    out = Q.forecast_quarters(bundle, hourly, f, quarters=q)
    assert (out["forecast_eur_mwh"] == 100.0).all()

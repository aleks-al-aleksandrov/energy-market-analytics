"""No-leakage tests for the whole feature builder (src/features.py).

For a delivery day D (market date, Europe/Brussels), gate closure is 12:00 CET on D-1. Changing any price at or
after the start of D, or any physical flow from D-1 on (flows are only used up to D-2), must leave every feature of D
unchanged. If a feature secretly used same-day actuals, these tests would fail.
"""
import numpy as np
import pandas as pd
import pytest

from src import features as F

DAY = pd.Timestamp("2025-03-20")      # delivery day under test (market date)


def _hourly(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", "2025-04-30 23:00", freq="1h", tz="UTC", name="timestamp_utc")
    n = len(idx)
    h = pd.DataFrame(index=idx)
    base = 100 + 30 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 10, n)
    for z in ("BG", "RO", "GR", "HU"):
        h[f"price_{z}"] = base + rng.normal(0, 5, n)
    h["load_forecast"] = 4000 + 500 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 50, n)
    h["wind_forecast"] = np.abs(rng.normal(300, 100, n))
    h["solar_forecast"] = np.abs(rng.normal(400, 200, n))
    h["residual_load_forecast"] = h["load_forecast"] - h["wind_forecast"] - h["solar_forecast"]
    h["net_import"] = rng.normal(0, 300, n)
    for var in ("temp", "wind100", "rad", "cloud"):
        for loc in ("sofia", "dobrich", "stara_zagora"):
            h[f"wfc_{var}_{loc}"] = rng.normal(10, 3, n)
    h["weather_fc_is_proxy"] = 0
    for c in ("fuel_brent_usd", "fuel_gas_eur_mwh", "fuel_ttf_eur_mwh"):
        h[c] = rng.normal(60, 5, n)
    return h


def _day_start_utc(day: pd.Timestamp) -> pd.Timestamp:
    return day.tz_localize("Europe/Brussels").tz_convert("UTC")


def _features_of_day(h: pd.DataFrame) -> pd.DataFrame:
    f = F.build_features(h)
    cols = F.feature_columns(respect_choice=False)
    return f.loc[f["delivery_day"] == DAY, cols]


@pytest.mark.parametrize("seed", [0, 1])
def test_prices_of_delivery_day_and_later_never_change_features(seed):
    h = _hourly(seed)
    base = _features_of_day(h)
    assert len(base) in (23, 24, 25) and not base.empty
    shocked = h.copy()
    cut = _day_start_utc(DAY)
    price_cols = [c for c in h if c.startswith("price_")]
    shocked.loc[shocked.index >= cut, price_cols] = 9999.0
    after = _features_of_day(shocked)
    pd.testing.assert_frame_equal(base, after)


def test_flows_are_used_only_up_to_d_minus_2():
    h = _hourly()
    base = _features_of_day(h)
    shocked = h.copy()
    shocked.loc[shocked.index >= _day_start_utc(DAY - pd.Timedelta(days=1)), "net_import"] = 9999.0   # D-1 and later
    pd.testing.assert_frame_equal(base, _features_of_day(shocked))


def test_price_of_previous_day_does_change_lag_features():
    """Sanity check of the test itself: yesterday's prices SHOULD be visible (otherwise the tests above prove nothing)."""
    h = _hourly()
    base = _features_of_day(h)
    shocked = h.copy()
    d1 = (shocked.index >= _day_start_utc(DAY - pd.Timedelta(days=1))) & (shocked.index < _day_start_utc(DAY))
    shocked.loc[d1, "price_BG"] = 9999.0
    after = _features_of_day(shocked)
    assert not np.allclose(base["price_lag24"], after["price_lag24"], equal_nan=True)
    assert not np.allclose(base["price_d1_mean"], after["price_d1_mean"], equal_nan=True)


def test_target_is_not_a_feature():
    cols = F.feature_columns(respect_choice=False)
    assert F.TARGET not in cols and "is_negative" not in cols
    assert not any(c.startswith(("gen_", "gencons_", "load_actual", "residual_load_actual")) for c in cols)

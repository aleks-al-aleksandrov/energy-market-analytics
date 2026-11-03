import numpy as np
import pandas as pd
import pytest

from src import validate as V


def _good(n=24 * 30):
    idx = pd.date_range("2026-09-01", periods=n, freq="1h", tz="UTC", name="timestamp_utc")
    rng = np.random.default_rng(0)
    h = pd.DataFrame(index=idx)
    for z in ("BG", "RO", "GR", "HU"):
        h[f"price_{z}"] = 100 + rng.normal(0, 20, n)
    for c in ("load_forecast", "wind_forecast", "solar_forecast"):
        h[c] = rng.normal(1000, 100, n)
    return h


NOW = pd.Timestamp("2026-10-01", tz="UTC")


def test_clean_data_has_no_issues():
    assert V.check_hourly(_good(), now=NOW) == []


def test_missing_required_column_is_an_error():
    h = _good().drop(columns="price_RO")
    assert any(i.severity == "error" and i.check == "schema" for i in V.check_hourly(h, now=NOW))


def test_duplicate_and_irregular_index_are_errors():
    h = _good()
    h2 = pd.concat([h, h.iloc[[5]]]).sort_index()
    assert any(i.check == "index" and i.severity == "error" for i in V.check_hourly(h2, now=NOW))
    h3 = h.drop(h.index[10])
    assert any(i.check == "index" and i.severity == "error" for i in V.check_hourly(h3, now=NOW))


def test_out_of_range_price_is_an_error():
    h = _good()
    h.iloc[7, h.columns.get_loc("price_BG")] = 99999.0
    assert any(i.check == "range" and i.severity == "error" for i in V.check_hourly(h, now=NOW))


def test_low_coverage_is_an_error():
    h = _good()
    h.iloc[::4, h.columns.get_loc("price_BG")] = np.nan
    assert any(i.check == "coverage" and i.severity == "error" for i in V.check_hourly(h, now=NOW))


def test_stuck_feed_and_stale_data_are_flagged():
    h = _good()
    h.iloc[-80:, h.columns.get_loc("price_BG")] = 55.0
    assert any(i.check == "stuck" and i.severity == "error" for i in V.check_hourly(h, now=NOW))
    old = pd.Timestamp("2026-12-01", tz="UTC")
    assert any(i.check == "freshness" for i in V.check_hourly(_good(), now=old))


def test_run_raises_on_error(monkeypatch, tmp_path):
    monkeypatch.setattr(V.C, "REPORTS", tmp_path)
    with pytest.raises(V.DataValidationError):
        V.run(_good().drop(columns="price_BG"), now=NOW)

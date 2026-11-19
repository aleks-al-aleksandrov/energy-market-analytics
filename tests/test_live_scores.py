"""Live scores of members and 15-minute forecasts: only pre-gate-closure forecasts count, the first one is kept."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import live_scores as L


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPORTS", tmp_path)
    return tmp_path


def _q15(day: str, value: float, shape: float = 4.0) -> pd.DataFrame:
    start = pd.Timestamp(day).tz_localize("Europe/Brussels").tz_convert("UTC")
    idx = pd.date_range(start, periods=96, freq="15min")
    sh = np.tile([-shape, -shape / 3, shape / 3, shape], 24)
    return pd.DataFrame({"forecast_eur_mwh": value + sh, "shape_eur_mwh": sh, "delivery_day": day}, index=idx)


def test_only_forecasts_made_before_gate_closure_count_and_the_first_is_kept(reports):
    day = "2026-11-20"
    L.append_quarters(_q15(day, 100.0), pd.Timestamp("2026-11-19 09:00", tz="UTC"))     # 10:00 CET: live
    L.append_quarters(_q15(day, 500.0), pd.Timestamp("2026-11-19 10:30", tz="UTC"))     # still before 12:00 CET, but later
    late = "2026-11-21"
    L.append_quarters(_q15(late, 100.0), pd.Timestamp("2026-11-20 12:30", tz="UTC"))    # 13:30 CET: after gate closure
    lg = pd.read_csv(reports / L.QLOG)
    assert (lg.loc[lg["delivery_day"] == day, "forecast_eur_mwh"] < 200).all()          # the first forecast was kept
    assert not lg.loc[lg["delivery_day"] == late, "is_live"].astype(str).str.lower().eq("true").any()


def test_scores_match_a_hand_calculation(reports):
    day = "2026-11-20"
    q15 = _q15(day, 100.0)
    L.append_quarters(q15, pd.Timestamp("2026-11-19 09:00", tz="UTC"))
    actual = pd.Series(100.0 + q15["shape_eur_mwh"].to_numpy() + 2.0, index=q15.index)   # the shape was right, the level 2 off
    members = pd.DataFrame({"timestamp_utc": actual.resample("1h").mean().index, "delivery_day": day, "source": "live",
                            "lightgbm": 101.0, "ridge": 110.0, "level": 102.0, "lear": 90.0})
    members.to_csv(reports / "member_log.csv", index=False)
    d = L.update(quarter_prices=actual)
    r = d.loc[day]
    assert r["quarters"] == 96
    assert r["mae_quarter"] == pytest.approx(2.0)
    assert r["mae_quarter_flat"] == pytest.approx(np.mean(np.abs(q15["shape_eur_mwh"] + 2.0)))
    assert r["mae_lightgbm"] == pytest.approx(1.0) and r["mae_ridge"] == pytest.approx(8.0) and r["mae_lear"] == pytest.approx(12.0)
    s = L.summary(d)
    assert s["days"] == 1 and s["mae_level"] == pytest.approx(0.0)


def test_days_without_published_prices_are_not_scored(reports):
    q15 = _q15("2026-11-20", 100.0)
    L.append_quarters(q15, pd.Timestamp("2026-11-19 09:00", tz="UTC"))
    earlier = pd.Series(50.0, index=pd.date_range("2026-11-01", periods=96, freq="15min", tz="UTC"))
    assert L.update(quarter_prices=earlier) is None

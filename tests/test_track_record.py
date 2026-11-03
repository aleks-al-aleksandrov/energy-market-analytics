"""The live track record must only count forecasts generated before day-ahead gate closure and never overwrite them."""
import pandas as pd

from src import track_record as TR


def _forecast(day="2026-11-03", value=100.0):
    idx = pd.date_range(f"{day} 00:00", periods=2, freq="1h", tz="UTC", name="timestamp_utc")
    return pd.DataFrame({"forecast_eur_mwh": value, "p10_eur_mwh": value - 10, "p90_eur_mwh": value + 10,
                         "spike_probability": 0.1, "delivery_day": day, "model_trained_until": "2026-11-01"}, index=idx)


def test_live_flag_and_first_live_forecast_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(TR, "LOG", tmp_path / "log.csv")
    # gate closure for delivery 2026-11-03 is 2026-11-02 12:00 CET = 11:00 UTC
    TR.append_forecast(_forecast(value=100.0), pd.Timestamp("2026-11-02 08:00", tz="UTC"))     # before: live
    TR.append_forecast(_forecast(value=555.0), pd.Timestamp("2026-11-02 15:00", tz="UTC"))     # after: not live, must not replace
    log = pd.read_csv(TR.LOG)
    assert len(log) == 2
    assert log["is_live"].all() and (log["forecast_eur_mwh"] == 100.0).all()


def test_late_forecast_is_logged_but_not_live(monkeypatch, tmp_path):
    monkeypatch.setattr(TR, "LOG", tmp_path / "log.csv")
    TR.append_forecast(_forecast(), pd.Timestamp("2026-11-02 14:00", tz="UTC"))
    log = pd.read_csv(TR.LOG)
    assert len(log) == 2 and not log["is_live"].any()

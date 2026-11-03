"""Leakage tests: fuel features for delivery day D may only use data published before 12:00 CET on D-1."""
import numpy as np
import pandas as pd

from src import fuels


def _fake_raw(monkeypatch, daily_dates, pink_months):
    brent = pd.Series(np.arange(len(daily_dates), dtype=float) + 100, index=daily_dates)
    pink = pd.DataFrame({"gas_eu_usd_mmbtu": np.arange(len(pink_months), dtype=float) + 10.0}, index=pink_months)
    monkeypatch.setattr(fuels, "_load", lambda: {"brent": brent, "carbon": brent * 0 + 80, "pink": pink})
    return brent, pink


def test_daily_series_uses_close_of_d_minus_2(monkeypatch):
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    brent, _ = _fake_raw(monkeypatch, dates, pd.date_range("2024-01-01", "2025-03-01", freq="MS"))
    idx = pd.date_range("2025-02-09 23:00", periods=48, freq="1h", tz="UTC")     # market days 10 and 11 Feb
    f = fuels.fuel_frame(idx)
    d = idx.tz_convert("Europe/Brussels").normalize().tz_localize(None)
    expected = brent.reindex(d - pd.Timedelta(days=2)).to_numpy()
    assert np.allclose(f["fuel_brent_usd"].to_numpy(), expected)
    # strictly older than the D-1 close, which settles after gate closure
    assert (f["fuel_brent_usd"].to_numpy() < brent.reindex(d - pd.Timedelta(days=1)).to_numpy()).all()


def test_changing_a_future_price_never_changes_a_feature(monkeypatch):
    dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    brent, pink = _fake_raw(monkeypatch, dates, pd.date_range("2024-01-01", "2025-03-01", freq="MS"))
    idx = pd.date_range("2025-02-09 23:00", periods=24, freq="1h", tz="UTC")     # exactly market day 10 Feb
    base = fuels.fuel_frame(idx)
    shocked = brent.copy()
    shocked.loc["2025-02-09":] = 9999.0                     # D-1 and later: unknown at gate closure
    pink2 = pink.copy()
    pink2.loc["2025-02-01":] = 999.0                        # the month of delivery itself: not yet published
    monkeypatch.setattr(fuels, "_load", lambda: {"brent": shocked, "carbon": shocked * 0 + 80, "pink": pink2})
    after = fuels.fuel_frame(idx)
    cols = ["fuel_brent_usd", "fuel_brent_chg20", "fuel_gas_eur_mwh", "fuel_gas_rel12", "fuel_gas_chg3"]
    assert np.allclose(base[cols].to_numpy(), after[cols].to_numpy(), equal_nan=True)


def test_pink_sheet_month_available_only_after_publication_lag(monkeypatch):
    months = pd.date_range("2024-01-01", "2025-03-01", freq="MS")
    _, pink = _fake_raw(monkeypatch, pd.date_range("2025-01-01", "2025-03-31", freq="D"), months)
    gas = fuels.monthly_gas_eur_mwh(pink, None)
    early = pd.date_range("2025-02-03 12:00", periods=1, freq="1h", tz="UTC")   # 3 Feb: January not yet published
    late = pd.date_range("2025-02-12 12:00", periods=1, freq="1h", tz="UTC")    # 12 Feb: January published
    f_early, f_late = fuels.fuel_frame(early), fuels.fuel_frame(late)
    assert np.isclose(f_early["fuel_gas_eur_mwh"].iloc[0], gas.loc["2024-12-01"])
    assert np.isclose(f_late["fuel_gas_eur_mwh"].iloc[0], gas.loc["2025-01-01"])


def test_daily_ttf_uses_d_minus_2_and_ignores_later_prices(monkeypatch):
    days = pd.date_range("2025-05-07", "2025-08-31", freq="D")
    ttf = pd.Series(np.arange(len(days), dtype=float) + 50, index=days)
    pink = pd.DataFrame({"gas_eu_usd_mmbtu": np.full(24, 12.0)}, index=pd.date_range("2024-01-01", periods=24, freq="MS"))
    monkeypatch.setattr(fuels, "_load", lambda: {"ttf": ttf, "pink": pink})
    idx = pd.date_range("2025-07-09 22:00", periods=24, freq="1h", tz="UTC")           # exactly market day 10 Jul
    base = fuels.fuel_frame(idx)
    assert np.allclose(base["fuel_ttf_eur_mwh"], ttf.loc["2025-07-08"])                  # D-2
    shocked = ttf.copy(); shocked.loc["2025-07-09":] = 9999.0                            # D-1 and later: unknown at gate closure
    monkeypatch.setattr(fuels, "_load", lambda: {"ttf": shocked, "pink": pink})
    after = fuels.fuel_frame(idx)
    cols = ["fuel_ttf_eur_mwh", "fuel_ttf_chg20", "fuel_ttf_vs_gasm"]
    assert np.allclose(base[cols].to_numpy(), after[cols].to_numpy(), equal_nan=True)


def test_ttf_falls_back_to_monthly_gas_before_daily_history(monkeypatch):
    days = pd.date_range("2025-05-07", "2025-08-31", freq="D")
    ttf = pd.Series(100.0, index=days)
    pink = pd.DataFrame({"gas_eu_usd_mmbtu": np.full(24, 12.0)}, index=pd.date_range("2024-01-01", periods=24, freq="MS"))
    monkeypatch.setattr(fuels, "_load", lambda: {"ttf": ttf, "pink": pink})
    idx = pd.date_range("2025-02-09 23:00", periods=24, freq="1h", tz="UTC")           # market day 10 Feb 2025, before daily data
    f = fuels.fuel_frame(idx)
    gas = fuels.monthly_gas_eur_mwh(pink, None)
    assert np.allclose(f["fuel_ttf_eur_mwh"], gas.loc["2025-01-01"])                     # monthly fallback, lag-safe
    assert f["fuel_ttf_chg20"].isna().all() and f["fuel_ttf_vs_gasm"].isna().all()       # changes stay NaN, no mixed definitions

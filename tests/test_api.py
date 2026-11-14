"""Serving layer and API: payload shape, JSON safety, battery plan physics, API key enforcement."""
import importlib
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from src import serving as S                   # noqa: E402


def test_forecast_payload_has_24ish_hours_and_an_ordered_band():
    f = S.forecast_latest()
    assert f["available"] and 23 <= len(f["hours"]) <= 25
    assert all(h["p10"] <= h["p90"] for h in f["hours"])
    s = f["summary"]
    assert s["min"] <= s["mean"] <= s["max"] and 0 <= s["min_hour_local"] <= 23
    assert s["cheapest_3h"]["mean"] <= s["dearest_3h"]["mean"]
    json.dumps(f, allow_nan=False)                                   # strictly valid JSON: no NaN / Infinity


def test_every_payload_is_strict_json():
    for name, fn in S.PAYLOADS.items():
        json.dumps(fn(), allow_nan=False)


def test_battery_plan_respects_limits_and_ends_empty():
    p = S.battery_plan(power_mw=2.0, duration_h=2.0, efficiency=0.9)
    assert p["available"]
    net = [r["net_mw"] for r in p["schedule"]]
    assert max(abs(x) for x in net) <= 2.0 + 1e-6
    soc = [r["soc_mwh"] for r in p["schedule"]]
    assert min(soc) >= -1e-6 and max(soc) <= p["battery"]["energy_mwh"] + 1e-6 and abs(soc[-1]) < 1e-6
    assert p["discharged_mwh"] <= p["charged_mwh"] + 1e-9            # losses: never more out than in
    assert p["expected_revenue_eur"] >= -1e-6                        # staying idle is always allowed


def test_battery_plan_rejects_nonsense():
    with pytest.raises(ValueError):
        S.battery_plan(power_mw=-1)


def _client(monkeypatch, key: str | None):
    if key is None:
        monkeypatch.delenv("EMA_API_KEY", raising=False)
    else:
        monkeypatch.setenv("EMA_API_KEY", key)
    monkeypatch.delenv("EMA_API_REQUIRE_KEY", raising=False)
    import api.main as m
    return TestClient(importlib.reload(m).app)


def test_api_routes_answer_and_health_reports_freshness(monkeypatch):
    c = _client(monkeypatch, None)
    assert "ok" in c.get("/health").json()
    for route in ("/v1/forecast/latest", "/v1/track-record", "/v1/metrics", "/v1/market/prices?days=3", "/v1/battery/plan"):
        r = c.get(route)
        assert r.status_code == 200, route
    assert c.get("/v1/battery/plan?efficiency=2").status_code == 422   # validated at the edge
    assert c.get("/v1/market/prices?days=100000").status_code == 422


def test_api_key_is_enforced_when_set(monkeypatch):
    c = _client(monkeypatch, "s3cret-key")
    assert c.get("/v1/forecast/latest").status_code == 401
    assert c.get("/v1/forecast/latest", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/forecast/latest", headers={"X-API-Key": "s3cret-key"}).status_code == 200
    assert c.get("/health").status_code == 200                         # health stays open for monitors


def test_hosted_mode_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("EMA_API_KEY", raising=False)
    monkeypatch.setenv("EMA_API_REQUIRE_KEY", "1")
    import api.main as m
    with pytest.raises(RuntimeError):
        importlib.reload(m)
    monkeypatch.delenv("EMA_API_REQUIRE_KEY")
    importlib.reload(m)

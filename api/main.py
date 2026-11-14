"""Forecast API (FastAPI). Serves the same payloads as the static export, plus an on-demand battery plan.

    uvicorn api.main:app --port 8000          # docs at http://localhost:8000/docs

Access: set EMA_API_KEY to require the header `X-API-Key` on every /v1 route. Without it the API is open, which is fine on
a laptop and wrong on the internet: api.main refuses to start with an empty key when EMA_API_REQUIRE_KEY=1 (use that in
any hosted deployment). CORS origins come from EMA_API_CORS (comma-separated; default: none).
"""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import serving as S

API_KEY = os.getenv("EMA_API_KEY", "")
if os.getenv("EMA_API_REQUIRE_KEY") == "1" and not API_KEY:
    raise RuntimeError("EMA_API_REQUIRE_KEY=1 but EMA_API_KEY is empty: refusing to start an open API")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")


app = FastAPI(title="BG Power forecast API", version=S.API_VERSION,
              description="Bulgarian day-ahead price forecast, its uncertainty band, live track record and a battery plan. "
                          "Forecast and analysis only; not trading or investment advice.")
origins = [o.strip() for o in os.getenv("EMA_API_CORS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["X-API-Key"])


@app.get("/health")
def health() -> dict:
    """Liveness and freshness. Returns ok=false (still HTTP 200) when the latest forecast is missing or stale."""
    return S.status()


@app.get("/v1/forecast/latest", dependencies=[Depends(require_key)])
def forecast_latest() -> dict:
    """Tomorrow's 24 hourly prices with the calibrated 10-90 % band, spike probability and a plain summary."""
    return S.forecast_latest()


@app.get("/v1/track-record", dependencies=[Depends(require_key)])
def track_record() -> dict:
    """Live forecasts (generated before gate closure) scored against published prices and the naive baseline."""
    return S.track_record()


@app.get("/v1/metrics", dependencies=[Depends(require_key)])
def metrics() -> dict:
    """Back-test quality: cross-validation and holdout tables, band calibration, and which changes passed the adoption rules."""
    return S.model_metrics()


@app.get("/v1/market/prices", dependencies=[Depends(require_key)])
def market_prices(days: int = Query(30, ge=1, le=400), zones: str = Query("BG,RO,GR,HU")) -> dict:
    """Recent published day-ahead prices (hourly and daily mean/min/max) for Bulgaria and its neighbours."""
    return S.market_recent(days, [z.strip().upper() for z in zones.split(",") if z.strip()])


@app.get("/v1/battery/plan", dependencies=[Depends(require_key)])
def battery_plan(power_mw: float = Query(1.0, gt=0, le=1000), duration_h: float = Query(2.0, ge=0.25, le=12),
                 efficiency: float = Query(0.90, ge=0.5, le=1.0), cycles_per_day: float = Query(1.0, gt=0, le=4),
                 degradation_eur_mwh: float = Query(0.0, ge=0, le=200)) -> dict:
    """Optimal charge/discharge schedule for tomorrow on the forecast, with revenue at the forecast and at each band edge."""
    try:
        return S.battery_plan(power_mw, duration_h, efficiency, cycles_per_day, degradation_eur_mwh)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

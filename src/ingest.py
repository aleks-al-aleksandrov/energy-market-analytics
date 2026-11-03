"""Download and cache ENTSO-E and Open-Meteo data.

Every request is cached as Parquet in data/raw/<dataset>/<key>_<YYYY-MM>.parquet,
so re-running never downloads the same chunk twice. All indices are stored in UTC.

Usage:
    python -m src.ingest            # full history START..today
    python -m src.ingest --quick    # last 7 days of BG prices (smoke test)
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from dotenv import load_dotenv

from src import config as C

log = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into calendar-month chunks (local time)."""
    edges = list(pd.date_range(start.normalize(), end, freq="MS", tz=start.tz))
    if not edges or edges[0] != start:
        edges = [start] + edges
    if edges[-1] != end:
        edges.append(end)
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _to_frame(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name)
    if isinstance(obj.columns, pd.MultiIndex):
        obj.columns = ["__".join(str(x) for x in c if str(x)) for c in obj.columns]
    obj.index = pd.DatetimeIndex(obj.index).tz_convert("UTC")
    obj.index.name = "timestamp_utc"
    return obj.astype("float64")


def _to_frame_docs(obj) -> pd.DataFrame:
    """Outage documents: keep text columns, put every timestamp in UTC (index = publication time)."""
    df = obj.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df.index.name = "created_doc_time"
    for c in ("start", "end"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df.astype({c: "string" for c in df.columns if df[c].dtype == object})


def cached(path: Path, fetch: Callable[[], pd.DataFrame | None], retries: int = 4,
           is_complete: bool = True) -> pd.DataFrame | None:
    """Return cached parquet or fetch with retry/backoff and save.

    A chunk that ENTSO-E reports as having no data is stored as an empty frame
    (so it is not re-requested). The current, incomplete month is never cached.
    """
    if path.exists():
        return pd.read_parquet(path)
    from entsoe.exceptions import NoMatchingDataError

    for attempt in range(1, retries + 1):
        try:
            df = fetch()
            break
        except NoMatchingDataError:
            log.warning("no data: %s", path.name)
            df = pd.DataFrame()
            break
        except Exception as exc:  # network / 5xx / rate limit
            wait = 5 * attempt
            log.warning("attempt %d failed for %s: %s (retry in %ds)", attempt, path.name, exc, wait)
            time.sleep(wait)
    else:
        log.error("giving up: %s", path.name)
        return None
    if is_complete:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        path.with_suffix(".partial.parquet").unlink(missing_ok=True)
    elif df is not None and len(df):
        # incomplete (current) month: stored as *.partial.parquet, refreshed on every run
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path.with_suffix(".partial.parquet"))
    return df


# ---------------------------------------------------------------------------
# ENTSO-E
# ---------------------------------------------------------------------------
def entsoe_client():
    from entsoe import EntsoePandasClient

    load_dotenv(C.ROOT / ".env")
    key = os.getenv("ENTSOE_API_KEY")
    if not key or key == "your_token_here":
        raise RuntimeError("ENTSOE_API_KEY missing - copy .env.example to .env and add your token")
    return EntsoePandasClient(api_key=key, retry_count=3, retry_delay=10)


def entsoe_jobs(client) -> dict[str, Callable]:
    """dataset_key -> function(start, end) returning Series/DataFrame."""
    jobs: dict[str, Callable] = {}
    for z in C.PRICE_ZONES:
        jobs[f"prices/{z}"] = lambda s, e, z=z: client.query_day_ahead_prices(z, start=s, end=e)
    bg = C.FOCUS_ZONE
    jobs[f"load/{bg}"] = lambda s, e: client.query_load(bg, start=s, end=e)
    jobs[f"load_forecast/{bg}"] = lambda s, e: client.query_load_forecast(bg, start=s, end=e)
    jobs[f"wind_solar_forecast/{bg}"] = lambda s, e: client.query_wind_and_solar_forecast(bg, start=s, end=e)
    jobs[f"generation/{bg}"] = lambda s, e: client.query_generation(bg, start=s, end=e)
    # day-ahead net transfer capacities (published before the auction) on the two SDAC-coupled borders
    for n in ("RO", "GR"):
        jobs[f"ntc/{n}_to_{bg}"] = lambda s, e, n=n: client.query_net_transfer_capacity_dayahead(n, bg, start=s, end=e)
        jobs[f"ntc/{bg}_to_{n}"] = lambda s, e, n=n: client.query_net_transfer_capacity_dayahead(bg, n, start=s, end=e)
    # planned/forced unavailability of generation units (documents with publication time -> usable without leakage)
    jobs[f"unavail_gen/{bg}"] = lambda s, e: client.query_unavailability_of_generation_units(bg, start=s, end=e)
    # balancing: contracted reserve capacity prices and volumes (daily agreements) for FCR / aFRR / mFRR
    for pt, name in (("A52", "fcr"), ("A51", "afrr"), ("A47", "mfrr")):
        jobs[f"reserve_price/{name}"] = lambda s, e, pt=pt: client.query_contracted_reserve_prices(
            bg, process_type=pt, start=s, end=e, type_marketagreement_type="A01")
        jobs[f"reserve_amount/{name}"] = lambda s, e, pt=pt: client.query_contracted_reserve_amount(
            bg, process_type=pt, start=s, end=e, type_marketagreement_type="A01")
    for n in C.FLOW_NEIGHBOURS:
        jobs[f"flows/{n}_to_{bg}"] = lambda s, e, n=n: client.query_crossborder_flows(n, bg, start=s, end=e)
        jobs[f"flows/{bg}_to_{n}"] = lambda s, e, n=n: client.query_crossborder_flows(bg, n, start=s, end=e)
    return jobs


def ingest_entsoe(start: pd.Timestamp, end: pd.Timestamp, only: list[str] | None = None,
                  workers: int = 6) -> None:
    """Download all datasets; datasets run in parallel threads (ENTSO-E limit: 400 req/min)."""
    from concurrent.futures import ThreadPoolExecutor

    client = entsoe_client()
    today = C.default_end()

    def run_dataset(key: str, fn: Callable) -> None:
        for s, e in month_chunks(start, end):
            path = C.DATA_RAW / "entsoe" / f"{key}_{s:%Y-%m}.parquet"
            complete = e <= today and (e - s) >= pd.Timedelta(days=27)
            if not path.exists():
                log.info("ENTSO-E %s %s", key, f"{s:%Y-%m}")
            conv = _to_frame_docs if key.startswith("unavail") else (lambda x: _to_frame(x, key.split("/")[0]))
            cached(path, lambda: conv(fn(s, e)), is_complete=complete)

    jobs = {k: fn for k, fn in entsoe_jobs(client).items()
            if not only or any(k.startswith(o) for o in only)}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in [ex.submit(run_dataset, k, fn) for k, fn in jobs.items()]:
            fut.result()


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
OM_ENDPOINTS = {
    # what actually happened (reanalysis / observations)
    "weather_actual": ("https://archive-api.open-meteo.com/v1/archive", ""),
    # forecast issued the day before (Previous Runs API, lead time 24-47h)
    "weather_fc_d1": ("https://previous-runs-api.open-meteo.com/v1/forecast", "_previous_day1"),
    # stitched short-lead forecasts (fallback proxy for periods without D-1 runs)
    "weather_fc_hist": ("https://historical-forecast-api.open-meteo.com/v1/forecast", ""),
}


def _om_fetch(url: str, suffix: str, lat: float, lon: float, s: str, e: str) -> pd.DataFrame:
    hourly = [v + suffix for v in C.WEATHER_VARS]
    r = requests.get(url, params=dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                                      hourly=",".join(hourly), timezone="UTC",
                                      wind_speed_unit="ms"), timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df.index = pd.to_datetime(df.pop("time"), utc=True)
    df.index.name = "timestamp_utc"
    df.columns = [c.replace(suffix, "") if suffix else c for c in df.columns]
    return df.astype("float64")


def ingest_weather(start: pd.Timestamp, end: pd.Timestamp, datasets: list[str] | None = None) -> None:
    today = C.default_end()
    for ds, (url, suffix) in OM_ENDPOINTS.items():
        if datasets and ds not in datasets:
            continue
        for name, (lat, lon) in C.WEATHER_POINTS.items():
            for s, e in month_chunks(start, end):
                path = C.DATA_RAW / "openmeteo" / ds / f"{name}_{s:%Y-%m}.parquet"
                # archive data lags ~5 days; only cache months that ended a week ago
                complete = e <= today - pd.Timedelta(days=7)
                last = (e - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if not path.exists():
                    log.info("Open-Meteo %s %s %s", ds, name, f"{s:%Y-%m}")
                cached(path, lambda: _om_fetch(url, suffix, lat, lon, s.strftime("%Y-%m-%d"), last),
                       is_complete=complete)


# ---------------------------------------------------------------------------
# Fuel and carbon prices (not available from ENTSO-E)
# ---------------------------------------------------------------------------
# * Brent, carbon ETC proxy (WisdomTree Carbon, tracks EU-allowance futures) and EUR/USD: Yahoo Finance daily closes
# * EU natural gas: World Bank "Pink Sheet" monthly average (daily TTF history is not freely available)
# * TTF front-month: forward-only daily snapshot log, so real daily gas history accumulates from now on
FUEL_YAHOO = {"brent": "BZ=F", "carbon": "CO2.L", "eurusd": "EURUSD=X"}
PINK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
PINK_COLUMNS = {"Crude oil, Brent": "brent_usd_bbl", "Natural gas, Europe": "gas_eu_usd_mmbtu",
                "Natural gas, US": "gas_us_usd_mmbtu", "Coal, Australian": "coal_aus_usd_mt"}
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, retries: int = 4, **kw) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            headers = {**UA, **kw.pop("headers", {})}
            r = requests.get(url, headers=headers, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == retries:
                raise
            log.warning("GET %s failed (%s), retry %d", url[:60], str(exc)[:60], attempt)
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def _yahoo_daily(symbol: str, start: str = "2022-09-01") -> pd.Series:
    """Final daily closes indexed by exchange-local trading date (today's partial bar is dropped)."""
    t = lambda x: int(pd.Timestamp(x, tz="UTC").timestamp())
    r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
             params=dict(period1=t(start), period2=int(time.time()) + 2 * 86400, interval="1d"))
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    close = pd.Series(res["indicators"]["quote"][0]["close"], index=ts, name=symbol).dropna()
    close = close[~close.index.duplicated(keep="last")]
    return close[close.index < pd.Timestamp.now(tz="UTC").tz_convert(tz).tz_localize(None).normalize()]


def _fresh(path: Path, hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < hours * 3600


def ingest_fuels(force: bool = False) -> None:
    d = C.DATA_RAW / "fuels"
    d.mkdir(parents=True, exist_ok=True)
    for name, sym in FUEL_YAHOO.items():
        path = d / f"{name}_daily.parquet"
        if _fresh(path, 12) and not force:
            continue
        try:
            s = _yahoo_daily(sym)
            s.rename(name).to_frame().to_parquet(path)
            log.info("fuels %s (%s): %d days to %s", name, sym, len(s), s.index.max().date())
        except Exception as exc:
            log.error("fuels %s failed: %s (keeping cached copy if any)", name, exc)
    # World Bank Pink Sheet: refreshed weekly; the landing page always links to the latest workbook
    path = d / "pinksheet_monthly.parquet"
    if not _fresh(path, 24 * 7) or force:
        try:
            import io
            import re

            page = _get(PINK_PAGE).text
            url = re.search(r"https://[^\"']+CMO-Historical-Data-Monthly[.]xlsx", page).group(0)
            raw = pd.read_excel(io.BytesIO(_get(url).content), sheet_name="Monthly Prices", header=None)
            updated = str(raw.iloc[3, 0])
            hdr = raw.iloc[4]
            cols = {i: PINK_COLUMNS[str(v).strip()] for i, v in hdr.items() if str(v).strip() in PINK_COLUMNS}
            m = raw.iloc[6:, [0, *cols]].copy()
            m.columns = ["month", *cols.values()]
            m["month"] = pd.to_datetime(m["month"].astype(str).str.replace("M", "-") + "-01", errors="coerce")
            m = m.dropna(subset=["month"]).set_index("month").apply(pd.to_numeric, errors="coerce")
            m.loc["2020-01-01":].to_parquet(path)
            (d / "pinksheet_meta.txt").write_text("\n".join([updated, url]) + "\n")
            log.info("fuels pink sheet: %s, last month %s", updated, m.dropna(how="all").index.max().date())
        except Exception as exc:
            log.error("fuels pink sheet failed: %s (keeping cached copy if any)", exc)
    ingest_ttf_daily(force=force)
    # forward-only TTF front-month snapshot (daily TTF history is not freely downloadable)
    try:
        r = _get("https://query1.finance.yahoo.com/v8/finance/chart/TTF=F", params=dict(range="1d", interval="1d"))
        px = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        snap = d / "ttf_snapshots.csv"
        today = pd.Timestamp.now(tz="Europe/Brussels").strftime("%Y-%m-%d")
        old = pd.read_csv(snap) if snap.exists() else pd.DataFrame(columns=["date", "ttf_front_eur_mwh"])
        old = old[old["date"] != today]
        pd.concat([old, pd.DataFrame([{"date": today, "ttf_front_eur_mwh": px}])]).sort_values("date").to_csv(snap, index=False)
    except Exception as exc:
        log.warning("TTF snapshot failed: %s", exc)


OILPRICE_URL = "https://api.oilpriceapi.com/v1/prices/past_month"


def ingest_ttf_daily(force: bool = False) -> None:
    """Daily TTF (EUR/MWh) from OilPriceAPI, merged into a local cache.

    The free tier returns the latest 500 daily averages whatever dates are requested, so every
    refresh is merged with the cached file and the local history keeps growing beyond 500 days.
    The key is read from OILPRICE_API_KEY in .env; without it this step is skipped.
    """
    import os

    from dotenv import load_dotenv

    d = C.DATA_RAW / "fuels"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "ttf_daily.parquet"
    if _fresh(path, 12) and not force:
        return
    load_dotenv(C.ROOT / ".env")
    key = os.getenv("OILPRICE_API_KEY")
    if not key:
        log.warning("OILPRICE_API_KEY missing in .env - daily TTF skipped (monthly gas only)")
        return
    try:
        today = pd.Timestamp.now(tz="UTC")
        r = _get(OILPRICE_URL, headers={"Authorization": f"Token {key}"},
                 params=dict(by_code="DUTCH_TTF_EUR", interval="1d", start_date="2023-01-01",
                             end_date=today.strftime("%Y-%m-%d")))
        rows = r.json()["data"]["prices"]
        new = pd.Series({pd.Timestamp(x["created_at"]).tz_convert("UTC").tz_localize(None).normalize(): float(x["price"])
                         for x in rows if not x.get("synthetic")}, name="ttf_eur_mwh").sort_index()
        new = new[new.index < today.tz_localize(None).normalize()]      # today's average is still incomplete
        old = pd.read_parquet(path).iloc[:, 0] if path.exists() else pd.Series(dtype=float)
        merged = pd.concat([old, new]).rename("ttf_eur_mwh")
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_frame().to_parquet(path)
        log.info("daily TTF: %d days cached (%s .. %s), last %.1f EUR/MWh", len(merged), merged.index.min().date(),
                 merged.index.max().date(), merged.iloc[-1])
    except Exception as exc:
        log.error("daily TTF failed: %s (keeping cached copy if any)", str(exc)[:120])


# ---------------------------------------------------------------------------
# loading cached raw data
# ---------------------------------------------------------------------------
def load_raw(dataset_key: str, source: str = "entsoe") -> pd.DataFrame:
    """Concatenate all cached monthly chunks for a dataset key, e.g. 'prices/BG'."""
    folder, _, prefix = dataset_key.rpartition("/")
    files = sorted((C.DATA_RAW / source / folder).glob(f"{prefix}_*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smoke test: last 7 days BG prices")
    p.add_argument("--only", nargs="*", help="dataset prefixes, e.g. prices load")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--weather-only", action="store_true")
    p.add_argument("--fuels-only", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = C.default_end()
    if a.quick:
        client = entsoe_client()
        s = end - pd.Timedelta(days=7)
        px = client.query_day_ahead_prices(C.FOCUS_ZONE, start=s, end=end)
        print(px.head(), "\nrows:", len(px), "| inferred freq:", px.index.to_series().diff().median())
        return
    if a.fuels_only:
        ingest_fuels(force=True)
        return
    if a.weather_only:
        ingest_weather(C.START, end)
        return
    ingest_entsoe(C.START, end, a.only)
    if not a.skip_weather and not a.only:
        ingest_weather(C.START, end)
    if not a.only:
        ingest_fuels()
    log.info("ingest done")


if __name__ == "__main__":
    main()


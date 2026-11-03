"""Generation-unit outages (ENTSO-E REMIT unavailability documents) for the BG bidding zone.

Why this is not a model feature (yet): the ENTSO-E API returns only the LATEST revision of every outage document
(852 documents with up to 48 revisions each). What was known at gate closure on a past day cannot be reconstructed:
documents of old days have long since been revised, so a "known before gate closure" filter would undercount old
days relative to live operation (train/live mismatch), and ignoring the filter would leak the future.

So this module provides
  * `outage_frame(idx, mode)`: hourly unavailable MW by plant type from the downloaded documents
        mode="oracle": final document windows. Uses information published after the day -> an UPPER BOUND for
                       analysis (src/outage_experiment.py), NEVER a model input
        mode="strict": only documents whose latest revision was created before gate closure (leak-free, but biased)
  * `record_point_in_time(day)`: logs what the API shows *now* for a delivery day, building a point-in-time-correct
    history (reports/outage_pit.csv) that can become a real feature once enough days have accumulated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C
from src.ingest import load_raw

log = logging.getLogger("outages")
MARKET_TZ = "Europe/Brussels"
PIT_FILE = C.REPORTS / "outage_pit.csv"
TYPE_GROUPS = {"Nuclear": "nuclear", "Fossil Brown coal/Lignite": "lignite"}


def load_docs() -> pd.DataFrame:
    """All downloaded outage documents, one row per (document, revision), with the unavailable MW."""
    d = load_raw("unavail_gen/BG")
    if d.empty:
        return d
    d = d.reset_index().drop_duplicates(["mrid", "revision", "created_doc_time"])
    d["nominal_power"] = pd.to_numeric(d["nominal_power"], errors="coerce")
    d["avail_qty"] = pd.to_numeric(d["avail_qty"], errors="coerce")
    d["unavail_mw"] = (d["nominal_power"] - d["avail_qty"]).clip(lower=0)
    d["group"] = d["plant_type"].map(TYPE_GROUPS).fillna("other")
    return d.dropna(subset=["start", "end", "unavail_mw"])


def _unavailable(docs: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Per hour: unavailable MW by group; overlapping documents of one unit count once (the maximum)."""
    res = {}
    for r in docs.itertuples():
        m = (hours >= r.start) & (hours < r.end)
        if not m.any():
            continue
        key = (r.production_resource_id, r.group)
        arr = res.setdefault(key, np.zeros(len(hours)))
        arr[m] = np.maximum(arr[m], r.unavail_mw)
    out = pd.DataFrame(0.0, index=hours, columns=["outage_nuclear_mw", "outage_lignite_mw", "outage_other_mw"])
    for (_, grp), arr in res.items():
        out[f"outage_{grp}_mw"] += arr
    out["outage_total_mw"] = out.sum(axis=1)
    return out


def outage_frame(idx_utc: pd.DatetimeIndex, mode: str = "oracle") -> pd.DataFrame:
    assert mode in ("oracle", "strict")
    docs = load_docs()
    if docs.empty:
        return pd.DataFrame(index=idx_utc)
    if mode == "oracle":
        return _unavailable(docs, idx_utc)
    mdate = idx_utc.tz_convert(MARKET_TZ).normalize().tz_localize(None)
    parts = []
    for day in pd.DatetimeIndex(mdate.unique()):
        hrs = idx_utc[mdate == day]
        gate = (day - pd.Timedelta(days=1)).tz_localize(MARKET_TZ) + pd.Timedelta(hours=12)
        known = docs[(docs["created_doc_time"] <= gate.tz_convert("UTC"))
                     & (docs["start"] < hrs.max() + pd.Timedelta(hours=1)) & (docs["end"] > hrs.min())]
        parts.append(_unavailable(known, hrs))
    return pd.concat(parts).reindex(idx_utc)


def record_point_in_time(day: pd.Timestamp, client=None) -> pd.DataFrame | None:
    """Log the outage state the API shows right now for delivery `day` (market date). Run before gate closure."""
    import os

    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv(C.ROOT / ".env")
    client = client or EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"), retry_count=2, retry_delay=5)
    day = pd.Timestamp(day).tz_localize(None).normalize()
    start = day.tz_localize(MARKET_TZ)
    try:
        raw = client.query_unavailability_of_generation_units(C.FOCUS_ZONE, start=start - pd.Timedelta(days=1),
                                                              end=start + pd.Timedelta(days=2))
    except Exception as exc:
        log.warning("outage snapshot failed (%s)", str(exc)[:80])
        return None
    d = raw.reset_index()
    d["created_doc_time"] = pd.to_datetime(d["created_doc_time"], utc=True)
    for c in ("start", "end"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d["nominal_power"] = pd.to_numeric(d["nominal_power"], errors="coerce")
    d["avail_qty"] = pd.to_numeric(d["avail_qty"], errors="coerce")
    d["unavail_mw"] = (d["nominal_power"] - d["avail_qty"]).clip(lower=0)
    d["group"] = d["plant_type"].map(TYPE_GROUPS).fillna("other")
    now = pd.Timestamp.now(tz="UTC")
    d = d[d["created_doc_time"] <= now]
    hours = pd.date_range(start.tz_convert("UTC"), (start + pd.DateOffset(days=1)).tz_convert("UTC"), freq="1h", inclusive="left")
    snap = _unavailable(d, hours)
    snap.insert(0, "snapshot_utc", now.floor("min"))
    snap.insert(1, "delivery_day", str(day.date()))
    snap = snap.rename_axis("timestamp_utc").reset_index()
    old = pd.read_csv(PIT_FILE, parse_dates=["snapshot_utc", "timestamp_utc"]) if PIT_FILE.exists() else None
    out = snap if old is None else pd.concat([old, snap])
    out.drop_duplicates(["delivery_day", "timestamp_utc", "snapshot_utc"]).round(1).to_csv(PIT_FILE, index=False)
    return snap

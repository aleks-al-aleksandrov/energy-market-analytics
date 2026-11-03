"""Outage frames: strict mode must never use a document created after gate closure; oracle mode ignores publication time."""
import pandas as pd

from src import outages as O


def _docs():
    return pd.DataFrame({
        "mrid": ["a", "b"], "revision": [1, 1], "production_resource_id": ["u1", "u2"],
        "created_doc_time": pd.to_datetime(["2026-03-01 08:00", "2026-03-10 18:00"], utc=True),     # b is created after the gate
        "start": pd.to_datetime(["2026-03-05", "2026-03-05"], utc=True), "end": pd.to_datetime(["2026-03-20", "2026-03-20"], utc=True),
        "nominal_power": [100.0, 300.0], "avail_qty": [0.0, 0.0], "unavail_mw": [100.0, 300.0],
        "group": ["lignite", "nuclear"], "plant_type": ["x", "y"],
    })


def test_strict_excludes_documents_created_after_gate_closure(monkeypatch):
    monkeypatch.setattr(O, "load_docs", _docs)
    # delivery day 11 Mar: gate closure = 10 Mar 12:00 CET = 11:00 UTC -> document b (created 10 Mar 18:00 UTC) is unknown
    idx = pd.date_range("2026-03-10 23:00", periods=24, freq="1h", tz="UTC")                      # market day 11 Mar
    strict = O.outage_frame(idx, "strict")
    oracle = O.outage_frame(idx, "oracle")
    assert (strict["outage_lignite_mw"] == 100.0).all() and (strict["outage_nuclear_mw"] == 0.0).all()
    assert (oracle["outage_nuclear_mw"] == 300.0).all() and (oracle["outage_total_mw"] == 400.0).all()


def test_overlapping_documents_of_one_unit_count_once(monkeypatch):
    d = _docs()
    d["production_resource_id"] = "u1"; d["group"] = "lignite"
    monkeypatch.setattr(O, "load_docs", lambda: d)
    idx = pd.date_range("2026-03-12", periods=24, freq="1h", tz="UTC")
    assert (O.outage_frame(idx, "oracle")["outage_lignite_mw"] == 300.0).all()                    # max, not 100 + 300

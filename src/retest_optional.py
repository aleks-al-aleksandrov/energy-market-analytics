"""Re-test the optional inputs that were rejected or unusable (capacity/NTC, outages) and record whether the preconditions changed.

Outages (ENTSO-E REMIT) can only become a leak-free feature if what was known at gate closure on past days can be
reconstructed. This script checks the two ways that could happen and writes the answer to reports/retest_status.json:

  1. the API now returns several revisions per outage document (a probe on a fully past month)
  2. enough point-in-time snapshots have been collected by src/predict_next_day.py (reports/outage_pit.csv)

Only when one of them holds does it re-run the outage experiment (CV months only, the holdout is never used).
Capacity/NTC is re-tested with the standing adoption rule (src/run_all.decide_optional_groups) when --groups is passed.

Usage: python -m src.retest_optional [--groups]
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

from src import config as C
from src import outages as O

log = logging.getLogger("retest")
MIN_PIT_DAYS = 180          # about half a year of daily snapshots: enough to cover seasons in a CV fold set
STATUS = C.REPORTS / "retest_status.json"


def probe_revisions(month_start: str | None = None) -> dict:
    """Ask the API for a fully past month and count documents that come back with more than one revision."""
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv(C.ROOT / ".env")
    key = os.getenv("ENTSOE_API_KEY")
    if not key:
        return {"probed": False, "reason": "no ENTSOE_API_KEY"}
    start = pd.Timestamp(month_start) if month_start else (pd.Timestamp.now().normalize() - pd.DateOffset(months=7)).replace(day=1)
    start = start.tz_localize(O.MARKET_TZ) if start.tzinfo is None else start
    end = start + pd.DateOffset(months=1)
    try:
        raw = EntsoePandasClient(api_key=key, retry_count=2, retry_delay=5).query_unavailability_of_generation_units(
            C.FOCUS_ZONE, start=start, end=end).reset_index()
    except Exception as exc:                   # network / API problems must not break a pipeline run
        return {"probed": False, "reason": str(exc)[:120]}
    per_doc = raw.groupby("mrid")["revision"].nunique()
    return {"probed": True, "month": f"{start:%Y-%m}", "documents": int(len(per_doc)),
            "documents_with_multiple_revisions": int((per_doc > 1).sum()), "max_revisions_per_document": int(per_doc.max() if len(per_doc) else 0)}


def pit_days() -> int:
    if not O.PIT_FILE.exists():
        return 0
    return int(pd.read_csv(O.PIT_FILE, usecols=["delivery_day"])["delivery_day"].nunique())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", action="store_true", help="also re-run the capacity/fuels adoption rule on the current data")
    args = ap.parse_args()

    status: dict = {"checked_utc": pd.Timestamp.now(tz="UTC").floor("min").isoformat()}
    probe = probe_revisions()
    n_pit = pit_days()
    status.update(revision_probe=probe, pit_days=n_pit, pit_days_needed=MIN_PIT_DAYS)
    api_ready = bool(probe.get("probed") and probe.get("documents_with_multiple_revisions", 0) > 0)
    pit_ready = n_pit >= MIN_PIT_DAYS
    status["outages_testable"] = api_ready or pit_ready
    if status["outages_testable"]:
        from src import outage_experiment
        log.info("preconditions met (api=%s, pit=%s): running the outage experiment on CV months", api_ready, pit_ready)
        outage_experiment.main()
        status["outage_experiment"] = "reports/outage_experiment.csv"
        status["outage_note"] = "Adopt outages only if the leak-free variant lowers CV MAE under the standing rule; the oracle row is an upper bound."
    else:
        status["outage_note"] = (f"Not testable yet: the API returns one revision per document "
                                 f"({probe.get('documents_with_multiple_revisions', 'n/a')} documents with several) and only {n_pit} of "
                                 f"{MIN_PIT_DAYS} point-in-time snapshot days exist. Snapshots are collected daily by the forecast workflow.")
    if args.groups:
        from src.evaluate import test_months
        from src.features import load_features
        from src.run_all import decide_optional_groups

        f = load_features()
        cv, _ = test_months(f)
        tuned = C.MODELS_DIR / "lgbm_best_params.json"
        used = decide_optional_groups(f, cv, json.loads(tuned.read_text()) if tuned.exists() else None)
        status["optional_groups_adopted"] = used
        status["optional_groups_file"] = "reports/feature_choice.json"
    STATUS.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

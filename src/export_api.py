"""Static JSON export of the serving payloads:  python -m src.export_api

Writes reports/api/v1/*.json (the same payloads as the FastAPI service). The daily workflow commits them, so a front end can
read them straight from the repository (or any static host) without a server that has to stay up.
"""
from __future__ import annotations

import json
import logging

from src import config as C
from src import serving as S

log = logging.getLogger("export_api")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = C.REPORTS / "api" / S.API_VERSION
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, fn in S.PAYLOADS.items():
        try:
            payload = fn()
        except Exception as exc:            # one broken payload must not stop the others (or the forecast run)
            log.warning("%s skipped: %s", name, str(exc)[:160])
            continue
        (out / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index[name] = f"{name}.json"
    (out / "index.json").write_text(json.dumps({"version": S.API_VERSION, "payloads": index}, indent=2), encoding="utf-8")
    log.info("wrote %d payloads to %s", len(index), out)


if __name__ == "__main__":
    main()

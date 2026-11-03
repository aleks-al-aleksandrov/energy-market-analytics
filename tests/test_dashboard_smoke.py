"""Headless smoke test: every dashboard page must run against the committed reports/ without raising."""
import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "overview.py", *sorted((APP / "views").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs_without_exception(page):
    if not (APP.parent / "reports" / "market_hourly.parquet").exists():
        pytest.skip("reports/ not generated (run python -m src.run_all --plus)")
    at = AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]

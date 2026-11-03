"""Headless smoke test: every dashboard page must run against the committed reports/ without raising,
in English/dark and in Bulgarian/light (the two configurations that differ in every code path that renders text or colour)."""
import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "home.py", APP / "overview.py", *sorted((APP / "views").glob("*.py"))]
CONFIGS = [("EN", "Dark"), ("BG", "Light")]


@pytest.mark.parametrize("lang,theme", CONFIGS, ids=lambda v: v)
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs_without_exception(page, lang, theme):
    if not (APP.parent / "reports" / "market_hourly.parquet").exists():
        pytest.skip("reports/ not generated (run python -m src.run_all --plus)")
    at = AppTest.from_file(str(page), default_timeout=180)
    at.session_state["_ui_init"] = True
    at.session_state["ui_lang"], at.session_state["ui_theme"] = lang, theme
    at.run()
    assert not at.exception, [e.value for e in at.exception]

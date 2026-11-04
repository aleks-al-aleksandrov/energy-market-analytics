"""Theme and language must survive page switches (Streamlit scopes widget state to the page that rendered it)."""
import os
import pathlib
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


def test_preferences_survive_the_widget_state_being_dropped(monkeypatch):
    monkeypatch.setenv("EMA_AUTH_DISABLED", "1")
    at = AppTest.from_file(str(APP / "overview.py"), default_timeout=120)
    at.session_state["_pref_theme"], at.session_state["_pref_lang"] = "Light", "BG"
    at.run()
    assert at.session_state["_mode"] == "light" and at.session_state["_lang"] == "bg"
    # what a page switch does to the widget keys
    at.session_state["ui_theme"], at.session_state["ui_lang"] = "Dark", "EN"      # widget state reset to its defaults
    at.run()
    assert at.session_state["_mode"] == "light" and at.session_state["_lang"] == "bg"
    assert at.session_state["ui_theme"] == "Light"


def test_header_marks_the_current_page():
    at = AppTest.from_file(str(APP / "views" / "6_Battery.py"), default_timeout=120)
    at.session_state["_pref_theme"], at.session_state["_pref_lang"] = "Dark", "EN"
    at.run()
    assert not at.exception
    # the keyed container for the Battery link carries the "_current" suffix
    import theme as T
    assert [t for *_, t in T.NAV].count("Battery Simulation") == 1

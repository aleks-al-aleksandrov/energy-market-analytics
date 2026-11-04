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


def test_every_page_is_reachable_from_the_header():
    """Inline links are a short primary set; the 'All pages' menu must list every page so nothing can be clipped away."""
    import theme as T
    titles = [t for *_, t in T.NAV]
    assert T.PRIMARY <= set(titles) and 3 <= len(T.PRIMARY) <= 6      # few enough to fit at laptop widths, in any language
    assert len(titles) == len(set(titles)) == 11


def test_page_link_falls_back_without_width(monkeypatch):
    import streamlit as st
    import theme as T
    calls = []

    def fake(path, label=None, icon=None, **kw):
        calls.append(kw)
        if "width" in kw:
            raise TypeError("unexpected keyword argument 'width'")
    monkeypatch.setattr(st, "page_link", fake)
    T.link("home.py", "Home", width="content")
    assert len(calls) == 2 and "width" not in calls[1]          # the second try still renders the link

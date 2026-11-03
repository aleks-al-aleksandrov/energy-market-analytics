"""Sign-in: hashing, verification, lock-out, and the gate in front of the real entry script."""
import pathlib
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import auth                                   # noqa: E402


def test_hash_roundtrip_and_salting():
    h1, h2 = auth.make_hash("correct horse battery"), auth.make_hash("correct horse battery")
    assert h1 != h2                                           # random salt
    assert auth.check_password("correct horse battery", h1)
    assert not auth.check_password("correct horse batterY", h1)
    assert not auth.check_password("x", "pbkdf2_sha256$broken")


def test_plain_text_fallback_is_still_constant_time_compare():
    assert auth.check_password("abc", "abc") and not auth.check_password("abd", "abc")


def test_disabled_without_configuration(monkeypatch):
    monkeypatch.delenv("EMA_AUTH_PASSWORD", raising=False)
    monkeypatch.delenv("EMA_AUTH_USERS", raising=False)
    monkeypatch.setattr(auth, "_config", lambda: {"password": None, "users": {}})
    assert not auth.enabled()


def test_user_table_and_unknown_user(monkeypatch):
    cfg = {"password": None, "users": {"alice": auth.make_hash("alice-passphrase")}}
    monkeypatch.setattr(auth, "_config", lambda: cfg)
    assert auth.verify("alice", "alice-passphrase")
    assert not auth.verify("alice", "nope") and not auth.verify("mallory", "alice-passphrase")


def _entry(monkeypatch):
    monkeypatch.setenv("EMA_AUTH_PASSWORD", auth.make_hash("open sesame 42"))
    at = AppTest.from_file(str(APP / "dashboard.py"), default_timeout=120)
    at.session_state["_ui_init"] = True
    at.session_state["ui_lang"], at.session_state["ui_theme"] = "EN", "Dark"
    return at


def test_gate_blocks_until_the_right_password(monkeypatch):
    at = _entry(monkeypatch).run()
    assert not at.exception
    assert len(at.text_input) == 1 and at.text_input[0].label == "Password"       # login screen, no dashboard
    assert not any(h for h in at.markdown if "Tomorrow" in h.value and "hero" in h.value)
    at.text_input[0].set_value("wrong").run()
    at.button[0].click().run()
    assert at.session_state["_auth_fails"] == 1 and "_auth_user" not in at.session_state

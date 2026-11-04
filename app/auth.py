"""Optional sign-in for the dashboard.

Access control is on by default and fails closed: until credentials are configured nobody can sign in. For local development set EMA_AUTH_DISABLED=1 (never on a hosted copy). To turn it on,
put either a shared password or per-user entries in `.streamlit/secrets.toml` (Streamlit Community Cloud: App settings -> Secrets)
or in environment variables:

    [auth]
    password = "pbkdf2_sha256$..."              # one shared password, or
    users = { alice = "pbkdf2_sha256$...", bob = "pbkdf2_sha256$..." }

Create the hash with `python -m src.make_password_hash`. A plain-text value works too but is stored readable, so prefer a hash.
Environment alternative: EMA_AUTH_PASSWORD, or EMA_AUTH_USERS as JSON ({"alice": "<hash>"}).

What this does and does not do: it keeps casual visitors out of the deployed app (salted PBKDF2 hashes, constant-time comparison,
a lock-out after repeated failures). It does not hide files that are in a public repository, and it is not a substitute for
the platform's own access control; for sensitive data use a private repository and the host's viewer allow-list as well.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets as _secrets
import time

import streamlit as st

ITERATIONS = 240_000
MAX_FAILS = 5
LOCK_SECONDS = 60


def make_hash(password: str, iterations: int = ITERATIONS) -> str:
    salt = _secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def check_password(password: str, stored: str) -> bool:
    """Constant-time check against a pbkdf2 hash (or a plain-text value)."""
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, it, salt, digest = stored.split("$")
            test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(it)).hex()
        except ValueError:
            return False
        return hmac.compare_digest(test, digest)
    return hmac.compare_digest(password.encode("utf-8"), stored.encode("utf-8"))


def _config() -> dict:
    """{'password': str | None, 'users': {name: stored}}; empty when access control is off."""
    if os.getenv("EMA_AUTH_PASSWORD") or os.getenv("EMA_AUTH_USERS"):
        password = os.getenv("EMA_AUTH_PASSWORD")
        users = {}
        if os.getenv("EMA_AUTH_USERS"):
            try:
                users = dict(json.loads(os.environ["EMA_AUTH_USERS"]))
            except ValueError:
                users = {}
        return {"password": password, "users": users}
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EMA_AUTH_DISABLED") == "1":
        return {"password": None, "users": {}}
    password, users = None, {}
    try:
        a = st.secrets.get("auth", {})
        password = a.get("password")
        users = dict(a.get("users", {}) or {})
    except Exception:           # no secrets file
        pass
    return {"password": password, "users": users}


def configured() -> bool:
    c = _config()
    return bool(c["password"] or c["users"])


def enabled() -> bool:
    """Access control is ON unless explicitly disabled for local development. Fail closed: with no credentials configured
    nobody gets in, so a forgotten secret can never leave a hosted copy open."""
    return os.getenv("EMA_AUTH_DISABLED") != "1"


def verify(username: str, password: str) -> bool:
    c = _config()
    if c["users"]:
        stored = c["users"].get((username or "").strip())
        if stored is None:      # still spend the hashing time, so unknown names are not distinguishable by speed
            check_password(password or "", make_hash("x", 1000))
            return False
        return check_password(password or "", str(stored))
    return bool(c["password"]) and check_password(password or "", str(c["password"]))


def user() -> str | None:
    return st.session_state.get("_auth_user")


def logout() -> None:
    st.session_state.pop("_auth_user", None)


def gate(render_login) -> None:
    """Call before the app runs. Shows `render_login(attempt)` and stops the script until the visitor has signed in."""
    if not enabled() or user():
        return
    ss = st.session_state
    now = time.time()
    locked = max(0.0, ss.get("_auth_lock", 0.0) - now)

    def attempt(username: str, password: str) -> str | None:
        """Returns an error message, or None on success."""
        if locked > 0:
            return f"Too many attempts. Try again in {int(locked) + 1} s."
        if verify(username, password):
            ss["_auth_user"], ss["_auth_fails"] = (username.strip() or "user"), 0
            return None
        ss["_auth_fails"] = ss.get("_auth_fails", 0) + 1
        time.sleep(min(2.0, 0.4 * ss["_auth_fails"]))               # slows down guessing
        if ss["_auth_fails"] >= MAX_FAILS:
            ss["_auth_lock"], ss["_auth_fails"] = time.time() + LOCK_SECONDS, 0
        return "Incorrect username or password."

    if not configured():
        render_login(None, False, 0.0)
    else:
        render_login(attempt, bool(_config()["users"]), locked)
    st.stop()

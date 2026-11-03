"""Bilingual UI (English / Bulgarian) without touching the page logic.

* `tr(text)` translates one English string. Keys in `i18n_bg.BG` are the English source with every computed value written
  as `{}`; the Bulgarian value refers to the captured values as {0}, {1}, ... so word order may change:
      "Average price was <b>{}</b>"  ->  "Средната цена беше <b>{0}</b>"
  Strings without computed values are plain dictionary hits.
* `install_hooks()` wraps the Streamlit text and widget calls once per process, so labels, captions, markdown, tabs,
  option lists and so on are translated at the boundary. The language is read per session at call time, which keeps
  concurrent viewers independent. English sessions pay one dictionary lookup and nothing else.
* Anything without a translation falls back to English (never to a blank), and `misses()` lists what was not found,
  which is how the translation was completed (see tests/test_i18n.py).
"""
from __future__ import annotations

import functools
import re

import streamlit as st

LANGS = ("en", "bg")
_EXACT: dict[str, str] = {}
_PATTERNS: list[tuple[str, re.Pattern, str]] = []      # (literal prefix, regex, bulgarian template)
_CACHE: dict[str, str] = {}
_MISSES: set[str] = set()
_loaded = False
_TAG = re.compile(r"(<[^>]+>)")
_LETTERS = re.compile(r"[A-Za-z]{3}")
_CYR = re.compile("[Ѐ-ӿ]")
_SLOT = re.compile(r"\{(\d+)\}")


def lang() -> str:
    try:
        v = st.session_state.get("_lang", "en")
    except Exception:
        v = "en"
    return v if v in LANGS else "en"


def _load() -> None:
    global _loaded
    if _loaded:
        return
    from i18n_bg import BG

    for key, val in BG.items():
        if "{}" in key:
            parts = key.split("{}")
            rx = re.compile("".join(re.escape(p) + ("(.*?)" if i < len(parts) - 1 else "") for i, p in enumerate(parts)), re.DOTALL)
            _PATTERNS.append((parts[0], rx, val))
        else:
            _EXACT[key] = val
    _PATTERNS.sort(key=lambda t: -len(t[0]))
    _loaded = True


# Bulgarian weekday and month names for dates that Python formatted in English (%A %B %a %b)
_DAYS = {"Monday": "понеделник", "Tuesday": "вторник", "Wednesday": "сряда", "Thursday": "четвъртък", "Friday": "петък",
         "Saturday": "събота", "Sunday": "неделя"}
_DAYS_SHORT = {"Mon": "пон", "Tue": "вт", "Wed": "ср", "Thu": "чет", "Fri": "пет", "Sat": "съб", "Sun": "нед"}
_MONTHS = {"January": "януари", "February": "февруари", "March": "март", "April": "април", "May": "май", "June": "юни",
           "July": "юли", "August": "август", "September": "септември", "October": "октомври", "November": "ноември",
           "December": "декември"}
_MONTHS_SHORT = {"Jan": "яну", "Feb": "фев", "Mar": "мар", "Apr": "апр", "May": "май", "Jun": "юни", "Jul": "юли", "Aug": "авг",
                 "Sep": "сеп", "Oct": "окт", "Nov": "ное", "Dec": "дек"}
_FULL = re.compile(r"\b(" + "|".join(list(_DAYS) + list(_MONTHS)) + r")\b")
# short names only next to a day number or year (so the word "Mar" or "May" in prose is never touched)


def dates(s: str) -> str:
    s = _FULL.sub(lambda m: _DAYS.get(m.group(1)) or _MONTHS[m.group(1)], s)
    s = re.sub(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?= \d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b)", lambda m: _DAYS_SHORT[m.group(1)], s)
    s = re.sub(r"(?<=\d )(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?= ?(?:20\d\d|,|\)|<|$|\s|\.))", lambda m: _MONTHS_SHORT[m.group(1)], s)
    s = re.sub(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?= 20\d\d\b)", lambda m: _MONTHS_SHORT[m.group(1)], s)
    return s


def _one(s: str) -> str | None:
    if s in _EXACT:
        return _EXACT[s]
    for prefix, rx, val in _PATTERNS:
        if prefix and not s.startswith(prefix):
            continue
        m = rx.fullmatch(s)
        if m:
            g = [_EXACT.get(x.strip(), x) for x in m.groups()]      # a captured word such as 'rose' or 'Hungary' is translated too
            return _SLOT.sub(lambda k: g[int(k.group(1))] if int(k.group(1)) < len(g) else k.group(0), val)
    return None


_SENT = re.compile(r"(?<=[.!?])\s+")
_PRE = re.compile(r"[\s:·]*(?:<b>)?$")
_EXTRA = re.compile(r"<extra>(.*?)</extra>")


def _sentences(s: str) -> str | None:
    """Composite strings (a headline built from optional sentences): translate sentence by sentence."""
    parts = _SENT.split(s.strip())
    if len(parts) < 2:
        return None
    out, hit = [], False
    for p in parts:
        t = _one(p)
        hit |= t is not None
        out.append(t if t is not None else p)
    return " ".join(out) if hit else None


def _hover(s: str) -> str | None:
    """Plotly hover templates: translate the series name before the first %{...} and the <extra> label."""
    i = s.find("%{")
    if i < 0:
        return None
    name = _PRE.sub("", s[:i])
    changed = False
    t = _one(name) if name else None
    if t is not None:
        s, changed = t + s[len(name):], True
    def ex(m):
        nonlocal changed
        v = _one(m.group(1)) if m.group(1) else None
        if v is None:
            return m.group(0)
        changed = True
        return f"<extra>{v}</extra>"
    s = _EXTRA.sub(ex, s)
    return s if changed else None


def _piece(p: str) -> str | None:
    return _one(p) or _sentences(p)


def _english_left(s: str) -> bool:
    """True if anything but English month/weekday names is still English (those are converted by dates())."""
    rest = re.sub(r"\b(" + "|".join([*_DAYS, *_DAYS_SHORT, *_MONTHS, *_MONTHS_SHORT]) + r")\b", " ", s)
    return bool(_LETTERS.search(re.sub(r"<[^>]+>", " ", rest)))


def tr(s):
    """Translate an English UI string (HTML allowed) for the current session. Non-strings pass through."""
    if not isinstance(s, str) or not s or lang() == "en":
        return s
    hit = _CACHE.get(s)
    if hit is not None:
        return hit
    if _CYR.search(s):                  # already translated (e.g. a composed string that is translated again downstream)
        return s
    _load()
    out = _one(s)
    if out is None and "%{" in s:
        out = _hover(s)
    if out is None:
        out = _sentences(s)
    if out is None and "<" in s and not s.lstrip().startswith("<style"):
        parts = _TAG.split(s)
        changed = False
        for i, p in enumerate(parts):
            if i % 2 == 0 and _LETTERS.search(p):
                t = _piece(p) or _piece(p.strip())
                if t is not None:
                    lead, trail = p[:len(p) - len(p.lstrip())], p[len(p.rstrip()):]
                    parts[i] = t if _piece(p) is not None else lead + t + trail
                    changed = True
        out = "".join(parts) if changed else None
    if out is None:
        if _english_left(s) and not s.lstrip().startswith("<style"):
            _MISSES.add(s)
        # an untranslated sentence stays English; a bare date (badge, axis label) still gets Bulgarian month names
        out = s if _english_left(s) else dates(s)
    else:
        out = dates(out)
    if len(_CACHE) < 20000:
        _CACHE[s] = out
    return out


def misses() -> list[str]:
    return sorted(_MISSES)


def tf(fn):
    """Wrap a format_func so the displayed option text is translated while the option value stays English."""
    if fn is None:
        return lambda x: tr(str(x))
    return lambda x: tr(fn(x))


# --- Streamlit boundary ------------------------------------------------------------------------------------------------------
_TEXT_FIRST = ("markdown", "caption", "subheader", "header", "title", "info", "warning", "success", "error", "write", "text", "code")
_LABEL_FIRST = ("slider", "select_slider", "selectbox", "radio", "multiselect", "toggle", "checkbox", "button", "download_button",
                "expander", "metric", "number_input", "text_input", "segmented_control", "pills", "date_input", "form_submit_button",
                "popover", "status")
_OPTIONS = ("selectbox", "radio", "multiselect", "segmented_control", "pills")


def _wrap(name: str, orig, kind: str):
    if getattr(orig, "_ema_wrapped", False):
        return orig

    @functools.wraps(orig)
    def inner(self, *a, **kw):
        if lang() == "bg":
            a = list(a)
            if kind == "text":
                if a and isinstance(a[0], str):
                    a[0] = tr(a[0])
                elif "body" in kw:
                    kw["body"] = tr(kw["body"])
            else:
                if a and isinstance(a[0], str):
                    a[0] = tr(a[0])
                elif "label" in kw:
                    kw["label"] = tr(kw["label"])
                if name == "metric" and len(a) > 2 and isinstance(a[2], str):
                    a[2] = tr(a[2])
                for k in ("help", "placeholder", "caption"):
                    if isinstance(kw.get(k), str):
                        kw[k] = tr(kw[k])
                if name in _OPTIONS:
                    kw["format_func"] = tf(kw.get("format_func"))
                elif name == "select_slider":
                    kw["format_func"] = tf(kw.get("format_func"))
        return orig(self, *a, **kw)
    inner._ema_wrapped = True
    inner._ema_orig = orig
    return inner


def _wrap_tabs(orig):
    if getattr(orig, "_ema_wrapped", False):
        return orig

    @functools.wraps(orig)
    def inner(self, tabs, *a, **kw):
        if lang() == "bg":
            tabs = [tr(t) for t in tabs]
        return orig(self, tabs, *a, **kw)
    inner._ema_wrapped = True
    return inner


def install_hooks() -> None:
    """Idempotent: wrap the Streamlit calls whose text a viewer sees."""
    from streamlit.delta_generator import DeltaGenerator as DG

    for n in _TEXT_FIRST:
        if hasattr(DG, n):
            setattr(DG, n, _wrap(n, getattr(DG, n), "text"))
    for n in _LABEL_FIRST:
        if hasattr(DG, n):
            setattr(DG, n, _wrap(n, getattr(DG, n), "label"))
    if hasattr(DG, "tabs"):
        DG.tabs = _wrap_tabs(DG.tabs)
    main = getattr(st, "_main", None)
    if main is not None:        # module-level st.markdown etc. were bound before the wrap: bind them again
        for n in (*_TEXT_FIRST, *_LABEL_FIRST, "tabs"):
            if hasattr(st, n) and hasattr(main, n):
                setattr(st, n, getattr(main, n))

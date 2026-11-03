"""Translation layer and theme palettes: structure checks that need no running dashboard."""
import pathlib
import re
import sys

import pytest

pytest.importorskip("streamlit")
APP = pathlib.Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import i18n                  # noqa: E402
import i18n_bg               # noqa: E402

TAG = re.compile(r"</?[a-zA-Z][^>]*>")
SLOT = re.compile(r"\{(\d+)\}")


def test_every_bulgarian_entry_keeps_its_placeholders_and_markup():
    for en, bg in i18n_bg.BG.items():
        n = en.count("{}")
        assert {int(x) for x in SLOT.findall(bg)} == set(range(n)), f"placeholder mismatch: {en[:80]!r}"
        # the same tags in the same quantity (attributes may be translated, tag names may not change)
        assert sorted(re.findall(r"</?([a-zA-Z0-9]+)", en)) == sorted(re.findall(r"</?([a-zA-Z0-9]+)", bg)), f"markup mismatch: {en[:80]!r}"


@pytest.fixture
def bg(monkeypatch):
    monkeypatch.setattr(i18n, "lang", lambda: "bg")
    i18n._CACHE.clear()
    i18n._MISSES.clear()
    yield
    i18n._CACHE.clear()


def test_exact_pattern_and_captured_words(bg):
    assert i18n.tr("Forecast") == "Прогноза"
    assert i18n.tr("Market Overview") == "Пазарен преглед"
    # computed numbers are kept, words captured from the English sentence are translated
    assert i18n.tr("Coupling with Romania fell to 71% in the latest month") == "Обвързването с Румъния спадна до 71% през последния месец"
    # a composite of several sentences is translated sentence by sentence
    out = i18n.tr("Over 02 Nov 2025 – 01 Nov 2026 the average price was <b>€129/MWh</b>. The cheapest hour was <b>€-79</b> (26 Apr); <b>2.0%</b> of hours were negative.")
    assert "€129/MWh" in out and "Най-евтиният час" in out and "cheapest" not in out
    # dates formatted by Python in English are converted when the sentence is translated
    assert "ное" in i18n.tr("Data through 01 Nov 2026") or "Данни до" in i18n.tr("Data through 01 Nov 2026")


def test_unknown_text_falls_back_to_english_and_is_recorded(bg):
    s = "A sentence that nobody translated yet, with 42 numbers"
    assert i18n.tr(s) == s
    assert s in i18n.misses()


def test_non_strings_and_english_pass_through(monkeypatch):
    monkeypatch.setattr(i18n, "lang", lambda: "en")
    assert i18n.tr("Forecast") == "Forecast"
    assert i18n.tr(3.5) == 3.5 and i18n.tr(None) is None


def test_dates_only_convert_real_date_words(bg):
    assert i18n.dates("Tuesday 03 November 2026") == "вторник 03 ноември 2026"
    assert i18n.dates("Mon 03 Nov") == "пон 03 ное"
    assert i18n.dates("You may call it maybe") == "You may call it maybe"      # prose is not a date


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def _contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_text_and_chrome_contrast_meets_wcag_aa(mode):
    import theme as T
    p = T.PALETTES[mode]
    for ink in ("TEXT", "TEXT_2", "MUTED"):
        for surface in ("SURFACE", "BG"):
            assert _contrast(p[ink], p[surface]) >= 4.5, (mode, ink, surface)
    assert _contrast(p["ACCENT_UI"], p["BG"]) >= 4.5, mode          # eyebrows, links and active states use the amber chrome as text
    assert _contrast(p["NEUTRAL"], p["SURFACE"]) >= 3.0, mode        # baseline series are non-text marks


def test_palettes_have_the_same_keys_and_resolve_per_mode(monkeypatch):
    import theme as T
    assert set(T.PALETTES["dark"]) == set(T.PALETTES["light"])
    monkeypatch.setattr(T, "mode", lambda: "light")
    assert T.TEXT == T.PALETTES["light"]["TEXT"] and T.MODEL["lightgbm"] == T.PALETTES["light"]["YELLOW"]
    monkeypatch.setattr(T, "mode", lambda: "dark")
    assert T.TEXT == T.PALETTES["dark"]["TEXT"]

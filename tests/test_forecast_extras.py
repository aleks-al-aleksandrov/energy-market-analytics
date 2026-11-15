"""The Forecast page's 15-minute chart and model-blend card only render once the pipeline has written their files.

The committed reports/ may not contain them yet, so this test writes small fixtures for the current delivery day, renders the
page in both configurations, and removes them again (files that were already there are left untouched).
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS, PAGE = ROOT / "reports", ROOT / "app" / "views" / "3_Forecast.py"


@pytest.fixture
def extras():
    nd_path = REPORTS / "forecast_next_day.csv"
    if not nd_path.exists():
        pytest.skip("no next-day forecast in reports/")
    nd = pd.read_csv(nd_path)
    day = str(nd["delivery_day"].iloc[0])
    q_path, e_path = REPORTS / "forecast_next_day_15min.csv", REPORTS / "ensemble_live.json"
    created = []
    if not q_path.exists():
        ts = pd.to_datetime(nd["timestamp_utc"], utc=True)
        idx = pd.DatetimeIndex([t + pd.Timedelta(minutes=15 * k) for t in ts for k in range(4)])
        shape = np.tile([-6.0, -2.0, 2.0, 6.0], len(nd))
        rep = nd.loc[nd.index.repeat(4)].reset_index(drop=True)
        q = pd.DataFrame({"timestamp_utc": idx, "forecast_eur_mwh": rep["forecast_eur_mwh"] + shape, "p10_eur_mwh": rep["p10_eur_mwh"] + shape - 8,
                          "p90_eur_mwh": rep["p90_eur_mwh"] + shape + 8, "shape_eur_mwh": shape,
                          "time_local": idx.tz_convert("Europe/Sofia").strftime("%Y-%m-%d %H:%M"), "delivery_day": day})
        q.to_csv(q_path, index=False)
        created.append(q_path)
    if not e_path.exists():
        e_path.write_text(json.dumps({"delivery_day": day, "weights": {"lightgbm": 0.22, "ridge": 0.28, "level": 0.30, "lear": 0.20}, "window_days": 14}))
        created.append(e_path)
    try:
        yield
    finally:
        for p in created:
            p.unlink(missing_ok=True)


@pytest.mark.parametrize("lang,theme", [("EN", "Dark"), ("BG", "Light")], ids=lambda v: v)
def test_forecast_page_renders_the_quarter_chart_and_the_blend(extras, lang, theme):
    at = AppTest.from_file(str(PAGE), default_timeout=180)
    at.session_state["_ui_init"] = True
    at.session_state["ui_lang"], at.session_state["ui_theme"] = lang, theme
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert ("Inside the hour" in html) if lang == "EN" else ("Вътре в часа" in html)       # the 15-minute card is there, and translated
    assert ("Tomorrow leans most on" in html) if lang == "EN" else ("Утре най-голяма тежест" in html)

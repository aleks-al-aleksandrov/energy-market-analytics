"""About / Method."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import theme as T

T.setup("About")
T.header("About", "Method, data and limitations",
         "How the numbers on every page are produced, what the models are allowed to know, and where they should not be trusted.")

steps = [("01", "Ingest", "ENTSO-E prices, load, wind/solar forecasts, generation, flows · Open-Meteo weather · monthly Parquet cache"),
         ("02", "Clean", "UTC index · 15-min → hourly mean · gaps ≤ 3 h interpolated · negative prices kept & flagged"),
         ("03", "Features", "Only what is known at 12:00 CET on D-1: price lags, neighbours D-1, ENTSO-E forecasts, D-1 weather, calendar"),
         ("04", "Models", "Naive ×2 · Ridge · LightGBM (Optuna) · MLP · ensemble · quantiles · spike classifier"),
         ("05", "Evaluate", "Expanding-window walk-forward · 3-month untouched holdout · MAE, skill, Diebold-Mariano"),
         ("06", "Value", "Battery LP settled at actual prices · SHAP · drift monitoring · next-day forecast")]
st.markdown("<div class='flow'>" + "<span class='arrow'>→</span>".join(
    f"<div class='step'><div class='n'>{n}</div><div class='t'>{t}</div><div class='d'>{d}</div></div>" for n, t, d in steps)
    + "</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("""<div class='info-card'><h3>Data</h3><ul>
<li><b>ENTSO-E Transparency Platform</b> (entsoe-py): day-ahead prices BG/RO/GR/HU, actual load, day-ahead load forecast,
day-ahead wind/solar forecast, actual generation per type, physical flows BG ↔ RO, GR, RS, MK, TR.</li>
<li><b>Open-Meteo</b>: observed weather (Historical Weather API) and forecasts issued the day before (Previous Runs API, D-1 run)
for Sofia (demand), Dobrich (wind) and Stara Zagora (solar).</li>
<li><b>Calendar</b>: Bulgarian public holidays and bridge days (<code>holidays</code>).</li></ul></div>
<div class='info-card'><h3>Cleaning</h3><ul>
<li>Everything is stored in <b>UTC</b>; 23/25-hour DST days need no special case. Display is Europe/Sofia.</li>
<li>15-minute data (BG day-ahead since October 2025) is <b>averaged to hourly</b>.</li>
<li>Gaps ≤ 3 h are linearly interpolated; longer gaps stay missing and those hours are dropped from training.</li>
<li>Negative and extreme prices are <b>kept</b> and flagged — they are real.</li></ul></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class='info-card'><h3>Forecasting</h3><ul>
<li>Target: hourly BG day-ahead price for delivery day D, forecast at gate closure (12:00 CET on D-1).</li>
<li><b>Expanding-window walk-forward</b>: retrain monthly, test on the next month. The last 3 months are a
<b>final holdout</b> never used for tuning, ensemble weights, thresholds or model choice.</li>
<li>Metrics: MAE, RMSE, skill vs best naive, Diebold-Mariano, pinball loss and coverage, PR-AUC with a cost-based threshold.</li></ul></div>
<div class='info-card'><h3>Limitations</h3><ul>
<li>No hydro reservoir levels. Interconnector capacities, REMIT outages, and fuel and carbon prices (monthly gas, daily TTF, Brent,
a carbon proxy) were tested and did not improve the forecast (capacity and fuels failed the pre-set CV-only adoption rule, so they are not used). Daily TTF history is only 16 months deep.</li>
<li>ENTSO-E wind/solar day-ahead forecasts may be published after gate closure (legal deadline 18:00 D-1); treated as
available because traders hold equivalent vendor forecasts.</li>
<li>Conformal-calibrated bands reach 80 % coverage on CV but only 68 % on the holdout, where the autumn-2026 surge pushed prices above the band.</li>
<li>Battery: day-ahead only, price-taker, no network fees or intraday revenue; degradation is a configurable cost and reserve capacity is indicative only.</li>
<li><b>Design and accessibility:</b> chart colours are a validated colour-blind-safe palette with a fixed order, every chart has a table view and CSV download, text meets WCAG AA contrast, and charts carry screen-reader descriptions.</li></ul></div>""",
                unsafe_allow_html=True)
st.markdown("<div class='muted'>The dashboard reads only precomputed files in <code>reports/</code> and never calls the ENTSO-E API. "
            "Rebuild everything with <code>python -m src.run_all --plus</code>.</div>", unsafe_allow_html=True)

"""Home: what this dashboard is, what it found, and where to go first."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import theme as T
from i18n import tr

T.setup("Home")
m = T.load_market()
nd = T.load_csv("forecast_next_day.csv")
sc = T.load_csv("skill_confidence.csv")
bat = T.load_csv("battery_results.csv")
mc = T.load_json("mc_summary.json")

# ---- hero --------------------------------------------------------------------------------------------------------------------
hero_nums = []
if nd is not None and len(nd) == 24:
    day = pd.Timestamp(nd["delivery_day"].iloc[0])
    hero_nums.append((f"€{nd['forecast_eur_mwh'].mean():.0f}", f"expected average price on {day:%d %b}"))
if sc is not None:
    g = sc[sc["variant"] == "lightgbm"].set_index("split")
    if "cv" in g.index:
        hero_nums.append((f"{100 * g.loc['cv', 'skill']:+.0f}%", "forecast error vs the naive baseline (validation months)"))
if bat is not None and (bat["strategy"] == "forecast_lightgbm").any():
    b = bat.set_index("strategy").loc["forecast_lightgbm"]
    hero_nums.append((f"€{b['eur_per_mw_year'] / 1000:.0f}k", f"per MW-year for a battery guided by the forecast ({b['pct_of_perfect']:.0f}% of perfect)"))
nums = "".join(f"<div class='hn'><b>{v}</b><span>{tr(t)}</span></div>" for v, t in hero_nums)
art = ""
if nd is not None and len(nd) == 24:
    from signature import level_color
    _f = nd["forecast_eur_mwh"].to_numpy()
    _lo, _sp = float(_f.min()), (float(_f.max() - _f.min()) or 1.0)
    art = "<div class='hero-art' aria-hidden='true'>" + "".join(
        f"<i style='--h:{28 + 72 * (v - _lo) / _sp:.0f}%;--c:{level_color((v - _lo) / _sp)};--i:{k}'></i>" for k, v in enumerate(_f)) + "</div>"
    art += f"<div class='hero-art-cap'>{tr('Tomorrow, hour by hour')}</div>"
st.markdown(
    f"<section class='hero'><div><div class='hero-eyebrow'>{tr('Bulgarian day-ahead power market')}</div>"
    f"<h1>{tr('Tomorrow’s electricity price, forecast in the open')}</h1>"
    f"<p class='hero-lede'>{tr('Hourly price forecasts for Bulgaria, tested without peeking at the answer, and a plain account of what flexibility such as a battery could earn from them. Every number on every page is computed from the data, and the models that lost are shown next to the one that won.')}</p>"
    f"<div class='hero-nums'>{nums}</div></div>{art}</section>", unsafe_allow_html=True)
with st.container(key="herocta"):
    c1, c2, c3, _ = st.columns([1.1, 1.1, 1.1, 2.2])
    with c1:
        T.link("views/0_Trader_Brief.py", "See tomorrow", icon=":material/bolt:")
    with c2:
        T.link("overview.py", "Explore the market", icon=":material/monitoring:")
    with c3:
        T.link("views/8_Beat_the_Model.py", "Play: beat the model", icon=":material/sports_esports:")

# ---- what is inside ----------------------------------------------------------------------------------------------------------
T.section("What you will find here", "Eight pages, from the market itself to what the forecasts are worth", anchor="inside")
CARDS = [
    ("views/0_Trader_Brief.py", ":material/assignment:", "Trader Brief", "Tomorrow’s forecast turned into actions, with how far to trust it."),
    ("overview.py", ":material/monitoring:", "Market Overview", "Prices since 2023, the daily shape, negative hours and the calendar."),
    ("views/1_Price_Drivers.py", ":material/bolt:", "Price Drivers", "Why prices move: solar, residual load, gas and carbon."),
    ("views/2_Cross_Border.py", ":material/compare_arrows:", "Cross-Border", "How tightly Bulgaria follows Romania, Greece and Hungary."),
    ("views/3_Forecast.py", ":material/query_stats:", "Forecast", "The back-test, the error map and the live track record."),
    ("views/5_Model_Lab.py", ":material/science:", "Model Lab", "Every model tried, with significance tests, SHAP and drift."),
    ("views/6_Battery.py", ":material/battery_charging_full:", "Battery Simulation", "What a battery earns, with a Monte Carlo revenue range."),
    ("views/4_Weather_Uncertainty.py", ":material/partly_cloudy_day:", "Weather Uncertainty", "Does knowing the weather better improve the forecast?"),
]
with st.container(key="homecards"):
    for row in range(0, len(CARDS), 4):
        cols = st.columns(4)
        for col, (path, icon, title, why) in zip(cols, CARDS[row:row + 4]):
            with col:
                T.link(path, title, icon=icon)
                st.caption(tr(why))

# ---- how it works --------------------------------------------------------------------------------------------------------------
T.section("How it works", "From raw market data to a decision, in four steps", anchor="how")
steps = [("01", "Collect", "Prices, load, wind and solar forecasts, flows and weather from ENTSO-E and Open-Meteo."),
         ("02", "Forecast", "Models that only use what is known at 12:00 on the day before delivery."),
         ("03", "Test", "Walk-forward back-test plus a final three months that were never used to choose anything."),
         ("04", "Value", "Settle a battery schedule at the real prices to see what the forecast is worth.")]
st.markdown("<div class='flow'>" + "<span class='arrow'>→</span>".join(
    f"<div class='step'><div class='n'>{n}</div><div class='t'>{tr(t)}</div><div class='d'>{tr(d)}</div></div>" for n, t, d in steps) + "</div>", unsafe_allow_html=True)

# ---- honest by design ---------------------------------------------------------------------------------------------------------
T.section("Honest by design", "What makes the numbers trustworthy, and where they stop", anchor="honest")
hold = ""
if sc is not None:
    g = sc[sc["variant"] == "lightgbm"].set_index("split")
    if {"cv", "holdout"} <= set(g.index):
        c, h = g.loc["cv"], g.loc["holdout"]
        hold = (f"On the validation months the model beat the naive baseline by <b>{100 * c['skill']:.0f}%</b>; on the untouched last three months the "
                f"edge shrank to <b>{100 * h['skill']:.0f}%</b> and its interval ({100 * h['skill_lo']:.0f} to {100 * h['skill_hi']:.0f}%) includes zero.")
k1, k2, k3 = st.columns(3)
with k1:
    T.takeaway("<b>No peeking.</b> Every feature is known by gate closure, and a leakage test fails the build if that ever changes.")
with k2:
    T.takeaway("<b>Losers stay visible.</b> Fuel prices, plant outages and interconnector capacity were tested and did not help, so they are not used.", "warning")
with k3:
    T.takeaway("<b>Small gains are called small.</b> " + hold if hold else "<b>Small gains are called small.</b> Intervals are shown, and a result that includes zero is labelled that way.", "action")

# ---- quick tips ----------------------------------------------------------------------------------------------------------------
T.section("Good to know", "A few things that make the dashboard easier to use", anchor="tips")
t1, t2, t3, t4 = st.columns(4)
for col, (icon, title, text) in zip((t1, t2, t3, t4), (
        ("◐", "Light or dark", "Open Settings at the top right to switch theme or language; the link keeps your choice."),
        ("▦", "Every chart has a table", "Open Table view under any chart for the numbers and a CSV download."),
        ("▶", "Press play", "The Overview animates how the daily price shape changed month by month."),
        ("?", "Ask the market", "On the Overview, pick a question and tomorrow’s forecast answers it in plain words."))):
    col.markdown(f"<div class='tip'><span class='tip-i' aria-hidden='true'>{icon}</span><b>{tr(title)}</b><p>{tr(text)}</p></div>", unsafe_allow_html=True)

through = f"{m.index.max():%d %b %Y}" if m is not None else "n/a"
st.markdown(f"<p class='home-foot'>{tr('Data through')} {through} · ENTSO-E Transparency Platform, Open-Meteo · "
            f"{tr('Forecasts and analysis only. This is not trading or investment advice.')}</p>", unsafe_allow_html=True)

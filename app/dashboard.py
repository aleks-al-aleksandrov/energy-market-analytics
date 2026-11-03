"""Dashboard entry point:  streamlit run app/dashboard.py"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st

import auth
import theme as T
from i18n import tr

T.bootstrap()
auth.gate(T.login_page)

st.set_page_config(page_title=tr("BG Power"), page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
P = HERE / "views"
nav = st.navigation({
    tr("Start"): [st.Page(HERE / "home.py", title=tr("Home"), icon=":material/home:", default=True),
                  st.Page(HERE / "overview.py", url_path="overview", title=tr("Market Overview"), icon=":material/monitoring:")],
    tr("Market"): [st.Page(P / "0_Trader_Brief.py", url_path="brief", title=tr("Trader Brief"), icon=":material/assignment:"),
                   st.Page(P / "1_Price_Drivers.py", url_path="price-drivers", title=tr("Price Drivers"), icon=":material/bolt:"),
                   st.Page(P / "2_Cross_Border.py", url_path="cross-border", title=tr("Cross-Border"), icon=":material/compare_arrows:")],
    tr("Models"): [st.Page(P / "3_Forecast.py", url_path="forecast", title=tr("Forecast"), icon=":material/query_stats:"),
                   st.Page(P / "4_Weather_Uncertainty.py", url_path="weather", title=tr("Weather Uncertainty"), icon=":material/partly_cloudy_day:"),
                   st.Page(P / "5_Model_Lab.py", url_path="model-lab", title=tr("Model Lab"), icon=":material/science:")],
    tr("Value"): [st.Page(P / "6_Battery.py", url_path="battery", title=tr("Battery Simulation"), icon=":material/battery_charging_full:"),
                  st.Page(P / "8_Beat_the_Model.py", url_path="play", title=tr("Beat the Model"), icon=":material/sports_esports:"),
                  st.Page(P / "7_About.py", url_path="about", title=tr("About / Method"), icon=":material/info:")],
}, position="top")
nav.run()

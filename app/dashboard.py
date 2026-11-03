"""Dashboard entry point:  streamlit run app/dashboard.py"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st

st.set_page_config(page_title="BG Power", page_icon="⚡", layout="wide")
P = HERE / "views"
nav = st.navigation({
    "Market": [st.Page(HERE / "overview.py", title="Market Overview", icon=":material/monitoring:", default=True),
               st.Page(P / "0_Trader_Brief.py", url_path="brief", title="Trader Brief", icon=":material/assignment:"),
               st.Page(P / "1_Price_Drivers.py", url_path="price-drivers", title="Price Drivers", icon=":material/bolt:"),
               st.Page(P / "2_Cross_Border.py", url_path="cross-border", title="Cross-Border", icon=":material/compare_arrows:")],
    "Models": [st.Page(P / "3_Forecast.py", url_path="forecast", title="Forecast", icon=":material/query_stats:"),
               st.Page(P / "4_Weather_Uncertainty.py", url_path="weather", title="Weather Uncertainty", icon=":material/partly_cloudy_day:"),
               st.Page(P / "5_Model_Lab.py", url_path="model-lab", title="Model Lab", icon=":material/science:")],
    "Value": [st.Page(P / "6_Battery.py", url_path="battery", title="Battery Simulation", icon=":material/battery_charging_full:"),
              st.Page(P / "7_About.py", url_path="about", title="About / Method", icon=":material/info:")],
})
nav.run()

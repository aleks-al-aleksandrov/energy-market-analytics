"""Trader brief: tomorrow's forecast turned into actions, with how much to trust it. Every number comes from reports/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T
from src import battery as B

T.setup("Trader Brief")
nd = T.load_csv("forecast_next_day.csv")
m = T.load_market()
if nd is None or m is None or not len(nd):
    T.header("Decide", "Trader brief")
    T.missing("Next-day forecast (python -m src.predict_next_day)")
    st.stop()

day = pd.Timestamp(nd["delivery_day"].iloc[0])
f = nd["forecast_eur_mwh"].to_numpy()
hrs = nd["time_local"].str[-5:].tolist()
base = float(np.mean(f))
p10, p90 = nd["p10_eur_mwh"].to_numpy(), nd["p90_eur_mwh"].to_numpy()
band_rel = float(np.mean(p90 - p10) / base)
price = m["price_BG"].dropna()
last7, last30, last365 = price.tail(24 * 7).mean(), price.tail(24 * 30).mean(), price.tail(24 * 365).mean()
spike = nd["spike_probability"].to_numpy()

T.header("Decide", f"Trader brief · delivery {day:%A %d %B %Y}",
         "What the forecast means for tomorrow, what to do about it, and how far to trust it. Rule-based and computed from the project's own "
         "forecasts and back-tests — a decision aid, not trading advice.",
         [f"Model trained until {nd['model_trained_until'].iloc[0]}", f"Generated {nd['generated_at'].iloc[0]}", "Gate closure 12:00 CET on D-1"],
         headline=(f"Expected base price <b>€{base:.0f}/MWh</b> ({100 * (base / last7 - 1):+.0f}% vs the last 7 days), cheapest at <b>{hrs[int(f.argmin())]}</b> "
                   f"(€{f.min():.0f}) and dearest at <b>{hrs[int(f.argmax())]}</b> (€{f.max():.0f}). Highest spike probability: {100 * spike.max():.0f}% at {hrs[int(spike.argmax())]}."))

# --- headline numbers -------------------------------------------------------------------------------------
k = st.columns(5)
T.kpi(k[0], "Expected base price", T.fmt_eur(base), "24h average, EUR/MWh", accent=True, spark=f,
      delta=T.delta_chip(100 * (base / last7 - 1), "% vs last 7 days"))
peak = f[7:19].mean() if len(f) >= 19 else f.mean()
T.kpi(k[1], "Peak block 08–20", T.fmt_eur(peak), f"{(peak - base) / base * 100:+.0f}% vs base")
T.kpi(k[2], "Daily spread", T.fmt_eur(f.max() - f.min(), 0), f"{hrs[int(f.argmin())]} low → {hrs[int(f.argmax())]} high")
T.kpi(k[3], "Uncertainty", f"±{band_rel * 50:.0f}%", f"10–90 % band is {np.mean(p90 - p10):.0f} EUR wide")
level = "high" if spike.max() > 0.2 else "medium" if spike.max() > 0.08 else "low"
T.kpi(k[4], "Spike risk", f"{100 * spike.max():.0f}%", f"<span style='color:{T.RISK[level]}'>●</span> {level} · peak hour {hrs[int(spike.argmax())]}")

# --- chart ------------------------------------------------------------------------------------------------------
net = B.schedule_day(f, B.Battery())
with T.card(f"Tomorrow: a €{f.max() - f.min():.0f} intraday spread, {'enough' if net.any() else 'not enough'} to cycle a battery",
            "Forecast with 10–90 % band · shaded bars = where a 1 MW / 2 MWh battery would charge (blue) and discharge (red)",
            T.legend_html([("Forecast", T.ACCENT), ("10–90 % band", T.rgba(T.ACCENT, .4)), ("Charge", T.DOWN), ("Discharge", T.UP)])):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hrs, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hrs, y=p10, fill="tonexty", fillcolor=T.rgba(T.ACCENT, .16), line=dict(width=0), name="band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hrs, y=f, name="forecast", line=dict(color=T.ACCENT, width=2.8),
                             hovertemplate="%{x} · <b>%{y:.0f}</b> EUR/MWh<extra></extra>"))
    for i, v in enumerate(net):
        if abs(v) > 1e-6:
            fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor=T.rgba(T.UP if v > 0 else T.DOWN, .14), line_width=0, layer="below")
    fig.update_xaxes(title="hour (local)", type="category", dtick=2)
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 380, legend=False)


# 24-hour risk strip: spike probability per hour, coloured by the same thresholds as the KPI
with T.card("Spike risk by hour", "Probability that the price exceeds recent level + the training-P95 excess · thresholds: 8 % medium, 20 % high",
            T.legend_html([("Low", T.RISK["low"]), ("Medium", T.RISK["medium"]), ("High", T.RISK["high"])], "box")):
    lvl_of = lambda v: "high" if v > 0.2 else "medium" if v > 0.08 else "low"
    fig = go.Figure(go.Bar(x=hrs, y=100 * spike, marker=dict(color=[T.RISK[lvl_of(v)] for v in spike], cornerradius=3),
                           customdata=[lvl_of(v) for v in spike],
                           hovertemplate="%{x} · <b>%{y:.1f}%</b> (%{customdata})<extra></extra>"))
    fig.add_hline(y=8, line=dict(color=T.MUTED, width=1, dash="dot"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="hour (local)", type="category", dtick=2, showspikes=False)
    fig.update_yaxes(title="spike probability (%)", rangemode="tozero")
    T.chart(fig, 220, legend=False)

# --- the playbook: rule-based actions -------------------------------------------------------------------------
T.section("What to do", "Rule-based reading of tomorrow's forecast")
charge = [hrs[i] for i, v in enumerate(net) if v < -1e-6]
sell = [hrs[i] for i, v in enumerate(net) if v > 1e-6]
rev = float(np.sum(net * f))
actions = []
if charge and sell:
    actions.append(("Battery", f"Charge {', '.join(charge)}; discharge {', '.join(sell)}.",
                    f"Expected €{rev:,.0f} per MW at forecast prices (1 MW / 2 MWh, 90 % efficiency, one cycle)."))
else:
    actions.append(("Battery", "No profitable cycle at forecast prices.", "The forecast spread does not cover the round-trip losses."))
gap = 100 * (base / last30 - 1)
if gap > 15:
    actions.append(("Price regime", f"Tomorrow's forecast is {gap:+.0f}% above the 30-day average (€{last30:.0f}): treat it as a high-price regime.",
                    "Tree models under-forecast when prices exceed what they were trained on; lean towards the upper half of the band."))
elif gap < -15:
    actions.append(("Price regime", f"Tomorrow's forecast is {gap:+.0f}% below the 30-day average (€{last30:.0f}).",
                    "Check for a one-off driver (holiday, very high solar) before assuming a lasting move."))
else:
    actions.append(("Price regime", f"Forecast is within {abs(gap):.0f}% of the 30-day average (€{last30:.0f}).", "No regime-shift warning."))
if "forecast_level_eur_mwh" in nd:
    lvl = float(nd["forecast_level_eur_mwh"].mean())
    blend = float(nd["forecast_blend_eur_mwh"].mean())
    gap_m = 100 * (lvl / base - 1)
    if abs(gap_m) > 10:
        actions.append(("Model disagreement", f"The level-adjusted cross-check says €{lvl:.0f} against €{base:.0f} ({gap_m:+.0f}%); a 50/50 blend gives €{blend:.0f}.",
                        "The two models disagree by more than 10 %: the price level is moving. The cross-check did better in the October-2026 surge but worse in most "
                        "earlier level shifts, so treat the blend as a hedge, not as a better forecast."))
    else:
        actions.append(("Model agreement", f"Level-adjusted cross-check (€{lvl:.0f}) is within {abs(gap_m):.0f}% of the main forecast.", "No sign of a regime problem."))
if band_rel > 0.25:
    actions.append(("Hedging", f"Wide band (±{band_rel * 50:.0f}% of the price).", "Do not rely on the point forecast; hedge more of the open position or cut size."))
else:
    actions.append(("Hedging", f"Band is moderate (±{band_rel * 50:.0f}%).", "The point forecast is usable for sizing; keep normal risk limits."))
hot = [hrs[i] for i in range(len(spike)) if spike[i] > 0.08]
actions.append(("Spike watch", f"Hours above 8 % spike probability: {', '.join(hot) if hot else 'none'}.",
                f"Spike = price more than the P95 excess ({nd['spike_definition'].iloc[0] if 'spike_definition' in nd else 'recent level + excess'})."))
lowest = int(f.argmin())
actions.append(("Midday floor", f"Cheapest forecast hour is {hrs[lowest]} at €{f[lowest]:.0f}.",
                "Cheap hours (< €5) happen on sunny weekends and holidays; avoid selling into them and consider charging."))
for title, headline, why in actions:
    st.markdown(f"<div class='takeaway action'><span class='ti'>→</span><div><b>{title}.</b> {headline}<br><span style='color:var(--muted)'>{why}</span></div></div>",
                unsafe_allow_html=True)

# --- context: gas, coupling ---------------------------------------------------------------------------------------
fd = T.load_csv("fuels_daily.csv", parse_dates=["date"])
cols = st.columns(3)
if fd is not None and "ttf" in fd and fd["ttf"].notna().any():
    s_ttf = fd.set_index("date")["ttf"].dropna()
    ch = 100 * (s_ttf.iloc[-1] / s_ttf.iloc[-min(30, len(s_ttf))] - 1)
    T.kpi(cols[0], "TTF gas", f"€{s_ttf.iloc[-1]:.0f}", f"/MWh · {ch:+.0f}% in 30 days · {s_ttf.index[-1]:%d %b}", spark=s_ttf.tail(90), spark_color=T.RED)
if "price_RO" in m:
    sp = (m["price_BG"] - m["price_RO"]).dropna()
    c30, c365 = (sp.tail(24 * 30).abs() < T.COUPLED_TOL).mean() * 100, (sp.tail(24 * 365).abs() < T.COUPLED_TOL).mean() * 100
    T.kpi(cols[1], "Coupling with Romania", f"{c30:.0f}%", f"hours within €1 · 12-month {c365:.0f}%",
          delta=T.delta_chip(c30 - c365, " pp", good_when_up=True, digits=0))
T.kpi(cols[2], "BG price level", f"€{last7:.0f}", f"7-day average · 30-day €{last30:.0f} · 12-month €{last365:.0f}")

# --- how much to trust it ---------------------------------------------------------------------------------------------
T.section("How much to trust it", "Recent performance of this model on days it had not seen")
pr = T.load_predictions()
if pr is not None:
    rec = pr[pr["delivery_day"] >= pr["delivery_day"].max() - pd.Timedelta(days=29)].dropna(subset=["actual", "lightgbm", "naive_d1"])
    mae_m, mae_n = (rec["actual"] - rec["lightgbm"]).abs().mean(), (rec["actual"] - rec["naive_d1"]).abs().mean()
    bias = (rec["actual"] - rec["lightgbm"]).mean()
    lo_c, hi_c = ("q10_cal", "q90_cal") if "q10_cal" in rec else ("q10", "q90")
    cov = ((rec["actual"] >= rec[lo_c]) & (rec["actual"] <= rec[hi_c])).mean()
    k = st.columns(4)
    T.kpi(k[0], "Back-test MAE, last 30 days", T.fmt_eur(mae_m), f"naive €{mae_n:.1f}",
          delta=T.delta_chip(100 * (mae_m - mae_n) / mae_n, "% vs naive", good_when_up=False), accent=True)
    T.kpi(k[1], "Average bias", f"{bias:+.1f}", "actual − forecast · positive = forecast too low")
    T.kpi(k[2], "Band coverage", f"{100 * cov:.0f}%", "target 80 %")
    tr = T.load_csv("live_track_record.csv")
    T.kpi(k[3], "Live days scored", f"{0 if tr is None else len(tr)}", "forecasts made before gate closure")
    if abs(bias) > 10:
        T.takeaway(f"Over the last 30 back-tested days the model was off by <b>{bias:+.0f} €/MWh on average</b>. "
                   f"Adjust your view of tomorrow's level accordingly ({'upward' if bias > 0 else 'downward'}): the bias comes from the price level moving "
                   "faster than the model can follow.", "warning")
T.takeaway("<b>Known blind spots:</b> plant outages (Kozloduy nuclear, lignite units), reservoir levels, sudden gas moves inside the day, and "
           "any interconnector constraint announced after the forecast was made. None of these are in the model — check them manually before acting.", "warning")

# --- general playbook (what the project found) -------------------------------------------------------------------------
T.section("Playbook for common situations", "What the 2023–2026 data say, and what a trader or analyst can do about it")
# computed, not asserted: which hours rose most between the same 61 days one year ago and now
now_w = price.index.max()
cur = price[price.index > now_w - pd.Timedelta(days=61)]
prev = price[(price.index > now_w - pd.Timedelta(days=61 + 365)) & (price.index <= now_w - pd.Timedelta(days=365))]
hr = (cur.groupby(cur.index.hour).mean() - prev.groupby(prev.index.hour).mean()).dropna()
gas_row = ("Gas price spike", f"Over the last 61 days vs a year ago prices rose by about €{hr.loc[0:5].mean():.0f} at night, "
           f"€{hr.loc[17:22].mean():.0f} in the evening and €{hr.loc[10:15].mean():.0f} at midday",
           "Baseload (night) exposure moved most: hedge that first; expect an upward forecast bias and lean to the upper band")
rows = [
    gas_row,
    ("Sunny weekend or holiday", "Midday prices fall towards zero or below (negative hours peak at 14:00, most often in spring)", "Charge a battery at midday, avoid selling into the trough, sell in the evening"),
    ("Low Romanian import capacity", "BG decouples from Romania: the average price gap was about twice as large when import capacity was in its lowest 20 %", "Treat BG as a local market that day; widen risk limits"),
    ("High wind forecast in the north-east", "Lower residual load (the wind forecast is among the top price drivers)", "Wind forecast misses mostly hit intraday and imbalance prices, not the day-ahead price"),
    ("Price level far above recent history", "Tree models stop extrapolating and under-forecast", "Use the level-adjusted forecast as a cross-check (it did best in Oct 2026)"),
]
T.table(pd.DataFrame(rows, columns=["Situation", "What happens", "What to do"]).set_index("Situation"))

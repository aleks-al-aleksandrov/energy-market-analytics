"""Forecast: next-day card, back-test explorer with calibrated band, diagnostics, live track record."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import theme as T

T.setup("Forecast")
top = st.container()
nd = T.load_csv("forecast_next_day.csv")
p = T.load_predictions()

rec_txt = ""
if p is not None:
    rec = p[p["delivery_day"] >= p["delivery_day"].max() - pd.Timedelta(days=29)].dropna(subset=["actual", "lightgbm", "naive_d1"])
    if len(rec):
        mm, mn = (rec["actual"] - rec["lightgbm"]).abs().mean(), (rec["actual"] - rec["naive_d1"]).abs().mean()
        rec_txt = f" Over the last 30 back-tested days its error was <b>€{mm:.0f}/MWh</b> against €{mn:.0f} for the naive 'same hour yesterday' baseline."
nd_txt = ""
if nd is not None and len(nd):
    nd_txt = (f"Tomorrow ({pd.Timestamp(nd['delivery_day'].iloc[0]):%d %b}) the model expects a base price of <b>€{nd['forecast_eur_mwh'].mean():.0f}/MWh</b> "
              f"(80% band €{nd['p10_eur_mwh'].mean():.0f}–{nd['p90_eur_mwh'].mean():.0f}).")
with top:
    T.header("Models", "Day-ahead price forecast",
             "Every forecast uses only what is known at gate closure (12:00 CET on D-1). The back-test is walk-forward: each month is predicted by a model trained on earlier data only.",
             ["Gate-closure information set", "Monthly expanding-window retrain", "Last 3 months = untouched holdout", "Bands: conformal-calibrated"],
             headline=(nd_txt + rec_txt).strip())

T.toc([("next-day", "Next day"), ("explorer", "Back-test explorer"), ("failures", "Where it fails"), ("live", "Live track record")])

# --- next-day forecast -------------------------------------------------------------------------------------------------------------------
if nd is not None and len(nd):
    T.section(f"Next-day forecast · delivery {pd.Timestamp(nd['delivery_day'].iloc[0]):%A %d %B %Y}", anchor="next-day", subtitle=
              f"Generated {nd['generated_at'].iloc[0]} · model trained until {nd['model_trained_until'].iloc[0]} · <code>python -m src.predict_next_day</code>")
    f = nd["forecast_eur_mwh"]
    hrs = nd["time_local"].str[-5:]
    peak = f.iloc[7:19] if len(f) >= 19 else f
    has_actual = "actual_eur_mwh" in nd and nd["actual_eur_mwh"].notna().any()
    k = st.columns(5)
    T.kpi(k[0], "Base (24h avg)", T.fmt_eur(f.mean()), f"band €{nd['p10_eur_mwh'].mean():.0f} – €{nd['p90_eur_mwh'].mean():.0f}", accent=True, spark=f)
    T.kpi(k[1], "Peak block 08–20", T.fmt_eur(peak.mean()), "average of peak hours")
    T.kpi(k[2], "Highest hour", T.fmt_eur(f.max(), 0), hrs.iloc[int(f.values.argmax())])
    T.kpi(k[3], "Daily spread", T.fmt_eur(f.max() - f.min(), 0), "max − min · battery opportunity")
    if has_actual:
        err = nd["actual_eur_mwh"] - f
        T.kpi(k[4], "vs published price", T.fmt_eur(err.abs().mean()), f"MAE · bias {err.mean():+.0f} €/MWh", delta=T.delta_chip(100 * err.mean() / nd["actual_eur_mwh"].mean(), "%"))
    else:
        sp = nd["spike_probability"].max()
        level = "high" if sp > 0.5 else "medium" if sp > 0.2 else "low"
        T.kpi(k[4], "Spike risk", f"{100 * sp:.0f}%", f"<span style='color:{T.RISK[level]}' aria-hidden='true'>●</span> {level} · "
              f"P({nd['spike_definition'].iloc[0] if 'spike_definition' in nd else 'price > ' + format(nd['spike_threshold_eur'].iloc[0], '.0f')})")
    items = [("Forecast", T.ACCENT), ("10–90 % band (calibrated)", T.rgba(T.ACCENT, .4))] + ([("Level-adjusted cross-check", T.VIOLET)] if "forecast_level_eur_mwh" in nd else []) \
        + ([("Published price", T.TEXT)] if has_actual else [])
    with T.card("Tomorrow's hourly curve", "Point forecast with the calibrated 10–90 % band; bars below show the spike probability per hour",
                T.legend_html(items), help="The cross-check forecasts the deviation from yesterday's average: it tracks level shifts better but is noisier."):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=hrs, y=nd["p90_eur_mwh"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=hrs, y=nd["p10_eur_mwh"], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .16), line=dict(width=0), name="P10–P90",
                                 customdata=nd["p90_eur_mwh"], hovertemplate="band %{y:.0f} – %{customdata:.0f}<extra></extra>"), 1, 1)
        if "forecast_level_eur_mwh" in nd:
            fig.add_trace(go.Scatter(x=hrs, y=nd["forecast_level_eur_mwh"], name="level-adjusted", line=dict(color=T.VIOLET, width=1.6),
                                     hovertemplate="level-adjusted %{y:.1f}<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=hrs, y=f, name="forecast", mode="lines+markers", line=dict(color=T.ACCENT, width=2.6),
                                 marker=dict(size=8, color=T.ACCENT, line=dict(color=T.SURFACE, width=2)), hovertemplate="forecast <b>%{y:.1f}</b><extra></extra>"), 1, 1)
        if has_actual:
            fig.add_trace(go.Scatter(x=hrs, y=nd["actual_eur_mwh"], name="published", line=dict(color=T.TEXT, width=1.8), hovertemplate="published <b>%{y:.1f}</b><extra></extra>"), 1, 1)
        sp_ = nd["spike_probability"]
        fig.add_trace(go.Bar(x=hrs, y=sp_ * 100, name="spike prob.", marker=dict(color=[T.RISK["high"] if v > .5 else T.RISK["medium"] if v > .2 else T.RISK["low"] for v in sp_], cornerradius=2),
                             hovertemplate="spike prob. <b>%{y:.0f}%</b><extra></extra>"), 2, 1)
        fig.update_yaxes(title="EUR/MWh", row=1, col=1)
        fig.update_yaxes(title="spike %", range=[0, 100], dtick=50, row=2, col=1)
        fig.update_xaxes(title="hour (local)", row=2, col=1)
        T.chart(fig, 430, legend=False)
    T.download(nd.set_index("time_local")[[c for c in nd.columns if c.startswith(("forecast", "p10", "p90", "spike"))]], "forecast-next-day", "Download tomorrow's forecast (CSV)", key="dl_nd")
    if has_actual:
        T.takeaway(f"The forecast was <b>€{abs(err.mean()):.0f}/MWh {'below' if err.mean() > 0 else 'above'}</b> the published price on average. Under-forecasting in the autumn-2026 "
                   "price surge is the known weakness: trees cannot extrapolate above the price levels seen in training.", "warning")
else:
    T.missing("Next-day forecast (python -m src.predict_next_day)")

# --- back-test explorer ----------------------------------------------------------------------------------------------------------------------
if p is None:
    T.missing("Back-test predictions")
    st.stop()
T.section("Back-test explorer", anchor="explorer", subtitle="Pick any window from the walk-forward back-test")
models = [c for c in ("lightgbm", "lightgbm_level", "lightgbm_blend", "ensemble", "ridge", "mlp", "naive_d1", "naive_w1", "lightgbm_nocapacity", "lightgbm_capacity",
                      "lightgbm_nofuel", "lightgbm_fuels") if c in p]
days = sorted(p["delivery_day"].dt.date.unique())
with T.controls("Window"):
    c1, c2, c3 = st.columns([1, 1, 2.4])
    day = c1.date_input("Start day", days[-7], min_value=days[0], max_value=days[-1])
    span = c2.segmented_control("Length", ["Day", "Week", "Month"], default="Week") or "Week"
    chosen = c3.multiselect("Models to compare", models, default=[m_ for m_ in ("lightgbm", "naive_d1") if m_ in models], max_selections=4,
                            format_func=lambda m_: T.MODEL_LABEL.get(m_, m_), help="Up to four models keep the chart readable.")
n = {"Day": 1, "Week": 7, "Month": 30}[span]
w = p[(p["delivery_day"].dt.date >= day) & (p["delivery_day"].dt.date < day + pd.Timedelta(days=n))]
split = w["split"].iloc[0] if len(w) else ""
lo_c, hi_c = ("q10_cal", "q90_cal") if "q10_cal" in w.columns and w["q10_cal"].notna().any() else ("q10", "q90")
mae_sel = {m_: (w["actual"] - w[m_]).abs().mean() for m_ in chosen} if len(w) else {}
best_sel = min(mae_sel, key=mae_sel.get) if mae_sel else None
with T.card(f"{T.MODEL_LABEL.get(best_sel, best_sel)} was closest in this {span.lower()} (MAE €{mae_sel[best_sel]:.0f})" if best_sel else "Actual vs predicted",
            "Holdout period: never used for any model choice" if split == "holdout" else "Walk-forward CV period",
            T.legend_html([("Actual", T.TEXT)] + [(T.MODEL_LABEL.get(m_, m_), T.MODEL.get(m_, T.MUTED)) for m_ in chosen] + [("10–90 % band", T.rgba(T.ACCENT, .4))])):
    fig = go.Figure()
    if {lo_c, hi_c} <= set(w.columns) and w[lo_c].notna().any():
        fig.add_trace(go.Scatter(x=w.index, y=w[hi_c], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=w.index, y=w[lo_c], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .10), line=dict(width=0), name="10–90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w.index, y=w["actual"], name="actual", line=dict(color=T.TEXT, width=2.4), hovertemplate="actual <b>%{y:.1f}</b><extra></extra>"))
    for m_ in chosen:
        fig.add_trace(go.Scatter(x=w.index, y=w[m_], name=T.MODEL_LABEL.get(m_, m_), line=dict(color=T.MODEL.get(m_, T.MUTED), width=1.8, dash="dot" if m_.startswith("naive") else "solid"),
                                 hovertemplate=T.MODEL_LABEL.get(m_, m_) + " %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 420, legend=False)
if len(w) and chosen:
    cols = st.columns(len(chosen) + 1)
    base = (w["actual"] - w["naive_d1"]).abs().mean() if "naive_d1" in w else None
    for c, m_ in zip(cols, chosen):
        e = mae_sel[m_]
        T.kpi(c, f"MAE · {T.MODEL_LABEL.get(m_, m_)}", T.fmt_eur(e), f"selected {span.lower()}", accent=m_ == "lightgbm",
              delta=T.delta_chip(100 * (e - base) / base, "% vs naive", good_when_up=False) if base and m_ != "naive_d1" else "")
    T.kpi(cols[-1], "Average price", T.fmt_eur(w["actual"].mean()), f"{len(w)} hours")

# --- diagnostics ----------------------------------------------------------------------------------------------------------------------------------
T.section("Where does the model fail?", anchor="failures", subtitle="LightGBM across the whole back-test (25 CV months + 3 holdout months)")
pp = p.dropna(subset=["actual", "lightgbm"]).copy()
pp["err"] = pp["actual"] - pp["lightgbm"]
pp["ae"] = pp["err"].abs()
pp["month_p"] = pp["delivery_day"].dt.to_period("M").dt.to_timestamp()
by_h = pp.groupby("hour")["err"].mean()
c1, c2 = st.columns([1.4, 1])
_mae_h = pp.groupby("hour")["ae"].mean()
_mae_m = pp.groupby("month_p")["ae"].mean().nlargest(2).index
with c1, T.card(f"The largest misses are at {_mae_h.idxmax()}:00 (MAE €{_mae_h.max():.0f}) and in {_mae_m[0]:%b %Y} and {_mae_m[1]:%b %Y}",
                "Mean absolute error (EUR/MWh) by hour of day and month · brighter = larger miss"):
    hm = pp.pivot_table(index="hour", columns="month_p", values="ae", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=T.SEQ, xgap=1, ygap=1, colorbar=dict(title=dict(text="MAE", side="right")),
                               hovertemplate="%{x|%b %Y}, %{y}:00<br>MAE <b>%{z:.1f}</b><extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", dtick=3, ticksuffix=":00", showgrid=False)
    fig.update_xaxes(showspikes=False, tickformat="%b %y")
    T.chart(fig, 380, legend=False)
with c2, T.card(f"Forecasts are too low by €{by_h.max():.0f} at {int(by_h.idxmax())}:00 and too high by €{abs(by_h.min()):.0f} at {int(by_h.idxmin())}:00 on average",
                "Mean error (actual − forecast) by hour: positive = forecast too low", T.legend_html([("Forecast too low", T.UP), ("Forecast too high", T.DOWN)], "box")):
    fig = go.Figure(go.Bar(x=by_h.index, y=by_h, name="mean error", marker=dict(color=np.where(by_h > 0, T.UP, T.DOWN), cornerradius=3),
                           hovertemplate="%{x}:00 · <b>%{y:+.1f}</b> EUR/MWh<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="hour (local)", dtick=3)
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 380, legend=False)

c1, c2 = st.columns(2)
q05, q95 = pp["err"].quantile([0.05, 0.95])
with c1, T.card(f"90% of hourly errors fall between {q05:+.0f} and {q95:+.0f} EUR/MWh", "Distribution of forecast errors (actual − forecast); bars cut at the 1st and 99th percentile"):
    lo1, hi1 = pp["err"].quantile([0.01, 0.99])
    fig = go.Figure(go.Histogram(x=pp["err"].clip(lo1, hi1), nbinsx=70, name="errors", marker=dict(color=T.ACCENT, line=dict(color=T.SURFACE, width=1)),
                                 hovertemplate="%{x:.0f} EUR/MWh: <b>%{y} hours</b><extra></extra>"))
    for v, txt in ((q05, "5 %"), (q95, "95 %"), (0, "")):
        fig.add_vline(x=v, line=dict(color=T.MUTED_DIM if v == 0 else T.TEXT_2, width=1, dash="solid" if v == 0 else "dash"), annotation_text=txt,
                      annotation_font=dict(size=12, color=T.TEXT_2))
    fig.update_layout(hovermode="closest", bargap=0.03)
    fig.update_xaxes(title="error (EUR/MWh)", showspikes=False)
    fig.update_yaxes(title="hours")
    T.chart(fig, 330, legend=False)
_lowp, _highp = pp[pp["actual"] < 150], pp[pp["actual"] >= 250]
with c2, T.card(f"Errors {'grow' if _highp['ae'].mean() > _lowp['ae'].mean() else 'do not grow'} with the price level: MAE €{_lowp['ae'].mean():.0f} when the price is below €150, €{_highp['ae'].mean():.0f} when it is above €250",
                "Forecast vs actual density; the dashed line is a perfect forecast"):
    lim = [float(min(pp["actual"].quantile(.005), pp["lightgbm"].quantile(.005))), float(max(pp["actual"].quantile(.995), pp["lightgbm"].quantile(.995)))]
    fig = go.Figure(go.Histogram2d(x=pp["actual"].clip(*lim), y=pp["lightgbm"].clip(*lim), nbinsx=55, nbinsy=55, showscale=False, hoverinfo="skip",
                                   colorscale=[[0, "rgba(0,0,0,0)"], [0.02, "#2A2110"], [0.3, "#A06C0F"], [1, "#FFE6B0"]]))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=T.TEXT_2, width=1, dash="dash"), name="perfect forecast", hoverinfo="skip"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="actual (EUR/MWh)", range=lim, showspikes=False, showgrid=True)
    fig.update_yaxes(title="forecast (EUR/MWh)", range=lim, showgrid=False)
    T.chart(fig, 330, legend=False)

mon = pp.groupby("month_p").agg(lgb=("ae", "mean"), price=("actual", "mean"), naive=("naive_d1", lambda s: (pp.loc[s.index, "actual"] - s).abs().mean()))
hold = p.loc[p["split"] == "holdout", "delivery_day"].min()
worst = mon["lgb"].idxmax()
with T.card(f"The worst month is {worst:%b %Y} (MAE €{mon.loc[worst, 'lgb']:.0f} at an average price of €{mon.loc[worst, 'price']:.0f}): level shifts, not the daily shape, cause the biggest misses",
            "MAE per month · hover for the average price · shaded = holdout", T.legend_html([("LightGBM", T.ACCENT), ("Naive · yesterday", T.NEUTRAL)], "box")):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mon.index, y=mon["naive"], name="naive", marker=dict(color=T.NEUTRAL, cornerradius=3), hovertemplate="naive MAE <b>%{y:.1f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=mon.index, y=mon["lgb"], name="LightGBM", marker=dict(color=T.ACCENT, cornerradius=3), customdata=mon["price"],
                         hovertemplate="LightGBM MAE <b>%{y:.1f}</b> · avg price %{customdata:.0f}<extra></extra>"))
    if pd.notna(hold):
        fig.add_vrect(x0=hold - pd.Timedelta(days=15), x1=mon.index.max() + pd.Timedelta(days=15), fillcolor=T.rgba(T.RED, .08), line_width=0, annotation_text="holdout",
                      annotation_position="top left", annotation_font=dict(size=12, color=T.TEXT_2))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MAE (EUR/MWh)")
    fig.update_xaxes(tickformat="%b %y")
    T.chart(fig, 330, legend=False)

# --- live track record ---------------------------------------------------------------------------------------------------------------------------------
T.section("Live track record", anchor="live", subtitle="Forecasts generated before gate closure, scored against the published day-ahead prices")
tr, fl = T.load_csv("live_track_record.csv"), T.load_csv("forecast_log.csv")
if tr is not None and len(tr):
    k = st.columns(4)
    T.kpi(k[0], "Days tracked", f"{len(tr)}", "live forecasts only", accent=True)
    mae_m, mae_n = (tr["mae_model"] * tr["hours"]).sum() / tr["hours"].sum(), (tr["mae_naive"] * tr["hours"]).sum() / tr["hours"].sum()
    T.kpi(k[1], "Live MAE", T.fmt_eur(mae_m), f"naive €{mae_n:.1f}", delta=T.delta_chip(100 * (mae_m - mae_n) / mae_n, "% vs naive", good_when_up=False))
    T.kpi(k[2], "Band coverage", f"{100 * (tr['coverage_10_90'] * tr['hours']).sum() / tr['hours'].sum():.0f}%", "target 80 %")
    T.kpi(k[3], "Mean bias", f"{(tr['bias'] * tr['hours']).sum() / tr['hours'].sum():+.1f}", "actual − forecast, EUR/MWh")
    with T.card("Daily error: model vs naive", "Mean absolute error per delivery day", T.legend_html([("Model", T.ACCENT), ("Naive · yesterday", T.NEUTRAL)], "box")):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tr["delivery_day"], y=tr["mae_naive"], name="naive", marker=dict(color=T.NEUTRAL, cornerradius=3)))
        fig.add_trace(go.Bar(x=tr["delivery_day"], y=tr["mae_model"], name="model", marker=dict(color=T.ACCENT, cornerradius=3)))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="MAE (EUR/MWh)")
        T.chart(fig, 300, legend=False)
else:
    n_log = 0 if fl is None else len(fl)
    st.markdown(f"<div class='empty' role='status'><div class='empty-icon' aria-hidden='true'>◌</div><div><b>The live track record is still empty.</b><br>"
                f"<span>Every run of <code>python -m src.predict_next_day</code> is logged ({n_log} hours so far). Only runs made before "
                f"12:00 CET on the day before delivery count as live, which the daily GitHub Action does automatically. "
                f"Scores appear here once the published prices for those days are available.</span></div></div>", unsafe_allow_html=True)

T.next_steps([("views/0_Trader_Brief.py", "Trader brief", "Tomorrow's forecast turned into actions."),
              ("views/4_Weather_Uncertainty.py", "Weather uncertainty", "Does better weather knowledge improve the forecast?"),
              ("views/5_Model_Lab.py", "Model lab", "Leaderboard, SHAP, calibration, drift.")])

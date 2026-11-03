"""Battery arbitrage simulation: configurable battery, value ladder, single-day dispatch, sensitivity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import theme as T
from src import battery as B

T.setup("Battery Simulation")
top = st.container()
p = T.load_predictions()
if p is None:
    with top:
        T.header("Value", "What is flexibility worth?", "A battery buys in cheap hours and sells in expensive ones.")
    T.missing("Back-test predictions")
    st.stop()

with T.controls("Battery configuration"):
    c1, c2, c3, c4 = st.columns(4)
    power = c1.slider("Power (MW)", 0.5, 10.0, 1.0, 0.5)
    dur = c2.select_slider("Duration (hours)", [1, 2, 3, 4], value=2)
    eff = c3.slider("Round-trip efficiency", 0.70, 0.98, 0.90, 0.01, format="%.2f")
    deg = c4.slider("Degradation (EUR/MWh)", 0.0, 20.0, 0.0, 1.0,
                    help="Wear cost charged on every MWh cycled; the schedule is optimised net of it. A new value needs about 30 s to simulate.")

strategies = {"perfect_foresight": "actual"}
for m in ("ensemble", "lightgbm"):
    if m in p:
        strategies[f"forecast_{m}"] = m
strategies["naive_yesterday"] = "naive_d1"
LABEL = {"perfect_foresight": "Perfect foresight", "forecast_lightgbm": "LightGBM forecast",
         "forecast_ensemble": "Ensemble forecast", "naive_yesterday": "Naive (yesterday)"}
COLOR = {"perfect_foresight": T.TEXT_2, "forecast_lightgbm": T.ACCENT, "forecast_ensemble": T.AQUA, "naive_yesterday": T.NEUTRAL}


@st.cache_data(show_spinner="Optimising daily schedules (about 30 s for a new duration/efficiency) …")
def run(dur: int, eff: float, deg: float, version: float | None) -> pd.DataFrame:
    """Daily revenue for a 1 MW battery. The schedule scales linearly with power, so power is applied outside."""
    pre = T.load_csv("battery_daily.csv")
    if (dur, round(eff, 2), deg) == (2, 0.90, 0.0) and pre is not None and set(strategies) <= set(pre.columns):
        return pre[["delivery_day", *strategies]]   # precomputed by run_all for the base case
    df = p.reset_index(drop=True).dropna(subset=list(strategies.values()))
    return B.simulate(df, B.Battery(power_mw=1.0, duration_h=dur, efficiency=eff, degradation_eur_mwh=deg), strategies)


daily = run(dur, eff, float(deg), T._mtime("predictions.csv")).copy()
daily[list(strategies)] *= power
if daily.empty:
    st.warning("No complete days to simulate.")
    st.stop()
bat = B.Battery(power_mw=power, duration_h=dur, efficiency=eff, degradation_eur_mwh=float(deg))
s = B.summarise(daily, bat)

_pl, _pn = s.loc["forecast_lightgbm", "pct_of_perfect"], s.loc["naive_yesterday", "pct_of_perfect"]
with top:
    T.header("Value", "What is flexibility worth?",
             "A battery buys in cheap hours and sells in expensive ones. Each day a small LP schedules charging on the schedule "
             "prices (perfect foresight, our forecast, or yesterday's prices) and the result is settled at actual day-ahead prices.",
             ["Price-taker", "≤ 1 cycle per day", "Day-ahead arbitrage", "Reserve revenue shown separately"],
             headline=(f"A {power:g} MW / {power * dur:g} MWh battery earns <b>€{s.loc['forecast_lightgbm', 'eur_per_mw_year'] / 1000:,.0f}k per MW-year</b> on the LightGBM "
                       f"schedule, <b>{_pl:.0f}%</b> of the perfect-foresight value; the naive rule gets {_pn:.0f}%."))

T.toc([("day", "One day"), ("monthly", "Monthly & design"), ("risk", "Monte Carlo risk"), ("costs", "Costs & investment"), ("reserve", "Reserve")])

k = st.columns(len(s))
for col, name in zip(k, strategies):
    r = s.loc[name]
    T.kpi(col, LABEL[name], f"€{r['eur_per_mw_year'] / 1000:,.1f}k", f"per MW-year · {r['pct_of_perfect']:.0f}% of perfect",
          accent=name == "forecast_lightgbm", spark=daily[name].rolling(30).mean().iloc[::7], spark_color=COLOR[name])

c1, c2 = st.columns([1, 1.5])
with c1, T.card(f"Forecasting recovers {100 * (_pl - _pn) / max(100 - _pn, 1e-9):.0f}% of the gap between naive and perfect", "EUR per MW-year · bars start at 0 · % of perfect foresight below the value"):
    names = list(strategies)
    vals = [s.loc[n, "eur_per_mw_year"] for n in names]
    fig = go.Figure(go.Bar(x=[LABEL[n] for n in names], y=vals, marker=dict(color=[COLOR[n] if n == 'forecast_lightgbm' else T.NEUTRAL for n in names], cornerradius=5),
                           text=[f"€{v / 1000:,.1f}k<br><span style='font-size:11px'>{s.loc[n, 'pct_of_perfect']:.0f}%</span>"
                                 for n, v in zip(names, vals)],
                           textposition="outside", textfont=dict(family=T.MONO, color=T.TEXT, size=12),
                           hovertemplate="%{x}: <b>€%{y:,.0f}</b> / MW-yr<extra></extra>"))
    gain = s.loc["forecast_lightgbm", "eur_per_mw_year"] - s.loc["naive_yesterday", "eur_per_mw_year"]
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(title="EUR / MW-year", range=[0, max(vals) * 1.22])
    fig.update_xaxes(showspikes=False)
    T.chart(fig, 380, legend=False)
with c2, T.card("Cumulative revenue per MW", "Settled at actual day-ahead prices",
                T.legend_html([(LABEL[n], COLOR[n]) for n in strategies])):
    cum = daily.set_index("delivery_day")[list(strategies)].cumsum() / power
    fig = go.Figure()
    for name in strategies:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[name], name=LABEL[name],
                                 line=dict(color=COLOR[name], width=2.6 if name == "forecast_lightgbm" else 1.6,
                                           dash="dot" if name == "perfect_foresight" else "solid"),
                                 hovertemplate=LABEL[name] + " <b>€%{y:,.0f}</b><extra></extra>"))
    fig.update_yaxes(title="EUR / MW", tickprefix="€")
    T.chart(fig, 380, legend=False)
T.takeaway(f"A {power:g} MW / {power * dur:g} MWh battery at {eff:.0%} efficiency earns <b>€{s.loc['perfect_foresight', 'eur_per_mw_year']:,.0f}"
           f"/MW-yr</b> with perfect foresight. The LightGBM schedule captures <b>{s.loc['forecast_lightgbm', 'pct_of_perfect']:.0f}%</b>, "
           f"the naive rule <b>{s.loc['naive_yesterday', 'pct_of_perfect']:.0f}%</b>: the forecast is worth <b>€{gain:,.0f} per MW-year</b>.",
           "action")

# --- one day in detail ------------------------------------------------------------------------------
T.section("One day in detail", anchor="day", subtitle="How the battery is dispatched, and what each schedule earns")
days = sorted(p["delivery_day"].dt.date.unique())
best_day = pd.Timestamp(daily.loc[daily["perfect_foresight"].idxmax(), "delivery_day"]).date()
with T.controls("Day and schedule"):
    c1, c2, _ = st.columns([1, 1.6, 1])
    day = c1.date_input("Delivery day", days[-1], min_value=days[0], max_value=days[-1])
    strat = c2.segmented_control("Schedule", list(strategies), default="forecast_lightgbm", format_func=lambda n: LABEL[n]) \
        or "forecast_lightgbm"
g = p[p["delivery_day"].dt.date == day].dropna(subset=list(strategies.values()))
if len(g):
    net = B.schedule_day(g[strategies[strat]].to_numpy(dtype=float), bat)
    se = np.sqrt(bat.efficiency)
    soc = np.cumsum(np.where(net < 0, -net * se, -net / se))
    rev = float(np.sum(net * g["actual"].to_numpy()))
    hrs = g.index.strftime("%H:%M")
    with T.card(f"{pd.Timestamp(day):%A %d %B %Y} · {LABEL[strat]}",
                f"Revenue settled at actual prices: €{rev:,.0f} · best day in the back-test: {best_day:%d %b %Y}",
                T.legend_html([("Actual price", T.TEXT), ("Schedule price", COLOR[strat]), ("Charge", T.DOWN),
                               ("Discharge", T.UP), ("State of charge", T.VIOLET)])):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=hrs, y=g["actual"], name="actual", line=dict(color=T.TEXT, width=2.4),
                                 hovertemplate="actual <b>%{y:.1f}</b><extra></extra>"), 1, 1)
        if strat != "perfect_foresight":
            fig.add_trace(go.Scatter(x=hrs, y=g[strategies[strat]], name="schedule", line=dict(color=COLOR[strat], width=1.8, dash="dot"),
                                     hovertemplate="schedule price %{y:.1f}<extra></extra>"), 1, 1)
        fig.add_trace(go.Bar(x=hrs, y=net, marker=dict(color=np.where(net > 0, T.UP, T.DOWN), cornerradius=3), name="dispatch",
                             hovertemplate="%{y:+.2f} MW (+ sell / − buy)<extra></extra>"), 2, 1)
        fig.add_trace(go.Scatter(x=hrs, y=soc, name="SoC", line=dict(color=T.VIOLET, width=2, shape="hv"),
                                 hovertemplate="state of charge %{y:.2f} MWh<extra></extra>"), 2, 1)
        fig.update_yaxes(title="EUR/MWh", row=1, col=1)
        fig.update_yaxes(title="MW · MWh", row=2, col=1)
        T.chart(fig, 460, legend=False)
    dd = daily[pd.to_datetime(daily["delivery_day"]).dt.date == day]
    if len(dd):
        cols = st.columns(len(strategies))
        for col, name in zip(cols, strategies):
            T.kpi(col, LABEL[name], T.fmt_eur(float(dd[name].iloc[0]) / power, 0), "revenue this day · per MW",
                  accent=name == strat)


@st.cache_data(show_spinner="Building the average dispatch pattern …")
def dispatch_profile(dur: int, eff: float, deg: float, version: float | None) -> pd.DataFrame:
    """Mean net dispatch (MW per 1 MW battery, + = sell) by month x hour for the LightGBM schedule."""
    b1 = B.Battery(power_mw=1.0, duration_h=dur, efficiency=eff, degradation_eur_mwh=deg)
    df = p.dropna(subset=["lightgbm"])
    rows = []
    for dday, g in df.groupby(df["delivery_day"].dt.date):
        if len(g) != 24:
            continue
        net = B.schedule_day(g["lightgbm"].to_numpy(dtype=float), b1)
        rows.append(pd.DataFrame({"month": pd.Timestamp(dday).to_period("M").to_timestamp(), "hour": g.index.hour, "net": net}))
    return pd.concat(rows).groupby(["month", "hour"])["net"].mean().unstack("hour") if rows else pd.DataFrame()


if "lightgbm" in p:
    prof = dispatch_profile(dur, eff, float(deg), T._mtime("predictions.csv"))
    if len(prof):
        ch_h, dis_h = prof.mean().idxmin(), prof.mean().idxmax()
        with T.card(f"On average the battery charges around {ch_h}:00 and sells around {dis_h}:00",
                    "Mean LightGBM-schedule dispatch per MW by month and hour · blue = buying, red = selling"):
            lim = float(np.nanmax(np.abs(prof.values)))
            fig = go.Figure(go.Heatmap(z=prof.values, x=[f"{h:02d}" for h in prof.columns], y=[m.strftime("%b %y") for m in prof.index], xgap=2, ygap=2,
                                       zmin=-lim, zmax=lim, colorscale=T.DIV, colorbar=dict(title=dict(text="MW", side="right")),
                                       hovertemplate="%{y} · %{x}:00<br>net <b>%{z:+.2f} MW</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="hour", showspikes=False)
            fig.update_yaxes(showgrid=False, autorange="reversed")
            T.chart(fig, 120 + 22 * len(prof), legend=False)

# --- monthly + sensitivity -------------------------------------------------------------------------
T.section("When and how much", anchor="monthly", subtitle="Monthly revenue and sensitivity to battery design")
c1, c2 = st.columns([1.5, 1])
d = daily.copy()
d["month"] = pd.to_datetime(d["delivery_day"]).dt.to_period("M").dt.to_timestamp()
mo = d.groupby("month")[list(strategies)].sum() / power
with c1, T.card("Monthly revenue per MW", "Perfect foresight (outline) vs the LightGBM schedule (filled)",
                T.legend_html([("Perfect foresight", T.TEXT_2), ("LightGBM forecast", T.ACCENT)], "box")):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mo.index, y=mo["perfect_foresight"], marker=dict(color="rgba(0,0,0,0)", line=dict(color=T.TEXT_2, width=1.2),
                         cornerradius=4), name="perfect", hovertemplate="perfect <b>€%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=mo.index, y=mo["forecast_lightgbm"], marker=dict(color=T.ACCENT, cornerradius=4), name="LightGBM",
                         customdata=100 * mo["forecast_lightgbm"] / mo["perfect_foresight"],
                         hovertemplate="LightGBM <b>€%{y:,.0f}</b> (%{customdata:.0f}% of perfect)<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="EUR / MW", tickprefix="€")
    fig.update_xaxes(tickformat="%b %y")
    T.chart(fig, 340, legend=False)
sens = T.load_csv("battery_sensitivity.csv")
if sens is not None:
    with c2, T.card("Design sensitivity", "Perfect-foresight revenue (k€ / MW-yr) by duration and efficiency"):
        pf = sens[sens.strategy == "perfect_foresight"].pivot(index="duration_h", columns="efficiency", values="eur_per_mw_year") / 1000
        fig = go.Figure(go.Heatmap(z=pf.values, x=[f"{e:.0%}" for e in pf.columns], y=[f"{int(h)} h" for h in pf.index],
                                   colorscale=T.SEQ, xgap=3, ygap=3, showscale=False, text=np.round(pf.values, 0),
                                   texttemplate="%{text:.0f}", textfont=dict(family=T.MONO, size=13, color=T.TEXT),
                                   hovertemplate="%{y} · %{x}: <b>€%{z:.1f}k</b><extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="round-trip efficiency", showspikes=False)
        fig.update_yaxes(title="duration", showgrid=False)
        T.chart(fig, 340, legend=False)
    fl = sens[(sens.strategy == "forecast_lightgbm") & (sens.efficiency == 0.9)]
    m = T.load_market()
    if m is not None:
        dspread = m["price_BG"].resample("D").agg(lambda x: x.max() - x.min())
        dspread.index = dspread.index.tz_localize(None)
        j = d.set_index(pd.to_datetime(d["delivery_day"]))["perfect_foresight"].to_frame().join(dspread.rename("spread")).dropna()
        qs = j.groupby(pd.qcut(j["spread"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]), observed=True)["perfect_foresight"].mean() / power
        T.takeaway(f"Longer batteries capture more of the value with a forecast ({', '.join(f'{int(r.duration_h)} h: {r.pct_of_perfect:.0f}%' for r in fl.itertuples())} "
                   f"at 90 % efficiency). Revenue is driven by the daily spread: €{qs.iloc[0]:,.0f} per MW on low-spread days vs "
                   f"<b>€{qs.iloc[-1]:,.0f}</b> on the highest-spread quarter of days.")


# --- Monte Carlo risk ---------------------------------------------------------------------------------------------
mc = T.load_json("mc_summary.json")
mcc = T.load_csv("mc_calibration.csv", parse_dates=["delivery_day"])
mcy = T.load_csv("mc_annual_draws.csv")
if mc and mcc is not None and mcy is not None:
    T.section("Revenue risk: Monte Carlo scenarios", anchor="risk", subtitle="How much could the forecast-driven schedule earn, and how far can the range be trusted? Fixed 1 MW / 2 MWh / 90 %, no degradation")
    meth = mc["chosen_method"]
    cal = mc["calibration"][meth]
    ann, nxt = mc["annual"], mc.get("next_day")
    k = st.columns(4)
    T.kpi(k[0], "Scenario-year median", f"€{ann['p50'] / 1000:,.0f}k", f"per MW-year · P5–P95 €{ann['p05'] / 1000:,.0f}k–€{ann['p95'] / 1000:,.0f}k", accent=True)
    T.kpi(k[1], "Back-test realised", f"€{ann['backtest_realised_eur_per_mw_year'] / 1000:,.0f}k", f"at the {100 * ann['backtest_realised_percentile']:.0f}th percentile of the scenarios")
    T.kpi(k[2], "Range check · P10–P90", f"{100 * cal['coverage_10_90_cv']:.0f}% CV", f"target 80% · holdout {100 * cal['coverage_10_90_holdout']:.0f}%",
          delta=T.delta_chip(100 * cal["coverage_10_90_holdout"] - 80, " pp holdout", good_when_up=True, digits=0))
    if nxt:
        T.kpi(k[3], "Tomorrow's revenue range", f"{'−' if nxt['p10'] < 0 else ''}€{abs(nxt['p10']):,.0f} to €{nxt['p90']:,.0f}", f"P10–P90 per MW · median €{nxt['p50']:,.0f} · expected at forecast €{nxt['expected_at_forecast']:,.0f}")
    c1, c2 = st.columns([1, 1.4])
    with c1, T.card(f"A scenario year earns €{ann['p05'] / 1000:,.0f}k–€{ann['p95'] / 1000:,.0f}k per MW (5th–95th percentile)",
                    f"{ann['n_years']:,} simulated years of 365 days · {ann['block_days']}-day blocks so bad regimes persist"):
        fig = go.Figure(go.Histogram(x=mcy["eur_per_mw_year"], nbinsx=40, marker=dict(color=T.rgba(T.ACCENT, .75), line=dict(color=T.SURFACE, width=1)),
                                     hovertemplate="€%{x:,.0f} per MW-year · %{y} scenarios<extra></extra>"))
        for x_, nm, col in ((ann["p05"], "P5", T.NEUTRAL), (ann["p50"], "median", T.TEXT_2), (ann["p95"], "P95", T.NEUTRAL)):
            fig.add_vline(x=x_, line=dict(color=col, width=1.4, dash="dot"), annotation_text=nm, annotation_font=dict(size=12, color=T.TEXT_2))
        fig.add_vline(x=ann["backtest_realised_eur_per_mw_year"], line=dict(color=T.AQUA, width=2.4),
                      annotation_text="back-test", annotation_position="top", annotation_font=dict(size=12, color=T.TEXT))
        fig.update_layout(bargap=0.04, hovermode="closest")
        fig.update_xaxes(title="annual revenue (EUR per MW-year)", tickprefix="€", showspikes=False)
        fig.update_yaxes(title="scenarios")
        T.chart(fig, 360, legend=False)
    cm = mcc[mcc["method"] == meth].sort_values("delivery_day").tail(120)
    with c2, T.card(f"Realised daily revenue sat inside the simulated P10–P90 range on {100 * cal['coverage_10_90']:.0f}% of back-test days",
                    f"Each day's range uses only errors known before that day · last {len(cm)} days shown · {cal['days']} days tested",
                    T.legend_html([("Realised", T.TEXT), ("P10–P90 range", T.rgba(T.ACCENT, .4)), ("Median", T.ACCENT)])):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cm["delivery_day"], y=cm["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cm["delivery_day"], y=cm["p10"], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .18), line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=cm["delivery_day"], y=cm["p50"], name="median", line=dict(color=T.ACCENT, width=1.6),
                                 hovertemplate="median <b>€%{y:,.0f}</b><extra></extra>"))
        out = ~cm["in_10_90"]
        fig.add_trace(go.Scatter(x=cm["delivery_day"], y=cm["realised"], mode="lines", name="realised", line=dict(color=T.TEXT, width=1.4),
                                 hovertemplate="realised <b>€%{y:,.0f}</b><extra></extra>"))
        fig.add_trace(go.Scatter(x=cm.loc[out, "delivery_day"], y=cm.loc[out, "realised"], mode="markers", name="outside range", showlegend=False,
                                 marker=dict(size=7, color=T.RED, line=dict(color=T.SURFACE, width=1.5)), hovertemplate="outside range <b>€%{y:,.0f}</b><extra></extra>"))
        fig.update_yaxes(title="EUR per MW-day", tickprefix="€")
        T.chart(fig, 360, legend=False)
    other = [m for m in mc["calibration"] if m != meth]
    alt = mc["calibration"][other[0]] if other else None
    T.takeaway(f"The range check passes on the CV months (<b>{100 * cal['coverage_10_90_cv']:.0f}%</b> vs 80% target) but is too narrow on the holdout "
               f"(<b>{100 * cal['coverage_10_90_holdout']:.0f}%</b>): the scenarios are built from recent misses and cannot anticipate a regime shift. "
               + (f"The generator was chosen on CV months only: a simpler one that draws error vectors independently of the schedule covered just "
                  f"{100 * alt['coverage_10_90_cv']:.0f}% because realised revenue falls systematically short of the revenue expected at forecast prices "
                  f"(the optimiser's curse: the schedule sells where the forecast is too high). " if alt else "")
               + f"Predicted loss probability per day averages {100 * cal['mean_predicted_prob_loss']:.0f}% against {100 * cal['realised_loss_rate']:.1f}% realised, so tails are conservative.",
               "warning")

# --- costs, filter, investment ------------------------------------------------------------------------------------
T.section("Costs and investment view", anchor="costs", subtitle="What wear costs, whether a confidence filter helps, and what the revenue means as a project")
dg, fl = T.load_csv("battery_degradation.csv"), T.load_csv("battery_filter.csv")
c1, c2 = st.columns(2)
if dg is not None:
    with c1, T.card("Degradation sensitivity", "Net revenue per MW-year for the 1 MW / 2 h / 90 % base case by wear cost per MWh",
                    T.legend_html([(LABEL[n], COLOR[n]) for n in ("perfect_foresight", "forecast_lightgbm", "naive_yesterday")])):
        fig = go.Figure()
        for name in ("perfect_foresight", "forecast_lightgbm", "naive_yesterday"):
            g = dg[dg.strategy == name]
            fig.add_trace(go.Scatter(x=g["degradation_eur_mwh"], y=g["eur_per_mw_year"], mode="lines+markers", name=LABEL[name],
                                     line=dict(color=COLOR[name], width=2.6 if name == "forecast_lightgbm" else 1.6),
                                     marker=dict(size=8, line=dict(color=T.SURFACE, width=1.5)),
                                     hovertemplate=LABEL[name] + " <b>€%{y:,.0f}</b><extra></extra>"))
        fig.update_xaxes(title="degradation cost (EUR per MWh throughput)")
        fig.update_yaxes(title="EUR / MW-year", tickprefix="€")
        T.chart(fig, 320, legend=False)
if fl is not None:
    with c2, T.card("Does a confidence filter help?", "Stay idle if the expected daily profit is below a threshold · chosen on CV months only",
                    T.legend_html([("Walk-forward CV", T.BLUE), ("Holdout", T.ACCENT)], "box")):
        fig = go.Figure()
        for split, col, nm in (("cv", T.BLUE, "CV"), ("holdout", T.ACCENT, "holdout")):
            g = fl[fl.split == split]
            fig.add_trace(go.Bar(x=[f"€{int(m)}" for m in g["min_expected_profit"]], y=g["eur_per_mw_year"], name=nm,
                                 marker=dict(color=col, cornerradius=3), customdata=g["days_traded"],
                                 hovertemplate="threshold %{x}: <b>€%{y:,.0f}</b> · %{customdata} days traded<extra>" + nm + "</extra>"))
        lo = fl["eur_per_mw_year"].min()
        fig.update_layout(barmode="group", hovermode="closest")
        fig.update_xaxes(title="minimum expected profit per MW-day", showspikes=False)
        fig.update_yaxes(title="EUR / MW-year — axis does not start at 0", range=[lo * 0.97, fl["eur_per_mw_year"].max() * 1.01])
        T.chart(fig, 320, legend=False)
if dg is not None and fl is not None:
    d0, d15 = (dg[(dg.degradation_eur_mwh == v) & (dg.strategy == "forecast_lightgbm")]["eur_per_mw_year"].iloc[0] for v in (0, 15))
    T.takeaway(f"Each €5/MWh of wear cost removes about <b>€{(d0 - d15) / 3:,.0f}</b> per MW-year (a €15/MWh cost cuts €{d0:,.0f} to €{d15:,.0f}). "
               "The confidence filter changes nothing: Bulgarian daily spreads are large enough that the best threshold on the CV months is "
               "<b>zero</b>, i.e. the battery should trade every day.")

net_year = float(s.loc["forecast_lightgbm", "eur_per_mw_year"]) if "forecast_lightgbm" in s.index else float(s.iloc[0]["eur_per_mw_year"])
with T.controls("Investment assumptions (illustrative, not market quotes)"):
    i1, i2, i3, i4 = st.columns(4)
    capex = i1.slider("Capex (EUR per kWh installed)", 100, 700, 300, 10)
    opex = i2.slider("Opex (% of capex per year)", 0.0, 6.0, 2.0, 0.5)
    life = i3.slider("Lifetime (years)", 5, 20, 12, 1)
    disc = i4.slider("Discount rate (%)", 3.0, 15.0, 8.0, 0.5)
case = B.investment_case(net_year / power, power, dur, float(capex), float(opex), int(life), disc / 100)
if True:
    k = st.columns(5)
    T.kpi(k[0], "Net revenue", f"€{net_year / power / 1000:,.0f}k", "per MW-year (LightGBM schedule)", accent=True)
    T.kpi(k[1], "Simple payback", f"{case['payback_years']:.1f} yr" if np.isfinite(case["payback_years"]) else "never", f"capex €{case['capex_eur'] / 1e6:,.2f}M")
    T.kpi(k[2], "IRR", f"{100 * case['irr']:.1f}%" if np.isfinite(case["irr"]) else "n/a", f"vs {disc:.1f}% discount rate")
    T.kpi(k[3], "NPV", f"€{case['npv_eur'] / 1e6:,.2f}M", f"{life} years")
    T.kpi(k[4], "Break-even capex", f"€{case['breakeven_capex_eur_per_kwh']:,.0f}/kWh", "NPV = 0 · needs no capex assumption")
T.takeaway(f"At the chosen assumptions this {power:g} MW / {power * dur:g} MWh battery has an IRR of "
           f"<b>{100 * case['irr']:.1f}%</b> and pays back in <b>{case['payback_years']:.1f} years</b>. The robust number is the break-even capex "
           f"(<b>€{case['breakeven_capex_eur_per_kwh']:,.0f}/kWh</b>): any real offer below it has a positive NPV on <i>2023–2026 arbitrage revenue alone</i>, "
           "which is a back-test, not a promise.", "action")

# --- reserve capacity (indicative) ---------------------------------------------------------------------------------
rv, rm = T.load_csv("battery_reserve.csv"), T.load_csv("reserve_monthly.csv", parse_dates=["month"])
if rv is not None and rm is not None:
    T.section("Balancing reserve capacity (indicative)", anchor="reserve", subtitle="ENTSO-E contracted reserve prices for Bulgaria — a different revenue stream, with big caveats")
    T.takeaway("<b>Read with care.</b> The published reserve prices are constant for long stretches (e.g. FCR €10 → €50 → €25/MW/h), which points to "
               "<b>administered prices, not a competitive market</b>, so they can change with regulation. The markets are tiny: FCR about "
               f"{rv.set_index('product').loc['FCR (symmetric)', 'contracted_mw']:.0f} MW and aFRR-up about {rv.set_index('product').loc['aFRR up', 'contracted_mw']:.0f} MW "
               "are contracted, so one battery is a large share and a small fleet saturates them (the price-taker assumption fails). "
               "Only the capacity payment is counted; activation, state-of-charge management and prequalification are ignored.", "warning")
    with T.controls("Reserve assumption"):
        haircut = st.slider("Share of the year actually awarded and available (%)", 5, 100, 30, 5,
                            help="Capacity payment only accrues for the hours the battery is awarded and available. 100 % is the theoretical maximum.")
    c1, c2 = st.columns([1.5, 1])
    prices = [c for c in rm.columns if c.endswith(" price")]
    cols_ = {"FCR (symmetric) price": T.ACCENT, "aFRR up price": T.RED, "aFRR down price": T.BLUE, "mFRR up price": T.VIOLET}
    with c1, T.card("Contracted reserve capacity price", "Monthly average, EUR per MW per hour",
                    T.legend_html([(c.replace(" price", ""), cols_.get(c, T.MUTED)) for c in prices])):
        fig = go.Figure()
        for c in prices:
            fig.add_trace(go.Scatter(x=rm["month"], y=rm[c], name=c.replace(" price", ""), line=dict(color=cols_.get(c, T.MUTED), width=2, shape="hv"),
                                     hovertemplate=c.replace(" price", "") + " <b>€%{y:.1f}</b><extra></extra>"))
        fig.update_yaxes(title="EUR / MW / h")
        T.chart(fig, 320, legend=False)
    with c2:
        rv2 = rv.set_index("product")
        eur = rv2["capacity_eur_per_mw_year_100pct"] * haircut / 100
        with T.card("Reserve vs day-ahead arbitrage", f"EUR per MW-year at {haircut}% awarded · latest 12 months of prices"):
            fig = go.Figure(go.Bar(x=list(eur.index) + ["Day-ahead arbitrage"], y=list(eur.values) + [float(rv2["arbitrage_eur_per_mw_year"].iloc[0])],
                                   marker=dict(color=[T.AQUA] * len(eur) + [T.ACCENT], cornerradius=4),
                                   text=[f"€{v / 1000:,.0f}k" for v in list(eur.values) + [float(rv2['arbitrage_eur_per_mw_year'].iloc[0])]],
                                   textposition="outside", textfont=dict(family=T.MONO, size=11, color=T.TEXT_2),
                                   hovertemplate="%{x}: <b>€%{y:,.0f}</b> / MW-yr<extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_yaxes(title="EUR / MW-year", range=[0, max(eur.max(), rv2["arbitrage_eur_per_mw_year"].iloc[0]) * 1.2])
            fig.update_xaxes(showspikes=False)
            T.chart(fig, 250, legend=False)
    depth = rv.set_index("product")
    T.table(depth[["price_eur_mw_h", "contracted_mw", "fleet_mw_that_would_be_10pct_of_market"]].rename(columns={
        "price_eur_mw_h": "Price (€/MW/h, 12-mo avg)", "contracted_mw": "Contracted (MW)",
        "fleet_mw_that_would_be_10pct_of_market": "Fleet = 10 % of market (MW)"}).rename_axis("Product"),
        fmt={"Price (€/MW/h, 12-mo avg)": "{:.1f}", "Contracted (MW)": "{:.0f}", "Fleet = 10 % of market (MW)": "{:.1f}"},
        bars={"Contracted (MW)": T.AQUA})

with st.expander("Assumptions and limitations"):
    st.markdown("""
- Day-ahead arbitrage in the base case: no intraday revenue; balancing reserve capacity is shown separately and indicatively.
- Price-taker: the battery does not move the price (fails for the thin reserve markets). No network charges or taxes; degradation is a configurable cost.
- At most one full cycle per day; start and end each day empty; efficiency split equally between charge and discharge.
- Forecast schedules use the walk-forward forecasts (made before gate closure), settled at the actual price.
""")

T.next_steps([("views/0_Trader_Brief.py", "Trader brief", "Tomorrow's battery schedule and its revenue range."),
              ("views/3_Forecast.py", "Forecast", "How good the prices behind the schedule are."),
              ("views/7_About.py", "About and method", "Assumptions and limitations of the simulation.")])

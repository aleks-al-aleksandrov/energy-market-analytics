"""Price drivers: hour x month profile, duck curve, residual load, generation mix, fuel and carbon."""
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

T.setup("Price Drivers")
top = st.container()
m = T.load_market()
if m is None or "price_BG" not in m:
    with top:
        T.header("Market", "What drives the price?")
    T.missing("Market data")
    st.stop()

years = sorted(m.index.year.unique())
with T.controls("Period"):
    yr = st.segmented_control("Years", ["All"] + [str(y) for y in years], default="All") or "All"
d = m if yr == "All" else m[m.index.year == int(yr)]
p = d["price_BG"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
prof = p.groupby(d.index.hour).mean()
warm = p[d.index.month.isin([5, 6, 7, 8])]
sp = warm.groupby(warm.index.hour).mean() if len(warm) else prof
rho = np.nan
if "residual_load_forecast" in d:
    rho = d[["residual_load_forecast", "price_BG"]].dropna().corr(method="spearman").iloc[0, 1]
with top:
    T.header("Market", "What drives the price?",
             "When power is expensive, how strongly price follows residual load (load − wind − solar), and which plants run in cheap versus expensive hours.",
             headline=(f"Price follows residual load (Spearman ρ = <b>{rho:.2f}</b>), and the solar midday trough in May–August (€{sp.min():.0f}) sits "
                       f"<b>€{sp.max() - sp.min():.0f}</b> below the evening peak (€{sp.max():.0f})." if np.isfinite(rho) else ""))

# --- hour x month + duck curve ----------------------------------------------------------------------------------
c1, c2 = st.columns([1.55, 1])
hm = p.groupby([d.index.hour, d.index.month]).mean().unstack()
pk_h, pk_m = np.unravel_index(np.nanargmax(hm.values), hm.shape)
with c1, T.card(f"The dearest hour of the year is {hm.index[pk_h]}:00 in {MONTHS[hm.columns[pk_m] - 1]} (€{np.nanmax(hm.values):.0f})",
                "Average price by hour of day and month · brighter = more expensive · local time",
                help="Dark band in the middle of the day from spring to autumn = the solar trough. Dark nights = low demand."):
    fig = go.Figure(go.Heatmap(z=hm.values, x=[MONTHS[i - 1] for i in hm.columns], y=hm.index, colorscale=T.SEQ, xgap=2, ygap=2,
                               colorbar=dict(title=dict(text="€/MWh", side="right")), hovertemplate="%{x}, %{y}:00<br><b>%{z:.1f} EUR/MWh</b><extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", dtick=3, ticksuffix=":00", showgrid=False)
    fig.update_xaxes(showspikes=False)
    T.chart(fig, 430, legend=False)

# validated 4-colour set for four seasons (yellow, blue, magenta, green pass the all-pairs checks)
SEASONS = {"Winter": ([12, 1, 2], T.BLUE), "Spring": ([3, 4, 5], T.GREEN), "Summer": ([6, 7, 8], T.YELLOW), "Autumn": ([9, 10, 11], T.MAGENTA)}
sprof = {k: p[d.index.month.isin(v[0])].groupby(d.index[d.index.month.isin(v[0])].hour).mean() for k, v in SEASONS.items() if d.index.month.isin(v[0]).any()}
deepest = min(sprof, key=lambda k: sprof[k].min()) if sprof else "Summer"
with c2, T.card(f"The duck curve is deepest in {deepest.lower()}: a €{sprof[deepest].min():.0f} midday trough",
                "Average hourly profile by season", T.legend_html([(k, SEASONS[k][1]) for k in sprof]),
                help="The 'duck': a solar-driven midday dip followed by a steep evening ramp. Batteries earn the difference."):
    fig = go.Figure()
    for name, (months, col) in SEASONS.items():
        if name in sprof:
            fig.add_trace(go.Scatter(x=sprof[name].index, y=sprof[name], name=name, line=dict(color=col, width=2.2, shape="spline"),
                                     hovertemplate=name + " <b>%{y:.0f}</b><extra></extra>"))
    fig.update_xaxes(title="hour (local)", dtick=3, range=[-0.3, 23.3])
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 430, legend=False)

# --- residual load vs price ---------------------------------------------------------------------------------------------
c1, c2 = st.columns([1.3, 1])
with c1:
    if "residual_load_forecast" in d:
        x = d[["residual_load_forecast", "price_BG"]].dropna()
        ylo, yhi = np.nanpercentile(x.price_BG, [0.5, 99.5])
        bins = pd.qcut(x["residual_load_forecast"], 25, duplicates="drop")
        g = x.groupby(bins, observed=True)["price_BG"].quantile([0.1, 0.5, 0.9]).unstack()
        mid = [b.mid for b in g.index]
        with T.card(f"Each step up in residual load lifts the median price: €{g[0.5].iloc[0]:.0f} at the bottom, €{g[0.5].iloc[-1]:.0f} at the top",
                    "Density of hours (blue) with the median and 10–90 % range per load bin",
                    T.legend_html([("Median", T.ACCENT), ("10–90 % range", T.rgba(T.ACCENT, .4))]),
                    help="Residual load = load − wind − solar (day-ahead forecasts). It is the demand that thermal and hydro plants must still cover."):
            fig = go.Figure()
            fig.add_trace(go.Histogram2d(x=x.iloc[:, 0], y=x.iloc[:, 1].clip(ylo, yhi), nbinsx=60, nbinsy=50, showscale=False, hoverinfo="skip",
                                         colorscale=[[0, "rgba(0,0,0,0)"], [0.02, "#162033"], [0.3, "#1C5CAB"], [1, "#86B6EF"]]))
            fig.add_trace(go.Scatter(x=mid, y=g[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=mid, y=g[0.1], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .16), line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=mid, y=g[0.5], name="median price", line=dict(color=T.ACCENT, width=2.6),
                                     hovertemplate="%{x:,.0f} MW → <b>%{y:.0f} EUR/MWh</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="residual load forecast (MW)", showspikes=False)
            fig.update_yaxes(title="EUR/MWh", range=[ylo, yhi], showgrid=False)
            T.chart(fig, 400, legend=False)
with c2:
    gen_cols = [c for c in d.columns if c.startswith("gen_")]
    if gen_cols:
        grp = d[gen_cols].clip(lower=0).T.groupby([T.gen_group(c[4:]) for c in gen_cols]).sum().T
        q = p.quantile([0.1, 0.9])
        cheap, exp = grp.loc[p <= q.iloc[0]].mean(), grp.loc[p >= q.iloc[1]].mean()
        order = [g_ for g_ in T.GEN_ORDER if g_ in grp.columns]
        diff = (exp - cheap)
        with T.card(f"{diff.idxmax()} runs {diff.max():+,.0f} MW more in the dearest hours; {diff.idxmin()} {diff.min():+,.0f} MW",
                    "Average output in the cheapest and the dearest 10 % of hours", T.legend_html([("Cheapest 10 %", T.DOWN), ("Dearest 10 %", T.UP)], "box"),
                    help="Dumbbell: the gap between the two dots is how much more (or less) a technology runs when power is expensive."):
            fig = go.Figure()
            for g_ in order:
                fig.add_trace(go.Scatter(x=[cheap[g_], exp[g_]], y=[g_, g_], mode="lines", line=dict(color=T.BORDER, width=4), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cheap[order], y=order, mode="markers", name="cheapest 10 %", marker=dict(size=13, color=T.DOWN, line=dict(color=T.SURFACE, width=2)),
                                     hovertemplate="%{y}: <b>%{x:,.0f} MW</b> in cheap hours<extra></extra>"))
            fig.add_trace(go.Scatter(x=exp[order], y=order, mode="markers", name="dearest 10 %", marker=dict(size=13, color=T.UP, line=dict(color=T.SURFACE, width=2)),
                                     hovertemplate="%{y}: <b>%{x:,.0f} MW</b> in expensive hours<extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="average output (MW)", showgrid=True, showspikes=False)
            fig.update_yaxes(showgrid=False, autorange="reversed")
            T.chart(fig, 400, legend=False)
if "residual_load_forecast" in d:
    T.takeaway("Residual load is the single most useful fundamental: it is what thermal and hydro plants still have to cover once wind and solar have been "
               "subtracted. Flexible plants set the price at the margin and solar sets the floor.", "action")

# --- generation mix over time ------------------------------------------------------------------------------------------------
gen = [c for c in m.columns if c.startswith("gen_")]
if gen:
    T.section("Generation mix over time", "Actual generation (ENTSO-E): analysis only, never a model feature")
    with T.controls("View"):
        mode = st.segmented_control("Show", ["Share of generation", "Megawatts"], default="Share of generation") or "Share of generation"
    w = m[gen].clip(lower=0).resample("W").mean()
    wg = w.T.groupby([T.gen_group(c[4:]) for c in gen]).sum().T
    order = [g_ for g_ in T.GEN_ORDER if g_ in wg.columns]
    share = wg.div(wg.sum(axis=1), axis=0) * 100
    first, last = share.iloc[:52].mean(), share.iloc[-52:].mean()
    mover = (last - first).abs().idxmax()
    with T.card(f"{mover}'s share moved from {first[mover]:.0f}% to {last[mover]:.0f}% between the first and the last year",
                "Weekly average generation by technology, stacked bottom → top in a fixed order; 7 groups (small sources folded into Other)",
                T.legend_html([(g_, T.GEN_GROUP_COLOR[g_]) for g_ in order], "box")):
        data = share if mode.startswith("Share") else wg
        fig = go.Figure()
        for g_ in order:
            fig.add_trace(go.Scatter(x=data.index, y=data[g_], stackgroup="g", name=g_, line=dict(width=1.5, color=T.SURFACE),
                                     fillcolor=T.rgba(T.GEN_GROUP_COLOR[g_], .9),
                                     hovertemplate=g_ + ": %{y:,.1f}" + ("%" if mode.startswith("Share") else " MW") + "<extra></extra>"))
        fig.update_yaxes(title="% of generation" if mode.startswith("Share") else "MW", range=[0, 100] if mode.startswith("Share") else None)
        T.range_buttons(fig)
        T.chart(fig, 440, legend=False)

# --- fuel and carbon prices -----------------------------------------------------------------------------------------------------
fm, fd, snap = T.load_csv("fuels_monthly.csv", parse_dates=["month"]), T.load_csv("fuels_daily.csv", parse_dates=["date"]), T.load_csv("ttf_snapshots.csv")
if fm is not None and fd is not None:
    T.section("Fuel and carbon prices", "What sets the marginal cost of gas- and lignite-fired power (not in ENTSO-E)")
    fm = fm.set_index("month")
    bg_m = m["price_BG"].resample("MS").mean()
    bg_m.index = bg_m.index.tz_localize(None)
    both = pd.concat([bg_m.rename("power"), fm["gas_eur_mwh"].rename("gas")], axis=1).dropna()
    fdd = fd.set_index("date")
    k = st.columns(4)
    T.kpi(k[0], "EU gas · latest month", f"€{fm['gas_eur_mwh'].dropna().iloc[-1]:.0f}", f"/MWh · {fm['gas_eur_mwh'].dropna().index[-1]:%b %Y} (World Bank)",
          accent=True, spark=fm["gas_eur_mwh"].dropna().tail(36))
    if "ttf" in fdd and fdd["ttf"].notna().any():
        ttf_d = fdd["ttf"].dropna()
        T.kpi(k[1], "TTF gas · daily", f"€{ttf_d.iloc[-1]:.0f}", f"/MWh · avg of {ttf_d.index[-1]:%d %b} (OilPriceAPI)", spark=ttf_d.tail(120), spark_color=T.RED)
    else:
        T.kpi(k[1], "TTF front month", f"€{snap['ttf_front_eur_mwh'].iloc[-1]:.0f}" if snap is not None and len(snap) else "n/a",
              f"/MWh · snapshot {snap['date'].iloc[-1]}" if snap is not None and len(snap) else "daily TTF not loaded")
    T.kpi(k[2], "Brent crude", f"${fdd['brent'].dropna().iloc[-1]:.0f}", f"/bbl · close {fdd['brent'].dropna().index[-1]:%d %b}", spark=fdd["brent"].dropna().tail(120), spark_color=T.VIOLET)
    T.kpi(k[3], "Carbon (ETC proxy)", f"{fdd['carbon'].dropna().iloc[-1]:.1f}", f"index · close {fdd['carbon'].dropna().index[-1]:%d %b}", spark=fdd["carbon"].dropna().tail(120), spark_color=T.AQUA)
    c1, c2 = st.columns([1.5, 1])
    fm_i = fm.loc["2023-01-01":]
    idx = pd.concat([bg_m.rename("BG power"), fm_i["gas_eur_mwh"].rename("EU gas"), fdd["brent"].resample("MS").mean().rename("Brent"),
                     fdd["carbon"].resample("MS").mean().rename("Carbon proxy")], axis=1).loc["2023-01-01":]
    idx = 100 * idx / idx.iloc[0]
    with c1, T.card(f"Gas is up {idx['EU gas'].dropna().iloc[-1] - 100:+.0f}% and power {idx['BG power'].dropna().iloc[-1] - 100:+.0f}% since January 2023",
                    "Monthly averages, indexed to January 2023 = 100 (one axis; no dual scales)",
                    T.legend_html([("BG power", T.ACCENT), ("EU gas", T.RED), ("Brent", T.VIOLET), ("Carbon proxy", T.AQUA)])):
        fig = go.Figure()
        for col, c, wd in (("EU gas", T.RED, 1.6), ("Brent", T.VIOLET, 1.6), ("Carbon proxy", T.AQUA, 1.6), ("BG power", T.ACCENT, 2.6)):
            fig.add_trace(go.Scatter(x=idx.index, y=idx[col], name=col, line=dict(color=c, width=wd), hovertemplate=col + " <b>%{y:.0f}</b><extra></extra>"))
        fig.add_hline(y=100, line=dict(color=T.MUTED_DIM, width=1))
        fig.update_yaxes(title="index (Jan 2023 = 100)")
        T.chart(fig, 360, legend=False)
    with c2, T.card(f"Monthly power and gas prices move together (r = {both['power'].corr(both['gas']):.2f})", "Each dot is a month · colour = year",
                    T.legend_html([(str(y_), c_) for y_, c_ in zip(sorted(both.index.year.unique()), [T.BLUE, T.GREEN, T.MAGENTA, T.YELLOW])], "dot")):
        both["year"] = both.index.year
        fig = go.Figure()
        for yr_, col in zip(sorted(both["year"].unique()), [T.BLUE, T.GREEN, T.MAGENTA, T.YELLOW]):
            g = both[both["year"] == yr_]
            fig.add_trace(go.Scatter(x=g["gas"], y=g["power"], mode="markers", name=str(yr_), marker=dict(size=10, color=col, line=dict(color=T.SURFACE, width=2)),
                                     customdata=g.index.strftime("%b %Y"), hovertemplate="%{customdata}: gas €%{x:.0f} → power <b>€%{y:.0f}</b><extra></extra>"))
        fig.update_layout(hovermode="closest", hoverdistance=60)
        fig.update_xaxes(title="EU gas (EUR/MWh)", showgrid=True, showspikes=False)
        fig.update_yaxes(title="BG power (EUR/MWh)")
        T.chart(fig, 360, legend=False)
    if "ttf" in fdd and fdd["ttf"].notna().sum() > 30:
        ttf_d = fdd["ttf"].dropna()
        pw = m["price_BG"].resample("D").mean()
        pw.index = pw.index.tz_localize(None)
        pw = pw.loc[ttf_d.index.min():]
        both_d = pd.concat([pw.rename("power"), ttf_d.rename("gas")], axis=1).dropna()
        rd, rchg = both_d["power"].corr(both_d["gas"]), both_d["power"].diff(20).corr(both_d["gas"].diff(20))
        with T.card(f"TTF rose from €{ttf_d.loc[:'2026-09-30'].iloc[-1]:.0f} to €{ttf_d.iloc[-1]:.0f} since September, and BG power followed",
                    f"Daily TTF and daily BG power since {ttf_d.index.min():%b %Y}: two panels share the time axis",
                    T.legend_html([("TTF gas", T.RED), ("BG power", T.ACCENT)])):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
            fig.add_trace(go.Scatter(x=ttf_d.index, y=ttf_d, name="TTF", line=dict(color=T.RED, width=2), hovertemplate="TTF <b>€%{y:.1f}</b><extra></extra>"), 1, 1)
            fig.add_trace(go.Scatter(x=pw.index, y=pw, name="BG power", line=dict(color=T.ACCENT, width=2), hovertemplate="BG power <b>€%{y:.1f}</b><extra></extra>"), 2, 1)
            fig.update_yaxes(title="TTF €/MWh", row=1, col=1)
            fig.update_yaxes(title="BG €/MWh", row=2, col=1)
            T.range_buttons(fig)
            T.chart(fig, 420, legend=False)
        T.takeaway(f"Over the {len(both_d)} days with daily TTF the correlation with BG power is <b>{rd:.2f}</b> in levels and <b>{rchg:.2f}</b> for 20-day changes. "
                   "The monthly gas series reports the autumn surge only weeks later.")
    T.takeaway(f"Monthly BG power and EU gas prices correlate at <b>{both['power'].corr(both['gas']):.2f}</b> over {len(both)} months. Gas and carbon set the "
               "marginal cost of gas- and lignite-fired plants, information ENTSO-E does not carry; in the forecasting tests it added no accuracy beyond the price lags.", "action")

T.next_steps([("views/2_Cross_Border.py", "Cross-border", "Spreads and market coupling with neighbours."),
              ("views/3_Forecast.py", "Forecast", "How these drivers feed the model."),
              ("overview.py", "Market overview", "Back to prices and KPIs.")])

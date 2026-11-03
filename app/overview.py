"""Market Overview page (entry point is app/dashboard.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import signature as S
import theme as T

T.setup("Market Overview")
m = T.load_market()
if m is None or "price_BG" not in m:
    T.header("Market", "Bulgarian Day-Ahead Market")
    T.missing("Market data (reports/market_hourly.parquet)")
    st.stop()

px_ = m["price_BG"].dropna()
lo, hi = px_.index.min().date(), px_.index.max().date()
top = st.container()    # the header is written here after the filters are read, so the page order is title -> filters -> content

# --- one filter row above everything it scopes --------------------------------------------------------------------
with T.controls("Period"):
    c1, c2, c3, _ = st.columns([1.5, 1.2, 1.3, 1.5])
    default_start = max(lo, (pd.Timestamp(hi) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).date())
    rng = c1.date_input("Date range", (default_start, hi), min_value=lo, max_value=hi)
    res = c2.segmented_control("Resolution", ["Hourly", "Daily", "Weekly"], default="Daily") or "Daily"
    compare = c3.toggle("Compare with previous year", value=False, help="Overlays the same dates one year earlier as a thin grey line.")
if not isinstance(rng, tuple) or len(rng) != 2:
    st.stop()
sel = m.loc[str(rng[0]):str(rng[1])]
p = sel["price_BG"].dropna()
if p.empty:
    st.warning("No prices in the selected period.")
    st.stop()

n_days = (pd.Timestamp(rng[1]) - pd.Timestamp(rng[0])).days + 1
prev_p = m.loc[str(pd.Timestamp(rng[0]) - pd.Timedelta(days=n_days)):str(pd.Timestamp(rng[0]) - pd.Timedelta(days=1)), "price_BG"].dropna()
pct = lambda a, b: 100 * (a - b) / abs(b) if len(prev_p) and b else None
ch = pct(p.mean(), prev_p.mean())
neg = p < 0

with top:
    T.header("Market overview", "Bulgarian Day-Ahead Market",
             "Hourly day-ahead prices for bidding zone BG (SDAC), with Romania, Greece and Hungary for context. Prices in EUR/MWh, times in Europe/Sofia.",
             [f"{lo:%b %Y} – {hi:%b %Y}", f"{len(px_):,} hours", "15-min MTU since Oct 2025 → hourly mean"],
             headline=(f"Over {rng[0]:%d %b %Y} – {rng[1]:%d %b %Y} the average price was <b>€{p.mean():.0f}/MWh</b>"
                       + (f", <b>{ch:+.0f}%</b> versus the {n_days} days before" if ch is not None else "")
                       + f". The cheapest hour was <b>€{p.min():.0f}</b> ({p.idxmin():%d %b}); <b>{100 * neg.mean():.1f}%</b> of hours were negative."))

S.tomorrow_hero(top)

daily = p.resample("D").mean()
dstd = p.groupby(p.index.date).std()
spark_n = min(len(daily), 120)
k = st.columns(6)
T.kpi(k[0], "Average price", T.fmt_eur(p.mean()), "EUR/MWh · base load", accent=True, spark=daily.tail(spark_n),
      delta=T.delta_chip(ch), help="Mean of all hourly prices in the selected period; the chip compares with the previous period of equal length.")
T.kpi(k[1], "Peak hour", T.fmt_eur(p.max(), 0), f"{p.idxmax():%d %b %Y, %H:%M}", spark=p.resample("D").max().tail(spark_n), spark_color=T.UP,
      delta=T.delta_chip(pct(p.max(), prev_p.max()) if len(prev_p) else None))
T.kpi(k[2], "Lowest hour", T.fmt_eur(p.min(), 0), f"{p.idxmin():%d %b %Y, %H:%M}", spark=p.resample("D").min().tail(spark_n), spark_color=T.DOWN)
T.kpi(k[3], "Volatility", T.fmt_eur(dstd.mean()), "mean intraday std dev", spark=pd.Series(dstd.values).tail(spark_n), spark_color=T.VIOLET,
      delta=T.delta_chip(pct(dstd.mean(), prev_p.groupby(prev_p.index.date).std().mean()) if len(prev_p) else None))
T.kpi(k[4], "Negative hours", f"{100 * neg.mean():.1f}%", f"{int(neg.sum())} hours in period", spark=neg.resample("W").sum().tail(60), spark_color=T.DOWN)
if "price_RO" in sel:
    spread = (sel["price_BG"] - sel["price_RO"]).dropna()
    coupled = spread.abs() < T.COUPLED_TOL
    T.kpi(k[5], "BG − RO spread", T.fmt_eur(spread.mean(), 2), f"{100 * coupled.mean():.0f}% of hours within €1", spark=coupled.resample("W").mean().tail(60),
          spark_color=T.ZONE["RO"], help="Average price difference to Romania; hours within €1 count as coupled (same price).")

# --- main price chart -----------------------------------------------------------------------------------------------
rule = {"Hourly": None, "Daily": "D", "Weekly": "W"}[res]
series = p if rule is None else p.resample(rule).mean()
direction = "rose" if (ch or 0) > 0 else "fell"
with T.card(f"Prices {direction} {abs(ch):.0f}% versus the previous period" if ch is not None else f"BG day-ahead price · {res.lower()}",
            f"{res} average with the lowest–highest hour in each bucket. Labels mark the period's extremes.",
            T.legend_html([("Average", T.ACCENT), ("Min–max range", T.rgba(T.ACCENT, .4))] + ([("Previous year", T.NEUTRAL)] if compare else [])),
            help="Shaded band: the cheapest and the dearest hour inside each day/week. Use the 1M/3M/1Y buttons to zoom."):
    fig = go.Figure()
    if rule is not None:
        band = p.resample(rule).agg(["min", "max"])
        fig.add_trace(go.Scatter(x=band.index, y=band["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=band.index, y=band["min"], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .10), line=dict(width=0), name="min–max",
                                 customdata=band["max"], hovertemplate="range %{y:.0f} – %{customdata:.0f}<extra></extra>"))
    if compare:
        prev = m["price_BG"].dropna().loc[str(pd.Timestamp(rng[0]) - pd.DateOffset(years=1)):str(pd.Timestamp(rng[1]) - pd.DateOffset(years=1))]
        prev = prev if rule is None else prev.resample(rule).mean()
        prev.index = prev.index + pd.DateOffset(years=1)
        fig.add_trace(go.Scatter(x=prev.index, y=prev, name="previous year", line=dict(color=T.NEUTRAL, width=1.5),
                                 hovertemplate="a year earlier <b>%{y:.1f}</b><extra></extra>"))
        T.direct_label(fig, prev.index[-1], prev.iloc[-1], "previous year", T.NEUTRAL, yshift=-10)
    fig.add_trace(go.Scatter(x=series.index, y=series, name="average", line=dict(color=T.ACCENT, width=2),
                             hovertemplate="<b>%{y:.1f} EUR/MWh</b><extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.MUTED_DIM, width=1))
    if rule is not None:
        T.annotate(fig, p.idxmax().normalize() if rule else p.idxmax(), series.max(), f"peak €{p.max():.0f}", ay=-30)
        if p.min() < 0:
            T.annotate(fig, p.idxmin().normalize(), series.min(), f"low €{p.min():.0f}", ay=34)
    fig.update_yaxes(title="EUR/MWh")
    T.range_buttons(fig)
    T.chart(fig, 430, legend=False)

# --- how prices are distributed ---------------------------------------------------------------------------------------
c1, c2 = st.columns(2)
srt = np.sort(p.to_numpy())[::-1]
share_neg = 100 * neg.mean()
with c1, T.card(f"{share_neg:.1f}% of hours cleared below zero; the top 5% cost above €{np.percentile(p, 95):.0f}",
                "Price-duration curve: every hour of the period sorted from dearest to cheapest",
                help="Reads left to right: how many hours exceed a given price. The flat tail on the right is negative-price hours."):
    pct_h = np.linspace(0, 100, len(srt))
    fig = go.Figure(go.Scatter(x=pct_h, y=srt, mode="lines", line=dict(color=T.ACCENT, width=2), fill="tozeroy", fillcolor=T.rgba(T.ACCENT, .10),
                               name="BG price", hovertemplate="%{x:.0f}% of hours are above <b>€%{y:.0f}</b><extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.MUTED_DIM, width=1))
    if share_neg > 0.2:
        fig.add_vline(x=100 - share_neg, line=dict(color=T.MUTED_DIM, width=1))
    fig.update_xaxes(title="share of hours (%)", ticksuffix="%", showspikes=False)
    fig.update_yaxes(title="EUR/MWh")
    fig.update_layout(hovermode="closest")
    T.chart(fig, 330, legend=False)
hour_stats = p.groupby(p.index.hour).quantile([0.1, 0.5, 0.9]).unstack()
with c2, T.card(f"A typical day: cheapest at {hour_stats[0.5].idxmin()}:00 (€{hour_stats[0.5].min():.0f}), dearest at {hour_stats[0.5].idxmax()}:00 (€{hour_stats[0.5].max():.0f})",
                "Median price by hour of day with the 10–90 % range", T.legend_html([("Median", T.ACCENT), ("10–90 % of days", T.rgba(T.ACCENT, .4))] +
                                                                                 ([("Previous period", T.NEUTRAL)] if len(prev_p) else [])),
                help="The spread between the midday trough and the evening peak is what a battery earns."):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hour_stats.index, y=hour_stats[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hour_stats.index, y=hour_stats[0.1], fill="tonexty", fillcolor=T.rgba(T.ACCENT, .14), line=dict(width=0), name="10–90 %",
                             customdata=hour_stats[0.9], hovertemplate="10–90%: %{y:.0f} – %{customdata:.0f}<extra></extra>"))
    if len(prev_p):
        pm = prev_p.groupby(prev_p.index.hour).median()
        fig.add_trace(go.Scatter(x=pm.index, y=pm, name="previous period", line=dict(color=T.NEUTRAL, width=1.5), hovertemplate="previous <b>%{y:.0f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=hour_stats.index, y=hour_stats[0.5], name="median", line=dict(color=T.ACCENT, width=2.4, shape="spline"),
                             hovertemplate="median <b>%{y:.0f}</b><extra></extra>"))
    fig.update_xaxes(title="hour (local)", dtick=3, range=[-0.3, 23.3])
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 330, legend=False)

# --- the daily shape through time (animated) ------------------------------------------------------------------------------------
_mk = px_.index.strftime("%Y-%m")
_mh = px_.groupby([_mk, px_.index.hour]).median().unstack()
_mh = _mh[_mh.notna().sum(axis=1) == 24]
_mh = _mh[px_.groupby(_mk).count().reindex(_mh.index) >= 24 * 20]     # only months with at least 20 days of prices
if len(_mh) >= 6:
    _depth = _mh[[18, 19, 20, 21]].max(axis=1) - _mh[[10, 11, 12, 13, 14, 15]].min(axis=1)
    _best = _depth.idxmax()
    _ref = _mh.median()
    _frames = [go.Frame(name=mo, data=[go.Scatter(y=_mh.loc[mo].tolist())], traces=[1],
                        layout=go.Layout(annotations=[dict(x=0.99, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                                                            text=f"<b>{mo}</b>  gap €{_depth.loc[mo]:.0f}", font=dict(size=14, color=T.TEXT))]))
               for mo in _mh.index]
    with T.card(f"The gap between evening peak and midday trough peaked at €{_depth.max():.0f} in {_best}; the latest month shows €{_depth.iloc[-1]:.0f}",
                "Median price by hour for each month since 2023. Press play, or drag the slider, to watch the solar dip deepen and fade with the seasons.",
                T.legend_html([("Selected month", T.ACCENT), ("Median of all months", T.NEUTRAL)])):
        fig = go.Figure(data=[go.Scatter(x=list(range(24)), y=_ref.tolist(), name="all months", line=dict(color=T.NEUTRAL, width=1.6, dash="dot"), hovertemplate="all months <b>%{y:.0f}</b><extra></extra>"),
                              go.Scatter(x=list(range(24)), y=_mh.iloc[-1].tolist(), name="selected month", line=dict(color=T.ACCENT, width=3, shape="spline"),
                                         fill="tonexty", fillcolor=T.rgba(T.ACCENT, .10), hovertemplate="%{x}:00 · <b>€%{y:.0f}</b><extra></extra>")],
                        frames=_frames)
        _lo, _hi = float(np.nanmin(_mh.to_numpy())), float(np.nanmax(_mh.to_numpy()))
        fig.update_yaxes(title="EUR/MWh", range=[min(_lo - 10, 0), _hi + 15])
        fig.update_xaxes(title="hour (local)", dtick=3, range=[-0.3, 23.3])
        fig.add_annotation(x=0.99, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False,
                           text=f"<b>{_mh.index[-1]}</b>  gap €{_depth.iloc[-1]:.0f}", font=dict(size=14, color=T.TEXT))
        fig.update_layout(hovermode="x unified",
                          updatemenus=[dict(type="buttons", direction="left", active=-1, x=0, xanchor="left", y=-0.16, yanchor="top", pad=dict(t=0, r=8),
                                            bgcolor=T.SURFACE_2, bordercolor=T.BORDER, font=dict(color=T.TEXT, size=12),
                                            buttons=[dict(label="▶ Play", method="animate",
                                                          args=[None, dict(frame=dict(duration=380, redraw=True), transition=dict(duration=260, easing="cubic-in-out"), fromcurrent=True)]),
                                                     dict(label="❚❚ Pause", method="animate",
                                                          args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])],
                          sliders=[dict(active=len(_mh) - 1, x=0.2, len=0.8, y=-0.14, yanchor="top", pad=dict(t=0, b=0), bgcolor=T.BORDER, bordercolor=T.BORDER,
                                        activebgcolor=T.ACCENT_UI, tickcolor=T.MUTED_DIM, font=dict(color=T.MUTED, size=11), currentvalue=dict(visible=False),
                                        steps=[dict(method="animate", label=mo[2:] if i % 3 == 0 else "", args=[[mo], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
                                               for i, mo in enumerate(_mh.index)])],
                          margin=dict(b=70))
        T.chart(fig, 430, legend=False, table=False, desc="Animated: one frame per month")
        with st.expander("Table view", icon=":material/table:"):
            T.table_html(_mh.round(1).set_axis([f"{h:02d}" for h in range(24)], axis=1).rename_axis("Month"), {f"{h:02d}": "{:.0f}" for h in range(24)})

# --- calendar -----------------------------------------------------------------------------------------------------------------
cal = daily.to_frame("price")
cal.index = cal.index.tz_localize(None)
cal["dow"] = cal.index.day_of_week
cal["week"] = cal.index - pd.to_timedelta(cal["dow"], unit="D")   # Monday of each week
piv = cal.pivot_table(index="dow", columns="week", values="price")
with T.card("Which days were dear? The daily price calendar", "One cell per delivery day: brighter = more expensive. Hover for the date.",
            help="Weekends (bottom rows) are usually the darkest cells: lower demand and more solar surplus."):
    txt = cal.assign(d=cal.index.strftime("%a %d %b %Y")).pivot_table(index="dow", columns="week", values="d", aggfunc="first")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                               customdata=txt.reindex(index=piv.index, columns=piv.columns).values, colorscale=T.SEQ, xgap=3, ygap=3,
                               colorbar=dict(title=dict(text="€/MWh", side="right")),
                               hovertemplate="%{customdata}<br><b>%{z:.1f} EUR/MWh</b><extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showspikes=False, tickformat="%b %y")
    T.chart(fig, 250, legend=False, table=False)

# --- full history ------------------------------------------------------------------------------------------------------------
T.section("Full history", "Monthly averages and negative-price hours since 2023; the selected period is highlighted")
monthly = px_.resample("MS").agg(["mean", lambda s: (s < 0).sum()])
monthly.columns = ["mean", "neg"]
avg_all = px_.mean()
in_sel = (monthly.index >= pd.Timestamp(rng[0]).replace(day=1).tz_localize(monthly.index.tz)) & \
         (monthly.index <= pd.Timestamp(rng[1], tz=monthly.index.tz))
c1, c2 = st.columns(2)
with c1, T.card(f"The latest month averaged €{monthly['mean'].iloc[-1]:.0f}, against €{avg_all:.0f} over the whole history",
                "Monthly average price · dashed line = full-history average"):
    f2 = go.Figure(go.Bar(x=monthly.index, y=monthly["mean"], name="monthly average", marker=dict(color=np.where(in_sel, T.ACCENT, T.NEUTRAL), cornerradius=4),
                          hovertemplate="%{x|%b %Y}: <b>%{y:.1f} EUR/MWh</b><extra></extra>"))
    f2.add_hline(y=avg_all, line=dict(color=T.TEXT_2, width=1, dash="dash"))
    f2.update_layout(hovermode="closest", bargap=0.35)
    f2.update_yaxes(title="EUR/MWh")
    T.chart(f2, 300, legend=False)
with c2, T.card(f"Negative prices: {int(monthly['neg'].sum())} hours since 2023, mostly in spring and on sunny middays",
                "Negative-price hours per month"):
    f3 = go.Figure(go.Bar(x=monthly.index, y=monthly["neg"], name="negative hours", marker=dict(color=T.DOWN, cornerradius=4),
                          hovertemplate="%{x|%b %Y}: <b>%{y} hours</b><extra></extra>"))
    f3.update_layout(hovermode="closest", bargap=0.35)
    f3.update_yaxes(title="hours")
    T.chart(f3, 300, legend=False)

yr = px_.groupby(px_.index.year).agg(["mean", "std", "min", "max", lambda s: int((s < 0).sum()), "count"])
yr.columns = ["Average", "Std dev", "Min", "Max", "Negative hours", "Hours"]
yr.index.name = "Year"
yr.index = [f"{y}{' (YTD)' if c < 8700 else ''}" for y, c in zip(yr.index, yr["Hours"])]
T.takeaway(f"The yearly average moved from <b>€{yr['Average'].iloc[0]:.0f}</b> to <b>€{yr['Average'].iloc[-1]:.0f}/MWh</b>, while negative-price hours rose "
           f"from <b>{int(yr['Negative hours'].iloc[0])}</b> to <b>{int(yr['Negative hours'].iloc[-1])}</b> a year: more solar makes midday cheaper and evenings relatively dearer.")
T.table(yr.drop(columns="Hours"), fmt={"Average": "{:.1f}", "Std dev": "{:.1f}", "Min": "{:.1f}", "Max": "{:.1f}", "Negative hours": "{:.0f}"},
        bars={"Average": T.ACCENT, "Negative hours": T.DOWN})
T.download(yr, "yearly-price-summary", key="dl_year")

T.next_steps([("views/0_Trader_Brief.py", "Trader brief", "Tomorrow's forecast and what to do about it."),
              ("views/1_Price_Drivers.py", "Price drivers", "Why prices move: solar, residual load, gas."),
              ("views/2_Cross_Border.py", "Cross-border", "How tightly Bulgaria follows Romania, Greece, Hungary.")])

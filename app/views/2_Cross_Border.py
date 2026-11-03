"""Cross-border: BG vs RO/GR/HU, regional map, spreads, coupling, flows."""
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

T.setup("Cross-Border")
top = st.container()
m = T.load_market()
if m is None or "price_BG" not in m:
    with top:
        T.header("Market", "Cross-border coupling")
    T.missing("Market data")
    st.stop()

zones = [z for z in ("BG", "RO", "GR", "HU") if f"price_{z}" in m]
P = m[[f"price_{z}" for z in zones]].rename(columns=lambda c: c[6:])
nb = [z for z in zones if z != "BG"]
ISO = {"BG": "BGR", "RO": "ROU", "GR": "GRC", "HU": "HUN"}
NAME = {"BG": "Bulgaria", "RO": "Romania", "GR": "Greece", "HU": "Hungary"}

with T.controls("Period"):
    period = st.segmented_control("Show", ["Last 30 days", "Last 12 months", "Full history"], default="Last 12 months") or "Last 12 months"
end = P.index.max()
start = {"Last 30 days": end - pd.Timedelta(days=30), "Last 12 months": end - pd.DateOffset(years=1), "Full history": P.index.min()}[period]
Pp = P.loc[start:]
coupled = pd.DataFrame({z: ((P["BG"] - P[z]).abs() < T.COUPLED_TOL).where(P[z].notna() & P["BG"].notna()) for z in nb}).astype(float)
shares = {z: coupled.loc[start:, z].mean() * 100 for z in nb}
hist = {z: coupled[z].mean() * 100 for z in nb}
best = max(shares, key=shares.get)
worst = min(shares, key=shares.get)
with top:
    T.header("Market", "Cross-border coupling",
             "Bulgaria is part of the single day-ahead coupling (SDAC). When interconnectors have spare capacity the zones clear at the same price; "
             f"when they congest, prices split. Coupled = |spread| below €{T.COUPLED_TOL:.0f}/MWh.",
             headline=(f"In the selected period BG cleared at the same price as <b>{NAME[best]}</b> in <b>{shares[best]:.0f}%</b> of hours but only "
                       f"<b>{shares[worst]:.0f}%</b> with {NAME[worst]}; coupling with Romania is <b>{shares.get('RO', np.nan) - hist.get('RO', np.nan):+.0f} pp</b> versus its full history."
                       if "RO" in shares else ""))

# --- KPIs --------------------------------------------------------------------------------------------------------------------
k = st.columns(len(nb) + 1)
for col, z in zip(k, nb):
    T.kpi(col, f"Coupled with {z}", f"{shares[z]:.0f}%", f"vs {hist[z]:.0f}% over the full history", spark=coupled[z].resample("MS").mean().tail(24), spark_color=T.ZONE[z],
          delta=T.delta_chip(shares[z] - hist[z], " pp", good_when_up=True, digits=0), help=f"Share of hours with |BG − {z}| below €{T.COUPLED_TOL:.0f}/MWh.")
if "net_import" in m:
    ni = m["net_import"].loc[start:]
    T.kpi(k[-1], "Net position", f"{ni.mean():+,.0f} MW", "importer" if ni.mean() > 0 else "net exporter", spark=m["net_import"].resample("W").mean().tail(60), spark_color=T.AQUA,
          help="Average physical net import into Bulgaria (positive) or export (negative).")

# --- map + price lines ----------------------------------------------------------------------------------------------------------
c1, c2 = st.columns([1, 1.6])
avg = Pp.mean()
dearest = avg.idxmax()
with c1, T.card(f"{NAME[dearest]} was the dearest of the four markets (€{avg[dearest]:.0f}); Bulgaria averaged €{avg['BG']:.0f}", f"Average day-ahead price · {period.lower()}"):
    fig = go.Figure(go.Choropleth(locations=[ISO[z] for z in zones], z=[avg[z] for z in zones], text=[NAME[z] for z in zones],
                                  colorscale=[[0, "#3a2c10"], [0.5, "#a06c0f"], [1, "#f0b232"]], marker=dict(line=dict(color=T.BG, width=1.2)),
                                  colorbar=dict(title=dict(text="€/MWh", side="right"), len=0.7),
                                  hovertemplate="%{text}<br><b>%{z:.1f} EUR/MWh</b><extra></extra>"))
    for z in zones:
        fig.add_trace(go.Scattergeo(locations=[ISO[z]], text=[f"<b>{z}</b><br>€{avg[z]:.0f}"], mode="text", textfont=dict(color=T.TEXT, size=13, family=T.FONT),
                                    hoverinfo="skip", showlegend=False))
    fig.update_geos(projection_type="mercator", lataxis_range=[34.5, 49], lonaxis_range=[15, 31], showframe=False, bgcolor="rgba(0,0,0,0)", showland=True,
                    landcolor="#161C27", showcountries=True, countrycolor="#2A3446", showcoastlines=True, coastlinecolor="#2A3446", showocean=True,
                    oceancolor="#0B0E14", showlakes=False)
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=0), hovermode="closest")
    T.chart(fig, 380, legend=False, desc="Map of Bulgaria, Romania, Greece and Hungary shaded by average price.")
with c2, T.card("The four markets move almost in lockstep, with Greece the one that drifts apart", "Weekly average day-ahead price; Bulgaria in bold",
                T.legend_html([(z, T.ZONE[z]) for z in zones]),
                help="Emphasis chart: Bulgaria is the thick line, neighbours are context. Names sit at the line ends."):
    w = Pp.resample("W" if period != "Last 30 days" else "D").mean()
    fig = go.Figure()
    for z in nb + ["BG"]:
        fig.add_trace(go.Scatter(x=w.index, y=w[z], name=z, line=dict(color=T.ZONE[z], width=3 if z == "BG" else 1.6), opacity=1 if z == "BG" else 0.9,
                                 hovertemplate=z + ": <b>%{y:.1f}</b><extra></extra>"))
    ends = w.iloc[-1].sort_values()
    for rank, (z, v) in enumerate(ends.items()):                       # nudge labels apart so close ends stay readable
        T.direct_label(fig, w.index[-1], v, z, T.ZONE[z], yshift=(rank - 1.5) * 9)
    fig.update_xaxes(range=[w.index[0], w.index[-1] + (w.index[-1] - w.index[0]) * 0.07])
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 380, legend=False)

# --- spreads small multiples ------------------------------------------------------------------------------------------------------
T.section("Spreads", "BG minus neighbour, daily average. Above zero = Bulgaria more expensive.")
mean_sp = {z: (Pp["BG"] - Pp[z]).mean() for z in nb}
with T.card(f"On average BG was {abs(mean_sp[max(mean_sp, key=lambda z: abs(mean_sp[z]))]):.0f} EUR/MWh "
            f"{'above' if mean_sp[max(mean_sp, key=lambda z: abs(mean_sp[z]))] > 0 else 'below'} {max(mean_sp, key=lambda z: abs(mean_sp[z]))} — the widest gap of the three",
            "Daily spread BG − neighbour · red = BG dearer, blue = BG cheaper; a flat line at zero = fully coupled",
            T.legend_html([("BG dearer", T.UP), ("BG cheaper", T.DOWN)], "box")):
    fig = make_subplots(rows=1, cols=len(nb), shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[f"BG − {z}" for z in nb])
    for i, z in enumerate(nb, start=1):
        s = (Pp["BG"] - Pp[z]).resample("D").mean()
        fig.add_trace(go.Bar(x=s.index, y=s, name=f"BG − {z}", marker=dict(color=np.where(s > 0, T.UP, T.DOWN), line=dict(width=0)),
                             hovertemplate="%{x|%d %b %Y}: <b>%{y:+.1f}</b><extra>" + z + "</extra>"), row=1, col=i)
        fig.update_xaxes(tickformat="%b %y", nticks=5, row=1, col=i)
    for i, a in enumerate(fig.layout.annotations):
        dom = fig.layout[f"xaxis{'' if i == 0 else i + 1}"].domain
        a.update(x=dom[0], xanchor="left", font=dict(size=13, color=T.TEXT_2))
    fig.update_layout(hovermode="closest", bargap=0)
    fig.update_yaxes(title="EUR/MWh", row=1, col=1)
    T.chart(fig, 300, legend=False)

c1, c2 = st.columns(2)
mo = 100 * coupled.resample("MS").mean()
with c1, T.card(f"Coupling with Romania {'fell' if mo['RO'].iloc[-1] < mo['RO'].dropna().iloc[:-1].tail(12).mean() else 'held'} to {mo['RO'].iloc[-1]:.0f}% in the latest month"
                if "RO" in mo else "Market coupling convergence", f"Share of hours each month with |BG − neighbour| < €{T.COUPLED_TOL:.0f}",
                T.legend_html([(z, T.ZONE[z]) for z in nb])):
    fig = go.Figure()
    for z in nb:
        fig.add_trace(go.Scatter(x=mo.index, y=mo[z], name=z, mode="lines+markers", line=dict(color=T.ZONE[z], width=2), marker=dict(size=6, line=dict(color=T.SURFACE, width=1.5)),
                                 hovertemplate=z + ": <b>%{y:.0f}%</b><extra></extra>"))
    last = mo.iloc[-1].sort_values()
    for rank, (z, v) in enumerate(last.items()):
        T.direct_label(fig, mo.index[-1], v, z, T.ZONE[z], yshift=(rank - 1) * 10)
    fig.update_xaxes(range=[mo.index[0], mo.index[-1] + (mo.index[-1] - mo.index[0]) * 0.06])
    fig.update_yaxes(title="% of hours", range=[0, 102], ticksuffix="%")
    T.chart(fig, 340, legend=False)
full = P.loc[start:].dropna()
others = full[[z for z in ("GR", "HU") if z in full]]
bg_high = full["BG"] >= others.max(axis=1) + T.COUPLED_TOL
bg_low = full["BG"] <= others.min(axis=1) - T.COUPLED_TOL
hour = full.index.hour
share_hi, share_lo = bg_high.groupby(hour).mean() * 100, bg_low.groupby(hour).mean() * 100
with c2, T.card(f"BG is the expensive side mostly around {int(share_hi.idxmax())}:00 and the cheap side around {int(share_lo.idxmax())}:00",
                "Share of hours BG (moving with RO) is above / below both GR and HU by at least €1", T.legend_html([("BG above both", T.UP), ("BG below both", T.DOWN)], "box")):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=share_hi.index, y=share_hi, name="BG above both", marker=dict(color=T.UP, cornerradius=3), hovertemplate="%{x}:00 above: <b>%{y:.1f}%</b><extra></extra>"))
    fig.add_trace(go.Bar(x=share_lo.index, y=-share_lo, name="BG below both", marker=dict(color=T.DOWN, cornerradius=3), customdata=share_lo,
                         hovertemplate="%{x}:00 below: <b>%{customdata:.1f}%</b><extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="hour (local)", dtick=3)
    fig.update_yaxes(title="% of hours", ticksuffix="%")
    T.chart(fig, 340, legend=False)

tab = []
for z in nb:
    s = (Pp["BG"] - Pp[z]).dropna()
    tab.append({"Neighbour": z, "Avg spread": s.mean(), "Mean |spread|": s.abs().mean(), "Coupled %": 100 * (s.abs() < T.COUPLED_TOL).mean(),
                "BG dearer %": 100 * (s >= T.COUPLED_TOL).mean(), "BG cheaper %": 100 * (s <= -T.COUPLED_TOL).mean()})
tab = pd.DataFrame(tab).set_index("Neighbour")
T.takeaway(f"BG is most tightly coupled with <b>{best}</b> ({tab.loc[best, 'Coupled %']:.0f}% of hours within €1 in this period). Decoupled hours are where local "
           "fundamentals and cross-border spread trades matter.")
T.table(tab, fmt={c: "{:.1f}" for c in tab.columns}, bars={"Coupled %": T.BLUE})

# --- flows -------------------------------------------------------------------------------------------------------------------------------------
flows = [c for c in m.columns if c.startswith("flow_")]
if "net_import" in m and flows:
    T.section("Physical flows", "ENTSO-E cross-border physical flows (MW)")
    c1, c2 = st.columns([1.6, 1])
    ni_w = m["net_import"].loc[start:].resample("W" if period != "Last 30 days" else "D").mean()
    with c1, T.card(f"Bulgaria was a net {'importer' if ni_w.mean() > 0 else 'exporter'} on average ({ni_w.mean():+,.0f} MW)", "Net physical import into Bulgaria · weekly average · above zero = importing",
                    T.legend_html([("Import", T.UP), ("Export", T.DOWN)], "box")):
        fig = go.Figure(go.Bar(x=ni_w.index, y=ni_w, name="net import", marker=dict(color=np.where(ni_w > 0, T.UP, T.DOWN), cornerradius=2),
                               hovertemplate="%{x|%d %b %Y}: <b>%{y:+,.0f} MW</b><extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(title="MW")
        T.chart(fig, 320, legend=False)
    with c2, T.card("Average flow per border", "Import into BG vs export out of BG"):
        nbs = sorted({c.split("_")[1] if c.endswith("_to_BG") else c.split("_")[-1] for c in flows})
        imp = [m[f"flow_{n}_to_BG"].loc[start:].mean() if f"flow_{n}_to_BG" in m else 0 for n in nbs]
        exp = [m[f"flow_BG_to_{n}"].loc[start:].mean() if f"flow_BG_to_{n}" in m else 0 for n in nbs]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=nbs, x=imp, orientation="h", name="import", marker=dict(color=T.UP, cornerradius=3), hovertemplate="%{y} → BG: <b>%{x:,.0f} MW</b><extra></extra>"))
        fig.add_trace(go.Bar(y=nbs, x=[-e for e in exp], orientation="h", name="export", marker=dict(color=T.DOWN, cornerradius=3), customdata=exp,
                             hovertemplate="BG → %{y}: <b>%{customdata:,.0f} MW</b><extra></extra>"))
        fig.update_layout(barmode="relative", hovermode="closest")
        fig.update_xaxes(title="← export · MW · import →", showgrid=True, showspikes=False)
        fig.update_yaxes(showgrid=False)
        T.chart(fig, 320, legend=False)

T.next_steps([("views/1_Price_Drivers.py", "Price drivers", "What sets the Bulgarian price level."),
              ("views/3_Forecast.py", "Forecast", "Neighbour prices are among the top model inputs."),
              ("overview.py", "Market overview", "Back to prices and KPIs.")])

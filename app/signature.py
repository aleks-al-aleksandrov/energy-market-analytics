"""Signature components for the Overview: the 24-hour market clock, the hour ribbon and "Ask the market".

All numbers are computed from the published next-day forecast and the project's own back-tests; nothing here is asserted.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T
from i18n import tr
from src import battery as B

QUESTIONS = ["When is power cheapest?", "When should I avoid it?", "Is a price spike likely?", "What would a battery earn?", "Can I trust this?"]


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def level_color(x: float) -> str:
    """Interpolate the sequential ramp of the current mode at position x in [0, 1]."""
    ramp = T.SEQ
    x = float(np.clip(x, 0, 1))
    for (a, ca), (b, cb) in zip(ramp[:-1], ramp[1:]):
        if x <= b:
            t = 0 if b == a else (x - a) / (b - a)
            ra, rb = _rgb(ca), _rgb(cb)
            return "#%02x%02x%02x" % tuple(int(round(u + (v - u) * t)) for u, v in zip(ra, rb))
    return ramp[-1][1]


def _lum(h: str) -> float:
    def f(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(v) for v in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ink_on(bg: str) -> str:
    """Black or white, whichever has the higher contrast on bg (the ribbon text is never colour-only)."""
    lb = _lum(bg)
    dark, light = "#0B0E14", "#FFFFFF"
    cd, cl = (lb + 0.05) / (_lum(dark) + 0.05), (_lum(light) + 0.05) / (lb + 0.05)
    return dark if cd >= cl else light


def _ranges(hours: list[str]) -> str:
    """['12:00','13:00','14:00','18:00'] -> '12:00–14:00 and 18:00'."""
    if not hours:
        return "—"
    idx = [int(h[:2]) for h in hours]
    out, start, prev = [], idx[0], idx[0]
    for v in idx[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        out.append(f"{start:02d}:00" if start == prev else f"{start:02d}:00–{prev:02d}:00")
        if v is not None:
            start = prev = v
    return " and ".join(out)


def _window(f: np.ndarray, n: int, cheapest: bool) -> int:
    means = np.convolve(f, np.ones(n) / n, mode="valid")
    return int(means.argmin() if cheapest else means.argmax())


def answer(q: str, nd: pd.DataFrame) -> dict:
    """Return {'text': html, 'hi': set(hour index), 'hi2': set(hour index)} for one question."""
    f = nd["forecast_eur_mwh"].to_numpy()
    hrs = nd["time_local"].str[-5:].tolist()
    sp = nd["spike_probability"].to_numpy()
    base = float(f.mean())
    if q == QUESTIONS[0]:
        i = _window(f, 3, True)
        w = f[i:i + 3].mean()
        return dict(hi=set(range(i, i + 3)), hi2=set(),
                    text=(f"The cheapest three hours are <b>{hrs[i]}–{(int(hrs[i + 2][:2]) + 1) % 24:02d}:00</b>, averaging <b>€{w:.0f}/MWh</b>, "
                          f"{100 * (1 - w / base):.0f}% below the day's average of €{base:.0f}. If you can shift a washing machine, an EV charge or a heat pump, this is the window."))
    if q == QUESTIONS[1]:
        i = _window(f, 3, False)
        w = f[i:i + 3].mean()
        return dict(hi=set(range(i, i + 3)), hi2=set(),
                    text=(f"The dearest three hours are <b>{hrs[i]}–{(int(hrs[i + 2][:2]) + 1) % 24:02d}:00</b>, averaging <b>€{w:.0f}/MWh</b>, "
                          f"{100 * (w / base - 1):.0f}% above the day's average of €{base:.0f}. Avoid flexible loads then."))
    if q == QUESTIONS[2]:
        j = int(sp.argmax())
        hot = {int(k) for k in np.where(sp >= max(0.10, 0.5 * sp.max()))[0]}
        thr = float(nd["spike_threshold_eur"].iloc[0]) if "spike_threshold_eur" in nd else float("nan")
        return dict(hi=hot, hi2=set(),
                    text=(f"The highest spike risk is <b>{100 * sp[j]:.0f}%</b> at <b>{hrs[j]}</b>"
                          + (f" (a spike means a price above about €{thr:.0f}/MWh)" if np.isfinite(thr) else "")
                          + f". The probability is the model's own estimate and is reasonably calibrated on the back-test; "
                            f"{'no hour is above 10%, so tomorrow looks calm.' if sp.max() < 0.10 else 'hours above 10% are marked.'}"))
    if q == QUESTIONS[3]:
        net = B.schedule_day(f, B.Battery())
        ch = [hrs[k] for k in np.where(net < -1e-6)[0]]
        dis = [hrs[k] for k in np.where(net > 1e-6)[0]]
        if not ch:
            return dict(hi=set(), hi2=set(), text="The spread is too small to cover round-trip losses, so a 1 MW / 2 MWh battery would stay idle tomorrow.")
        val = float((net * f).sum())
        mc = (T.load_json("mc_summary.json") or {}).get("next_day")
        rng = (f" A Monte Carlo of past forecast errors gives a realistic range of <b>{T.fmt_eur(mc['p10'], 0)}–{T.fmt_eur(mc['p90'], 0)}</b>."
               if mc and str(mc.get("delivery_day", ""))[:10] == str(nd["delivery_day"].iloc[0])[:10] else "")
        return dict(hi={int(k) for k in np.where(net < -1e-6)[0]}, hi2={int(k) for k in np.where(net > 1e-6)[0]},
                    text=(f"A 1 MW / 2 MWh battery would charge at <b>{_ranges(ch)}</b> and discharge at <b>{_ranges(dis)}</b>, "
                          f"earning about <b>{T.fmt_eur(val, 0)}</b> at the forecast prices.{rng}"))
    sc = T.load_csv("skill_confidence.csv")
    if sc is not None:
        g = sc[sc["variant"] == "lightgbm"].set_index("split")
        if {"cv", "holdout"} <= set(g.index):
            c, h = g.loc["cv"], g.loc["holdout"]
            return dict(hi=set(), hi2=set(),
                        text=(f"On the validation months the model beat the \"same as yesterday\" baseline by <b>{100 * c['skill']:.0f}%</b> "
                              f"(95% interval {100 * c['skill_lo']:.0f} to {100 * c['skill_hi']:.0f}%). On the untouched last three months the edge shrank to "
                              f"<b>{100 * h['skill']:.0f}%</b> (interval {100 * h['skill_lo']:.0f} to {100 * h['skill_hi']:.0f}%), which includes zero. "
                              f"So treat the shape of the day as reliable and the exact level with care."))
    return dict(hi=set(), hi2=set(), text="Skill statistics are not available yet.")


def ribbon_html(nd: pd.DataFrame, hi: set[int], hi2: set[int]) -> str:
    f = nd["forecast_eur_mwh"].to_numpy()
    hrs = nd["time_local"].str[-5:].tolist()
    sp = nd["spike_probability"].to_numpy()
    lo, hi_ = float(f.min()), float(f.max())
    span = (hi_ - lo) or 1.0
    cells, bar = [], []
    for i, (h, v, s) in enumerate(zip(hrs, f, sp)):
        col = level_color((v - lo) / span)
        mark = "⚡" if s >= 0.10 else "▼" if i == int(f.argmin()) else "▲" if i == int(f.argmax()) else ""
        what = tr("spike risk") + f" {100 * s:.0f}%"
        cls = "rb-cell" + (" hi" if i in hi else "") + (" hi2" if i in hi2 else "")
        night = int(h[:2]) >= 22 or int(h[:2]) < 6
        cells.append(f"<li class='{cls}' style='--c:{col};--ink:{ink_on(col)}' title='{h} · €{v:.0f} · {what}' "
                     f"aria-label='{h}, €{v:.0f} per megawatt-hour, {what}'><span class='rb-h'>{h[:2]}</span><span class='rb-p'>{v:.0f}</span>"
                     f"<span class='rb-m' aria-hidden='true'>{mark}</span></li>")
        bar.append(f"<i class='{'n' if night else 'd'}'></i>")
    return (f"<div class='ribbon' role='group' aria-label='{html.escape(tr('Forecast price by hour, EUR/MWh'))}'><ol>{''.join(cells)}</ol>"
            f"<div class='rb-day' aria-hidden='true'><span>☾</span>{''.join(bar)}<span>☀</span></div>"
            f"<div class='rb-key'><span>▼ {tr('cheapest')}</span><span>▲ {tr('dearest')}</span><span>⚡ {tr('spike risk')} ≥ 10%</span>"
            f"<span class='k-hi'>{tr('highlighted by your question')}</span></div></div>")


def clock_figure(nd: pd.DataFrame, hi: set[int], hi2: set[int]) -> go.Figure:
    f = nd["forecast_eur_mwh"].to_numpy()
    hrs = nd["time_local"].str[-5:].tolist()
    sp = nd["spike_probability"].to_numpy()
    lo, top = float(f.min()), float(f.max())
    span = (top - lo) or 1.0
    base = min(lo, 0) - 0.12 * span
    colors = [level_color((v - lo) / span) for v in f]
    edge = [T.ACCENT_UI if i in hi else T.DOWN if i in hi2 else T.SURFACE for i in range(len(f))]
    width = [3.5 if (i in hi or i in hi2) else 1.5 for i in range(len(f))]
    labels = [h[:2] for h in hrs]
    fig = go.Figure(go.Barpolar(
        r=f - base, theta=labels, marker=dict(color=colors, line=dict(color=edge, width=width)), name="forecast €/MWh", customdata=np.c_[f, 100 * sp],
        hovertemplate="%{theta}:00 · <b>€%{customdata[0]:.0f}</b><br>spike risk %{customdata[1]:.0f}%<extra></extra>"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", hole=0.34, radialaxis=dict(visible=False), sector=[0, 360],
                   angularaxis=dict(direction="clockwise", rotation=90, categoryarray=labels, tickmode="array", tickvals=labels[::3],
                                    tickfont=dict(size=12, color=T.TEXT_2), gridcolor=T.GRID, linecolor=T.GRID, ticks="")),
        hovermode="closest", margin=dict(l=18, r=18, t=18, b=18))
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
                       text=f"<b style='font-size:22px;color:{T.TEXT}'>€{f.mean():.0f}</b><br><span style='color:{T.MUTED}'>{tr('day average')}</span>")
    return fig


def tomorrow_hero(target) -> None:
    """The Overview's centrepiece. `target` is the container the header was written into."""
    nd = T.load_csv("forecast_next_day.csv")
    if nd is None or len(nd) != 24:
        return
    day = pd.Timestamp(nd["delivery_day"].iloc[0])
    with target:
        T.section(f"Tomorrow, {day:%A %d %B}", "The published next-day forecast as a clock, an hour ribbon and a few plain questions", anchor="tomorrow")
        c1, c2 = st.columns([1, 1.15])
        with c2, T.card("Ask the market", "Pick a question: the answer is computed from tomorrow's forecast and lights up the clock and the ribbon"):
            q = st.pills("Question", QUESTIONS, selection_mode="single", default=QUESTIONS[0], key="ask_q", label_visibility="collapsed") or QUESTIONS[0]
            ans = answer(q, nd)
            T.takeaway(ans["text"], "action" if q == QUESTIONS[3] else "insight")
            T.link("views/0_Trader_Brief.py", "Open the full trader brief")
        with c1, T.card(f"Tomorrow around the clock · €{nd['forecast_eur_mwh'].min():.0f} to €{nd['forecast_eur_mwh'].max():.0f}",
                        "Each wedge is one hour, midnight at the top; longer and brighter means dearer"):
            T.chart(clock_figure(nd, ans["hi"], ans["hi2"]), 330, legend=False)
        st.markdown(ribbon_html(nd, ans["hi"], ans["hi2"]), unsafe_allow_html=True)

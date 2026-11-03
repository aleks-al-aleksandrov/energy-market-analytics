"""Beat the model: schedule a 1 MW / 2 MWh battery for a past day and see how it settles against the model and perfect foresight."""
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

T.setup("Beat the Model")
pred = T.load_predictions()
if pred is None:
    T.header("Play", "Beat the model")
    T.missing("Back-test predictions (reports/predictions.csv)")
    st.stop()

BAT = B.Battery()
SE = float(np.sqrt(BAT.efficiency))
HOURS = [f"{h:02d}" for h in range(24)]


@st.cache_data(show_spinner=False)
def _days(mtime: float) -> list[dict]:
    """One record per complete back-test day: actual prices, the model's forecast, yesterday and last week at the same hour."""
    out = []
    for day, g in pred.groupby("delivery_day"):
        g = g.sort_index()
        cols = ["actual", "lightgbm", "naive_d1", "naive_w1"]
        if len(g) != 24 or g[cols].isna().any().any():
            continue
        out.append(dict(day=pd.Timestamp(day), split=str(g["split"].iloc[0]), hours=[t.strftime("%H") for t in g.index],
                        **{c: g[c].to_numpy(float) for c in cols}))
    return out


days = _days((T.REPORTS / "predictions.csv").stat().st_mtime)
if len(days) < 30:
    T.header("Play", "Beat the model")
    T.missing("Enough complete back-test days")
    st.stop()


def revenue(net: np.ndarray, actual: np.ndarray) -> float:
    return float((net * actual).sum())


def lp_value(sched_prices: np.ndarray, actual: np.ndarray) -> tuple[float, np.ndarray]:
    net = B.schedule_day(sched_prices, BAT)
    return revenue(net, actual), net


def player_net(charge: list[int], discharge: list[int]) -> np.ndarray:
    """Charge at full power in the chosen hours; the energy that survives the round trip is sold evenly over the chosen hours."""
    net = np.zeros(24)
    if charge and discharge:
        net[charge] = -BAT.power_mw
        net[discharge] = len(charge) * BAT.power_mw * SE * SE / len(discharge)
    return net


ss = st.session_state
if "game" not in ss:
    order = np.random.default_rng().permutation(len(days)).tolist()
    ss["game"] = dict(order=order, i=0, scores=[], last=None)
g = ss["game"]
d = days[g["order"][g["i"] % len(days)]]

scores = g["scores"]
n = len(scores)
share = lambda k: (100 * sum(s[k] for s in scores) / max(sum(s["perfect"] for s in scores), 1e-9)) if n else float("nan")

T.header("Play", "Beat the model",
         "Schedule a 1 MW / 2 MWh battery for a day from the back-test. You see yesterday's and last week's prices, as the baseline forecasters do. "
         "Pick two charge hours and two discharge hours, then the real prices are revealed and your plan is settled against LightGBM, the naive forecaster and perfect foresight.",
         ["Past days only", "Same battery for everyone: 90% round trip, one cycle", "Your scores stay in this browser session"],
         headline=((f"After {n} rounds you capture <b>{share('you'):.0f}%</b> of the perfect-foresight revenue; "
                    f"LightGBM captures <b>{share('model'):.0f}%</b> on the same days." if n > 1 else
                    f"After 1 round you capture <b>{share('you'):.0f}%</b> of the perfect-foresight revenue; "
                    f"LightGBM captures <b>{share('model'):.0f}%</b> on the same days.") if n else
                   "Can you out-schedule a forecasting model? Pick the hours, lock in your plan and find out."))

# ----- the round ---------------------------------------------------------------------------------------------------------
T.section(f"Round {n + 1}: {d['day']:%A %d %B %Y}", "Look at the two reference days, decide when the battery should buy and sell, then lock in", anchor="round")
yday, lweek = d["naive_d1"], d["naive_w1"]
legend = T.legend_html([("Yesterday", T.ACCENT), ("Same day last week", T.NEUTRAL)])
revealed = g["last"] is not None and g["last"]["day"] == d["day"]
c1, c2 = st.columns([1.5, 1])
with c1, T.card("What you can see before the day: yesterday and last week", "Hourly prices in EUR/MWh · local time", legend):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["hours"], y=lweek, name="same day last week", line=dict(color=T.NEUTRAL, width=1.6, shape="hv"), hovertemplate="last week <b>%{y:.0f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=d["hours"], y=yday, name="yesterday", line=dict(color=T.ACCENT, width=2.4, shape="hv"), hovertemplate="yesterday <b>%{y:.0f}</b><extra></extra>"))
    if revealed:
        r = g["last"]
        fig.add_trace(go.Scatter(x=d["hours"], y=d["actual"], name="actual", line=dict(color=T.TEXT, width=2.6, shape="hv"), hovertemplate="actual <b>%{y:.0f}</b><extra></extra>"))
        for hs, nm, col, sym in ((r["you_c"], "your charge", T.DOWN, "triangle-up"), (r["you_d"], "your discharge", T.UP, "triangle-down")):
            if hs:
                fig.add_trace(go.Scatter(x=[d["hours"][h] for h in hs], y=[d["actual"][h] for h in hs], mode="markers", name=nm,
                                         marker=dict(size=13, color=col, symbol=sym, line=dict(color=T.SURFACE, width=2)),
                                         hovertemplate=nm + " %{x}:00 · <b>%{y:.0f}</b><extra></extra>"))
    fig.update_xaxes(title="hour (local)", dtick=3)
    fig.update_yaxes(title="EUR/MWh")
    T.chart(fig, 340, legend=False, table=False)

with c2, T.card("Your plan", "Tap the hours. The battery starts empty and ends empty; all charging must come before all selling."):
    ch = st.pills("Charge: buy 1 MW in 2 hours", HOURS, selection_mode="multi", key=f"ch_{g['i']}", disabled=revealed)
    di = st.pills("Discharge: sell in 2 hours", HOURS, selection_mode="multi", key=f"di_{g['i']}", disabled=revealed)
    ci, dj = sorted(int(h) for h in (ch or [])), sorted(int(h) for h in (di or []))
    problem = ""
    if ci or dj:
        if len(ci) != 2 or len(dj) != 2:
            problem = "Pick exactly two charge hours and two discharge hours, or leave both empty to sit this day out."
        elif max(ci) >= min(dj):
            problem = "Charge hours must all come before the discharge hours."
    if problem and not revealed:
        st.caption(problem)
    b1, b2 = st.columns(2)
    if b1.button("Lock in my plan", type="primary", disabled=bool(problem) or revealed, key=f"lock_{g['i']}", width="stretch"):
        mv, mnet = lp_value(d["lightgbm"], d["actual"])
        nv, nnet = lp_value(d["naive_d1"], d["actual"])
        pv, pnet = lp_value(d["actual"], d["actual"])
        you = revenue(player_net(ci, dj), d["actual"])
        g["last"] = dict(day=d["day"], you=you, model=mv, naive=nv, perfect=pv, you_c=ci, you_d=dj,
                         model_c=np.where(mnet < -1e-6)[0].tolist(), model_d=np.where(mnet > 1e-6)[0].tolist(),
                         perf_c=np.where(pnet < -1e-6)[0].tolist(), perf_d=np.where(pnet > 1e-6)[0].tolist())
        g["scores"].append({k: g["last"][k] for k in ("you", "model", "naive", "perfect")})
        st.rerun()
    if b2.button("Next day", disabled=not revealed, key=f"next_{g['i']}", width="stretch"):
        g["i"] += 1
        g["last"] = None
        st.rerun()

# ----- the reveal --------------------------------------------------------------------------------------------------------
if revealed:
    r = g["last"]
    T.section("The result", "Everyone is settled at the real day-ahead prices", anchor="result")
    k = st.columns(4)
    T.kpi(k[0], "You", T.fmt_eur(r["you"], 0), f"{100 * r['you'] / r['perfect']:.0f}% of perfect" if r["perfect"] > 0 else "no profitable trade existed", accent=True)
    T.kpi(k[1], "LightGBM", T.fmt_eur(r["model"], 0), f"{100 * r['model'] / r['perfect']:.0f}% of perfect" if r["perfect"] > 0 else "")
    T.kpi(k[2], "Naive (yesterday)", T.fmt_eur(r["naive"], 0), f"{100 * r['naive'] / r['perfect']:.0f}% of perfect" if r["perfect"] > 0 else "")
    T.kpi(k[3], "Perfect foresight", T.fmt_eur(r["perfect"], 0), "the ceiling for this day")
    diff = r["you"] - r["model"]
    if abs(diff) < 1:
        msg, kind = "A tie with the model: you picked effectively the same revenue.", "insight"
    elif diff > 0:
        msg, kind = f"You beat the model by <b>{T.fmt_eur(diff, 0)}</b> on this day.", "insight"
    else:
        msg, kind = f"The model beat you by <b>{T.fmt_eur(-diff, 0)}</b> on this day.", "action"
    fmt = lambda hs: ", ".join(f"{h:02d}" for h in hs) or "idle"
    T.takeaway(msg + f" Your plan: charge {fmt(r['you_c'])}, discharge {fmt(r['you_d'])}. LightGBM chose to charge {fmt(r['model_c'])} and discharge {fmt(r['model_d'])}; "
               f"with hindsight the best schedule charges {fmt(r['perf_c'])} and discharges {fmt(r['perf_d'])}.", kind)

# ----- the scoreboard ----------------------------------------------------------------------------------------------------
T.section("Scoreboard", "Share of the perfect-foresight revenue captured over the days you played", anchor="score")
if n:
    wins = sum(1 for s in scores if s["you"] > s["model"] + 1)
    k = st.columns(4)
    T.kpi(k[0], "Rounds played", f"{n}", f"{wins} won against the model", accent=True)
    T.kpi(k[1], "You", f"{share('you'):.0f}%", "of perfect revenue")
    T.kpi(k[2], "LightGBM", f"{share('model'):.0f}%", "on the same days")
    T.kpi(k[3], "Naive", f"{share('naive'):.0f}%", "on the same days")
    if n < 10:
        T.takeaway("Ten days is a small sample: one volatile day can swing these shares by many points. Play more rounds before drawing conclusions.", "warning")
    if st.button("Reset my score", key="reset_game", icon=":material/restart_alt:"):
        del ss["game"]
        st.rerun()
else:
    T.takeaway("Your first round starts above. The scoreboard compares you with the model over every day you play, so the comparison is always like for like.", "insight")

T.takeaway("The model's schedule is the optimiser run on its forecast and settled at actual prices, the same way the battery study does it. "
           "Your plan is limited to two charge and two discharge hours at full power, which is simpler than the optimiser's freedom to split hours, "
           "so the model has a small structural advantage. Perfect foresight is a ceiling, not something any forecaster can reach.", "warning")

T.next_steps([("views/6_Battery.py", "Battery simulation", "The full year, with degradation and the Monte Carlo revenue range."),
              ("views/3_Forecast.py", "Forecast", "How good the prices behind the model's schedule are."),
              ("views/0_Trader_Brief.py", "Trader brief", "Tomorrow's real schedule and its range.")])

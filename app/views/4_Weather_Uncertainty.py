"""Weather uncertainty: oracle vs realistic, noise injection, forecast vs actual weather, RES error and price."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.setup("Weather Uncertainty")
top = st.container()
ex = T.load_csv("weather_experiment.csv")
noise = T.load_csv("weather_noise_injection.csv")
e = ex.set_index("setup")["MAE"] if ex is not None else None
with top:
    T.header("Models", "Weather uncertainty",
             "Chain: weather forecast error → wind/solar forecast error → residual-load error → price error. Only the weather forecast issued on D-1 is ever used as a model input.",
             ["Open-Meteo Previous Runs API (D-1)", "Oracle = observed weather (upper bound)", "3 seeds per noise level"],
             headline=(f"D-1 weather forecasts are worth <b>€{e['no_weather'] - e['realistic']:.2f}/MWh</b> of price error, and perfect weather knowledge would change it by "
                       f"only <b>€{e['oracle'] - e['realistic']:+.2f}</b> more: the ENTSO-E forecasts already carry most of the weather signal." if e is not None else ""))

# --- value of weather information ----------------------------------------------------------------------------------------------------
if ex is not None:
    k = st.columns(4)
    T.kpi(k[0], "Oracle weather", T.fmt_eur(e["oracle"], 2), "observed weather of day D · not realistic")
    T.kpi(k[1], "Realistic (D-1 forecast)", T.fmt_eur(e["realistic"], 2), "headline setup", accent=True)
    T.kpi(k[2], "No weather features", T.fmt_eur(e["no_weather"], 2), "weather group removed")
    T.kpi(k[3], "Value of weather info", T.fmt_eur(e["no_weather"] - e["realistic"], 2), "MAE saved by D-1 weather",
          delta=T.delta_chip(100 * (e["no_weather"] - e["realistic"]) / e["no_weather"], "%", good_when_up=True))
if noise is not None:
    scales = noise["noise_scale"].tolist()
    with T.controls("What-if"):
        s_noise = st.select_slider("Injected weather-forecast error", options=scales, value=1.0 if 1.0 in scales else scales[-1],
                                   format_func=lambda v: f"{v:g}× the measured forecast error", help="Gaussian noise added to the observed weather, scaled by the measured D-1 forecast error.")
c1, c2 = st.columns([1, 1.3])
if ex is not None:
    with c1, T.card("Same model, three weather inputs: the gaps are small", "Price MAE, LightGBM with identical settings and months · axis does not start at 0"):
        order = ["oracle", "realistic", "no_weather"]
        lab = {"oracle": "Oracle (observed)", "realistic": "Realistic (D-1)", "no_weather": "No weather"}
        fig = go.Figure(go.Bar(y=[lab[s] for s in order], x=[e[s] for s in order], orientation="h", name="MAE", marker=dict(color=[T.NEUTRAL, T.ACCENT, T.BLUE], cornerradius=4),
                               text=[f"€{e[s]:.2f}" for s in order], textposition="outside", textfont=dict(size=13, color=T.TEXT), hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>"))
        lo = min(e[order]) - 1
        fig.update_xaxes(range=[lo, max(e[order]) + 0.7], title="MAE (EUR/MWh)", showgrid=True, showspikes=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(hovermode="closest")
        T.chart(fig, 300, legend=False)
if noise is not None:
    row = noise.set_index("noise_scale").loc[s_noise]
    with c2, T.card(f"At {s_noise:g}× the weather error, price MAE is €{row['mean']:.2f} (± {row['std']:.2f} across seeds)",
                    "Noise injection: observed weather + Gaussian noise, mean ± sd over seeds", help="Flat line = the model is insensitive to weather-forecast quality."):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=noise["noise_scale"], y=noise["mean"] + noise["std"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=noise["noise_scale"], y=noise["mean"] - noise["std"], mode="lines", fill="tonexty", fillcolor=T.rgba(T.ACCENT, .15), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=noise["noise_scale"], y=noise["mean"], mode="lines+markers", name="price MAE", line=dict(color=T.ACCENT, width=2.4),
                                 marker=dict(size=8, color=T.ACCENT, line=dict(color=T.SURFACE, width=2)), hovertemplate="%{x}× → MAE <b>%{y:.2f}</b><extra></extra>"))
        fig.add_trace(go.Scatter(x=[s_noise], y=[row["mean"]], mode="markers", name="selected", marker=dict(size=16, color=T.ACCENT, line=dict(color=T.TEXT, width=2)), hoverinfo="skip"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="noise scale (0 = oracle)", showspikes=False)
        fig.update_yaxes(title="price MAE (EUR/MWh)")
        T.chart(fig, 300, legend=False)
if ex is not None:
    T.takeaway(f"Perfect weather would improve MAE by only €{e['realistic'] - e['oracle']:.2f} (oracle €{e['oracle']:.2f} vs realistic €{e['realistic']:.2f}), and injected noise "
               f"from 0× to 2× moves it by €{noise['mean'].max() - noise['mean'].min():.2f}. The ENTSO-E load/wind/solar forecasts already carry the weather signal. "
               "Caveat: the noise-injection runs use a separate training setup, so their level (about €24.5) is not directly comparable with the three bars on the left.")

# --- forecast vs observed weather ----------------------------------------------------------------------------------------------------------
T.section("Weather forecast vs observed", "Open-Meteo D-1 forecast against reanalysis, three representative points")
we = T.load_csv("weather_fc_error.csv")
smp = T.load_csv("weather_fc_vs_actual_sample.csv")
weh = T.load_csv("weather_fc_error_by_hour.csv", index_col=0)
units = {"temp": "°C", "wind100": "m/s", "rad": "W/m²", "cloud": "%"}
vname = {"temp": "Temperature", "wind100": "Wind speed 100 m", "rad": "Solar radiation", "cloud": "Cloud cover"}
pname = {"sofia": "Sofia", "dobrich": "Dobrich", "stara_zagora": "Stara Zagora"}
if we is not None and smp is not None:
    vars_ = sorted({c[4:] for c in smp.columns if c.startswith("wfc_")})
    label = lambda v: f"{vname[v.split('_', 1)[0]]} · {pname[v.split('_', 1)[1]]}"
    with T.controls("Variable"):
        choice = st.selectbox("Variable and location", vars_, index=vars_.index("wind100_dobrich") if "wind100_dobrich" in vars_ else 0, format_func=label)
    v = choice.split("_")[0]
    row_e = we[(we.season == "all") & (we.variable == v) & (we.point == choice.split("_", 1)[1])].iloc[0]
    c1, c2 = st.columns(2)
    x, y = smp["wact_" + choice], smp["wfc_" + choice]
    with c1, T.card(f"{label(choice)}: the forecast is off by {row_e['MAE']:.1f} {units[v]} on average (bias {row_e['bias']:+.1f})",
                    f"Forecast vs observed, {len(x):,} sampled hours · dashed = perfect forecast"):
        lim = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
        fig = go.Figure(go.Histogram2d(x=x, y=y, nbinsx=45, nbinsy=45, showscale=False, hoverinfo="skip",
                                       colorscale=[[0, "rgba(0,0,0,0)"], [0.01, "#2A2110"], [0.3, "#A06C0F"], [1, "#FFE6B0"]]))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect forecast", line=dict(color=T.TEXT_2, dash="dash", width=1), hoverinfo="skip"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title=f"observed ({units[v]})", range=lim, showspikes=False, showgrid=True)
        fig.update_yaxes(title=f"D-1 forecast ({units[v]})", range=lim)
        T.chart(fig, 380, legend=False)
    if weh is not None and choice in weh:
        hr = weh[choice]
        with c2, T.card(f"The error is largest at {int(hr.idxmax())}:00 ({hr.max():.1f} {units[v]}) and smallest at {int(hr.idxmin())}:00 ({hr.min():.1f})", f"Mean absolute forecast error by hour · {units[v]}"):
            fig = go.Figure(go.Bar(x=hr.index, y=hr, name="MAE", marker=dict(color=T.ACCENT, cornerradius=3), hovertemplate="%{x}:00 · <b>%{y:.2f} " + units[v] + "</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="hour (local)", dtick=3)
            fig.update_yaxes(title=units[v])
            T.chart(fig, 380, legend=False)
    t = we[we["season"] == "all"].copy()
    t.index = [f"{vname[a]} · {pname[b]}" for a, b in zip(t["variable"], t["point"])]
    t["Unit"] = t["variable"].map(units)
    t = t[["Unit", "bias", "MAE", "RMSE", "n"]].rename(columns={"bias": "Bias", "n": "Hours"})
    t.index.name = "Variable"
    T.table(t, fmt={"Bias": "{:+.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "Hours": "{:,.0f}"})
else:
    T.missing("Weather forecast error tables")

# --- RES forecast error + price impact ----------------------------------------------------------------------------------------------------
T.section("Wind/solar forecast error and the price", "ENTSO-E day-ahead forecast vs actual generation")
rs = T.load_csv("res_forecast_error_summary.csv")
rt = T.load_csv("res_forecast_error_by_hour_month.csv", index_col=0)
link = T.load_csv("res_error_vs_price_error.csv")
if rs is not None:
    r = rs.iloc[0]
    k = st.columns(4)
    T.kpi(k[0], "Solar forecast MAE", f"{r['solar_mae_mw']:,.0f} MW", f"{r['solar_mae_pct_of_mean']:.0f}% of mean output")
    T.kpi(k[1], "Wind forecast MAE", f"{r['wind_mae_mw']:,.0f} MW", f"{r['wind_mae_pct_of_mean']:.0f}% of mean output")
    T.kpi(k[2], "Solar error explained by weather", f"{100 * r.get('solar_err_r2_vs_weather_err', np.nan):.1f}%", "R² vs point radiation error")
    T.kpi(k[3], "Wind error explained by weather", f"{100 * r.get('wind_err_r2_vs_weather_err', np.nan):.1f}%", "R² vs point wind-speed error")
c1, c2 = st.columns(2)
SOLAR_C, WIND_C = T.YELLOW, T.AQUA
if rt is not None:
    hcols = [c for c in rt.columns if not c.startswith("m_")]
    with c1, T.card("Solar forecast errors peak in daylight; wind errors are flatter through the day", "Mean absolute error of the ENTSO-E day-ahead forecast (MW) by hour",
                    T.legend_html([("Solar", SOLAR_C), ("Wind", WIND_C)], "box")):
        fig = go.Figure()
        for c in hcols:
            kind = c.split("_")[0]
            fig.add_trace(go.Bar(x=rt.index, y=rt[c], name=kind, marker=dict(color=SOLAR_C if kind == "solar" else WIND_C, cornerradius=3),
                                 hovertemplate=kind + " <b>%{y:,.0f} MW</b><extra></extra>"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="hour (local)", dtick=3)
        fig.update_yaxes(title="MW")
        T.chart(fig, 340, legend=False)
if link is not None and len(link) > 100:
    b = pd.qcut(link["res_load_err_mw"], 15, duplicates="drop")
    g = link.groupby(b, observed=True)["price_err"].agg(["mean", "count"])
    slope = np.polyfit(link["res_load_err_mw"], link["price_err"], 1)[0]
    with c2, T.card(f"Each 100 MW of residual-load miss moves the day-ahead price error by only {100 * slope:+.2f} EUR/MWh", "Mean LightGBM price error per bin of residual-load forecast error"):
        fig = go.Figure(go.Bar(x=[f"{i.mid:,.0f}" for i in g.index], y=g["mean"], name="price error", marker=dict(color=np.where(g["mean"] > 0, T.UP, T.DOWN), cornerradius=3),
                               hovertemplate="miss %{x} MW → price error <b>%{y:+.1f}</b><extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="actual − forecast residual load (MW, bin centre)", tickangle=-40)
        fig.update_yaxes(title="price error (EUR/MWh)")
        T.chart(fig, 340, legend=False)
    T.takeaway("The slope is essentially zero, and that is market logic: the day-ahead price clears on D-1 <i>forecasts</i>, so a realised wind/solar/load miss moves intraday "
               "and imbalance prices, not the day-ahead price. <b>So what:</b> renewable-forecast risk is an intraday/balancing exposure, so hedge it there.", "action")
wd = T.load_csv("worst_res_forecast_days.csv")
bw = T.load_csv("battery_weather_scenarios.csv", index_col=0)
c1, c2 = st.columns(2)
if wd is not None:
    with c1:
        st.markdown("<div class='card-title' style='margin:6px 0'>Worst 10 days by residual-load miss</div>", unsafe_allow_html=True)
        wd = wd.set_index("delivery_day")
        wd.index = pd.to_datetime(wd.index).strftime("%d %b %Y")
        wd.index.name = "Day"
        wd.columns = ["Mean |miss| (MW)", "Mean price error", "Price MAE"]
        T.table(wd, fmt={"Mean |miss| (MW)": "{:,.0f}", "Mean price error": "{:+.1f}", "Price MAE": "{:.1f}"}, bars={"Mean |miss| (MW)": T.BLUE})
if bw is not None:
    with c2:
        st.markdown("<div class='card-title' style='margin:6px 0'>Battery revenue under each weather scenario</div>", unsafe_allow_html=True)
        t = bw[["eur_per_mw_year", "pct_of_perfect"]].rename(columns={"eur_per_mw_year": "EUR / MW-year", "pct_of_perfect": "% of perfect"})
        t.index = [i.replace("_", " ").capitalize() for i in t.index]
        t.index.name = "Schedule"
        T.table(t, fmt={"EUR / MW-year": "{:,.0f}", "% of perfect": "{:.1f}"}, bars={"% of perfect": T.ACCENT})

T.next_steps([("views/3_Forecast.py", "Forecast", "The model these weather inputs feed."),
              ("views/5_Model_Lab.py", "Model lab", "Feature groups, SHAP and ablation."),
              ("views/6_Battery.py", "Battery simulation", "What forecast errors cost in revenue.")])

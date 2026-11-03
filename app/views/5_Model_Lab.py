"""Model Lab (Data Scientist view): leaderboard, SHAP, calibration, classifier, monitoring."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

T.setup("Model Lab")
top = st.container()
_mc, _mh = T.load_csv("metrics_cv.csv"), T.load_csv("metrics_holdout.csv")
_head = ""
if _mc is not None and _mh is not None:
    _a, _b = _mc.set_index("model"), _mh.set_index("model")
    _nb = _a["MAE"].idxmin()
    _hb = _b["MAE"].idxmin()
    _head = (f"On walk-forward CV the best model is <b>{T.MODEL_LABEL.get(_nb, _nb)}</b> (MAE €{_a.loc[_nb, 'MAE']:.2f}, "
             f"{100 * _a.loc[_nb, 'skill_vs_best_naive']:+.1f}% vs naive). On the untouched holdout the best is <b>{T.MODEL_LABEL.get(_hb, _hb)}</b> "
             f"(€{_b.loc[_hb, 'MAE']:.2f}, {100 * _b.loc[_hb, 'skill_vs_best_naive']:+.1f}%): the ranking changes when the price regime does.")
with top:
    T.header("Models", "Model lab",
             "Every model that was tried, including the ones that lost, evaluated on 25 walk-forward months and an untouched "
             "3-month holdout, with Diebold-Mariano significance against the best naive baseline.",
             ["Optuna · 50 trials", "MLflow tracked", "SHAP", "Quantile LightGBM", "PSI drift"], headline=_head)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Leaderboard", "Explainability", "Probabilistic", "Spike classifier", "Monitoring"])

# ================================================================== leaderboard
with tab1:
    mc, mh = T.load_csv("metrics_cv.csv"), T.load_csv("metrics_holdout.csv")
    if mc is None or mh is None:
        T.missing("Model metrics")
    else:
        mc, mh = mc.set_index("model"), mh.set_index("model")
        split = st.segmented_control("Evaluation set", ["Walk-forward CV", "Final holdout"], default="Walk-forward CV") \
            or "Walk-forward CV"
        mt = mc if split.startswith("Walk") else mh
        best = mt["MAE"].idxmin()
        k = st.columns(4)
        T.kpi(k[0], "Best model", T.MODEL_LABEL.get(best, best), split.lower(), accent=True)
        T.kpi(k[1], "Best MAE", T.fmt_eur(mt.loc[best, "MAE"], 2), f"RMSE {mt.loc[best, 'RMSE']:.1f}")
        T.kpi(k[2], "Skill vs naive", f"{100 * mt.loc[best, 'skill_vs_best_naive']:+.1f}%", "1 − MAE / MAE(naive · yesterday)")
        nsig = int(pd.Series(mt.get("significant_5pct")).fillna(False).astype(bool).sum())
        T.kpi(k[3], "Significant improvements", f"{nsig} / {len(mt) - 1}", "Diebold-Mariano, 5 % level")
        show = pd.DataFrame({
            "MAE": mt["MAE"], "RMSE": mt["RMSE"],
            "Fold MAE ± sd": [f"{a:.1f} ± {b:.1f}" for a, b in zip(mt["MAE_fold_mean"], mt["MAE_fold_std"].fillna(0))],
            "Skill": 100 * mt["skill_vs_best_naive"], "DM p": pd.to_numeric(mt["DM_p"], errors="coerce"),
            "Significant": mt.get("significant_5pct"),
            "Spike-hour MAE": mt["MAE_spike_hours"], "Negative-hour MAE": mt["MAE_negative_hours"]}).sort_values("MAE")
        sc = T.load_csv("skill_confidence.csv")
        if sc is not None:
            sc = sc[sc.split == ("cv" if split.startswith("Walk") else "holdout")].set_index("variant")
            show["Skill 95% CI"] = [f"[{100 * sc.loc[i, 'skill_lo']:+.0f}%, {100 * sc.loc[i, 'skill_hi']:+.0f}%]" if i in sc.index and pd.notna(sc.loc[i, "skill_lo"]) else "—"
                                    for i in show.index]
        show.index = [T.MODEL_LABEL.get(i, i) for i in show.index]
        show.index.name = "Model"

        def sig(v):
            if v is True or v == "True":
                return "✓ better", "good"
            if v is False or v == "False":
                return "not significant", "neutral"
            return "baseline", "neutral"

        T.table(show, fmt={"MAE": "{:.2f}", "RMSE": "{:.2f}", "Skill": "{:+.1f}%", "DM p": "{:.3g}",
                           "Spike-hour MAE": "{:.1f}", "Negative-hour MAE": "{:.1f}"},
                bars={"MAE": T.ACCENT}, badges={"Significant": sig}, highlight=T.MODEL_LABEL.get(best, best))

        if sc is not None and len(sc):
            with T.card("Skill vs the best naive baseline, with 95% intervals",
                        "Moving-block bootstrap over days · an interval crossing 0 means the gain is not distinguishable from noise",
                        T.legend_html([("Interval excludes 0", T.AQUA), ("Interval includes 0", T.NEUTRAL)], "box")):
                f = sc.dropna(subset=["skill_lo"]).sort_values("skill", ascending=False)
                ok = (f["skill_lo"] > 0)
                fig = go.Figure()
                labs_f = [T.MODEL_LABEL.get(i, i.replace("_", " ")) for i in f.index]
                for lab_, (_, rr), good in zip(labs_f, f.iterrows(), ok):
                    col = T.AQUA if good else T.NEUTRAL
                    fig.add_trace(go.Scatter(x=[100 * rr.skill_lo, 100 * rr.skill_hi], y=[lab_, lab_], mode="lines", line=dict(color=col, width=4),
                                             hoverinfo="skip", showlegend=False))
                    fig.add_trace(go.Scatter(x=[100 * rr.skill], y=[lab_], mode="markers", marker=dict(size=12, color=col, line=dict(color=T.SURFACE, width=2)),
                                             showlegend=False, customdata=[[100 * rr.skill_lo, 100 * rr.skill_hi]],
                                             hovertemplate="%{y}: skill <b>%{x:+.1f}%</b> [%{customdata[0]:+.1f}, %{customdata[1]:+.1f}]<extra></extra>"))
                fig.add_vline(x=0, line=dict(color=T.MUTED, width=1))
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(title="skill vs naive (%)", showgrid=True, showspikes=False, zeroline=False)
                fig.update_yaxes(autorange="reversed", showgrid=False)
                T.chart(fig, 60 + 38 * len(f), legend=False)
        c1, c2 = st.columns([1.2, 1])
        with c1, T.card("CV vs holdout", "Same models, two evaluation periods — the gap is the October-2026 level shift",
                        T.legend_html([("Walk-forward CV", T.BLUE), ("Final holdout", T.ACCENT)], "box")):
            order = mc["MAE"].sort_values().index
            labs = [T.MODEL_LABEL.get(i, i) for i in order]
            fig = go.Figure()
            for lab, i in zip(labs, order):
                fig.add_trace(go.Scatter(x=[mc.loc[i, "MAE"], mh.loc[i, "MAE"]], y=[lab, lab], mode="lines",
                                         line=dict(color=T.BORDER, width=4), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=mc.loc[order, "MAE"], y=labs, mode="markers", name="CV",
                                     marker=dict(size=13, color=T.BLUE, line=dict(color=T.SURFACE, width=2)),
                                     hovertemplate="%{y} · CV <b>%{x:.2f}</b><extra></extra>"))
            fig.add_trace(go.Scatter(x=mh.loc[order, "MAE"], y=labs, mode="markers", name="holdout",
                                     marker=dict(size=13, color=T.ACCENT, line=dict(color=T.SURFACE, width=2)),
                                     hovertemplate="%{y} · holdout <b>%{x:.2f}</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="MAE (EUR/MWh)", showgrid=True, showspikes=False)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            T.chart(fig, 360, legend=False)
        ab = T.load_csv("ablation.csv")
        with c2, T.card("Feature-group ablation", "MAE change when one group is dropped (last 12 CV months)",
                        T.legend_html([("Hurts when removed", T.UP), ("Helps when removed", T.DOWN)], "box")):
            if ab is not None:
                ab = ab.iloc[1:].sort_values("delta_MAE")
                fig = go.Figure(go.Bar(y=ab["dropped"].str.replace("_", " "), x=ab["delta_MAE"], orientation="h",
                                       marker=dict(color=np.where(ab["delta_MAE"] > 0, T.UP, T.DOWN), cornerradius=4),
                                       text=[f"{v:+.2f}" for v in ab["delta_MAE"]], textposition="outside",
                                       textfont=dict(family=T.MONO, size=11, color=T.TEXT_2),
                                       hovertemplate="drop %{y}: <b>%{x:+.2f}</b><extra></extra>"))
                fig.update_layout(hovermode="closest")
                span = ab["delta_MAE"].max() - min(ab["delta_MAE"].min(), 0)
                fig.update_xaxes(title="Δ MAE (EUR/MWh)", showgrid=True, showspikes=False, zeroline=True, zerolinecolor=T.MUTED,
                                 range=[min(ab["delta_MAE"].min(), 0) - 0.25 * span, ab["delta_MAE"].max() + 0.25 * span])
                fig.update_yaxes(showgrid=False)
                T.chart(fig, 360, legend=False)
        c1, c2 = st.columns([1.2, 1])
        ot = T.load_csv("optuna_trials.csv")
        with c1, T.card("Optuna optimisation history", "TPE sampler, seed 42 · objective = MAE on the last 4 CV months",
                        T.legend_html([("Trial", T.NEUTRAL), ("Best so far", T.ACCENT)])):
            if ot is not None:
                ot = ot.sort_values("number")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ot["number"], y=ot["value"], mode="markers", name="trial",
                                         marker=dict(color=T.NEUTRAL, size=8, line=dict(color=T.SURFACE, width=1)),
                                         hovertemplate="trial %{x}: %{y:.3f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=ot["number"], y=ot["value"].cummin(), name="best", line=dict(color=T.ACCENT, width=2.4, shape="hv"),
                                         hovertemplate="best so far %{y:.3f}<extra></extra>"))
                fig.update_xaxes(title="trial")
                fig.update_yaxes(title="CV MAE")
                T.chart(fig, 300, legend=False)
        w = T.load_json("ensemble_weights.json")
        with c2, T.card("Ensemble weights", "Chosen on CV months only (grid on the simplex)"):
            if w:
                ws = pd.Series(w).sort_values()
                fig = go.Figure(go.Bar(y=[T.MODEL_LABEL.get(k_, k_) for k_ in ws.index], x=ws.values, orientation="h",
                                       marker=dict(color=[T.MODEL.get(k_, T.MUTED) for k_ in ws.index], cornerradius=4),
                                       text=[f"{100 * v:.0f}%" for v in ws.values], textposition="outside", textfont=dict(size=12, color=T.TEXT),
                                       hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>"))
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(range=[0, max(ws.values) * 1.2], title="weight", showspikes=False)
                fig.update_yaxes(showgrid=False)
                T.chart(fig, 300, legend=False)
        fx, fc_ = T.load_csv("fuel_experiment.csv"), T.load_json("feature_choice.json")
        if fx is not None:
            none = fx[fx.variant == "none"].set_index("model")["MAE"]
            with T.card("Do fuel and carbon prices help?",
                        f"MAE on CV months {fx['months'].iloc[0]} · fixed set of encodings · holdout never used · "
                        "dashed line = no fuel features",
                        T.legend_html([("LightGBM", T.ACCENT), ("Level-adjusted LightGBM", T.MAGENTA)], "box")):
                order = fx["variant"].drop_duplicates().tolist()
                fig = go.Figure()
                for mname, col in (("LightGBM", T.ACCENT), ("LightGBM · level-adj.", T.MAGENTA)):
                    g = fx[fx.model == mname].set_index("variant").reindex(order)
                    fig.add_trace(go.Bar(x=order, y=g["MAE"], name=mname, marker=dict(color=col, cornerradius=4),
                                         customdata=g["delta_vs_none"], hovertemplate="%{x}<br>MAE <b>%{y:.2f}</b> (%{customdata:+.2f} vs none)<extra>" + mname + "</extra>"))
                    fig.add_hline(y=none[mname], line=dict(color=col, dash="dash", width=1))
                lo = fx["MAE"].min()
                fig.update_layout(barmode="group", hovermode="closest")
                fig.update_yaxes(title="MAE (EUR/MWh) — axis does not start at 0", range=[lo - 0.6, fx["MAE"].max() + 0.4])
                fig.update_xaxes(showspikes=False)
                T.chart(fig, 330, legend=False)
            best = fx.loc[fx["delta_vs_none"].idxmin()]
            verdict = ("adopted" if fc_ and fc_.get("use_fuels") else "not adopted")
            T.takeaway(f"Fuel and carbon features were <b>{verdict}</b> by the pre-set rule (CV MAE with €{fc_['cv_mae_with_fuels']:.2f} vs without "
                       f"€{fc_['cv_mae_without_fuels']:.2f}). Across {len(order)} encodings the best change versus no fuel data is "
                       f"<b>{best['delta_vs_none']:+.2f} €/MWh</b> ({best['variant']}, {best['model']}). Price lags and neighbour prices already carry "
                       "the fuel level, and the daily TTF history is only 16 months deep with a single high-gas episode that tree models cannot extrapolate "
                       "to. Re-test once a year of daily TTF has accumulated.", "warning")
        rm_, re_ = T.load_csv("robust_metrics.csv"), T.load_csv("robust_episodes.csv")
        if rm_ is not None:
            variants = [v for v in ("lightgbm", "lightgbm_level", "lightgbm_blend", "lightgbm_guarded", "lightgbm_levelshape", "lightgbm_linbase", "lightgbm_recency")
                        if v in set(rm_.variant)]
            lab_ = {"lightgbm": "LightGBM (baseline)", "lightgbm_level": "level-adjusted", "lightgbm_blend": "50/50 blend", "lightgbm_guarded": "guarded switch",
                    "lightgbm_levelshape": "level + shape", "lightgbm_linbase": "Ridge + tree", "lightgbm_recency": "recency-weighted"}
            with T.card("Regime robustness: do level-anchored models help?", "MAE on the CV months vs the final holdout (Aug–Oct 2026, dominated by the October surge). "
                        "A variant replaces the baseline only if it wins on CV by ≥ 0.1 with p < 0.05.",
                        T.legend_html([("Walk-forward CV", T.BLUE), ("Holdout", T.ACCENT)], "box")):
                fig = go.Figure()
                for split, col, nm in (("cv", T.BLUE, "CV"), ("holdout", T.ACCENT, "holdout")):
                    g = rm_[(rm_.split == split)].set_index("variant").reindex(variants)
                    fig.add_trace(go.Bar(x=[lab_[v] for v in variants], y=g["MAE"], name=nm, marker=dict(color=col, cornerradius=4),
                                         customdata=g["skill"], text=[f"{v:.1f}" for v in g["MAE"]], textposition="outside",
                                         textfont=dict(family=T.MONO, size=10, color=T.TEXT_2),
                                         hovertemplate="%{x}: MAE <b>%{y:.2f}</b> · skill %{customdata:+.1%}<extra>" + nm + "</extra>"))
                fig.update_layout(barmode="group", hovermode="closest")
                fig.update_yaxes(title="MAE (EUR/MWh)")
                fig.update_xaxes(showspikes=False)
                T.chart(fig, 330, legend=False)
            if re_ is not None and len(re_):
                with T.card("Level-shift episodes", "Months where the average price moved ≥ 25 % versus the previous three months · MAE in EUR/MWh"):
                    ep = re_.set_index("month")[["split", "price_change", "naive_d1", "lightgbm", "lightgbm_level", "lightgbm_blend"]].rename(columns={
                        "split": "Set", "price_change": "Price change", "naive_d1": "Naive", "lightgbm": "LightGBM", "lightgbm_level": "Level-adj.", "lightgbm_blend": "Blend"})
                    ep["Set"] = ep["Set"].str.upper()
                    ep.index.name = "Month"
                    T.table(ep, fmt={"Price change": "{:+.0%}", "Naive": "{:.1f}", "LightGBM": "{:.1f}", "Level-adj.": "{:.1f}", "Blend": "{:.1f}"})
            ch_ = T.load_json("robust_choice.json")
            T.takeaway(f"Pre-set rule: <b>{(ch_ or {}).get('variant', 'lightgbm')}</b> stays the headline model — no variant wins on the CV months. The level-anchored "
                       "variants lose a little on CV (the blend by only 0.08, p = 0.36) but gain about 3.5 MAE in the October surge, and in the earlier "
                       "level-shift episodes they mostly do not help. They are cheap insurance against extrapolation, not a tested improvement; "
                       "recency weighting is clearly worse on the holdout, and the level/shape candidate was added after the first results were seen.", "warning")
        T.takeaway("On CV every model beats the naive baseline significantly. On the holdout only Ridge and the level-adjusted "
                   "LightGBM hold up: the October-2026 price surge pushed prices above the training range, and tree models "
                   "and the MLP cannot extrapolate. The level-adjusted variant was added after this diagnosis, so its holdout "
                   "score is not a clean test.", "warning")

# ================================================================== explainability
with tab2:
    imp = T.load_csv("shap_importance.csv")
    sv = T.load_csv("shap_values_sample.csv")
    if imp is None:
        T.missing("SHAP results")
    else:
        def _grp(f):
            f = f.lower()
            if f.startswith("price_") or "spread" in f or "_lag" in f and "price" in f:
                return "Price history & neighbours"
            if any(t in f for t in ("hour", "dow", "month", "weekend", "holiday", "doy", "sin", "cos")):
                return "Calendar"
            if any(t in f for t in ("wind", "solar", "temp", "rad", "cloud")):
                return "Wind, solar & weather"
            if any(t in f for t in ("load", "resid")):
                return "Load & residual load"
            if any(t in f for t in ("gas", "ttf", "brent", "carbon", "fuel")):
                return "Fuels"
            return "Other"
        _g = imp.assign(group=imp["feature"].map(_grp)).groupby("group")["mean_abs_shap"].sum().sort_values()
        with T.card(f"{_g.index[-1]} drives {100 * _g.iloc[-1] / _g.sum():.0f}% of the model's total attribution",
                    "Mean |SHAP| summed over features in each group (grouping by feature name)"):
            fig = go.Figure(go.Bar(y=_g.index, x=_g.values, orientation="h", marker=dict(color=[T.ACCENT if i == len(_g) - 1 else T.NEUTRAL for i in range(len(_g))], cornerradius=4),
                                   text=[f"{100 * v / _g.sum():.0f}%" for v in _g.values], textposition="outside", textfont=dict(size=12, color=T.TEXT),
                                   hovertemplate="%{y}: <b>%{x:.1f}</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(range=[0, _g.max() * 1.15], title="summed mean |SHAP| (EUR/MWh)", showspikes=False)
            fig.update_yaxes(showgrid=False)
            T.chart(fig, 60 + 36 * len(_g), legend=False)
        c1, c2 = st.columns([1, 1.4])
        with c1, T.card("Global importance", "Mean |SHAP| on the holdout (EUR/MWh)"):
            top = imp.head(12).iloc[::-1]
            fig = go.Figure(go.Bar(y=top["feature"], x=top["mean_abs_shap"], orientation="h",
                                   marker=dict(color=T.ACCENT, cornerradius=4),
                                   hovertemplate="%{y}: <b>%{x:.2f}</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(showgrid=True, showspikes=False)
            fig.update_yaxes(showgrid=False)
            T.chart(fig, 460, legend=False)
        with c2, T.card("SHAP beeswarm", "Each dot = one hour · colour = feature value (blue low → red high)"):
            if sv is not None:
                order = sv.groupby("feature")["shap"].apply(lambda s: s.abs().mean()).sort_values().index
                rng = np.random.default_rng(0)
                fig = go.Figure()
                for i, feat in enumerate(order):
                    g = sv[sv.feature == feat]
                    fig.add_trace(go.Scattergl(x=g["shap"], y=i + rng.uniform(-0.32, 0.32, len(g)), mode="markers",
                                               marker=dict(size=4, color=g["value_pct"], cmin=0, cmax=1, opacity=0.75,
                                                           colorscale=[[0, T.BLUE], [0.5, "#8A8A99"], [1, T.RED]],
                                                           showscale=i == 0, colorbar=dict(title=dict(text="value", side="right"),
                                                                                          tickvals=[0, 1], ticktext=["low", "high"])),
                                               hovertemplate=feat + ": SHAP <b>%{x:.1f}</b><extra></extra>"))
                fig.update_layout(hovermode="closest")
                fig.update_yaxes(tickvals=list(range(len(order))), ticktext=list(order), showgrid=False)
                fig.update_xaxes(title="SHAP value (EUR/MWh)", zeroline=True, zerolinecolor=T.MUTED, showspikes=False)
                T.chart(fig, 460, legend=False)
        cases = T.load_csv("shap_cases.csv")
        c1, c2 = st.columns([1.4, 1])
        if cases is not None:
            with c1:
                case = st.segmented_control("Explain a single hour", cases["case"].unique().tolist(), default=cases["case"].iloc[0]) \
                    or cases["case"].iloc[0]
                g = cases[cases.case == case]
                base = g.base_value.iloc[0]
                g = g.sort_values("shap", key=abs, ascending=False)
                with T.card(f"Why the model predicted this · {case} hour",
                            f"{pd.Timestamp(g.timestamp_utc.iloc[0]).tz_convert(T.LOCAL_TZ):%d %b %Y %H:%M} · actual €{g.actual.iloc[0]:.1f} · "
                            f"model base value €{base:.1f}"):
                    fig = go.Figure(go.Waterfall(orientation="h", y=[f"{a} = {b:.1f}" for a, b in zip(g.feature, g.feature_value)][::-1]
                                                 + ["prediction"], x=list(g["shap"][::-1]) + [0],
                                                 measure=["relative"] * len(g) + ["total"], base=base,
                                                 increasing=dict(marker=dict(color=T.UP)), decreasing=dict(marker=dict(color=T.DOWN)),
                                                 totals=dict(marker=dict(color=T.ACCENT)), connector=dict(line=dict(color=T.BORDER, width=1)),
                                                 hovertemplate="%{y}: <b>%{x:+.1f}</b><extra></extra>"))
                    fig.update_layout(hovermode="closest")
                    fig.update_xaxes(title="EUR/MWh", showgrid=True, showspikes=False)
                    fig.update_yaxes(showgrid=False)
                    T.chart(fig, 420, legend=False)
        pdp = T.load_csv("pdp_residual_load.csv")
        if pdp is not None:
            with c2:
                with T.card("Partial dependence", "Average prediction while sweeping the residual-load forecast"):
                    fig = go.Figure(go.Scatter(x=pdp.iloc[:, 0], y=pdp.iloc[:, 1], line=dict(color=T.ACCENT, width=2.6, shape="spline"),
                                               fill="tozeroy", fillcolor=T.rgba(T.ACCENT, .08),
                                               hovertemplate="%{x:,.0f} MW → <b>%{y:.1f}</b><extra></extra>"))
                    fig.update_xaxes(title="residual load forecast (MW)")
                    fig.update_yaxes(title="avg predicted price", rangemode="normal")
                    T.chart(fig, 420, legend=False)
        T.takeaway("Drivers match market logic: yesterday's price at the same hour, D-1 prices in Romania and Hungary (market coupling), "
                   "the same-weekday price, day of week and residual load. No same-day actual quantity is a feature, so SHAP shows no sign of leakage.")

# ================================================================== probabilistic
with tab3:
    qm = T.load_csv("quantile_metrics.csv")
    qh = T.load_csv("quantile_by_hour.csv", index_col=0)
    if qm is None:
        T.missing("Quantile forecast")
    else:
        q = qm.set_index("split")
        k = st.columns(4)
        T.kpi(k[0], "10–90 % coverage · CV", f"{100 * q.loc['cv', 'coverage_10_90']:.0f}%", "target 80 %",
              delta=T.delta_chip(100 * q.loc['cv', 'coverage_10_90'] - 80, " pp", good_when_up=True, digits=0))
        T.kpi(k[1], "10–90 % coverage · holdout", f"{100 * q.loc['holdout', 'coverage_10_90']:.0f}%", "target 80 %",
              delta=T.delta_chip(100 * q.loc['holdout', 'coverage_10_90'] - 80, " pp", good_when_up=True, digits=0))
        T.kpi(k[2], "Mean band width", T.fmt_eur(q.loc["cv", "mean_band_width"], 0), "CV · EUR/MWh")
        T.kpi(k[3], "Pinball loss q50", f"{q.loc['cv', 'pinball_q50']:.2f}", f"holdout {q.loc['holdout', 'pinball_q50']:.2f}")
        qc = T.load_csv("quantile_conformal.csv")
        if qc is not None:
            ch = qc[qc.chosen].iloc[0]["variant"]
            with T.card("Conformal recalibration of the 10–90 % band",
                        f"Share of outcomes inside the band · window chosen on CV months only ({ch}) · target 80 %",
                        T.legend_html([("Walk-forward CV", T.BLUE), ("Final holdout", T.ACCENT)], "box")):
                order = ["raw quantile band"] + [v for v in qc["variant"].drop_duplicates() if v.startswith("conformal")]
                fig = go.Figure()
                for split, col, nm in (("cv", T.BLUE, "CV"), ("holdout", T.ACCENT, "holdout")):
                    g = qc[qc.split == split].set_index("variant").reindex(order)
                    fig.add_trace(go.Bar(x=[o.replace("conformal ", "") if o != "raw quantile band" else "raw band" for o in order],
                                         y=100 * g["coverage"], name=nm, marker=dict(color=col, cornerradius=4),
                                         customdata=g["width"], text=[f"{100 * v:.0f}%" for v in g["coverage"]], textposition="outside",
                                         textfont=dict(family=T.MONO, size=11, color=T.TEXT_2),
                                         hovertemplate="%{x}: <b>%{y:.1f}%</b> inside · mean width €%{customdata:.0f}<extra>" + nm + "</extra>"))
                fig.add_hline(y=80, line=dict(color=T.TEXT_2, dash="dash", width=1))
                fig.update_layout(barmode="group", hovermode="closest")
                fig.update_yaxes(title="% of outcomes inside the band", range=[0, 100])
                fig.update_xaxes(title="calibration window (days of recent out-of-sample misses)", showspikes=False)
                T.chart(fig, 320, legend=False)
        c1, c2 = st.columns(2)
        with c1, T.card("Calibration", "Share of outcomes below each quantile forecast · on the diagonal = calibrated",
                        T.legend_html([("Walk-forward CV", T.BLUE), ("Holdout", T.ACCENT)])):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.MUTED, dash="dash", width=1), hoverinfo="skip"))
            for split, col in (("cv", T.BLUE), ("holdout", T.ACCENT)):
                r = q.loc[split]
                fig.add_trace(go.Scatter(x=[0.1, 0.5, 0.9], y=[r.share_below_q10, r.share_below_q50, r.share_below_q90],
                                         mode="lines+markers", marker=dict(size=11, line=dict(color=T.SURFACE, width=2)),
                                         line=dict(color=col, width=2.4), hovertemplate="nominal %{x} → observed <b>%{y:.2f}</b><extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="nominal quantile", range=[0, 1], showspikes=False, showgrid=True)
            fig.update_yaxes(title="observed share below", range=[0, 1])
            T.chart(fig, 380, legend=False)
        if qh is not None:
            with c2, T.card("Band width and coverage by hour", "Bars = width of the 10–90 % band · line = coverage (target 80 %)",
                            T.legend_html([("Band width (EUR/MWh)", T.ACCENT), ("Coverage", T.BLUE)])):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=qh.index, y=qh["band_width"], marker=dict(color=T.rgba(T.ACCENT, .75), cornerradius=3),
                                     name="width", hovertemplate="%{x}:00 width <b>%{y:.0f}</b><extra></extra>"))
                fig.add_trace(go.Scatter(x=qh.index, y=qh["coverage_10_90"] * qh["band_width"].max(), mode="lines+markers",
                                         line=dict(color=T.BLUE, width=2), marker=dict(size=5), customdata=100 * qh["coverage_10_90"],
                                         name="coverage", hovertemplate="coverage <b>%{customdata:.0f}%</b><extra></extra>"))
                fig.add_hline(y=0.8 * qh["band_width"].max(), line=dict(color=T.BLUE, dash="dot", width=1),
                              annotation_text="80 % coverage", annotation_font=dict(size=10, color=T.BLUE))
                fig.update_xaxes(title="hour (local)", dtick=3)
                fig.update_yaxes(title="band width (EUR/MWh) · coverage scaled")
                T.chart(fig, 380, legend=False)
        if qc is not None:
            raw_cv, raw_ho = qc[(qc.variant == "raw quantile band")].set_index("split")["coverage"].reindex(["cv", "holdout"])
            cal_cv, cal_ho = qc[qc.chosen].set_index("split")["coverage"].reindex(["cv", "holdout"])
            T.takeaway(f"The raw band covered only <b>{100 * raw_cv:.0f}%</b> of outcomes in CV. Conformal recalibration (widening each side by its recent "
                       f"out-of-sample misses) lifts CV coverage to <b>{100 * cal_cv:.0f}%</b>. On the holdout it rises from {100 * raw_ho:.0f}% to "
                       f"<b>{100 * cal_ho:.0f}%</b> but still falls short: the October-2026 surge put about a quarter of outcomes above the band, "
                       "so a calibration built from calmer months cannot fully catch a regime shift. Shorter windows adapt faster at the cost of "
                       "stability.", "warning")
        else:
            T.takeaway(f"The 10–90 % band covers only <b>{100 * q.loc['cv', 'coverage_10_90']:.0f}%</b> of outcomes in CV and "
                       f"<b>{100 * q.loc['holdout', 'coverage_10_90']:.0f}%</b> on the holdout (target 80 %).", "warning")

# ================================================================== classifier
with tab4:
    cm = T.load_csv("classifier_metrics.csv")
    if cm is None:
        T.missing("Classifier results")
    else:
        kinds = [k_ for k_ in ("spike_rel", "spike", "cheap", "negative") if k_ in set(cm.kind)]
        with T.controls("Event"):
            kind = st.segmented_control("Event", kinds, default=kinds[0], label_visibility="collapsed",
                                        format_func=lambda s: {"spike_rel": "Spike vs recent level", "spike": "Spike above fixed € level",
                                                               "cheap": "Cheap hour (< €5)", "negative": "Negative price"}[s]) or kinds[0]
        r = cm[(cm.kind == kind) & (cm.split == "cv")].iloc[0]
        k = st.columns(4)
        T.kpi(k[0], "PR-AUC · CV", f"{r.pr_auc:.2f}", f"vs base rate {100 * r.base_rate:.1f}%", accent=True)
        T.kpi(k[1], "Precision", f"{100 * r.precision:.0f}%", "alarms that were real")
        T.kpi(k[2], "Recall", f"{100 * r.recall:.0f}%", "events caught")
        T.kpi(k[3], "Decision threshold", f"{r.threshold:.2f}", "min cost: miss = 5 × false alarm")
        c1, c2 = st.columns(2)
        pr = T.load_csv(f"classifier_{kind}_pr_curve.csv")
        with c1, T.card("Precision–recall curve", "CV months · dashed = no-skill (base rate)"):
            if pr is not None:
                fig = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], line=dict(color=T.ACCENT, width=2.4), fill="tozeroy",
                                           fillcolor=T.rgba(T.ACCENT, .08), hovertemplate="recall %{x:.2f} · precision <b>%{y:.2f}</b><extra></extra>"))
                fig.add_hline(y=r.base_rate, line=dict(color=T.MUTED, dash="dash", width=1))
                fig.add_trace(go.Scatter(x=[r.recall], y=[r.precision], mode="markers", marker=dict(size=13, color=T.TEXT,
                                         line=dict(color=T.ACCENT, width=3)), hovertemplate="chosen threshold<extra></extra>"))
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(title="recall", range=[0, 1], showspikes=False, showgrid=True)
                fig.update_yaxes(title="precision", range=[0, 1.02])
                T.chart(fig, 360, legend=False)
        ct = T.load_csv(f"classifier_{kind}_threshold_costs.csv")
        with c2, T.card("Business cost vs threshold", "Cost = 5 × missed events + 1 × false alarms (CV months)"):
            if ct is not None:
                b = ct.loc[ct.cost.idxmin()]
                fig = go.Figure(go.Scatter(x=ct["threshold"], y=ct["cost"], line=dict(color=T.UP, width=2.4), fill="tozeroy",
                                           fillcolor=T.rgba(T.UP, .06), hovertemplate="threshold %{x:.2f} · cost <b>%{y:,.0f}</b><extra></extra>"))
                fig.add_trace(go.Scatter(x=[b.threshold], y=[b.cost], mode="markers+text", text=[f"  optimum {b.threshold:.2f}"],
                                         textposition="middle right", textfont=dict(color=T.TEXT, size=11),
                                         marker=dict(size=13, color=T.ACCENT, line=dict(color=T.SURFACE, width=2)), hoverinfo="skip"))
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(title="probability threshold", showspikes=False)
                fig.update_yaxes(title="cost units")
                T.chart(fig, 360, legend=False)
        t = cm[cm.kind == kind].set_index("split").drop(columns="kind")
        t.index = t.index.str.upper()
        t.index.name = "Split"
        t = t.rename(columns={"events": "Events", "base_rate": "Base rate", "pr_auc": "PR-AUC", "threshold": "Threshold",
                              "precision": "Precision", "recall": "Recall"})
        T.table(t, fmt={"Events": "{:,.0f}", "Base rate": "{:.1%}", "PR-AUC": "{:.3f}", "Threshold": "{:.2f}",
                        "Precision": "{:.2f}", "Recall": "{:.2f}"})
        T.takeaway({"spike_rel": "<b>Spike vs recent level</b>: price above yesterday's average by more than the training-P95 excess. Because it is "
                                 "relative, the event rate stays near 7 % even after the autumn-2026 level shift, and PR-AUC is about 7× the base rate on the holdout.",
                    "spike": "<b>Fixed-EUR spike</b>: price above the training P95 level. After the 2026 price surge it is exceeded in 43 % of holdout hours, "
                             "so it stops being a spike signal — the level-relative definition fixes this.",
                    "cheap": "<b>Cheap hour</b> (price below €5): about ten times more events than strictly negative prices, so the classifier has "
                             "something to learn from — and it is the event a battery cares about when choosing when to charge.",
                    "negative": "<b>Negative price</b>: very few events on the holdout (8), so only the CV result is informative."}[kind] +
                   " Accuracy would be misleading on rare events, so PR-AUC is compared with the base rate and the threshold minimises an explicit cost "
                   "(missed event = 5 × false alarm).")

# ================================================================== monitoring
with tab5:
    rm = T.load_csv("monitor_rolling_mae.csv", index_col=0, parse_dates=True)
    if rm is None:
        T.missing("Monitoring")
    else:
        lvl = rm["cv_level"].iloc[0]
        k = st.columns(3)
        T.kpi(k[0], "Current 30-day MAE", T.fmt_eur(rm["lightgbm"].iloc[-1]), f"CV level €{lvl:.1f}",
              delta=T.delta_chip(100 * (rm["lightgbm"].iloc[-1] / lvl - 1), "%", good_when_up=False), spark=rm["lightgbm"].tail(120))
        T.kpi(k[1], "Retrain trigger", T.fmt_eur(1.2 * lvl), "120 % of CV MAE for 14 days")
        T.kpi(k[2], "Alert days in back-test", f"{int(rm['retrain_alert'].sum())}", "three episodes")
        with T.card("Rolling 30-day MAE", "Shaded = days the retraining rule would have fired",
                    T.legend_html([("LightGBM", T.ACCENT), ("Naive · yesterday", T.NEUTRAL), ("Ensemble", T.AQUA), ("Trigger", T.UP)])):
            fig = go.Figure()
            al = rm["retrain_alert"] == 1
            if al.any():
                fig.add_trace(go.Bar(x=rm.index[al], y=[rm[["lightgbm", "naive_d1"]].max().max() * 1.05] * int(al.sum()),
                                     marker=dict(color=T.rgba(T.UP, .12)), width=86400000, hoverinfo="skip", name="alert"))
            for mname in [c for c in ("naive_d1", "ensemble", "lightgbm") if c in rm]:
                fig.add_trace(go.Scatter(x=rm.index, y=rm[mname], name=T.MODEL_LABEL[mname],
                                         line=dict(color=T.MODEL[mname], width=2.4 if mname == "lightgbm" else 1.5,
                                                   dash="dot" if mname.startswith("naive") else "solid"),
                                         hovertemplate=T.MODEL_LABEL[mname] + " <b>%{y:.1f}</b><extra></extra>"))
            fig.add_hline(y=1.2 * lvl, line=dict(color=T.UP, dash="dash", width=1))
            fig.update_layout(bargap=0)
            fig.update_yaxes(title="EUR/MWh")
            T.chart(fig, 360, legend=False)
        pm = T.load_csv("monitor_psi_by_month.csv")
        if pm is not None:
            with T.card("Feature drift (PSI vs training reference)", "< 0.1 stable · 0.1–0.25 moderate · > 0.25 significant shift"):
                piv = pm.pivot(index="feature", columns="month", values="psi")
                fig = go.Figure(go.Heatmap(z=piv.values.clip(0, 1), x=piv.columns, y=piv.index, zmin=0, zmax=1, xgap=2, ygap=2,
                                           customdata=piv.values, colorscale=[[0, "#16243A"], [0.1, "#1C5CAB"], [0.25, T.ACCENT],
                                                                              [0.6, T.UP], [1, "#B83A3A"]],
                                           colorbar=dict(title=dict(text="PSI", side="right")),
                                           hovertemplate="%{y} · %{x}<br>PSI <b>%{customdata:.2f}</b><extra></extra>"))
                fig.update_layout(hovermode="closest")
                fig.update_xaxes(showspikes=False)
                fig.update_yaxes(showgrid=False)
                T.chart(fig, 300, legend=False)
            st.caption("PSI compares each month with the full training years, so seasonal features (load, solar) also flag seasonal mix; "
                       "price features drifting in late 2026 is a genuine level shift.")
        T.takeaway("<b>Retraining rule:</b> retrain monthly on the expanding window; retrain immediately if the 30-day rolling MAE stays more "
                   "than 20 % above its CV level for 14 consecutive days, or a key feature's PSI exceeds 0.25.", "action")

"""Generate and execute the analysis notebooks from the precomputed reports.

Usage: python -m src.build_notebooks [--no-exec]
The notebooks read data/processed/hourly.parquet and reports/*; they never call an API.
Takeaway sentences are computed from the data, so they stay true after a re-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

from src import config as C

NB_DIR = C.ROOT / "notebooks"

SETUP = """\
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
R = ROOT / "reports"
plt.rcParams.update({"figure.figsize": (11, 4.2), "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
ACC, BLUE, TEAL, RED, GREY = "#E08E0B", "#3F6FD8", "#1E9E92", "#D64545", "#7A8194"
pd.set_option("display.float_format", "{:,.2f}".format)
"""
LOAD_H = """\
h = pd.read_parquet(ROOT / "data/processed/hourly.parquet")
h.index = h.index.tz_convert("Europe/Sofia")   # UTC internally, local only for display
p = h["price_BG"]
"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def nb_data_quality():
    return [
        md("# 01 · Data quality\nCoverage, gaps, resolution changes and DST handling for every series "
           "(produced by `src/clean.py`). Rules: UTC internally; 15-min data averaged to hourly; gaps ≤ 3 h "
           "linearly interpolated, longer gaps left missing; negative and extreme prices kept and flagged."),
        code(SETUP + LOAD_H + "print(h.shape, h.index.min(), '→', h.index.max())"),
        md("## Coverage and gaps per series"),
        code('q = pd.read_csv(R / "data_quality.csv")\n'
             'q[~q.series.str.startswith(("wact_", "wfc_"))].sort_values("coverage_pct").reset_index(drop=True)'),
        code('w = q[q.series.str.startswith(("wact_", "wfc_"))]\n'
             'print(f"Weather: {len(w)} series, coverage {w.coverage_pct.min():.2f}-{w.coverage_pct.max():.2f}%")\n'
             'print(f"Share of hours where the D-1 weather forecast is a stitched proxy: {h.weather_fc_is_proxy.mean():.1%}")\n'
             'print("Proxy period:", h.index[h.weather_fc_is_proxy == 1].min(), "→", h.index[h.weather_fc_is_proxy == 1].max())'),
        md("## Resolution changes (15-minute MTU)\nMedian sampling interval per month. The day-ahead market moved "
           "to 15-minute products during the period; everything is resampled to the hourly mean so history is consistent."),
        code('res = pd.read_csv(R / "resolution_by_month.csv", index_col=0)\n'
             'ax = res.plot(drawstyle="steps-mid", figsize=(11, 3.5), title="Median sampling interval (minutes)")\n'
             'ax.set_ylabel("minutes"); plt.show()\n'
             'for c in res: \n'
             '    ch = res[c][res[c].diff().fillna(0) != 0]\n'
             '    if len(ch): print(c, "changes:", ch.to_dict())'),
        md("## DST: 23- and 25-hour days\nBecause the index is UTC, the spring and autumn clock changes appear as "
           "local days with 23 and 25 hours and need no special treatment."),
        code('n = p.groupby(h.index.date).size()\nprint(n.value_counts().sort_index())\nprint(n[n != 24])'),
        md("## Negative and extreme prices are real"),
        code('print(f"Negative-price hours: {int((p < 0).sum())} ({(p < 0).mean():.2%}), min {p.min():.1f} EUR/MWh")\n'
             'print(f"P99 = {p.quantile(.99):.1f}, max = {p.max():.1f} EUR/MWh")\n'
             'ax = p.plot(lw=.4, color=ACC, title="BG hourly day-ahead price (EUR/MWh)"); ax.axhline(0, color=GREY, lw=.8); plt.show()'),
        md("**Takeaway.** The price series is complete; system forecasts have small gaps (mostly interpolated); some "
           "cross-border flow borders are not reported by ENTSO-E for parts of the period, which is why the net-import "
           "feature uses only the borders that are available. Weather forecasts before the D-1 archive starts are "
           "flagged as a proxy and excluded from the weather-uncertainty experiments."),
    ]


def nb_market_insights():
    return [
        md("# 02 · Market insights\nSix hypotheses about the Bulgarian day-ahead market, each answered with a chart, "
           "a one-sentence takeaway computed from the data, and a *so what* for a trader or analyst."),
        code(SETUP + LOAD_H),
        md("## 1. Price profile by hour and month\n*Hypothesis:* a midday solar dip and morning/evening peaks."),
        code('hm = p.groupby([h.index.hour, h.index.month]).mean().unstack()\n'
             'fig, ax = plt.subplots(figsize=(11, 5))\n'
             'im = ax.imshow(hm, aspect="auto", cmap="viridis"); plt.colorbar(im, label="EUR/MWh")\n'
             'ax.set_xticks(range(12), ["J","F","M","A","M","J","J","A","S","O","N","D"]); ax.set_ylabel("hour (local)")\n'
             'ax.set_title("Average BG price by hour and month"); ax.grid(False); plt.show()\n'
             'prof = p.groupby(h.index.hour).mean(); summer = hm[[5, 6, 7, 8]].mean(axis=1)\n'
             'print(f"Takeaway: cheapest hour {prof.idxmin()}:00 ({prof.min():.0f}), dearest {prof.idxmax()}:00 ({prof.max():.0f}); "\n'
             '      f"in May-Aug the midday trough is {summer.min():.0f} vs evening peak {summer.max():.0f} EUR/MWh.")'),
        md("**So what:** the intraday shape (cheap midday, expensive evening ramp) is the core of any battery or "
           "demand-shifting business case, and the evening ramp is where hedging matters most."),
        md("## 2. Residual load vs price\nResidual load = load − wind − solar (day-ahead forecasts)."),
        code('x = h[["residual_load_forecast", "price_BG"]].dropna()\n'
             'b = pd.qcut(x.residual_load_forecast, 20)\n'
             'g = x.groupby(b, observed=True).price_BG.agg(["median", lambda s: s.quantile(.1), lambda s: s.quantile(.9)])\n'
             'mid = [i.mid for i in g.index]\n'
             'plt.scatter(x.residual_load_forecast[::7], x.price_BG[::7], s=2, alpha=.15, color=GREY)\n'
             'plt.fill_between(mid, g.iloc[:, 1], g.iloc[:, 2], alpha=.3, color=ACC); plt.plot(mid, g["median"], color=ACC, lw=2)\n'
             'plt.ylim(x.price_BG.quantile(.005), x.price_BG.quantile(.995)); plt.xlabel("residual load forecast (MW)"); plt.ylabel("EUR/MWh")\n'
             'plt.title("Price vs residual load (median and P10-P90 per bin)"); plt.show()\n'
             'rho = x.corr("spearman").iloc[0, 1]\n'
             'print(f"Takeaway: Spearman rho = {rho:.2f}; median price rises from {g[\'median\'].iloc[0]:.0f} in the lowest to "\n'
             '      f"{g[\'median\'].iloc[-1]:.0f} EUR/MWh in the highest residual-load bin.")'),
        md("**So what:** residual load is the single most informative fundamental; a good residual-load view is a "
           "good price view."),
        md("## 3. Generation mix over time, cheap vs expensive hours\n(Actual generation — analysis only, never a feature.)"),
        code('gen = [c for c in h if c.startswith("gen_")]\n'
             'h[gen].resample("MS").mean().rename(columns=lambda c: c[4:]).plot.area(figsize=(11, 4.5), lw=0, cmap="tab20")\n'
             'plt.ylabel("MW"); plt.title("Monthly average generation by type"); plt.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12)); plt.show()\n'
             'lo, hi = p.quantile([.1, .9])\n'
             'cmp = pd.DataFrame({"cheapest 10%": h.loc[p <= lo, gen].mean(), "dearest 10%": h.loc[p >= hi, gen].mean()})\n'
             'cmp["difference"] = cmp.iloc[:, 1] - cmp.iloc[:, 0]\n'
             'cmp = cmp.sort_values("difference"); display(cmp.round(0))\n'
             'print(f"Takeaway: in the dearest hours {cmp.index[-1][4:]} is {cmp.difference.iloc[-1]:+.0f} MW and "\n'
             '      f"{cmp.index[0][4:]} {cmp.difference.iloc[0]:+.0f} MW compared with the cheapest hours.")'),
        md("**So what:** flexible thermal and hydro plants are marginal in expensive hours; solar defines the floor. "
           "Outages or water levels of these plants are key information not in our data."),
        md("## 4. Volatility and extremes"),
        code('spk = p.quantile(.99)\n'
             'fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))\n'
             'p.clip(-50, p.quantile(.999)).hist(bins=120, ax=ax[0], color=ACC); ax[0].set_title("Price distribution")\n'
             'neg = (p < 0).groupby(h.index.hour).sum(); neg.plot.bar(ax=ax[1], color=TEAL); ax[1].set_title("Negative hours by hour of day")\n'
             '(p > spk).groupby(h.index.hour).sum().plot.bar(ax=ax[2], color=RED); ax[2].set_title(f"Spikes (> P99 = {spk:.0f}) by hour")\n'
             'plt.tight_layout(); plt.show()\n'
             'ny = (p < 0).groupby(h.index.year).sum(); print("Negative hours per year:", ny.to_dict())\n'
             'nm = (p < 0).groupby(h.index.month).sum()\n'
             'print(f"Takeaway: {int((p<0).sum())} negative hours, concentrated at {neg.idxmax()}:00 and in month {nm.idxmax()}; "\n'
             '      f"spikes above {spk:.0f} EUR/MWh cluster at {(p > spk).groupby(h.index.hour).sum().idxmax()}:00.")'),
        md("**So what:** negative prices are a spring/summer, sunny-weekend-midday phenomenon; spikes are an evening "
           "phenomenon. Different hours need different risk treatment."),
        md("## 5. Cross-border coupling"),
        code('rows = []\nfor z in ["RO", "GR", "HU"]:\n'
             '    s = (h.price_BG - h[f"price_{z}"]).dropna()\n'
             '    rows.append(dict(zone=z, mean_spread=s.mean(), coupled_pct=100 * (s.abs() < 1).mean(),\n'
             '                     BG_higher_pct=100 * (s >= 1).mean(), BG_lower_pct=100 * (s <= -1).mean()))\n'
             'cb = pd.DataFrame(rows).set_index("zone"); display(cb)\n'
             'cpl = pd.DataFrame({z: ((h.price_BG - h[f"price_{z}"]).abs() < 1) for z in ["RO", "GR", "HU"]}).resample("MS").mean() * 100\n'
             'cpl.plot(marker="o", title="Share of hours with |BG - neighbour| < 1 EUR/MWh (%)"); plt.ylabel("%"); plt.show()\n'
             'P = h[["price_BG", "price_RO", "price_GR", "price_HU"]].dropna()\n'
             'gh = P[["price_GR", "price_HU"]]\n'
             'top = P.price_BG >= gh.max(axis=1) + 1; low = P.price_BG <= gh.min(axis=1) - 1\n'
             'print(f"Takeaway: BG is within 1 EUR/MWh of RO in {cb.loc[\'RO\',\'coupled_pct\']:.0f}% of hours and of GR in "\n'
             '      f"{cb.loc[\'GR\',\'coupled_pct\']:.0f}%; BG (moving with RO) is above both GR and HU in {top.mean():.0%} of hours and below both in {low.mean():.0%}.")\n'
             'print("Coupling with RO by month (last 8):", {k.strftime("%Y-%m"): v for k, v in cpl["RO"].tail(8).round(0).items()})'),
        md("*Note on the definition:* since mid-2026 BG and RO prices in clearly coupled hours differ by a few cents "
           "(e.g. 161.82 vs 161.79 EUR/MWh), so an exact-equality test understates convergence. Coupling is "
           "therefore defined as |spread| < 1 EUR/MWh. Even with this tolerance, decoupling increases strongly in 2026."),
        md("**So what:** most of the time BG imports the regional price; decoupled hours (congested interconnectors) "
           "are where local fundamentals matter and where cross-border spread trades live."),
        md("## 6. Daily spread (max − min) by season"),
        code('ds = p.groupby(h.index.date).agg(lambda s: s.max() - s.min()); ds.index = pd.to_datetime(ds.index)\n'
             'ds.resample("MS").mean().plot.bar(color=ACC, figsize=(11, 3.8), title="Average daily max-min spread (EUR/MWh)")\n'
             'plt.xticks(range(0, len(ds.resample("MS").mean()), 3), [d.strftime("%Y-%m") for d in ds.resample("MS").mean().index[::3]], rotation=45); plt.show()\n'
             'season = ds.groupby(ds.index.month).mean()\n'
             'print(f"Takeaway: the daily spread averages {ds.mean():.0f} EUR/MWh, highest in month {season.idxmax()} "\n'
             '      f"({season.max():.0f}) and lowest in month {season.idxmin()} ({season.min():.0f}); latest-year average "\n'
             '      f"{ds[ds.index.year == ds.index.year.max()].mean():.0f} vs first-year {ds[ds.index.year == ds.index.year.min()].mean():.0f}.")'),
        md("**So what:** the daily spread is the battery's raw revenue pool — notebook 05 turns it into EUR/MW-year."),
    ]


def nb_forecasting():
    return [
        md("# 03 · Forecasting: base models\nTarget: hourly BG day-ahead price of delivery day D, using only information "
           "available at gate closure on D-1. Expanding-window walk-forward (retrain monthly, test the next month); "
           "the last 3 months are a final holdout never used for tuning or model choices."),
        code(SETUP + 'fg = json.loads((R / "feature_groups.json").read_text())\n'
             'for g, cols in fg.items(): print(f"{g:18s} {len(cols):2d}  {cols[:6]}{\' ...\' if len(cols) > 6 else \'\'}")'),
        md("## Model comparison\nNaive 1 = same hour yesterday, Naive 2 = same hour last week, Ridge, LightGBM "
           "(plus the ML-Plus models from notebook 04). Skill = 1 − MAE / MAE(best naive). DM = Diebold-Mariano test vs best naive."),
        code('cv = pd.read_csv(R / "metrics_cv.csv"); ho = pd.read_csv(R / "metrics_holdout.csv")\n'
             'print("Walk-forward CV months"); display(cv.set_index("model").round(3))\n'
             'print("Final holdout (last 3 months)"); display(ho.set_index("model").round(3))'),
        code('pr = pd.read_csv(R / "predictions.csv", parse_dates=["delivery_day"])\n'
             'pr["timestamp_utc"] = pd.to_datetime(pr.timestamp_utc, utc=True)\n'
             'wk = pr[pr.delivery_day >= pr.delivery_day.max() - pd.Timedelta(days=13)].set_index("timestamp_utc")\n'
             'ax = wk.actual.plot(color="k", lw=1.6, label="actual")\n'
             'for m, c in [("naive_d1", GREY), ("ridge", BLUE), ("lightgbm", ACC)]: wk[m].plot(ax=ax, color=c, lw=1, label=m)\n'
             'ax.legend(); ax.set_ylabel("EUR/MWh"); ax.set_title("Last two weeks of the holdout"); plt.show()'),
        md("## Where does it fail?"),
        code('eh = pd.read_csv(R / "error_by_hour.csv", index_col=0); em = pd.read_csv(R / "error_by_month.csv", index_col=0)\n'
             'fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n'
             'eh[["naive_d1", "ridge", "lightgbm"]].plot(ax=ax[0], marker="o", title="MAE by hour (local)")\n'
             'em[["naive_d1", "ridge", "lightgbm"]].plot.bar(ax=ax[1], title="MAE by month"); plt.tight_layout(); plt.show()\n'
             'print(cv.set_index("model")[["MAE", "MAE_spike_hours", "MAE_negative_hours"]].round(1))'),
        code('h_ = ho.set_index("model"); best_naive = h_.loc[h_.index.str.startswith("naive"), "MAE"].idxmin()\n'
             'lg = h_.loc["lightgbm"]; rd = h_.loc["ridge"]\n'
             'print(f"Holdout: best naive = {best_naive} (MAE {h_.loc[best_naive, \'MAE\']:.2f}); Ridge MAE {rd.MAE:.2f} "\n'
             '      f"(skill {rd.skill_vs_best_naive:+.1%}); LightGBM MAE {lg.MAE:.2f} (skill {lg.skill_vs_best_naive:+.1%}, "\n'
             '      f"DM p = {lg.DM_p:.3g}).")'),
        md("**Honest conclusion.** The model beats the naive baselines clearly in normal hours, but its error is "
           "several times larger in spike and negative-price hours, and largest in the evening ramp. The missing "
           "drivers are known: plant outages (REMIT), hydro reservoir levels and interconnector capacities are not in the "
           "data, and fuel/carbon prices (monthly gas, daily TTF, Brent, a carbon proxy) were tested in `src/fuel_experiment.py` "
           "and did not improve the forecast. The model is a fundamentals-plus-persistence model, not an oracle."),
    ]


def nb_ml_deep_dive():
    return [
        md("# 04 · ML deep dive\nFeature ablation, Optuna tuning, neural network, ensemble, quantile forecast, "
           "spike/negative-price classifier, SHAP, Diebold-Mariano and drift monitoring. All runs are logged in MLflow "
           "(`mlflow ui --backend-store-uri sqlite:///mlflow.db`)."),
        code(SETUP),
        md("## 1. Feature ablation (last 12 CV months)"),
        code('ab = pd.read_csv(R / "ablation.csv"); display(ab.round(3))\n'
             'a = ab.iloc[1:].sort_values("delta_MAE"); plt.barh(a.dropped, a.delta_MAE, color=[RED if v > 0 else TEAL for v in a.delta_MAE])\n'
             'plt.xlabel("MAE change when group dropped (EUR/MWh)"); plt.title("Ablation"); plt.show()\n'
             'print(f"Takeaway: removing \'{a.dropped.iloc[-1]}\' hurts most ({a.delta_MAE.iloc[-1]:+.2f} EUR/MWh).")'),
        md("## 2. Optuna tuning\nTPE sampler, fixed seed, objective = MAE over the 4 most recent CV months "
           "(expanding window). The holdout is never seen."),
        code('ot = pd.read_csv(R / "optuna_trials.csv"); print(len(ot), "trials")\n'
             'plt.plot(ot.number, ot.value, "o", color=GREY, ms=4); plt.plot(ot.number, ot.value.cummin(), color=ACC)\n'
             'plt.xlabel("trial"); plt.ylabel("CV MAE"); plt.title("Optuna optimisation history"); plt.show()\n'
             'print(json.loads((ROOT / "models/lgbm_best_params.json").read_text()))'),
        md("## 3-4. Neural network and ensemble"),
        code('ho = pd.read_csv(R / "metrics_holdout.csv").set_index("model"); cv = pd.read_csv(R / "metrics_cv.csv").set_index("model")\n'
             'cols = ["MAE", "RMSE", "MAE_fold_mean", "MAE_fold_std", "skill_vs_best_naive", "DM_stat", "DM_p", "significant_5pct"]\n'
             'print("CV (mean ± sd across monthly folds)"); display(cv[cols].round(3))\n'
             'print("Holdout"); display(ho[cols].round(3))\n'
             'print("Ensemble weights (fitted on CV only):", json.loads((R / "ensemble_weights.json").read_text()))\n'
             'print(f"Takeaway: MLP MAE {ho.loc[\'mlp\',\'MAE\']:.2f} vs LightGBM {ho.loc[\'lightgbm\',\'MAE\']:.2f}; "\n'
             '      f"ensemble {ho.loc[\'ensemble\',\'MAE\']:.2f} on the holdout.")'),
        md("On tabular data with strong, monotone hand-made features, gradient boosting usually matches or beats a small "
           "MLP: trees handle heterogeneous scales, missing values and threshold effects (e.g. residual load above which "
           "thermal units set the price) natively, while the MLP needs more data to learn them."),
        md("### Skill with 95 % intervals and regime robustness\nSkill = 1 - MAE / MAE(naive). Intervals: moving-block bootstrap over "
           "delivery days (7-day blocks). Candidate variants and the adoption rule were fixed in advance (see `src/robust_experiment.py`): "
           "a variant replaces the baseline only if CV MAE is >= 0.1 lower and Diebold-Mariano p < 0.05 on the CV months."),
        code('sc = pd.read_csv(R / "skill_confidence.csv"); rm = pd.read_csv(R / "robust_metrics.csv")\n'
             'order = ["ridge", "lightgbm", "ensemble", "mlp", "lightgbm_level", "lightgbm_blend", "lightgbm_guarded", "lightgbm_levelshape", "lightgbm_linbase", "lightgbm_recency"]\n'
             'tab = sc.assign(ci=lambda d: d.apply(lambda r: f"{r.skill:+.1%} [{r.skill_lo:+.0%}, {r.skill_hi:+.0%}]", axis=1)).pivot(index="variant", columns="split", values="ci")\n'
             'display(tab.reindex([o for o in order if o in tab.index]))\n'
             'mae = rm.pivot(index="variant", columns="split", values="MAE").reindex([o for o in order if o in set(rm.variant)])\n'
             'mae.plot.bar(figsize=(11, 3.8), color=[BLUE, ACC]); plt.ylabel("MAE (EUR/MWh)"); plt.title("CV vs holdout MAE by variant"); plt.xticks(rotation=25); plt.show()\n'
             'print("Chosen by the rule:", json.loads((R / "robust_choice.json").read_text())["variant"])'),
        code('ep = pd.read_csv(R / "robust_episodes.csv"); display(ep.round(1))'),
        md("**Reading it.** No variant meets the rule, so LightGBM stays the headline model. The level-anchored variants lose a little on CV and "
           "gain about 3.5 MAE in the October-2026 surge, but in the earlier level-shift episodes they mostly do not help: they are insurance against "
           "extrapolation, not a tested improvement. Recency weighting is clearly worse on the holdout. The level/shape candidate was added after the "
           "first results were seen."),
        md("### Other inputs tested: capacity, fuels, outages\nEach optional group must lower CV MAE (last 12 CV months) to enter the model."),
        code('fc = json.loads((R / "feature_choice.json").read_text()); display(pd.Series(fc).to_frame("value"))\n'
             'ox = pd.read_csv(R / "outage_experiment.csv"); display(ox.round(3))\n'
             'fx = pd.read_csv(R / "fuel_experiment.csv"); display(fx.pivot(index="variant", columns="model", values="MAE").round(2))'),
        md("Interconnector capacities (NTC) and fuel/carbon prices did not pass the rule. Plant outages cannot be used point-in-time from the API "
           "(only the latest revision of each document is returned); as an upper bound even perfect (leaky) knowledge of the final outage windows "
           "does not improve the forecast."),
        md("## 5. Probabilistic forecast (quantile LightGBM 10/50/90)"),
        code('qm = pd.read_csv(R / "quantile_metrics.csv"); display(qm.round(3))\n'
             'qh = pd.read_csv(R / "quantile_by_hour.csv", index_col=0)\n'
             'fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n'
             'ax[0].plot([0, 1], [0, 1], "--", color=GREY)\n'
             'for _, r in qm.iterrows(): ax[0].plot([.1, .5, .9], [r.share_below_q10, r.share_below_q50, r.share_below_q90], "o-", label=r.split)\n'
             'ax[0].set_xlabel("nominal quantile"); ax[0].set_ylabel("observed share below"); ax[0].legend(); ax[0].set_title("Calibration")\n'
             'qh.coverage_10_90.plot.bar(ax=ax[1], color=ACC); ax[1].axhline(.8, color="k", ls="--"); ax[1].set_title("10-90% coverage by hour")\n'
             'plt.tight_layout(); plt.show()'),
        md("### Conformal recalibration\nThe raw band is too narrow. Each side is widened by the 90 % conformal quantile of "
           "its recent out-of-sample misses (days up to D-2 only; window chosen on CV months only)."),
        code('qc = pd.read_csv(R / "quantile_conformal.csv"); display(qc.round(3))\n'
             'order = ["raw quantile band"] + [v for v in qc.variant.unique() if v.startswith("conformal")]\n'
             'piv = qc.pivot(index="variant", columns="split", values="coverage").loc[order]\n'
             'ax = (100 * piv).plot.bar(figsize=(10, 3.6), color=[BLUE, ACC]); ax.axhline(80, color="k", ls="--", lw=.8)\n'
             'ax.set_ylabel("% of outcomes inside the 10-90 % band"); ax.set_title("Coverage: raw band vs conformal windows")\n'
             'plt.xticks(rotation=20); plt.show()\n'
             'ch = qc[qc.chosen].set_index("split").coverage\n'
             'print(f"Takeaway: coverage CV {ch[\'cv\']:.1%} (target 80 %), holdout {ch[\'holdout\']:.1%}; "\n'
             '      "the October-2026 surge pushed outcomes above the band.")'),
        md("**Business use:** the band is widest in the evening ramp and on solar-driven midday hours; in those hours "
           "a trader should hedge or reduce position size rather than rely on the point forecast."),
        md("## 6. Spike and negative-price classifier\nTargets: (a) price > P95 of each fold's training period (fixed EUR level), "
           "(b) **level-relative spike**: price above yesterday's daily mean by more than the training-P95 excess, and (c) price < 0. "
           "The fixed definition stops working after the autumn-2026 level shift (43 % of holdout hours); the relative one does not. "
           "LightGBM with balanced class weights. Threshold chosen on CV months by minimising "
           "**cost = 5 × missed events + 1 × false alarms** (a missed spike = unhedged exposure / missed battery "
           "discharge; a false alarm = an unnecessary hedge)."),
        code('cm = pd.read_csv(R / "classifier_metrics.csv"); display(cm.round(3))\n'
             'fig, ax = plt.subplots(1, 3, figsize=(15, 4))\n'
             'for i, k in enumerate(["spike_rel", "spike", "negative"]):\n'
             '    f = R / f"classifier_{k}_pr_curve.csv"\n'
             '    if f.exists():\n'
             '        c = pd.read_csv(f); ax[i].plot(c.recall, c.precision, color=ACC); ax[i].set_title(f"PR curve: {k}")\n'
             '        ax[i].set_xlabel("recall"); ax[i].set_ylabel("precision")\n'
             'plt.tight_layout(); plt.show()'),
        md("## 7. Explainability (SHAP on the holdout)"),
        code('display(Image(R / "figures/shap_summary.png", width=750))\n'
             'display(pd.read_csv(R / "shap_importance.csv").head(12).round(2))'),
        code('for c in ["spike", "negative", "normal"]:\n'
             '    f = R / f"figures/shap_waterfall_{c}.png"\n'
             '    if f.exists(): display(Image(f, width=700))'),
        code('pdp = pd.read_csv(R / "pdp_residual_load.csv"); plt.plot(pdp.iloc[:, 0], pdp.iloc[:, 1], color=ACC, lw=2)\n'
             'plt.xlabel("residual load forecast (MW)"); plt.ylabel("avg predicted price"); plt.title("Partial dependence"); plt.show()'),
        md("The drivers line up with market logic: recent price level (yesterday's same hour and daily mean), "
           "neighbour prices (market coupling), hour of day and residual load. The partial dependence on residual "
           "load is monotone increasing and steepens at high load (merit-order). Nothing points to leakage: no "
           "same-day actual quantity is in the feature set."),
        md("## 9-10. Monitoring and drift"),
        code('rm = pd.read_csv(R / "monitor_rolling_mae.csv", index_col=0, parse_dates=True)\n'
             'ax = rm[["lightgbm", "naive_d1"]].plot(title="Rolling 30-day MAE"); ax.axhline(1.2 * rm.cv_level.iloc[0], color=RED, ls="--", label="retrain trigger")\n'
             'ax.legend(); plt.show()\n'
             'print("Days with retrain alert:", int(rm.retrain_alert.sum()))\n'
             'display(pd.read_csv(R / "monitor_psi.csv").round(3))'),
        md("**Retraining rule:** retrain monthly on the expanding window; retrain immediately if the 30-day rolling "
           "MAE stays more than 20 % above its CV level for 14 consecutive days, or if a key feature's PSI exceeds 0.25."),
    ]


def nb_weather():
    return [
        md("# 04b · Weather forecast vs actual: error analysis and simulation\n"
           "Chain: weather forecast error → wind/solar forecast error → residual-load error → price error. "
           "Only the archived D-1 forecast (Open-Meteo Previous Runs API, day-1 run) is used as a model feature."),
        code(SETUP),
        md("## 1. Weather forecast (D-1) vs observed"),
        code('we = pd.read_csv(R / "weather_fc_error.csv"); display(we[we.season == "all"].round(2))\n'
             'display(we.pivot_table(index=["variable", "point"], columns="season", values="MAE").round(2))'),
        code('s = pd.read_csv(R / "weather_fc_vs_actual_sample.csv")\n'
             'fig, ax = plt.subplots(1, 3, figsize=(14, 4))\n'
             'for a, v in zip(ax, ["temp_sofia", "wind100_dobrich", "rad_stara_zagora"]):\n'
             '    a.scatter(s["wact_" + v], s["wfc_" + v], s=3, alpha=.25, color=ACC); lim = [s["wact_" + v].min(), s["wact_" + v].max()]\n'
             '    a.plot(lim, lim, "k--", lw=.8); a.set_xlabel("observed"); a.set_ylabel("D-1 forecast"); a.set_title(v)\n'
             'plt.tight_layout(); plt.show()\n'
             'pd.read_csv(R / "weather_fc_error_by_hour.csv", index_col=0)[["wind100_dobrich", "rad_stara_zagora"]].plot(subplots=True, figsize=(11, 4), title="MAE by hour")\n'
             'plt.show()'),
        md("## 2. ENTSO-E wind/solar forecast vs actual generation"),
        code('rs = pd.read_csv(R / "res_forecast_error_summary.csv"); display(rs.T.round(3))\n'
             'rt = pd.read_csv(R / "res_forecast_error_by_hour_month.csv", index_col=0)\n'
             'rt[[c for c in rt if not c.startswith("m_")]].plot.bar(figsize=(11, 3.8), title="Forecast MAE by hour (MW)"); plt.show()\n'
             'print("Solar error by hour is concentrated in daylight hours:", (rt.solar_mae_mw.idxmax()), ":00 peak")'),
        md("## 3. From residual-load forecast error to price error"),
        code('l = pd.read_csv(R / "res_error_vs_price_error.csv")\n'
             'b = pd.qcut(l.res_load_err_mw, 15); g = l.groupby(b, observed=True).price_err.mean()\n'
             'plt.scatter(l.res_load_err_mw[::5], l.price_err[::5], s=2, alpha=.15, color=GREY)\n'
             'plt.plot([i.mid for i in g.index], g.values, "o-", color=ACC, lw=2); plt.ylim(l.price_err.quantile(.01), l.price_err.quantile(.99))\n'
             'plt.xlabel("actual − forecast residual load (MW)"); plt.ylabel("price error actual − LightGBM (EUR/MWh)"); plt.show()\n'
             'slope = np.polyfit(l.res_load_err_mw, l.price_err, 1)[0]\n'
             'print(f"Takeaway: slope = {100 * slope:+.2f} EUR/MWh per 100 MW of residual-load miss, i.e. about zero.")\n'
             'display(pd.read_csv(R / "worst_res_forecast_days.csv").round(1))'),
        md("Why is there no relationship? The **day-ahead price is cleared on D-1 forecasts**, so a miss in the realised "
           "residual load cannot change it; it moves intraday and imbalance prices instead. The price-forecast error "
           "comes from what the market knew on D-1 and our model did not (fuel prices, outages, cross-border capacity)."),
        md("## 4. Oracle vs realistic vs no weather (key comparison)\nSame LightGBM settings and months; only the "
           "weather inputs differ. Months where the D-1 archive is incomplete are excluded."),
        code('ex = pd.read_csv(R / "weather_experiment.csv").set_index("setup").MAE; display(ex.round(3))\n'
             'print(f"Cost of weather-forecast uncertainty (realistic − oracle): {ex.realistic - ex.oracle:+.3f} EUR/MWh")\n'
             'print(f"Value of weather information (no weather − realistic): {ex.no_weather - ex.realistic:+.3f} EUR/MWh")'),
        md("## 5. Noise-injection simulation\nOracle weather + Gaussian noise with the measured forecast-error SD × "
           "scale, 3 seeds per level, last 6 months."),
        code('nz = pd.read_csv(R / "weather_noise_injection.csv"); display(nz.round(3))\n'
             'plt.errorbar(nz.noise_scale, nz["mean"], yerr=nz["std"], marker="o", color=ACC, capsize=4)\n'
             'plt.xlabel("noise scale (× measured weather-forecast error SD)"); plt.ylabel("price MAE (EUR/MWh)"); plt.show()'),
        md("## 6. Business takeaway"),
        code('display(pd.read_csv(R / "battery_weather_scenarios.csv", index_col=0)[["eur_per_mw_year", "pct_of_perfect"]].round(1))'),
        md("**What the data says.** Point weather forecasts add a little accuracy (no-weather vs realistic), but the "
           "oracle barely beats the realistic D-1 forecast, and injecting up to 2× the measured weather error leaves "
           "price MAE almost unchanged. Battery revenue is also nearly identical across the weather scenarios. "
           "The reasons: (1) the ENTSO-E load/wind/solar forecasts already carry the weather signal; (2) the day-ahead "
           "price itself is set on D-1 forecasts, so the realised weather cannot move it; (3) the model is dominated by "
           "price lags and neighbour prices. The ENTSO-E solar forecast error is large and growing but almost "
           "uncorrelated with point-weather error, which points to capacity growth and curtailment rather than weather.\n\n"
           "**So what for a trader:** weather-forecast risk is mainly an intraday/balancing exposure. Hedge more on "
           "high-solar and volatile-wind days, widen the risk band in the midday solar hours, and re-optimise after "
           "intraday weather updates."),
    ]


def nb_battery():
    return [
        md("# 05 · Battery value\n1 MW / 2 MWh, 90 % round-trip efficiency, max one cycle per day, no degradation "
           "(all configurable in `src/battery.py`). Each day a small LP picks charge/discharge hours on the *schedule* "
           "prices; revenue is settled at *actual* prices."),
        code(SETUP),
        code('br = pd.read_csv(R / "battery_results.csv", index_col=0); display(br.round(1))\n'
             'bd = pd.read_csv(R / "battery_daily.csv", parse_dates=["delivery_day"]).set_index("delivery_day")\n'
             'bd.resample("MS").sum().plot.bar(figsize=(12, 4), title="Monthly revenue (EUR, 1 MW)")\n'
             'plt.xticks(range(len(bd.resample("MS").sum())), [d.strftime("%Y-%m") for d in bd.resample("MS").sum().index], rotation=45); plt.show()\n'
             'pf, fc, nv = br.loc["perfect_foresight"], br.loc["forecast_lightgbm"], br.loc["naive_yesterday"]\n'
             'print(f"Takeaway: perfect foresight €{pf.eur_per_mw_year:,.0f}/MW-yr; LightGBM schedule captures {fc.pct_of_perfect:.0f}% "\n'
             '      f"(€{fc.eur_per_mw_year:,.0f}), naive {nv.pct_of_perfect:.0f}% (€{nv.eur_per_mw_year:,.0f}). "\n'
             '      f"Value of the forecast over naive: €{fc.eur_per_mw_year - nv.eur_per_mw_year:,.0f}/MW-yr.")'),
        md("## Sensitivity: duration and efficiency"),
        code('se = pd.read_csv(R / "battery_sensitivity.csv")\n'
             'display(se.pivot_table(index=["duration_h", "efficiency"], columns="strategy", values="eur_per_mw_year").round(0))\n'
             'display(se[se.efficiency == .9].pivot_table(index="duration_h", columns="strategy", values="pct_of_perfect").round(1))'),
        md("## Effect of price level / spread"),
        code('h = pd.read_parquet(ROOT / "data/processed/hourly.parquet"); h.index = h.index.tz_convert("Europe/Sofia")\n'
             'd = h.price_BG.groupby(h.index.date).agg(["mean", lambda s: s.max() - s.min()]); d.columns = ["level", "spread"]\n'
             'd.index = pd.to_datetime(d.index); j = bd.join(d, how="inner")\n'
             'for c in ["level", "spread"]:\n'
             '    q = pd.qcut(j[c], 4, labels=["Q1", "Q2", "Q3", "Q4"])\n'
             '    print(f"Avg daily perfect-foresight revenue by {c} quartile:", j.groupby(q, observed=True).perfect_foresight.mean().round(0).to_dict())'),
        md("## Costs, confidence filter and reserve capacity"),
        code('dg = pd.read_csv(R / "battery_degradation.csv")\n'
             'display(dg.pivot(index="degradation_eur_mwh", columns="strategy", values="eur_per_mw_year").round(0))\n'
             'dg.pivot(index="degradation_eur_mwh", columns="strategy", values="eur_per_mw_year").plot(marker="o", figsize=(9, 3.6), title="Net revenue per MW-year vs degradation cost")\n'
             'plt.ylabel("EUR / MW-year"); plt.show()\n'
             'fl = pd.read_csv(R / "battery_filter.csv"); display(fl.pivot(index="min_expected_profit", columns="split", values="eur_per_mw_year").round(0))\n'
             'print("Confidence-filter threshold chosen on CV:", json.loads((R / "battery_filter_choice.json").read_text()))'),
        md("Each EUR 5/MWh of wear cost removes roughly EUR 3.6k per MW-year. The confidence filter adds nothing: Bulgarian daily spreads always cover the "
           "round-trip losses, so the best threshold on the CV months is zero."),
        code('rv = pd.read_csv(R / "battery_reserve.csv"); display(rv.round(1))\n'
             'rm = pd.read_csv(R / "reserve_monthly.csv", parse_dates=["month"]).set_index("month")\n'
             'rm[[c for c in rm if c.endswith(" price")]].plot(drawstyle="steps-post", figsize=(10, 3.6), title="Contracted reserve capacity price (EUR/MW/h)"); plt.show()'),
        md("**Read with care.** The reserve prices are constant for long stretches (administered, not competitively cleared) and the markets are tiny "
           "(FCR ~29 MW, aFRR-up ~110 MW contracted), so one battery is a large share and a fleet saturates them; the price-taker assumption fails. Only the "
           "capacity payment is counted. Break-even capex and payback are available as sliders on the dashboard."),
        md("**Limitations:** day-ahead arbitrage only (no intraday, aFRR/mFRR or capacity revenue, which dominate real "
           "battery business cases), price-taker, no network charges or degradation, one cycle per day. The numbers are "
           "a floor for the arbitrage component, not a full investment case."),
    ]


NOTEBOOKS = {
    "01_data_quality.ipynb": nb_data_quality,
    "02_market_insights.ipynb": nb_market_insights,
    "03_forecasting.ipynb": nb_forecasting,
    "04_ml_deep_dive.ipynb": nb_ml_deep_dive,
    "04b_weather_uncertainty.ipynb": nb_weather,
    "05_battery_value.ipynb": nb_battery,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB_DIR / name
        if not a.no_exec:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, path)
        print("wrote", Path(path).relative_to(C.ROOT))


if __name__ == "__main__":
    main()

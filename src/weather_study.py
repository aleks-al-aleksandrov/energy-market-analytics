"""Weather-forecast uncertainty study (notebook 04b).

Chain: weather forecast error -> wind/solar forecast error -> residual-load error -> price error.
Experiments (identical LightGBM settings, identical train/test months):
  * oracle    : observed weather of day D as features (upper bound, NOT realistic)
  * realistic : archived D-1 forecast (headline setup)
  * no weather: weather features removed
  * noise     : oracle weather + Gaussian noise (sd = measured forecast error x scale)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C
from src import battery as B
from src.evaluate import mae, walk_forward
from src.features import FEATURE_GROUPS, TARGET, feature_columns
from src.models import make_lgbm

log = logging.getLogger("weather")
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def weather_error_tables(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    genuine = h["weather_fc_is_proxy"] == 0
    rows, by_hour = [], []
    local = h.index.tz_convert(C.LOCAL_TZ)
    for c in [c for c in h.columns if c.startswith("wfc_")]:
        a = "wact_" + c[4:]
        if a not in h:
            continue
        m = genuine & h[c].notna() & h[a].notna()
        e = (h.loc[m, c] - h.loc[m, a])
        var, point = c[4:].split("_", 1)
        seasons = pd.Series(local[m].month, index=e.index).map(SEASON)
        for s, es in [("all", e)] + list(e.groupby(seasons)):
            rows.append(dict(variable=var, point=point, season=s, bias=es.mean(), MAE=es.abs().mean(),
                             RMSE=np.sqrt((es ** 2).mean()), error_sd=es.std(), n=len(es)))
        hh = e.abs().groupby(local[m].hour).mean()
        by_hour.append(hh.rename(f"{var}_{point}"))
    return pd.DataFrame(rows), pd.concat(by_hour, axis=1).rename_axis("hour_local")


def res_forecast_error(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    local = h.index.tz_convert(C.LOCAL_TZ)
    out = pd.DataFrame(index=h.index)
    wind_act = [c for c in h.columns if c.startswith("gen_wind")]
    solar_act = [c for c in h.columns if c.startswith("gen_solar")]
    if wind_act:
        out["wind_err_mw"] = h["wind_forecast"] - h[wind_act].sum(axis=1, min_count=1)
        out["wind_actual"] = h[wind_act].sum(axis=1, min_count=1)
    if solar_act:
        out["solar_err_mw"] = h["solar_forecast"] - h[solar_act].sum(axis=1, min_count=1)
        out["solar_actual"] = h[solar_act].sum(axis=1, min_count=1)
    out["hour"] = local.hour
    out["month"] = local.month
    by_hour = out.groupby("hour")[[c for c in out if c.endswith("_err_mw")]].agg(lambda s: s.abs().mean())
    by_month = out.groupby("month")[[c for c in out if c.endswith("_err_mw")]].agg(lambda s: s.abs().mean())
    by_hour.columns = [c.replace("_err_mw", "_mae_mw") for c in by_hour.columns]
    by_month.columns = [c.replace("_err_mw", "_mae_mw") for c in by_month.columns]
    summary = {}
    for k in ("wind", "solar"):
        if f"{k}_err_mw" in out:
            e, a = out[f"{k}_err_mw"], out[f"{k}_actual"]
            summary[f"{k}_mae_mw"] = float(e.abs().mean())
            summary[f"{k}_bias_mw"] = float(e.mean())
            summary[f"{k}_mae_pct_of_mean"] = float(100 * e.abs().mean() / a.mean())
    # link to weather error (genuine D-1 period): R^2 of a 1-D linear fit
    g = h["weather_fc_is_proxy"] == 0
    for k, w in (("solar", "rad_stara_zagora"), ("wind", "wind100_dobrich")):
        if f"{k}_err_mw" in out and f"wfc_{w}" in h:
            we = (h[f"wfc_{w}"] - h[f"wact_{w}"])[g]
            re = out.loc[g, f"{k}_err_mw"]
            m = we.notna() & re.notna()
            if k == "solar":
                m &= h.loc[g, f"wact_{w}"].reindex(m.index).fillna(0) > 0
            if m.sum() > 100:
                summary[f"{k}_err_r2_vs_weather_err"] = float(np.corrcoef(we[m], re[m])[0, 1] ** 2)
    return pd.DataFrame([summary]), pd.concat([by_hour, by_month.add_prefix("m_")], axis=1), out


def experiments(f: pd.DataFrame, h: pd.DataFrame, months: list[pd.Period], params: dict | None,
                noise_scales=(0.0, 0.5, 1.0, 2.0), seeds=(0, 1, 2), noise_months: int = 6):
    """Oracle / realistic / no-weather + noise injection. Returns (summary, noise table, preds)."""
    wfc = FEATURE_GROUPS["weather"]
    wact = ["wact_" + c[4:] for c in wfc]
    feats = feature_columns()
    no_w = feature_columns(exclude=["weather"])
    fo = f.copy()
    for c, a in zip(wfc, wact):
        fo[c] = h[a]  # oracle: observed weather in the forecast columns

    preds = pd.DataFrame(index=f.index)
    preds["realistic"] = walk_forward(f, lambda: make_lgbm(params), feats, months)
    preds["oracle"] = walk_forward(fo, lambda: make_lgbm(params), feats, months)
    preds["no_weather"] = walk_forward(f, lambda: make_lgbm(params), no_w, months)
    preds = preds.dropna()
    preds["actual"] = f.loc[preds.index, TARGET]
    preds["delivery_day"] = f.loc[preds.index, "delivery_day"]
    preds["naive_d1"] = f.loc[preds.index, "price_lag24"]
    summary = pd.DataFrame([{"setup": s, "MAE": mae(preds["actual"], preds[s])}
                            for s in ("oracle", "realistic", "no_weather")])

    # noise injection on top of oracle weather (train + test, mimicking forecast quality)
    genuine = h["weather_fc_is_proxy"] == 0
    sd = {c: float((h.loc[genuine, c] - h.loc[genuine, a]).std()) for c, a in zip(wfc, wact)}
    lo = {c: 0.0 for c in wfc}
    hi = {c: (100.0 if "cloud" in c else np.inf) for c in wfc}
    nm = months[-noise_months:]
    rows = []
    noisy_preds = {}
    for scale in noise_scales:
        for seed in (seeds if scale > 0 else (0,)):
            fn = fo.copy()
            rng = np.random.default_rng(1000 + seed)
            for c in wfc:
                noise = rng.normal(0, sd[c] * scale, len(fn))
                v = fn[c] + noise
                if not c.startswith("wfc_temp"):
                    v = v.clip(lo[c], hi[c])
                if c.startswith("wfc_rad"):
                    v = v.where(fo[c] > 0, 0)  # no sun at night
                fn[c] = v
            p = walk_forward(fn, lambda: make_lgbm(params), feats, nm)
            rows.append(dict(noise_scale=scale, seed=seed, MAE=mae(f.loc[p.index, TARGET], p)))
            if seed == 0:
                noisy_preds[scale] = p
    noise = pd.DataFrame(rows).groupby("noise_scale")["MAE"].agg(["mean", "std", "count"]).reset_index()
    noise["std"] = noise["std"].fillna(0)
    return summary, noise, preds, noisy_preds


def battery_under_weather(preds: pd.DataFrame, noisy: dict) -> pd.DataFrame:
    bat = B.Battery()
    df = preds.copy()
    cols = {"perfect_foresight": "actual", "oracle_weather": "oracle", "realistic_weather": "realistic",
            "no_weather": "no_weather"}
    for s, p in noisy.items():
        if s > 0:
            df[f"noise_{s}x"] = p.reindex(df.index)
            cols[f"noisy_{s}x"] = f"noise_{s}x"
    df = df.dropna(subset=list(cols.values()))
    daily = B.simulate(df, bat, cols)
    return B.summarise(daily, bat)


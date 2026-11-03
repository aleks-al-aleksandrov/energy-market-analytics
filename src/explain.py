"""SHAP explainability for the LightGBM point model."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C


def shap_analysis(model, X: pd.DataFrame, y: pd.Series, out_prefix: str = "shap",
                  sample: int = 4000) -> dict:
    """Global summary + waterfalls for a spike, a negative-price and a normal hour.

    Saves PNGs to reports/figures and tables to reports/. Returns chosen timestamps.
    """
    import shap

    rng = np.random.default_rng(C.SEED)
    Xs = X.iloc[np.sort(rng.choice(len(X), size=min(sample, len(X)), replace=False))]
    expl = shap.TreeExplainer(model)
    sv = expl(Xs)

    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv.values).mean(0)})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(C.REPORTS / f"{out_prefix}_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(C.FIGURES / f"{out_prefix}_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    # sample SHAP values for the dashboard (beeswarm-like scatter), top 12 features
    top = imp.sort_values("mean_abs_shap", ascending=False)["feature"].head(12).tolist()
    long = []
    for c in top:
        j = list(X.columns).index(c)
        v = Xs[c].to_numpy(dtype=float)
        rank = pd.Series(v).rank(pct=True).to_numpy()
        long.append(pd.DataFrame({"feature": c, "shap": sv.values[:, j], "value_pct": rank}))
    pd.concat(long).sample(frac=0.5, random_state=C.SEED).to_csv(C.REPORTS / f"{out_prefix}_values_sample.csv", index=False)

    # three illustrative hours
    picks = {}
    if (y > y.quantile(0.99)).any():
        picks["spike"] = y.idxmax()
    if (y < 0).any():
        picks["negative"] = y.idxmin()
    med = (y - y.median()).abs()
    picks["normal"] = med.idxmin()
    rows = []
    for label, ts in picks.items():
        one = expl(X.loc[[ts]])
        plt.figure()
        shap.plots.waterfall(one[0], max_display=12, show=False)
        plt.title(f"{label}: {ts:%Y-%m-%d %H:%M} UTC, actual {y.loc[ts]:.1f} EUR/MWh")
        plt.savefig(C.FIGURES / f"{out_prefix}_waterfall_{label}.png", dpi=130, bbox_inches="tight")
        plt.close("all")
        contrib = pd.Series(one.values[0], index=X.columns)
        for feat in contrib.abs().sort_values(ascending=False).index[:10]:
            rows.append(dict(case=label, timestamp_utc=ts, actual=float(y.loc[ts]),
                             base_value=float(one.base_values[0]), feature=feat,
                             feature_value=float(X.loc[ts, feat]) if pd.notna(X.loc[ts, feat]) else np.nan,
                             shap=float(contrib[feat])))
    pd.DataFrame(rows).to_csv(C.REPORTS / f"{out_prefix}_cases.csv", index=False)
    return picks


def partial_dependence(model, X: pd.DataFrame, feature: str, grid_n: int = 30, sample: int = 3000) -> pd.DataFrame:
    """Manual 1-D partial dependence (average prediction while sweeping one feature)."""
    Xs = X.sample(min(sample, len(X)), random_state=C.SEED)
    grid = np.nanpercentile(X[feature], np.linspace(2, 98, grid_n))
    out = []
    for g in grid:
        Xg = Xs.copy()
        Xg[feature] = g
        out.append((g, float(np.mean(model.predict(Xg)))))
    return pd.DataFrame(out, columns=[feature, "avg_predicted_price"])


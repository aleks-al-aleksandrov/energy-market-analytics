"""Thin MLflow helper: one run per model/configuration, local SQLite backend."""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from src import config as C

log = logging.getLogger("tracking")
TRACKING_URI = f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "bg_day_ahead_price"


@contextlib.contextmanager
def mlflow_run(name: str, params: dict | None = None, metrics: dict | None = None,
               artifacts: list[Path] | None = None):
    """Log params/metrics/artifacts; silently degrades if MLflow is not installed."""
    try:
        import mlflow
    except ImportError:
        yield None
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("seed", C.SEED)
        if params:
            mlflow.log_params({k: (str(v)[:500]) for k, v in params.items()})
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for a in artifacts or []:
            if Path(a).exists():
                mlflow.log_artifact(str(a))
        yield run



def export_comparison_figure(out: Path | None = None) -> Path | None:
    """Bar chart of CV vs holdout MAE for the latest run of every model in the MLflow store."""
    try:
        import mlflow
    except ImportError:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if runs.empty or "metrics.cv_mae" not in runs:
        return None
    runs = runs.dropna(subset=["metrics.cv_mae"]).sort_values("start_time").groupby("tags.mlflow.runName").last()
    runs = runs.sort_values("metrics.cv_mae")
    out = out or C.FIGURES / "mlflow_comparison.png"
    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar([i - 0.2 for i in x], runs["metrics.cv_mae"], 0.4, label="walk-forward CV", color="#3F6FD8")
    ax.bar([i + 0.2 for i in x], runs["metrics.holdout_mae"], 0.4, label="final holdout", color="#E08E0B")
    ax.set_xticks(list(x), runs.index, rotation=20)
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("MLflow runs: MAE by model (experiment bg_day_ahead_price)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out

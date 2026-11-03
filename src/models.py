"""Forecast models. Every model exposes fit(X, y) / predict(X)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C

LGBM_DEFAULT = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=C.SEED, verbose=-1, n_jobs=-1,
)


class Naive:
    """Copies a lag column: 'price_lag24' (same hour yesterday) or 'price_lag168' (last week)."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


def make_ridge(features: list[str], alpha: float = 10.0) -> Pipeline:
    cat = [c for c in ("hour", "dow", "month") if c in features]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def make_lgbm(params: dict | None = None, objective: str = "regression", alpha: float | None = None):
    import lightgbm as lgb

    p = {**LGBM_DEFAULT, **(params or {})}
    if objective == "quantile":
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)
    return lgb.LGBMRegressor(objective=objective, **p)


def make_lgbm_classifier(params: dict | None = None):
    import lightgbm as lgb

    p = {**LGBM_DEFAULT, "n_estimators": 400, **(params or {})}
    return lgb.LGBMClassifier(objective="binary", class_weight="balanced", **p)


class LevelAdjusted:
    """Fit a model on the deviation from a price level known at gate closure (default: D-1 mean).

    Tree models cannot extrapolate beyond the target range seen in training; when the price
    level jumps (e.g. a fuel-price shock), modelling `price - level` keeps the forecast anchored
    to the latest observed level while the trees learn the shape.
    """

    def __init__(self, model, base: str = "price_d1_mean", fallback: str = "price_7d_mean"):
        self.model, self.base, self.fallback = model, base, fallback

    def _level(self, X: pd.DataFrame) -> pd.Series:
        return X[self.base].fillna(X[self.fallback])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        lvl = self._level(X)
        ok = lvl.notna().to_numpy()
        self.model.fit(X[ok], (y - lvl)[ok])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X) + self._level(X).fillna(0).to_numpy()


class Recency:
    """Exponentially down-weight old training rows (half-life in days) so the model follows the current regime."""

    def __init__(self, model, half_life_days: float = 90.0):
        self.model, self.half_life = model, half_life_days

    def fit(self, X: pd.DataFrame, y: pd.Series):
        age = np.asarray((X.index.max() - X.index).total_seconds(), dtype=float) / 86400
        self.model.fit(X, y, sample_weight=np.power(0.5, age / self.half_life))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class LinearBase:
    """Ridge first, then a tree model on the residual: the linear part extrapolates beyond the training price range."""

    def __init__(self, linear, tree):
        self.linear, self.tree = linear, tree

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.linear.fit(X, y)
        self.tree.fit(X, y - self.linear.predict(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.linear.predict(X) + self.tree.predict(X)


LEVEL_FEATURES = ["price_d1_mean", "price_d1_max", "price_d1_min", "price_7d_mean", "price_7d_std", "price_RO_d1_mean",
                  "price_GR_d1_mean", "price_HU_d1_mean", "load_forecast", "residual_load_forecast", "wind_forecast",
                  "solar_forecast", "dow", "month", "is_holiday", "is_weekend"]


class LevelShape:
    """Level / shape decomposition.

    * level: a regularised LINEAR model of the delivery day's mean price, expressed relative to yesterday's mean
      (so it can follow the price level out of the training range, which trees cannot)
    * shape: LightGBM on the hourly deviation from that day's mean, a quantity that is far more stable across regimes
    Forecast = level + shape. Needs `price_d1_mean` among the features; rows are grouped by market delivery day.
    """

    def __init__(self, tree, features: list[str], alpha: float = 10.0):
        self.tree = tree
        self.cols = [c for c in LEVEL_FEATURES if c in features]
        self.alpha = alpha

    @staticmethod
    def _day(X: pd.DataFrame) -> np.ndarray:
        return X.index.tz_convert("Europe/Brussels").normalize().tz_localize(None).to_numpy()

    def _agg(self, X: pd.DataFrame, day: np.ndarray) -> pd.DataFrame:
        return X[self.cols].groupby(day).mean()

    def _anchor(self, agg: pd.DataFrame) -> pd.Series:
        return agg["price_d1_mean"].fillna(agg.get("price_7d_mean", np.nan)).fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        day = self._day(X)
        agg = self._agg(X, day)
        ymean = y.groupby(day).mean().reindex(agg.index)
        self.linear = make_ridge(self.cols, alpha=self.alpha)
        self.linear.fit(agg, ymean - self._anchor(agg))
        self.tree.fit(X, y - ymean.reindex(day).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        day = self._day(X)
        agg = self._agg(X, day)
        level = pd.Series(self._anchor(agg).to_numpy() + self.linear.predict(agg), index=agg.index)
        return level.reindex(day).to_numpy() + self.tree.predict(X)


class TorchMLP:
    """Small feed-forward net (PyTorch) with the same feature set as the other models.

    Hour/dow/month are one-hot encoded, numeric features median-imputed and scaled.
    The target is standardised; Huber loss; early stopping on the most recent 10 %
    of the training window (time-ordered, no shuffling across the split).
    """

    def __init__(self, features: list[str], hidden=(128, 64), dropout=0.1, lr=1e-3,
                 epochs=80, batch_size=512, patience=8, seed=C.SEED):
        self.features, self.hidden, self.dropout, self.lr = features, hidden, dropout, lr
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    def _net(self, n_in):
        import torch.nn as nn

        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cat = [c for c in ("hour", "dow", "month") if c in self.features]
        num = [c for c in self.features if c not in cat]
        self.pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ])
        Xa = self.pre.fit_transform(X).astype(np.float32)
        Xa = np.nan_to_num(Xa)
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())
        ya = ((y.to_numpy() - self.y_mu) / self.y_sd).astype(np.float32)
        n_val = max(int(0.1 * len(Xa)), 24)
        Xt, yt = torch.tensor(Xa[:-n_val]), torch.tensor(ya[:-n_val])
        Xv, yv = torch.tensor(Xa[-n_val:]), torch.tensor(ya[-n_val:])
        self.net = self._net(Xa.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = torch.nn.HuberLoss(delta=1.0)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = torch.mean(torch.abs(self.net(Xv).squeeze(1) - yv)).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        Xa = np.nan_to_num(self.pre.transform(X).astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(Xa)).squeeze(1).numpy()
        return out * self.y_sd + self.y_mu


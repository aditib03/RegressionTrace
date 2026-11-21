"""LightGBM training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import LGB_PARAMS, N_JOBS
from .data import SLICE_COLUMN


def train(X: pd.DataFrame, y: np.ndarray, weight: np.ndarray | None = None, seed: int = 0,
          **overrides) -> lgb.LGBMClassifier:
    params = {**LGB_PARAMS, **overrides, "n_jobs": N_JOBS, "random_state": seed}
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=weight)
    return model


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


@dataclass
class Metrics:
    auc: float
    logloss: float
    ece: float
    slice_auc: dict[int, float]

    def as_dict(self) -> dict:
        d = {"auc": self.auc, "logloss": self.logloss, "ece": self.ece}
        d.update({f"auc_slice_{k}": v for k, v in self.slice_auc.items()})
        return d


def evaluate(model, X: pd.DataFrame, y: np.ndarray) -> Metrics:
    p = model.predict_proba(X)[:, 1]
    return metrics_from_proba(y, p, X[SLICE_COLUMN].to_numpy() if SLICE_COLUMN in X else None)


def metrics_from_proba(y, p, slices=None) -> Metrics:
    sl = {}
    if slices is not None:
        for s in np.unique(slices):
            m = slices == s
            if m.sum() >= 200 and len(np.unique(y[m])) == 2:
                sl[int(s)] = float(roc_auc_score(y[m], p[m]))
    return Metrics(
        auc=float(roc_auc_score(y, p)),
        logloss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        ece=ece(y, p),
        slice_auc=sl,
    )


def paired_bootstrap_auc_delta(y, p_new, p_old, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap CI for AUC(new) - AUC(old) on the same evaluation rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        yb = y[i]
        if yb.min() == yb.max():
            deltas[b] = 0.0
            continue
        deltas[b] = roc_auc_score(yb, p_new[i]) - roc_auc_score(yb, p_old[i])
    return {
        "delta": float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }

"""Label-quality diagnostics via confident learning (Northcutt et al., 2021).

Out-of-fold probabilities are computed with K-fold LightGBM. A row is flagged as
a likely label error when the model confidently predicts the *other* class:
its probability for the other class exceeds that class's per-class threshold
(the mean self-confidence of rows labelled with that class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import model as M


def out_of_fold_proba(X: pd.DataFrame, y: np.ndarray, folds: int = 3, seed: int = 0,
                      n_estimators: int = 100) -> np.ndarray:
    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = M.train(X.iloc[tr], y[tr], seed=seed, n_estimators=n_estimators)
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def label_issues(y: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_issue bool array, self_confidence array)."""
    probs = np.column_stack([1 - p1, p1])
    self_conf = probs[np.arange(len(y)), y]
    thresholds = np.array([probs[y == c, c].mean() for c in (0, 1)])
    other = 1 - y
    is_issue = (probs[np.arange(len(y)), other] >= thresholds[other]) & (self_conf < 0.5)
    return is_issue, self_conf


def batch_label_quality(y: np.ndarray, p1: np.ndarray, batch: np.ndarray) -> pd.DataFrame:
    issue, conf = label_issues(y, p1)
    n_b = int(batch.max()) + 1
    cnt = np.maximum(np.bincount(batch, minlength=n_b), 1)
    pos_rate = np.bincount(batch, weights=y, minlength=n_b) / cnt
    return pd.DataFrame({
        "batch": np.arange(n_b),
        "label_issue_frac": np.bincount(batch, weights=issue.astype(float), minlength=n_b) / cnt,
        "mean_self_conf": np.bincount(batch, weights=conf, minlength=n_b) / cnt,
        "pos_rate": pos_rate,
    })

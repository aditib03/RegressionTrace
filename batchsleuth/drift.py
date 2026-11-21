"""Per-batch data-validation signals against a trusted reference sample.

* ``oor_rate``: fraction of rows with any numeric feature outside the reference's
  [0.1%, 99.9%] range (catches zero-fills, unit changes, sentinel values).
* ``drift_max_psi``: max Population Stability Index across features (catches
  distribution shift, but also fires on natural source-to-source variation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import numeric_columns


def _psi(ref_hist: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(ref_hist, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def batch_drift(X: pd.DataFrame, batch: np.ndarray, X_ref: pd.DataFrame, bins: int = 10,
                top: int = 3) -> pd.DataFrame:
    cols = numeric_columns(X_ref)
    n_b = int(batch.max()) + 1
    order_rows = np.argsort(batch, kind="stable")
    bounds = np.searchsorted(batch[order_rows], np.arange(n_b + 1))
    groups = [order_rows[bounds[b]:bounds[b + 1]] for b in range(n_b)]

    psi = np.zeros((n_b, len(cols)))
    oor_any = np.zeros(len(X), dtype=bool)
    oor_col = np.zeros((n_b, len(cols)))
    for j, c in enumerate(cols):
        ref = X_ref[c].to_numpy()
        lo, hi = np.quantile(ref, [0.001, 0.999])
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_hist = np.histogram(ref, edges)[0] / len(ref)
        col = X[c].to_numpy()
        out = (col < lo) | (col > hi)
        oor_any |= out
        for b, g in enumerate(groups):
            psi[b, j] = _psi(ref_hist, col[g], edges)
            oor_col[b, j] = out[g].mean() if len(g) else 0.0

    top_psi = np.argsort(-psi, axis=1)[:, :top]
    top_oor = np.argmax(oor_col, axis=1)
    return pd.DataFrame({
        "batch": np.arange(n_b),
        "oor_rate": [oor_any[g].mean() if len(g) else 0.0 for g in groups],
        "oor_top_feature": [cols[k] for k in top_oor],
        "drift_max_psi": psi.max(axis=1),
        "drift_top_features": [[cols[k] for k in row if psi[b, k] > 0.1] for b, row in enumerate(top_psi)],
    })

"""Dataset loading and batching.

Uses the UCI Covertype dataset (581k rows, 54 features, ~11 MB download) as a
binary task: "is this Lodgepole Pine?" (~49% positive). Rows are shuffled with a
fixed seed and cut into equally sized *ingestion batches* that play the role of
daily data drops in a production pipeline. A disjoint holdout set acts as the
trusted evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PATHS

TARGET = "label"
SLICE_COLUMN = "wilderness"  # used for per-slice promotion checks


@dataclass
class BatchedData:
    """Training pool split into batches plus a holdout set (all in memory)."""

    X: pd.DataFrame
    y: np.ndarray
    batch: np.ndarray  # batch index per training row
    X_hold: pd.DataFrame
    y_hold: np.ndarray

    @property
    def n_batches(self) -> int:
        return int(self.batch.max()) + 1

    def copy(self) -> BatchedData:
        return BatchedData(self.X.copy(), self.y.copy(), self.batch.copy(), self.X_hold, self.y_hold)


def _load_covtype_frame() -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype

    d = fetch_covtype(data_home=str(PATHS.data / "sk"), as_frame=True)
    df = d.frame.copy()
    df[TARGET] = (df.pop("Cover_Type") == 2).astype(np.int8)
    df.columns = [c.lower() for c in df.columns]
    # Collapse one-hot wilderness area into a single categorical slice column.
    wa = [c for c in df.columns if c.startswith("wilderness_area")]
    df[SLICE_COLUMN] = df[wa].to_numpy().argmax(axis=1).astype(np.int8)
    return df.astype({c: np.float32 for c in df.columns if c not in (TARGET, SLICE_COLUMN)})


def _synthetic_frame(n: int, seed: int) -> pd.DataFrame:
    """Offline fallback / test fixture with a covtype-like shape."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10, n_redundant=4, flip_y=0.02,
        class_sep=0.8, random_state=seed,
    )
    df = pd.DataFrame(X.astype(np.float32), columns=[f"f{i}" for i in range(X.shape[1])])
    df[TARGET] = y.astype(np.int8)
    df[SLICE_COLUMN] = (np.abs(X[:, 0] * 1000).astype(int) % 4).astype(np.int8)
    return df


def load_batched(
    n_batches: int = 100,
    batch_size: int = 2000,
    holdout_size: int = 30000,
    seed: int = 0,
    synthetic: bool = False,
    heterogeneity: float = 0.35,
) -> BatchedData:
    """Load data and cut the training pool into batches.

    ``heterogeneity`` controls how non-i.i.d. batches are: rows are ordered by a
    noisy geographic key (elevation + wilderness area), so consecutive batches
    come from overlapping but different regions, mimicking data drops from
    different sources/time periods. 0 = i.i.d. shuffle; larger = more distinct.
    Clean batches therefore show natural drift and loss differences, which is
    what makes naive baselines produce false alarms.
    """
    need = n_batches * batch_size + holdout_size
    df = _synthetic_frame(need, seed) if synthetic else _load_covtype_frame()
    if len(df) < need:
        raise ValueError(f"dataset has {len(df)} rows, need {need}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))[:need]
    df = df.iloc[idx].reset_index(drop=True)
    hold, train = df.iloc[:holdout_size], df.iloc[holdout_size:].reset_index(drop=True)
    if heterogeneity > 0:
        geo = train.iloc[:, 0].rank(pct=True).to_numpy() + 0.25 * train[SLICE_COLUMN].to_numpy()
        geo = (geo - geo.mean()) / geo.std()
        key = heterogeneity * geo + rng.normal(0, 1, len(train)) * (1 - heterogeneity)
        train = train.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)
    feats = [c for c in df.columns if c != TARGET]
    return BatchedData(
        X=train[feats].reset_index(drop=True),
        y=train[TARGET].to_numpy(),
        batch=np.repeat(np.arange(n_batches), batch_size),
        X_hold=hold[feats].reset_index(drop=True),
        y_hold=hold[TARGET].to_numpy(),
    )


def feature_columns(X: pd.DataFrame) -> list[str]:
    return list(X.columns)


def numeric_columns(X: pd.DataFrame) -> list[str]:
    """Continuous columns (more than two distinct values, not the slice id)."""
    out = []
    for c in X.columns:
        if c == SLICE_COLUMN:
            continue
        if X[c].nunique(dropna=True) > 2:
            out.append(c)
    return out

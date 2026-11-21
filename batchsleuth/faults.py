"""Fault injectors that simulate realistic data-pipeline bugs inside a batch.

Severity = the fraction of a corrupted batch's rows that are affected
(``low`` 5%, ``med`` 15%, ``high`` 40%). Partial corruption is the realistic and
hard case: most rows in a bad batch still look normal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .data import BatchedData, numeric_columns

SEVERITY = {"low": 0.05, "med": 0.15, "high": 0.4}


def _affected(d: BatchedData, batches, sev, rng) -> np.ndarray:
    r = np.flatnonzero(np.isin(d.batch, batches))
    return r[rng.random(len(r)) < sev]


def _top_numeric(d: BatchedData, k: int) -> list[str]:
    # Bugs in predictive columns are the ones that hurt; rank by |corr| with the label.
    cols = numeric_columns(d.X)
    corr = {c: abs(np.nan_to_num(np.corrcoef(d.X[c].to_numpy(), d.y)[0, 1])) for c in cols}
    return sorted(corr, key=lambda c: -corr[c])[:k]


def label_flip(d, batches, sev, rng) -> str:
    """Annotation bug: labels flipped at random."""
    hit = _affected(d, batches, sev, rng)
    d.y[hit] = 1 - d.y[hit]
    return f"flipped {len(hit)} labels"


def class_flip(d, batches, sev, rng) -> str:
    """Systematic labelling bug: positives recorded as negatives."""
    hit = _affected(d, batches, sev, rng)
    hit = hit[d.y[hit] == 1]
    d.y[hit] = 0
    return f"relabelled {len(hit)} positives as negative"


def feature_shuffle(d, batches, sev, rng) -> str:
    """Broken join: the top predictive columns are misaligned with their rows."""
    cols = _top_numeric(d, 3)
    hit = _affected(d, batches, sev, rng)
    for c in cols:
        j = d.X.columns.get_loc(c)
        d.X.iloc[hit, j] = rng.permutation(d.X.iloc[hit, j].to_numpy())
    return f"misaligned {cols} on {len(hit)} rows"


def unit_change(d, batches, sev, rng) -> str:
    """New upstream source: units changed (ft -> m) and its labels follow a different policy."""
    cols = _top_numeric(d, 2)
    hit = _affected(d, batches, sev, rng)
    for c in cols:
        j = d.X.columns.get_loc(c)
        d.X.iloc[hit, j] = d.X.iloc[hit, j].to_numpy() * 0.3048
    flip = hit[rng.random(len(hit)) < 0.5]
    d.y[flip] = 1 - d.y[flip]
    return f"rescaled {cols} and flipped {len(flip)} labels on {len(hit)} rows"


def missing_burst(d, batches, sev, rng) -> str:
    """Upstream outage: predictive columns null-filled with a sentinel 0."""
    cols = _top_numeric(d, 3)
    hit = _affected(d, batches, sev, rng)
    for c in cols:
        d.X.iloc[hit, d.X.columns.get_loc(c)] = 0.0
    return f"zero-filled {cols} on {len(hit)} rows"


FAULTS: dict[str, Callable] = {
    "label_flip": label_flip,
    "class_flip": class_flip,
    "feature_shuffle": feature_shuffle,
    "unit_change": unit_change,
    "missing_burst": missing_burst,
}


def inject(d: BatchedData, fault: str, batches: list[int], severity: str | float, seed: int = 0):
    """Return (corrupted copy, description)."""
    sev = SEVERITY[severity] if isinstance(severity, str) else float(severity)
    out = d.copy()
    desc = FAULTS[fault](out, list(batches), sev, np.random.default_rng(seed))
    return out, desc

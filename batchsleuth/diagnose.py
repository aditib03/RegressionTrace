"""Root-cause ranking: combine influence, label quality and drift into one score per batch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import drift, influence, labelq

# column, weight, absolute floor for the robust scale.
# Chosen on tuning seeds 100-101 (scratch experiment), which the benchmark never uses.
# PSI drift is reported for explanation but excluded from the score: natural
# source-to-source variation makes it fire on clean batches (P@k 0.10 when used alone).
SIGNALS = {
    "influence": ("influence_total", 1.0, 1e-3),
    "label_quality": ("label_issue_frac", 1.0, 0.005),
    "validation": ("oor_rate", 1.0, 0.005),
}


def robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    scale = max(mad, floor, 0.05 * abs(med))
    return (x - med) / scale


@dataclass
class Diagnosis:
    table: pd.DataFrame  # one row per batch, sorted by suspicion
    timings: dict = field(default_factory=dict)
    row_influence: np.ndarray | None = None
    label_issue: np.ndarray | None = None
    oof_proba: np.ndarray | None = None

    def ranking(self, by: str = "score") -> list[int]:
        return self.table.sort_values(by, ascending=False)["batch"].astype(int).tolist()

    def top(self, k: int) -> list[int]:
        return self.ranking()[:k]


def _explain(r: pd.Series) -> str:
    parts = []
    if r["z_influence"] > 3:
        parts.append(f"raises eval loss (influence z={r['z_influence']:.1f})")
    if r["z_label_quality"] > 3:
        parts.append(f"{r['label_issue_frac']:.0%} likely label errors")
    if r["z_validation"] > 3:
        parts.append(f"{r['oor_rate']:.0%} rows out of range (e.g. {r['oor_top_feature']})")
    if parts and r["drift_top_features"]:
        parts.append("distribution shift in " + ", ".join(r["drift_top_features"]))
    return "; ".join(parts) or "no strong signal"


def diagnose(model, X: pd.DataFrame, y: np.ndarray, batch: np.ndarray, X_eval: pd.DataFrame,
             y_eval: np.ndarray, X_ref: pd.DataFrame | None = None, weight=None, seed: int = 0,
             oof_proba: np.ndarray | None = None) -> Diagnosis:
    X_ref = X_eval if X_ref is None else X_ref
    tm = {}
    # Work with compact 0..B-1 ids internally; map back to real batch ids at the end.
    batch_ids, batch = np.unique(np.asarray(batch), return_inverse=True)

    t = time.perf_counter()
    row_inf = influence.row_influence(model, X, y, X_eval, y_eval, weight=weight)
    inf_tab = influence.batch_influence(row_inf, batch)
    tm["influence_s"] = time.perf_counter() - t

    t = time.perf_counter()
    p_oof = labelq.out_of_fold_proba(X, y, seed=seed) if oof_proba is None else oof_proba
    issue, _ = labelq.label_issues(y, p_oof)
    lq_tab = labelq.batch_label_quality(y, p_oof, batch)
    tm["label_quality_s"] = time.perf_counter() - t

    t = time.perf_counter()
    dr_tab = drift.batch_drift(X, batch, X_ref)
    tm["drift_s"] = time.perf_counter() - t

    # Loss-only baseline: mean in-sample training loss per batch.
    p_in = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p_in) + (1 - y) * np.log(1 - p_in))
    n_b = int(batch.max()) + 1
    loss_b = np.bincount(batch, weights=loss, minlength=n_b) / np.maximum(np.bincount(batch, minlength=n_b), 1)

    tab = inf_tab.merge(lq_tab, on="batch").merge(dr_tab, on="batch")
    tab["train_loss"] = loss_b
    score = np.zeros(len(tab))
    for name, (col, w, floor) in SIGNALS.items():
        z = robust_z(tab[col].to_numpy(), floor)
        tab[f"z_{name}"] = z
        score += w * np.clip(z, 0, None)
    tab["score"] = score
    tab["explanation"] = tab.apply(_explain, axis=1)
    tab["batch"] = batch_ids[tab["batch"].to_numpy()]
    tab = tab.sort_values("score", ascending=False).reset_index(drop=True)
    tm["total_s"] = sum(tm.values())
    return Diagnosis(tab, tm, row_inf, issue, p_oof)

"""Counterfactual repairs: what would the model look like without / with fixed suspect batches?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnose import Diagnosis

ACTIONS = ("drop", "relabel", "reweight")
SUSPECT_THRESHOLD = 6.0  # chosen on tuning seeds: ~0.07 false positives per scenario
MAX_SUSPECT_FRAC = 0.10


def select_suspects(diag: Diagnosis, threshold: float = SUSPECT_THRESHOLD,
                    max_frac: float = MAX_SUSPECT_FRAC) -> list[int]:
    t = diag.table
    cap = max(1, round(max_frac * len(t)))
    return t[t["score"] >= threshold].sort_values("score", ascending=False)["batch"].astype(int).tolist()[:cap]


@dataclass
class RepairPlan:
    action: str
    suspects: list[int]
    keep_mask: np.ndarray          # rows kept in the training set
    labels: np.ndarray             # labels after repair (full length)
    weights: np.ndarray            # weights after repair (full length)
    removed_frac: float
    changed_rows: int


def plan_repair(diag: Diagnosis, y: np.ndarray, batch: np.ndarray, suspects: list[int],
                action: str = "drop") -> RepairPlan:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action}; choose from {ACTIONS}")
    in_suspect = np.isin(batch, suspects)
    keep = np.ones(len(y), dtype=bool)
    labels = y.copy()
    weights = np.ones(len(y))
    changed = 0
    if action == "drop":
        keep = ~in_suspect
        changed = int(in_suspect.sum())
    elif action == "relabel":
        # Flip confident-learning label issues inside suspect batches only.
        flip = in_suspect & diag.label_issue
        labels[flip] = 1 - labels[flip]
        changed = int(flip.sum())
    elif action == "reweight":
        # Down-weight suspect rows by how much the out-of-fold model trusts their label.
        conf = np.where(y == 1, diag.oof_proba, 1 - diag.oof_proba)
        weights[in_suspect] = np.clip(conf[in_suspect], 0.05, 1.0)
        changed = int(in_suspect.sum())
    return RepairPlan(action, list(suspects), keep, labels, weights,
                      removed_frac=float(1 - keep.mean()), changed_rows=changed)

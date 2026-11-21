"""Model-promotion safeguards.

A candidate may replace the champion only if every check passes. Thresholds are
deliberately conservative: a repair that silently hurts a slice, removes too
much data, or degrades calibration is blocked even if overall AUC looks fine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .model import metrics_from_proba, paired_bootstrap_auc_delta


@dataclass
class GateRules:
    max_auc_drop_ci: float = 0.002     # lower 95% CI of AUC delta must exceed -this
    max_logloss_increase: float = 0.002
    max_ece_increase: float = 0.01
    max_slice_auc_drop: float = 0.01
    max_removed_frac: float = 0.15


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    candidate: dict = field(default_factory=dict)
    champion: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)

    def summary(self) -> str:
        lines = [f"PROMOTION {'ALLOWED' if self.passed else 'BLOCKED'}"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
        return "\n".join(lines)


def evaluate_gate(y: np.ndarray, p_candidate: np.ndarray, p_champion: np.ndarray,
                  slices: np.ndarray | None = None, removed_frac: float = 0.0,
                  rules: GateRules | None = None, seed: int = 0) -> GateResult:
    r = rules or GateRules()
    cand = metrics_from_proba(y, p_candidate, slices)
    champ = metrics_from_proba(y, p_champion, slices)
    boot = paired_bootstrap_auc_delta(y, p_candidate, p_champion, seed=seed)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    add("auc_non_inferior", boot["ci_low"] > -r.max_auc_drop_ci,
        f"dAUC={boot['delta']:+.4f} (95% CI {boot['ci_low']:+.4f}..{boot['ci_high']:+.4f}), "
        f"need CI low > {-r.max_auc_drop_ci:+.4f}")
    dll = cand.logloss - champ.logloss
    add("logloss", dll <= r.max_logloss_increase, f"dLogLoss={dll:+.4f}, limit +{r.max_logloss_increase}")
    dece = cand.ece - champ.ece
    add("calibration", dece <= r.max_ece_increase, f"dECE={dece:+.4f}, limit +{r.max_ece_increase}")
    worst = min(((cand.slice_auc[s] - champ.slice_auc[s], s) for s in cand.slice_auc if s in champ.slice_auc),
                default=(0.0, None))
    add("slice_auc", worst[0] >= -r.max_slice_auc_drop,
        f"worst slice {worst[1]} dAUC={worst[0]:+.4f}, limit -{r.max_slice_auc_drop}")
    add("data_removed", removed_frac <= r.max_removed_frac,
        f"removed {removed_frac:.1%} of training rows, limit {r.max_removed_frac:.0%}")
    return GateResult(all(c["passed"] for c in checks), checks, cand.as_dict(), champ.as_dict())

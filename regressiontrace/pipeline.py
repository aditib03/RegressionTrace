"""End-to-end workflow shared by the CLI and the API.

Story of the demo:
  1. ``init``: 80 historical batches -> dataset version v1 -> champion model.
  2. 20 new batches arrive; 4 of them carry pipeline bugs (logged in ``fault_log``,
     which diagnosis never reads). A refresh v2 = v1 + new batches is trained.
  3. ``check``: the refreshed candidate regresses against the champion.
  4. ``diagnose``: rank batches by suspicion, with explanations.
  5. ``repair``: build a counterfactual dataset version (drop/relabel/reweight
     suspects), retrain, and evaluate the promotion gate.
  6. ``promote``: only if the gate passes.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import faults as F
from . import gate as G
from . import model as M
from . import repair as R
from . import tracking
from .config import PATHS
from .data import SLICE_COLUMN, load_batched
from .diagnose import Diagnosis, diagnose
from .provenance import Store

DIAG_ROWS = 5000        # holdout rows reserved for diagnosis (influence eval set + drift reference)
INFLUENCE_EVAL = 2000   # sample of the diagnosis slice used as influence evaluation set

DEMO_FAULTS = [("class_flip", "high"), ("unit_change", "high"), ("label_flip", "high"),
               ("missing_burst", "high"), ("feature_shuffle", "high")]


@contextmanager
def open_store(read_only: bool = False):
    s = Store(PATHS.store, read_only=read_only)
    try:
        yield s
    finally:
        s.close()


def _holdout_split(store: Store):
    Xh, yh = store.load_holdout()
    return (Xh.iloc[:DIAG_ROWS], yh[:DIAG_ROWS]), (Xh.iloc[DIAG_ROWS:].reset_index(drop=True), yh[DIAG_ROWS:])


def init_demo(seed: int = 0, n_new: int = 20, synthetic: bool = False, force: bool = False) -> dict:
    if PATHS.store.exists():
        if not force:
            raise FileExistsError(f"{PATHS.store} exists; pass force=True / --force to rebuild")
        # Reset all generated project-local state (store + MLflow tracking/artifacts).
        import shutil
        PATHS.store.unlink()
        PATHS.mlflow_db.unlink(missing_ok=True)
        shutil.rmtree(PATHS.mlflow_artifacts, ignore_errors=True)
    d = load_batched(seed=seed, synthetic=synthetic)
    rng = np.random.default_rng(seed)
    order = rng.permutation(d.n_batches)
    history, new = order[:-n_new], order[-n_new:]
    bad = rng.choice(new, len(DEMO_FAULTS), replace=False)
    log = []
    for (fault, sev), b in zip(DEMO_FAULTS, bad):
        d, desc = F.inject(d, fault, [int(b)], sev, seed=int(b))
        log.append((int(b), fault, sev, desc))

    with open_store() as s:
        for day, b in enumerate(order):
            m = d.batch == b
            src = "historical" if b in history else f"daily-drop-{day - len(history) + 1:02d}"
            s.ingest_batch(int(b), d.X[m].reset_index(drop=True), d.y[m], source=src)
        s.set_holdout(d.X_hold, d.y_hold)
        for row in log:
            s.log_fault(*row)
        v1 = s.create_version(history, note="initial training set (historical batches)")
        v2 = s.create_version(order, parent=v1, note=f"refresh: v1 + {n_new} new daily batches")

    champ = train_version(v1, run_name="champion-v1")
    tracking.set_champion(champ)
    cand = train_version(v2, run_name="refresh-v2")
    return {"v1": v1, "v2": v2, "champion_run": champ, "candidate_run": cand,
            "injected_faults": [dict(zip(["batch_id", "fault", "severity", "description"], r)) for r in log]}


def train_version(version_id: str, run_name: str | None = None, tags: dict | None = None,
                  extra_metrics: dict | None = None) -> str:
    with open_store(read_only=True) as s:
        X, y, batch, w, _ = s.load_version(version_id)
        _, (Xg, yg) = _holdout_split(s)
        v = s.get_version(version_id)
    model = M.train(X, y, weight=w if not np.allclose(w, 1) else None)
    metrics = M.evaluate(model, Xg, yg).as_dict()
    metrics.update(extra_metrics or {})
    return tracking.log_model_run(
        model, version_id=version_id, metrics=metrics, run_name=run_name,
        params={"n_train_rows": len(y), "n_batches": len(np.unique(batch))},
        tags={"parent_version": str(v["parent_version"]), "repair": json.dumps(v["repair"]), **(tags or {})},
    )


def _gate_inputs(run_id: str):
    with open_store(read_only=True) as s:
        _, (Xg, yg) = _holdout_split(s)
    champ = tracking.champion_run_id()
    if champ is None:
        raise RuntimeError("no champion model registered; run init first")
    p_new = tracking.load_model(run_id).predict_proba(Xg)[:, 1]
    p_old = tracking.load_model(champ).predict_proba(Xg)[:, 1]
    return yg, p_new, p_old, Xg[SLICE_COLUMN].to_numpy(), champ


def check(run_id: str) -> dict:
    """Regression check / promotion gate for a run against the current champion."""
    y, p_new, p_old, sl, champ = _gate_inputs(run_id)
    run = tracking.get_run(run_id)
    removed = float(run.data.metrics.get("removed_frac", 0.0))
    res = G.evaluate_gate(y, p_new, p_old, sl, removed_frac=removed)
    return {"run_id": run_id, "champion_run": champ, **res.as_dict(), "summary": res.summary()}


def diagnose_run(run_id: str, seed: int = 0, log: bool = True) -> tuple[Diagnosis, str]:
    run = tracking.get_run(run_id)
    vid = run.data.tags["dataset_version"]
    with open_store(read_only=True) as s:
        X, y, batch, w, _ = s.load_version(vid)
        (Xd, yd), _ = _holdout_split(s)
    model = tracking.load_model(run_id)
    ev = np.random.default_rng(seed).choice(len(yd), min(INFLUENCE_EVAL, len(yd)), replace=False)
    diag = diagnose(model, X, y, batch, Xd.iloc[ev], yd[ev], X_ref=Xd, weight=w, seed=seed)
    if log:
        tracking.log_artifact_text(run_id, "diagnosis.csv", diag.table.to_csv(index=False))
    return diag, vid


def repair_run(run_id: str, action: str = "drop", suspects: list[int] | None = None) -> dict:
    diag, vid = diagnose_run(run_id)
    suspects = R.select_suspects(diag) if suspects is None else [int(b) for b in suspects]
    if not suspects:
        return {"run_id": run_id, "suspects": [], "message": "no batch exceeds the suspicion threshold"}
    with open_store(read_only=True) as s:
        _, y, batch, _, row_id = s.load_version(vid)
        parent = s.get_version(vid)
    # diagnosis rows are aligned with load_version order (row_id ascending)
    plan = R.plan_repair(diag, y, batch, suspects, action)
    keep_batches = sorted(set(parent["batch_ids"]) - (set(suspects) if action == "drop" else set()))
    overrides = {int(r): int(lab) for r, lab, old in zip(row_id, plan.labels, y) if lab != old}
    weights = {int(r): float(x) for r, x in zip(row_id, plan.weights) if x != 1.0}
    with open_store() as s:
        new_vid = s.create_version(keep_batches, parent=vid, label_overrides=overrides or None,
                                   weights=weights or None,
                                   repair={"action": action, "suspects": suspects, "source_run": run_id},
                                   note=f"counterfactual: {action} {suspects}")
    new_run = train_version(new_vid, run_name=f"repair-{action}", tags={"repair_of": run_id},
                            extra_metrics={"removed_frac": plan.removed_frac, "changed_rows": plan.changed_rows})
    gate = check(new_run)
    return {"source_run": run_id, "suspects": suspects, "action": action, "new_version": new_vid,
            "new_run": new_run, "removed_frac": plan.removed_frac, "changed_rows": plan.changed_rows,
            "gate": gate}


def promote(run_id: str) -> dict:
    res = check(run_id)
    if res["passed"]:
        tracking.set_champion(run_id)
    return {**res, "promoted": res["passed"]}


def status() -> dict:
    with open_store(read_only=True) as s:
        versions = s.list_versions()
        n_batches = len(s.list_batches())
    runs = tracking.list_runs()
    champ = tracking.champion_run_id()
    rows = [{
        "run_id": r.info.run_id, "name": r.info.run_name, "dataset_version": r.data.tags.get("dataset_version"),
        "auc": round(r.data.metrics.get("auc", float("nan")), 4),
        "logloss": round(r.data.metrics.get("logloss", float("nan")), 4),
        "champion": r.info.run_id == champ,
    } for r in runs]
    versions = json.loads(versions.to_json(orient="records", date_format="iso"))  # NaN/None/timestamps -> JSON-safe
    return {"n_batches": n_batches, "versions": versions, "runs": rows, "champion_run": champ}


def diagnosis_to_records(diag: Diagnosis, top: int | None = None) -> list[dict]:
    cols = ["batch", "score", "influence_total", "label_issue_frac", "oor_rate", "drift_max_psi",
            "z_influence", "z_label_quality", "z_validation", "explanation"]
    t = diag.table[cols].head(top) if top else diag.table[cols]
    return json.loads(t.to_json(orient="records"))


def fault_log() -> pd.DataFrame:
    with open_store(read_only=True) as s:
        return s.fault_log()

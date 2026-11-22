"""FastAPI service exposing provenance, diagnosis, counterfactual repair and gated promotion.

Run with ``regressiontrace serve`` and open http://127.0.0.1:8000/docs for interactive docs.
Set ``REGRESSIONTRACE_READ_ONLY=1`` for public demos: mutating endpoints then return 403
(diagnosis and checks still work).
"""

from __future__ import annotations

import os
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import __version__
from . import pipeline as P

READ_ONLY = os.environ.get("REGRESSIONTRACE_READ_ONLY") == "1"
_cache: dict = {}  # read-only deployments: results are immutable, so cache them
_lock = threading.Lock()  # one heavy job at a time: keeps CPU/RAM bounded and DuckDB single-writer

app = FastAPI(
    title="RegressionTrace",
    version=__version__,
    description="Trace model regressions to suspicious training-data batches, test repairs "
                "counterfactually, and promote only models that pass safeguards.",
)


class RepairRequest(BaseModel):
    run_id: str
    action: Literal["drop", "relabel", "reweight"] = "drop"
    batch_ids: list[int] | None = Field(None, description="defaults to auto-selected suspects")


class RunRequest(BaseModel):
    run_id: str


def _guard_write():
    if READ_ONLY:
        raise HTTPException(403, "this deployment is read-only")


def _cached(key, fn, *args, **kw):
    if READ_ONLY and key in _cache:
        return _cache[key]
    out = _wrap(fn, *args, **kw)
    if READ_ONLY and len(_cache) < 256:
        _cache[key] = out
    return out


def _wrap(fn, *args, **kw):
    try:
        with _lock:
            return fn(*args, **kw)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}") from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "read_only": READ_ONLY,
            "store_ready": P.PATHS.store.exists()}


@app.get("/status")
def status():
    return _wrap(P.status)


@app.get("/batches")
def list_batches():
    def f():
        with P.open_store(read_only=True) as s:
            df = s.list_batches()
        df["ingested_at"] = df["ingested_at"].astype(str)
        return df.to_dict("records")
    return _wrap(f)


@app.get("/batches/{batch_id}")
def get_batch(batch_id: int):
    def f():
        with P.open_store(read_only=True) as s:
            b = s.get_batch(batch_id)
            b["hash_verified"] = s.verify_batch(batch_id)
        return b
    return _wrap(f)


@app.get("/versions")
def list_versions():
    def f():
        with P.open_store(read_only=True) as s:
            df = s.list_versions()
        df["created_at"] = df["created_at"].astype(str)
        return df.to_dict("records")
    return _wrap(f)


@app.get("/versions/{version_id}")
def get_version(version_id: str):
    def f():
        with P.open_store(read_only=True) as s:
            v = s.get_version(version_id)
        v["created_at"] = str(v["created_at"])
        return v
    return _wrap(f)


@app.get("/runs/{run_id}/check")
def check(run_id: str):
    """Regression check: evaluate the promotion gate for a run against the champion."""
    return _cached(("check", run_id), P.check, run_id)


@app.post("/diagnose")
def diagnose(req: RunRequest, top: int = 10):
    """Rank the run's training batches by suspicion (influence + label quality + validation)."""
    def f():
        from .repair import select_suspects
        diag, vid = P.diagnose_run(req.run_id, log=not READ_ONLY)
        return {"run_id": req.run_id, "dataset_version": vid, "timings_s": diag.timings,
                "suspects": select_suspects(diag), "batches": P.diagnosis_to_records(diag, top)}
    return _cached(("diagnose", req.run_id, top), f)


@app.post("/repair")
def repair(req: RepairRequest):
    """Build a counterfactual dataset version, retrain, and evaluate the promotion gate."""
    _guard_write()
    return _wrap(P.repair_run, req.run_id, req.action, req.batch_ids)


@app.post("/promote")
def promote(req: RunRequest):
    """Promote a run to champion only if every gate check passes."""
    _guard_write()
    return _wrap(P.promote, req.run_id)

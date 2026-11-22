"""End-to-end: demo init -> check -> diagnose -> repair -> promote, through the HTTP API."""

import pytest
from fastapi.testclient import TestClient

from regressiontrace import pipeline as P


@pytest.fixture(scope="module")
def demo():
    return P.init_demo(synthetic=True, force=True)


@pytest.fixture(scope="module")
def client(demo):
    from regressiontrace.api import app
    return TestClient(app)


def test_health_and_status(client, demo):
    assert client.get("/health").json()["store_ready"]
    st = client.get("/status").json()
    assert st["n_batches"] == 100 and st["champion_run"] == demo["champion_run"]


def test_provenance_endpoints(client, demo):
    bad = demo["injected_faults"][0]["batch_id"]
    b = client.get(f"/batches/{bad}").json()
    assert b["hash_verified"] and demo["v2"] in b["versions"]
    assert client.get(f"/versions/{demo['v2']}").json()["parent_version"] == demo["v1"]
    assert client.get("/batches/9999").status_code == 404


def test_diagnose_finds_injected_faults(client, demo):
    r = client.post("/diagnose", json={"run_id": demo["candidate_run"]}).json()
    truth = {f["batch_id"] for f in demo["injected_faults"]}
    top = {row["batch"] for row in r["batches"][: len(truth)]}
    assert len(top & truth) >= len(truth) - 1
    assert r["batches"][0]["explanation"] != "no strong signal"


def test_repair_then_promote(client, demo):
    r = client.post("/repair", json={"run_id": demo["candidate_run"], "action": "drop"}).json()
    assert r["suspects"] and r["new_version"] != demo["v2"]
    assert {c["name"] for c in r["gate"]["checks"]} >= {"auc_non_inferior", "logloss", "slice_auc"}
    p = client.post("/promote", json={"run_id": r["new_run"]}).json()
    assert p["promoted"] == r["gate"]["passed"]


def test_read_only_mode_blocks_writes(client, demo, monkeypatch):
    from regressiontrace import api
    monkeypatch.setattr(api, "READ_ONLY", True)
    assert client.post("/promote", json={"run_id": demo["candidate_run"]}).status_code == 403

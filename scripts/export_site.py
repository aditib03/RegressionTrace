"""Export real demo + benchmark outputs to docs/data/*.json for the GitHub Pages site.

Runs the full demo story in an isolated temporary BATCHSLEUTH_HOME (your local
store and MLflow runs are untouched), then copies the benchmark summary.

    python scripts/export_site.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
home = Path(tempfile.mkdtemp(prefix="batchsleuth-site-"))
(home / "data").mkdir()
if (REPO / "data" / "sk").exists():  # reuse the downloaded dataset
    os.symlink(REPO / "data" / "sk", home / "data" / "sk")
os.environ["BATCHSLEUTH_HOME"] = str(home)
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
sys.path.insert(0, str(REPO))

from batchsleuth import pipeline as P
from batchsleuth.repair import select_suspects


def gate(res: dict) -> dict:
    return {"passed": res["passed"], "checks": res["checks"],
            "candidate": res["candidate"], "champion": res["champion"]}


def main():
    out_dir = REPO / "docs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    init = P.init_demo(force=True)
    cand = init["candidate_run"]
    refresh_gate = P.check(cand)
    diag, _ = P.diagnose_run(cand)
    relabel = P.repair_run(cand, "relabel")
    drop = P.repair_run(cand, "drop")
    promoted = P.promote(drop["new_run"])
    with P.open_store(read_only=True) as s:
        batches = s.list_batches()
        bad = int(init["injected_faults"][0]["batch_id"])
        prov = s.get_batch(bad)
        prov["hash_verified"] = s.verify_batch(bad)
        versions = json.loads(s.list_versions().to_json(orient="records", date_format="iso"))

    demo = {
        "versions": versions,
        "n_batches": len(batches),
        "rows_per_batch": int(batches.row_count.iloc[0]),
        "injected_faults": init["injected_faults"],
        "refresh_gate": gate(refresh_gate),
        "diagnosis": {"timings_s": diag.timings, "suspects": select_suspects(diag),
                      "batches": P.diagnosis_to_records(diag, 12)},
        "repairs": {
            a: {"suspects": r["suspects"], "removed_frac": r["removed_frac"], "changed_rows": r["changed_rows"],
                "new_version": r["new_version"], "gate": gate(r["gate"])}
            for a, r in (("relabel", relabel), ("drop", drop))
        },
        "promoted": promoted["promoted"],
        "provenance_example": {k: prov[k] for k in ("batch_id", "source", "row_count", "content_sha256",
                                                     "versions", "hash_verified")},
    }
    (out_dir / "demo.json").write_text(json.dumps(demo, indent=1, default=str))

    res = REPO / "benchmark" / "results"
    if (res / "summary.json").exists():
        shutil.copy(res / "summary.json", out_dir / "benchmark.json")
    else:
        print("warning: benchmark/results/summary.json missing; run `make bench` first")
    shutil.rmtree(home, ignore_errors=True)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()

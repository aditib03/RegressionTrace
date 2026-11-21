"""Command line interface: ``batchsleuth <command>``."""

from __future__ import annotations

import argparse
import sys


def _print_table(rows: list[dict], cols: list[str]):
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols} if rows else {}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def cmd_init(a):
    from . import pipeline as P
    out = P.init_demo(seed=a.seed, synthetic=a.synthetic, force=a.force)
    print(f"dataset versions: v1={out['v1']} (champion)  v2={out['v2']} (refresh candidate)")
    print(f"champion run : {out['champion_run']}")
    print(f"candidate run: {out['candidate_run']}")
    print("\nRegression check for the refresh candidate:")
    print(P.check(out["candidate_run"])["summary"])
    print("\nNext: batchsleuth diagnose", out["candidate_run"])


def cmd_status(a):
    from . import pipeline as P
    st = P.status()
    print(f"{st['n_batches']} batches ingested; champion run = {st['champion_run']}\n")
    _print_table(st["runs"], ["run_id", "name", "dataset_version", "auc", "logloss", "champion"])


def cmd_train(a):
    from . import pipeline as P
    print(P.train_version(a.version))


def cmd_check(a):
    from . import pipeline as P
    print(P.check(a.run_id)["summary"])


def cmd_diagnose(a):
    from . import pipeline as P
    diag, vid = P.diagnose_run(a.run_id)
    print(f"Diagnosis of run {a.run_id} (dataset {vid}) in {diag.timings['total_s']:.1f}s\n")
    _print_table(P.diagnosis_to_records(diag, a.top),
                 ["batch", "score", "z_influence", "z_label_quality", "z_validation", "explanation"])
    from .repair import select_suspects
    print(f"\nSuspects above threshold: {select_suspects(diag)}")


def cmd_repair(a):
    from . import pipeline as P
    sus = [int(x) for x in a.batches.split(",")] if a.batches else None
    out = P.repair_run(a.run_id, a.action, sus)
    if not out.get("suspects"):
        print(out["message"])
        return
    print(f"{a.action} suspects {out['suspects']} -> version {out['new_version']}, run {out['new_run']}")
    print(out["gate"]["summary"])
    if out["gate"]["passed"]:
        print(f"\nNext: batchsleuth promote {out['new_run']}")


def cmd_promote(a):
    from . import pipeline as P
    out = P.promote(a.run_id)
    print(out["summary"])
    print("promoted to champion" if out["promoted"] else "not promoted")


def cmd_faults(a):
    from . import pipeline as P
    print("Ground-truth injected faults (hidden from diagnosis):")
    print(P.fault_log().to_string(index=False))


def cmd_serve(a):
    import uvicorn
    uvicorn.run("batchsleuth.api:app", host=a.host, port=a.port)


def cmd_bench(a):
    from benchmark import run as B
    B.main(["--quick"] if a.quick else [])


def main(argv=None):
    p = argparse.ArgumentParser(prog="batchsleuth", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="build the demo store, champion and refresh candidate")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--synthetic", action="store_true", help="use offline synthetic data")
    s.add_argument("--force", action="store_true", help="rebuild an existing store")
    s.set_defaults(f=cmd_init)
    sub.add_parser("status", help="list runs and the champion").set_defaults(f=cmd_status)
    s = sub.add_parser("train", help="train a dataset version")
    s.add_argument("version")
    s.set_defaults(f=cmd_train)
    s = sub.add_parser("check", help="regression check / promotion gate vs champion")
    s.add_argument("run_id")
    s.set_defaults(f=cmd_check)
    s = sub.add_parser("diagnose", help="rank training batches by suspicion")
    s.add_argument("run_id")
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(f=cmd_diagnose)
    s = sub.add_parser("repair", help="counterfactual retrain with repaired suspects")
    s.add_argument("run_id")
    s.add_argument("--action", choices=["drop", "relabel", "reweight"], default="drop")
    s.add_argument("--batches", help="comma-separated batch ids (default: auto-selected suspects)")
    s.set_defaults(f=cmd_repair)
    s = sub.add_parser("promote", help="promote a run if it passes the gate")
    s.add_argument("run_id")
    s.set_defaults(f=cmd_promote)
    sub.add_parser("faults", help="show injected ground-truth faults").set_defaults(f=cmd_faults)
    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_serve)
    s = sub.add_parser("bench", help="run the fault-injection benchmark")
    s.add_argument("--quick", action="store_true")
    s.set_defaults(f=cmd_bench)
    a = p.parse_args(argv)
    try:
        a.f(a)
    except (FileExistsError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

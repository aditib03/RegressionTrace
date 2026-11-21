"""Reproducible fault-injection benchmark.

For every scenario (fault type x severity x #corrupted batches x seed):
  1. corrupt K of 100 training batches, train the candidate model;
  2. rank batches with every method (combined + each signal + baselines);
  3. repair (drop / relabel / reweight auto-selected suspects; oracle drop of the
     true bad batches), retrain, and measure recovery of the regression;
  4. run the promotion gate for the corrupted and repaired candidates.
On a subset, exact leave-one-batch-out (LOBO) retraining is run to measure how
well the fast influence approximation tracks the real counterfactual and what it
costs. Clean "control" candidates measure the gate's false-block rate.

Usage:  python -m benchmark.run [--quick]   (results -> benchmark/results/)
Seeds 100-101 were used for tuning and are never used here.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import log_loss

from batchsleuth import faults as F
from batchsleuth import gate as G
from batchsleuth import model as M
from batchsleuth import repair as R
from batchsleuth.config import N_JOBS
from batchsleuth.data import SLICE_COLUMN, load_batched
from batchsleuth.diagnose import diagnose

OUT = Path(__file__).parent / "results"
METHODS = {
    "combined (BatchSleuth)": "score",
    "influence only": "influence_total",
    "label quality only": "label_issue_frac",
    "validation (out-of-range) only": "oor_rate",
    "PSI drift only": "drift_max_psi",
    "loss-only baseline": "train_loss",
}
MIN_GAP = 0.001  # minimum logloss regression for recovery to be meaningful


def precision_at_k(ranking, bad, k):
    return len(set(ranking[:k]) & set(bad)) / k


def average_precision(ranking, bad):
    bad, hits, s = set(bad), 0, 0.0
    for i, b in enumerate(ranking, 1):
        if b in bad:
            hits += 1
            s += hits / i
    return s / len(bad)


def ll(model, X, y):
    return log_loss(y, np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7))


def lobo(c, model_full, Xe, ye):
    """Exact leave-one-batch-out: change in mean eval log-loss when each batch is removed.

    Note: for GBDTs, removing one 2% batch perturbs tree structure; on clean batches
    this effect is mostly noise (see README), so this is a noisy reference.
    """
    t = time.perf_counter()
    base = ll(model_full, Xe, ye)
    deltas = np.zeros(c.n_batches)
    for b in range(c.n_batches):
        keep = c.batch != b
        m = M.train(c.X[keep], c.y[keep])
        deltas[b] = ll(m, Xe, ye) - base
    return deltas, time.perf_counter() - t


def run(quick: bool):
    seeds = [0] if quick else [0, 1, 2]
    sevs = ["med", "high"] if quick else ["low", "med", "high"]
    ks = [3] if quick else [1, 3, 5]
    lobo_budget = 1 if quick else 10
    scen, lobo_rows, controls = [], [], []
    t_start = time.time()

    for seed in seeds:
        d = load_batched(seed=seed)
        rng = np.random.default_rng(seed)
        Xd, yd = d.X_hold.iloc[:5000], d.y_hold[:5000]          # diagnosis slice
        Xg, yg = d.X_hold.iloc[5000:], d.y_hold[5000:]          # gate / evaluation slice
        slices = Xg[SLICE_COLUMN].to_numpy()
        ev = rng.choice(5000, 2000, replace=False)
        Xe, ye = Xd.iloc[ev], yd[ev]

        t = time.perf_counter()
        clean = M.train(d.X, d.y, seed=seed)
        train_s = time.perf_counter() - t
        p_clean = clean.predict_proba(Xg)[:, 1]
        ll_clean = ll(clean, Xg, yg)

        # Controls: clean candidates (5 random clean batches removed, different seed) vs clean champion.
        for j in range(5 if not quick else 2):
            drop = rng.choice(d.n_batches, 5, replace=False)
            keep = ~np.isin(d.batch, drop)
            m = M.train(d.X[keep], d.y[keep], seed=seed + 10 + j)
            g = G.evaluate_gate(yg, m.predict_proba(Xg)[:, 1], p_clean, slices, removed_frac=0.05)
            controls.append({"seed": seed, "passed": g.passed})

        for fault in F.FAULTS:
            for sev in sevs:
                for K in ks:
                    bad = sorted(rng.choice(d.n_batches, K, replace=False).tolist())
                    c, _ = F.inject(d, fault, bad, sev, seed=seed * 1000 + K)
                    m_c = M.train(c.X, c.y, seed=seed)
                    p_c = m_c.predict_proba(Xg)[:, 1]
                    ll_c = ll(m_c, Xg, yg)
                    diag = diagnose(m_c, c.X, c.y, c.batch, Xe, ye, X_ref=Xd, seed=seed)
                    tab = diag.table
                    row = {"seed": seed, "fault": fault, "severity": sev, "k": K,
                           "ll_clean": ll_clean, "ll_corrupt": ll_c, "gap": ll_c - ll_clean,
                           "diagnose_s": diag.timings["total_s"], "train_s": train_s,
                           **{f"t_{k}": v for k, v in diag.timings.items()}}
                    for name, col in METHODS.items():
                        ranking = tab.sort_values(col, ascending=False)["batch"].tolist()
                        row[f"p@k|{name}"] = precision_at_k(ranking, bad, K)
                        row[f"ap|{name}"] = average_precision(ranking, bad)
                    rnd = rng.permutation(d.n_batches).tolist()
                    row["p@k|random"] = precision_at_k(rnd, bad, K)
                    row["ap|random"] = average_precision(rnd, bad)

                    g_c = G.evaluate_gate(yg, p_c, p_clean, slices)
                    row["gate_blocks_corrupt"] = not g_c.passed

                    suspects = R.select_suspects(diag)
                    row["n_suspects"] = len(suspects)
                    row["suspect_precision"] = (len(set(suspects) & set(bad)) / len(suspects)) if suspects else np.nan
                    row["suspect_recall"] = len(set(suspects) & set(bad)) / K
                    for action in R.ACTIONS:
                        if not suspects:
                            row[f"ll|{action}"] = ll_c
                            row[f"gate_pass|{action}"] = False
                            continue
                        plan = R.plan_repair(diag, c.y, c.batch, suspects, action)
                        k_ = plan.keep_mask
                        m_r = M.train(c.X[k_], plan.labels[k_], weight=plan.weights[k_], seed=seed)
                        row[f"ll|{action}"] = ll(m_r, Xg, yg)
                        g_r = G.evaluate_gate(yg, m_r.predict_proba(Xg)[:, 1], p_clean, slices,
                                              removed_frac=plan.removed_frac)
                        row[f"gate_pass|{action}"] = g_r.passed
                    keep = ~np.isin(c.batch, bad)
                    row["ll|oracle drop"] = ll(M.train(c.X[keep], c.y[keep], seed=seed), Xg, yg)
                    scen.append(row)

                    if lobo_budget > 0 and K == 3 and sev in ("med", "high") and seed == 0:
                        lobo_budget -= 1
                        deltas, lobo_s = lobo(c, m_c, Xg, yg)  # large gate set: less eval noise
                        pred = -tab.sort_values("batch")["influence_total"].to_numpy() / len(ye)
                        lobo_rank = np.argsort(deltas).tolist()  # most loss reduction first
                        clean_b = ~np.isin(np.arange(c.n_batches), bad)
                        lobo_rows.append({
                            "fault": fault, "severity": sev, "spearman": spearmanr(pred, deltas)[0],
                            "pearson": pearsonr(pred, deltas)[0], "lobo_s": lobo_s,
                            "lobo_bad_mean": deltas[~clean_b].mean(), "lobo_clean_sd": deltas[clean_b].std(),
                            "infl_bad_mean": pred[~clean_b].mean(), "infl_clean_sd": pred[clean_b].std(),
                            "sign_agree_bad": float(np.mean(np.sign(deltas[~clean_b]) == np.sign(pred[~clean_b]))),
                            "influence_s": diag.timings["influence_s"], "diagnose_s": diag.timings["total_s"],
                            "p@k|exact LOBO": precision_at_k(lobo_rank, bad, K),
                            "p@k|influence": row["p@k|influence only"],
                            "p@k|combined": row["p@k|combined (BatchSleuth)"],
                            "pred": pred.tolist(), "actual": deltas.tolist(), "bad": bad,
                        })
                    print(f"[{time.time() - t_start:6.0f}s] seed={seed} {fault:15s} {sev:4s} k={K} "
                          f"gap={row['gap']:+.4f} P@k={row['p@k|combined (BatchSleuth)']:.2f}", flush=True)

    return pd.DataFrame(scen), pd.DataFrame(lobo_rows), pd.DataFrame(controls), time.time() - t_start


def summarize(scen: pd.DataFrame, lob: pd.DataFrame, ctrl: pd.DataFrame, wall: float, quick: bool) -> dict:
    methods = list(METHODS) + ["random"]
    loc = {m: {"p@k": scen[f"p@k|{m}"].mean(), "ap": scen[f"ap|{m}"].mean()} for m in methods}
    loc_by_fault = scen.groupby("fault")[[f"p@k|{m}" for m in methods]].mean()
    loc_by_fault.columns = methods
    reg = scen[scen.gap >= MIN_GAP]
    rec = {}
    for a in list(R.ACTIONS) + ["oracle drop"]:
        r = (reg.ll_corrupt - reg[f"ll|{a}"]) / reg.gap
        rec[a] = {"median_recovery": float(r.median()) if len(r) else np.nan,
                  "mean_recovery": float(r.clip(-1, 1.5).mean()) if len(r) else np.nan}
    big = scen[scen.gap >= 0.004]
    out = {
        "quick": quick, "n_scenarios": len(scen), "n_regressed": len(reg), "wall_clock_s": wall,
        "n_jobs": N_JOBS, "machine": f"{platform.machine()} / {platform.system()}",
        "localization": loc, "localization_by_fault": loc_by_fault.round(3).to_dict(orient="index"),
        "recovery": rec,
        "suspects": {"precision": float(scen.suspect_precision.mean()), "recall": float(scen.suspect_recall.mean()),
                     "mean_n": float(scen.n_suspects.mean())},
        "gate": {
            "block_rate_corrupt_all": float(scen.gate_blocks_corrupt.mean()),
            "block_rate_corrupt_gap>=0.004": float(big.gate_blocks_corrupt.mean()) if len(big) else np.nan,
            "n_gap>=0.004": len(big),
            "pass_rate_repaired_drop": float(reg["gate_pass|drop"].mean()) if len(reg) else np.nan,
            "false_block_rate_clean_controls": float(1 - ctrl.passed.mean()),
            "n_controls": len(ctrl),
        },
        "overhead": {
            "mean_diagnose_s": float(scen.diagnose_s.mean()), "mean_train_s": float(scen.train_s.mean()),
            "mean_influence_s": float(scen.t_influence_s.mean()),
        },
    }
    if len(lob):
        out["lobo"] = {
            "n": len(lob), "mean_spearman": float(lob.spearman.mean()), "mean_pearson": float(lob.pearson.mean()),
            "mean_lobo_s": float(lob.lobo_s.mean()), "mean_influence_s": float(lob.influence_s.mean()),
            "speedup_influence_vs_lobo": float(lob.lobo_s.mean() / lob.influence_s.mean()),
            "speedup_full_diagnosis_vs_lobo": float(lob.lobo_s.mean() / lob.diagnose_s.mean()),
            "p@k_exact_lobo": float(lob["p@k|exact LOBO"].mean()),
            "p@k_influence": float(lob["p@k|influence"].mean()),
            "p@k_combined": float(lob["p@k|combined"].mean()),
            "lobo_bad_mean": float(lob.lobo_bad_mean.mean()), "lobo_clean_sd": float(lob.lobo_clean_sd.mean()),
            "lobo_snr": float((-lob.lobo_bad_mean / lob.lobo_clean_sd).mean()),
            "infl_snr": float((-lob.infl_bad_mean / lob.infl_clean_sd).mean()),
            "sign_agree_bad": float(lob.sign_agree_bad.mean()),
        }
    return out


def plots(scen: pd.DataFrame, lob: pd.DataFrame, summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    gray, accent = "#9aa0a6", "#1a73e8"

    # 1. precision@k by method
    loc = summary["localization"]
    names = sorted(loc, key=lambda m: loc[m]["p@k"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = [accent if n.startswith("combined") else gray for n in names]
    ax.barh(names, [loc[n]["p@k"] for n in names], color=colors)
    for i, n in enumerate(names):
        ax.text(loc[n]["p@k"] + 0.01, i, f"{loc[n]['p@k']:.2f}", va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("precision@k (k = # corrupted batches), mean over scenarios")
    ax.set_title("Root-cause localization: which batches are corrupted?", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "localization.png")
    plt.close(fig)

    # 2. influence vs exact LOBO
    if len(lob):
        r = lob.iloc[0]
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        pred, act = np.array(r.pred), np.array(r.actual)
        isbad = np.isin(np.arange(len(pred)), r.bad)
        ax.scatter(pred[~isbad], act[~isbad], s=12, color=gray, label="clean batch")
        ax.scatter(pred[isbad], act[isbad], s=30, color=accent, label="corrupted batch")
        ax.axhline(0, color="#ddd", lw=0.8)
        ax.axvline(0, color="#ddd", lw=0.8)
        ax.set_xlabel("predicted Δ mean eval log-loss if removed\n(influence)")
        ax.set_ylabel("actual Δ (exact retrain)")
        ax.set_title(f"{r.fault} ({r.severity}): Spearman ρ={r.spearman:.2f}", loc="left")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "influence_vs_lobo.png")
        plt.close(fig)

    # 3. recovery by repair action
    rec = summary["recovery"]
    acts = list(rec)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    vals = [rec[a]["median_recovery"] for a in acts]
    ax.bar(acts, vals, color=[gray if a == "oracle drop" else accent for a in acts])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax.set_ylabel("median recovery of\nlog-loss regression")
    ax.set_ylim(min(0, min(vals)) - 0.05, max(1.1, max(vals) + 0.15))
    ax.set_title("Counterfactual repair (auto-selected suspects)", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "recovery.png")
    plt.close(fig)


def write_markdown(s: dict, path: Path):
    L = []
    L.append(f"# Benchmark results{' (quick mode)' if s['quick'] else ''}\n")
    L.append(f"{s['n_scenarios']} fault-injection scenarios on UCI Covertype (100 batches x 2,000 rows), "
             f"{s['n_regressed']} of which produced a measurable regression (log-loss gap >= {MIN_GAP}). "
             f"Wall clock {s['wall_clock_s'] / 60:.1f} min on {s['machine']} with {s['n_jobs']} LightGBM threads. "
             "Reproduce with `make bench`.\n")
    L.append("## Root-cause localization\n\n| method | precision@k | average precision |\n|---|---|---|")
    for m, v in sorted(s["localization"].items(), key=lambda kv: -kv[1]["p@k"]):
        L.append(f"| {m} | {v['p@k']:.3f} | {v['ap']:.3f} |")
    L.append("\n### precision@k by fault type\n")
    methods = list(next(iter(s["localization_by_fault"].values())).keys())
    L.append("| fault | " + " | ".join(methods) + " |\n|---|" + "---|" * len(methods))
    for f, row in s["localization_by_fault"].items():
        L.append(f"| {f} | " + " | ".join(f"{row[m]:.2f}" for m in methods) + " |")
    L.append("\n## Counterfactual repair (regressed scenarios)\n")
    L.append(f"Auto-selected suspects: precision {s['suspects']['precision']:.2f}, recall "
             f"{s['suspects']['recall']:.2f}, {s['suspects']['mean_n']:.1f} batches flagged on average.\n")
    L.append("| repair | median recovery | mean recovery (clipped) |\n|---|---|---|")
    for a, v in s["recovery"].items():
        L.append(f"| {a} | {v['median_recovery']:.0%} | {v['mean_recovery']:.0%} |")
    g = s["gate"]
    L.append("\n## Promotion gate\n")
    L.append(f"- Blocks {g['block_rate_corrupt_all']:.0%} of all corrupted candidates and "
             f"{g['block_rate_corrupt_gap>=0.004']:.0%} of those with log-loss regression >= 0.004 "
             f"(n={g['n_gap>=0.004']}).")
    L.append(f"- Passes {g['pass_rate_repaired_drop']:.0%} of drop-repaired candidates.")
    L.append(f"- False-block rate on clean control candidates: {g['false_block_rate_clean_controls']:.0%} "
             f"(n={g['n_controls']}).")
    o = s["overhead"]
    L.append("\n## Overhead\n")
    L.append(f"- Full diagnosis (influence + 3-fold label quality + validation): {o['mean_diagnose_s']:.1f}s; "
             f"influence alone {o['mean_influence_s']:.2f}s; one full training run {o['mean_train_s']:.2f}s.")
    if "lobo" in s:
        lb = s["lobo"]
        L.append(f"- Exact leave-one-batch-out (100 retrains): {lb['mean_lobo_s']:.0f}s -> influence is "
                 f"{lb['speedup_influence_vs_lobo']:.0f}x faster, full diagnosis {lb['speedup_full_diagnosis_vs_lobo']:.0f}x.")
        L.append("\n## Influence approximation vs exact leave-one-batch-out\n")
        L.append(f"Over {lb['n']} scenarios: Spearman {lb['mean_spearman']:.2f}, Pearson {lb['mean_pearson']:.2f} "
                 f"between predicted and actual change in evaluation loss. precision@k: exact LOBO "
                 f"{lb['p@k_exact_lobo']:.2f}, influence {lb['p@k_influence']:.2f}, combined {lb['p@k_combined']:.2f}.\n")
        L.append(f"Why the low correlation: removing one corrupted batch changes mean eval log-loss by only "
                 f"{lb['lobo_bad_mean']:+.5f}, versus a spread of {lb['lobo_clean_sd']:.5f} across clean-batch removals "
                 f"(signal-to-noise {lb['lobo_snr']:.1f}); single GBDT retrains are dominated by tree-structure "
                 f"changes. The influence score separates corrupted batches with signal-to-noise {lb['infl_snr']:.1f}, "
                 f"and agrees with LOBO on the sign of the effect for {lb['sign_agree_bad']:.0%} of corrupted batches. "
                 "The group counterfactual (dropping all suspects, see recovery) is the more reliable check.")
    L.append("\n![localization](figures/localization.png)\n![recovery](figures/recovery.png)\n"
             "![influence vs lobo](figures/influence_vs_lobo.png)\n")
    path.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="~2 min smoke run")
    a = ap.parse_args(argv)
    scen, lob, ctrl, wall = run(a.quick)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "quick_" if a.quick else ""
    scen.to_csv(OUT / f"{tag}scenarios.csv", index=False)
    lob.drop(columns=["pred", "actual"], errors="ignore").to_csv(OUT / f"{tag}lobo.csv", index=False)
    s = summarize(scen, lob, ctrl, wall, a.quick)
    (OUT / f"{tag}summary.json").write_text(json.dumps(s, indent=2, default=float))
    plots(scen, lob, s)
    write_markdown(s, OUT / f"{tag}results.md")
    print((OUT / f"{tag}results.md").read_text())


if __name__ == "__main__":
    main()

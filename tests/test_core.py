import numpy as np
import pytest

from regressiontrace import diagnose as D
from regressiontrace import faults as F
from regressiontrace import gate as G
from regressiontrace import influence as I
from regressiontrace import labelq as LQ
from regressiontrace import model as M
from regressiontrace import repair as R
from regressiontrace.provenance import Store


def test_influence_recomputes_lightgbm_leaf_values(small, eval_set):
    m = M.train(small.X, small.y)
    _, err = I.row_influence(m, small.X, small.y, *eval_set, return_check=True)
    assert err < 1e-6, "gradient/hessian bookkeeping must reproduce the model's own leaf values"


def test_influence_flags_mislabeled_rows(small, eval_set):
    c, _ = F.inject(small, "label_flip", [3], 0.5, seed=1)
    m = M.train(c.X, c.y)
    inf = I.row_influence(m, c.X, c.y, *eval_set)
    flipped = c.y != small.y
    assert inf[flipped].mean() > inf[~flipped].mean()


def test_confident_learning_finds_flipped_labels():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.05, 2000)
    y_noisy = y.copy()
    y_noisy[:100] = 1 - y_noisy[:100]
    issue, _ = LQ.label_issues(y_noisy, np.clip(p, 0, 1))
    recall, precision = issue[:100].mean(), issue[:100].sum() / max(issue.sum(), 1)
    assert recall > 0.6 and precision > 0.9


@pytest.mark.parametrize("fault", list(F.FAULTS))
def test_injectors_only_touch_target_batches(small, fault):
    c, desc = F.inject(small, fault, [2, 5], "high", seed=0)
    other = ~np.isin(small.batch, [2, 5])
    assert (c.y[other] == small.y[other]).all()
    assert np.allclose(c.X[other].to_numpy(), small.X[other].to_numpy())
    changed = (c.y != small.y).any() or not np.allclose(c.X.to_numpy(), small.X.to_numpy())
    assert changed and desc


def test_diagnosis_ranks_corrupted_batches_first(small, eval_set):
    bad = [4, 11]
    c, _ = F.inject(small, "label_flip", bad, "high", seed=2)
    m = M.train(c.X, c.y)
    diag = D.diagnose(m, c.X, c.y, c.batch, *eval_set, X_ref=small.X_hold)
    assert set(diag.top(2)) == set(bad)
    assert set(R.select_suspects(diag)) >= set(bad)


def test_diagnosis_maps_noncontiguous_batch_ids(small, eval_set):
    keep = small.batch >= 10
    X, y, b = small.X[keep], small.y[keep], small.batch[keep] + 100
    m = M.train(X, y)
    diag = D.diagnose(m, X, y, b, *eval_set)
    assert set(diag.table.batch) == set(np.unique(b))


def test_repair_actions(small, eval_set):
    c, _ = F.inject(small, "label_flip", [1], "high", seed=0)
    m = M.train(c.X, c.y)
    diag = D.diagnose(m, c.X, c.y, c.batch, *eval_set)
    drop = R.plan_repair(diag, c.y, c.batch, [1], "drop")
    assert drop.removed_frac == pytest.approx(1 / 20)
    rel = R.plan_repair(diag, c.y, c.batch, [1], "relabel")
    assert rel.changed_rows > 0 and (rel.labels[c.batch != 1] == c.y[c.batch != 1]).all()
    rw = R.plan_repair(diag, c.y, c.batch, [1], "reweight")
    assert (rw.weights[c.batch == 1] < 1).any() and (rw.weights[c.batch != 1] == 1).all()
    with pytest.raises(ValueError):
        R.plan_repair(diag, c.y, c.batch, [1], "delete-everything")


def test_gate_blocks_worse_model_and_passes_identical():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    good = np.clip(np.where(y == 1, 0.75, 0.25) + rng.normal(0, 0.15, 5000), 0.01, 0.99)
    bad = np.clip(good + rng.normal(0, 0.3, 5000), 0.01, 0.99)
    assert G.evaluate_gate(y, good, good).passed
    res = G.evaluate_gate(y, bad, good)
    assert not res.passed and "BLOCKED" in res.summary()
    assert not G.evaluate_gate(y, good, good, removed_frac=0.5).passed


def test_provenance_versions_are_content_addressed(tmp_path, small):
    s = Store(tmp_path / "s.duckdb")
    for b in range(3):
        m = small.batch == b
        s.ingest_batch(b, small.X[m].reset_index(drop=True), small.y[m], source="t")
    s.set_holdout(small.X_hold, small.y_hold)
    v1 = s.create_version([0, 1, 2])
    assert s.create_version([2, 1, 0]) == v1, "same inputs -> same id"
    first_row = int(s.load_version(v1)[4][0])
    v2 = s.create_version([0, 1], parent=v1, label_overrides={first_row: 1}, weights={first_row: 0.5})
    X, y, batch, w, _ = s.load_version(v2)
    assert set(batch) == {0, 1} and y[0] == 1 and w[0] == 0.5 and len(X) == 2 * 600
    assert s.verify_batch(0)
    assert set(s.get_batch(1)["versions"]) == {v1, v2}
    assert s.get_batch(2)["versions"] == [v1]
    s.close()

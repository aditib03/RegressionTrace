"""Batch-level influence estimation for LightGBM binary classifiers.

LightGBM is not differentiable in its parameters, so classic influence
functions (Koh & Liang, 2017) do not apply directly. We use a first-order,
leaf-local approximation in the spirit of BoostIn (Brophy et al., JMLR 2023):

For tree t, a leaf's value is   v = -lr * G / (H + lambda)
where G, H are the sums of gradients / hessians of the training rows routed to
that leaf, evaluated at the model's prediction *before* tree t. Up-weighting a
single training row i by epsilon changes that leaf value by

    dv/de_i = -(lr * g_i + h_i * v) / (H + lambda).

The resulting change in total loss on an evaluation set E is approximated by
summing, over trees, the evaluation gradient mass that falls in the same leaf:

    I(i) = sum_t  S_t[leaf_t(i)] * dv_t/de_i,    S_t[l] = sum_{j in E, leaf_t(j)=l} g_j^(t)

I(i) > 0 means "this row increases evaluation loss" (harmful). Removing row i
changes evaluation loss by roughly -I(i). A batch's score is the sum over its
rows (an additive approximation of leave-one-batch-out).

Effects that propagate through later trees are ignored; that is what makes it
an approximation, and the benchmark quantifies how well it tracks exact
leave-one-batch-out retraining.

Cost: O((n_train + n_eval) * n_trees) with tiny memory (leaf ids as int16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _tree_leaf_values(booster) -> list[np.ndarray]:
    dump = booster.dump_model()
    out = []
    for tree in dump["tree_info"]:
        vals = np.zeros(tree["num_leaves"])

        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                vals[node["leaf_index"]] = node["leaf_value"]
            else:
                stack += [node["left_child"], node["right_child"]]
        out.append(vals)
    return out


def _init_score(booster, y_train: np.ndarray, weight=None) -> float:
    # LightGBM boost_from_average: first tree's leaves include the log-odds of the mean label.
    p = np.average(y_train, weights=weight)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def row_influence(model, X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame,
                  y_eval: np.ndarray, weight: np.ndarray | None = None,
                  return_check: bool = False):
    """Per-training-row influence on total evaluation log-loss (positive = harmful)."""
    booster = model.booster_
    params = booster.params
    lr = float(params.get("learning_rate", 0.1))
    lam = float(params.get("lambda_l2", params.get("reg_lambda", 0.0)))
    w = np.ones(len(y_train)) if weight is None else np.asarray(weight, float)

    leaf_tr = booster.predict(X_train, pred_leaf=True).astype(np.int16)
    leaf_ev = booster.predict(X_eval, pred_leaf=True).astype(np.int16)
    leaf_vals = _tree_leaf_values(booster)
    T = len(leaf_vals)

    init = _init_score(booster, y_train, weight)
    f_tr = np.full(len(y_train), init)
    f_ev = np.full(len(y_eval), init)
    infl = np.zeros(len(y_train))
    max_abs_err = 0.0

    for t in range(T):
        lt, le = leaf_tr[:, t], leaf_ev[:, t]
        n_leaves = len(leaf_vals[t])
        p_tr = _sigmoid(f_tr)
        g = (p_tr - y_train) * w
        h = np.maximum(p_tr * (1 - p_tr), 1e-12) * w
        G = np.bincount(lt, weights=g, minlength=n_leaves)
        H = np.bincount(lt, weights=h, minlength=n_leaves)
        v = -lr * G / (H + lam)  # recomputed leaf values (no init offset)

        # sanity: recomputed values should match the model's (tree 0 contains the init score)
        model_v = leaf_vals[t] - (init if t == 0 else 0.0)
        occupied = H > 0
        if occupied.any():
            max_abs_err = max(max_abs_err, float(np.max(np.abs(model_v[occupied] - v[occupied]))))

        g_ev = _sigmoid(f_ev) - y_eval
        S = np.bincount(le, weights=g_ev, minlength=n_leaves)
        dv = -(lr * g + h * v[lt]) / (H[lt] + lam)
        infl += S[lt] * dv

        # advance staged predictions using the model's actual leaf values
        f_tr = f_tr + model_v[lt]
        f_ev = f_ev + model_v[le]

    if return_check:
        return infl, max_abs_err
    return infl


def batch_influence(row_infl: np.ndarray, batch: np.ndarray) -> pd.DataFrame:
    n_b = int(batch.max()) + 1
    total = np.bincount(batch, weights=row_infl, minlength=n_b)
    count = np.bincount(batch, minlength=n_b)
    harmful = np.bincount(batch, weights=(row_infl > 0).astype(float), minlength=n_b)
    return pd.DataFrame({
        "batch": np.arange(n_b),
        "influence_total": total,
        "influence_mean": total / np.maximum(count, 1),
        "harmful_frac": harmful / np.maximum(count, 1),
    })

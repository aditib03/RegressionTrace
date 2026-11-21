import os
import tempfile

# Isolate all state (DuckDB store, MLflow) in a temp dir before batchsleuth is imported.
os.environ["BATCHSLEUTH_HOME"] = tempfile.mkdtemp(prefix="batchsleuth-test-")
os.environ.setdefault("BATCHSLEUTH_N_JOBS", "2")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pytest

from batchsleuth.data import load_batched


@pytest.fixture(scope="session")
def small():
    return load_batched(n_batches=20, batch_size=600, holdout_size=3000, synthetic=True, seed=0)


@pytest.fixture(scope="session")
def eval_set(small):
    idx = np.random.default_rng(0).choice(len(small.y_hold), 1000, replace=False)
    return small.X_hold.iloc[idx], small.y_hold[idx]

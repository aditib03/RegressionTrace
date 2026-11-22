"""Central configuration.

Defaults are deliberately laptop-friendly: LightGBM uses at most 4 threads
(override with REGRESSIONTRACE_N_JOBS) and all state lives inside the project folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("REGRESSIONTRACE_HOME", Path.cwd())).resolve()


def _default_jobs() -> int:
    env = os.environ.get("REGRESSIONTRACE_N_JOBS")
    if env:
        return max(1, int(env))
    return max(1, min(4, (os.cpu_count() or 2) // 2))


N_JOBS = _default_jobs()


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = field(default_factory=lambda: ROOT / "data")
    store: Path = field(default_factory=lambda: ROOT / "data" / "regressiontrace.duckdb")
    mlflow_db: Path = field(default_factory=lambda: ROOT / "mlflow.db")
    mlflow_artifacts: Path = field(default_factory=lambda: ROOT / "mlartifacts")
    results: Path = field(default_factory=lambda: ROOT / "benchmark" / "results")

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{self.mlflow_db}"


PATHS = Paths()

# Model hyper-parameters (kept modest so a full fit takes a few seconds).
LGB_PARAMS: dict = {
    "n_estimators": 150,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_lambda": 1.0,
    "subsample": 1.0,  # influence math assumes no row subsampling
    "colsample_bytree": 1.0,
    "verbose": -1,
}

REGISTERED_MODEL = "regressiontrace-covtype"
CHAMPION_ALIAS = "champion"

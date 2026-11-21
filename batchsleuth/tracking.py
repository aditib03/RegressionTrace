"""Thin MLflow wrapper: experiment tracking + model registry with a champion alias."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", module="mlflow")

import mlflow
from mlflow import MlflowClient

from .config import CHAMPION_ALIAS, PATHS, REGISTERED_MODEL

EXPERIMENT = "batchsleuth"


def setup():
    mlflow.set_tracking_uri(PATHS.tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        PATHS.mlflow_artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=PATHS.mlflow_artifacts.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def log_model_run(model, *, version_id: str, metrics: dict, params: dict | None = None,
                  tags: dict | None = None, artifacts: dict[str, str] | None = None,
                  run_name: str | None = None) -> str:
    setup()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"dataset_version": version_id, **(tags or {})})
        mlflow.log_params({**model.get_params(), **(params or {})})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for name, text in (artifacts or {}).items():
            mlflow.log_text(text, name)
        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=REGISTERED_MODEL)
        mlflow.set_tag("model_uri", info.model_uri)
        mlflow.set_tag("model_version", str(info.registered_model_version))
        return run.info.run_id


def log_artifact_text(run_id: str, name: str, text: str):
    setup()
    MlflowClient().log_text(run_id, text, name)


def get_run(run_id: str):
    setup()
    return MlflowClient().get_run(run_id)


def load_model(run_id: str):
    return mlflow.lightgbm.load_model(get_run(run_id).data.tags["model_uri"])


def champion_run_id() -> str | None:
    setup()
    try:
        return MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).run_id
    except mlflow.exceptions.MlflowException:
        return None


def set_champion(run_id: str):
    c = MlflowClient()
    version = get_run(run_id).data.tags["model_version"]
    c.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)
    c.set_tag(run_id, "promoted", "true")


def list_runs(max_results: int = 50):
    setup()
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    return MlflowClient().search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"],
                                      max_results=max_results)

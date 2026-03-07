"""
MLflow logging for retrieval evaluation runs.

Best-effort: silently skips if MLflow is unavailable.
"""

import csv
import logging
import tempfile

from config import settings
from evaluation.recall import EvalResult

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "eval_retrieval"


def log_eval_to_mlflow(
    result: EvalResult,
    run_name: str | None = None,
    dataset_params: dict | None = None,
) -> bool:
    """
    Log an EvalResult to MLflow.

    Returns True if logging succeeded, False otherwise.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            # Metrics
            mlflow.log_metric("recall_at_1", result.recall_at(1))
            mlflow.log_metric("recall_at_3", result.recall_at(3))
            mlflow.log_metric("recall_at_5", result.recall_at(5))
            mlflow.log_metric("pairs_total", result.total)
            mlflow.log_metric("pairs_hit_at_5", result.hits_at(5))
            mlflow.log_metric("pairs_with_errors", result.errors)
            mlflow.log_metric("duration_s", result.duration_s)

            # Params
            for key, val in result.weights.items():
                mlflow.log_param(f"weight_{key}", val)
            mlflow.log_param("top_k", result.top_k)
            mlflow.log_param("embedding_model", settings.embedding_model)

            if dataset_params:
                for key, val in dataset_params.items():
                    mlflow.log_param(key, val)

            # Artifact: per-pair CSV
            _log_results_csv(mlflow, result)

        return True
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return False


def _log_results_csv(mlflow, result: EvalResult) -> None:
    """Write per-pair results to a temp CSV and log as artifact."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="eval_results_", delete=False
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query",
                "target",
                "rank",
                "sim_overall",
                "sim_macro_visual",
                "sim_structural",
                "sim_flesh_sensory",
                "sim_microscopic_lab",
                "sim_ecological",
                "sim_taxonomic",
                "sim_numeric",
                "error",
            ]
        )
        for r in result.pair_results:
            writer.writerow(
                [
                    r.query,
                    r.target,
                    r.rank,
                    r.sim_overall,
                    r.sim_macro_visual,
                    r.sim_structural,
                    r.sim_flesh_sensory,
                    r.sim_microscopic_lab,
                    r.sim_ecological,
                    r.sim_taxonomic,
                    r.sim_numeric,
                    r.error,
                ]
            )
    mlflow.log_artifact(f.name, "eval_results.csv")

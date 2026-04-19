"""MLflow experiment setup and training run logging for vision heads."""

import mlflow


def setup_experiment(feature_name: str) -> str:
    """Create or get MLflow experiment for a vision head.

    Naming convention: vision/morpho/{feature_name}

    Returns:
        experiment_id
    """
    experiment_name = f"vision/morpho/{feature_name}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)
    return experiment.experiment_id


def log_training_run(
    experiment_id: str,
    params: dict,
    metrics: dict,
    artifacts: dict[str, str] | None = None,
) -> str:
    """Log a complete training run to MLflow.

    Args:
        experiment_id: MLflow experiment ID.
        params: Hyperparameters dict.
        metrics: Final metrics dict.
        artifacts: Optional {name: file_path} of artifacts to log.

    Returns:
        run_id
    """
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
        return run.info.run_id

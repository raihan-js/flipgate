"""MLflow integration for FlipGate eval runs."""

from pathlib import Path
from typing import Any


def log_run_to_mlflow(
    run_id: str,
    model: str,
    engine: str,
    dataset: str,
    accuracy: float,
    flip_rate: float,
    metadata: dict[str, Any] | None = None,
    tracking_uri: str = "./mlruns",
) -> str:
    """Log an evaluation run to MLflow.
    
    Args:
        run_id: FlipGate run ID
        model: Model name/ID
        engine: Inference engine name
        dataset: Dataset name
        accuracy: Overall accuracy
        flip_rate: Right-to-wrong flip rate
        metadata: Additional parameters to log
        tracking_uri: MLflow tracking URI
    
    Returns:
        MLflow run ID
    """
    import mlflow
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("flipgate")
    
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_param("model", model)
        mlflow.log_param("engine", engine)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("flipgate_run_id", run_id)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("flip_rate_right_to_wrong", flip_rate)
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
    
    return run.info.run_id


def compare_runs_mlflow(
    baseline_mlflow_id: str,
    candidate_mlflow_id: str,
    tracking_uri: str = "./mlruns",
) -> dict[str, Any]:
    """Compare two MLflow runs and return metrics diff."""
    import mlflow
    
    mlflow.set_tracking_uri(tracking_uri)
    
    baseline_run = mlflow.get_run(baseline_mlflow_id)
    candidate_run = mlflow.get_run(candidate_mlflow_id)
    
    baseline_metrics = baseline_run.data.metrics
    candidate_metrics = candidate_run.data.metrics
    
    return {
        "baseline_accuracy": baseline_metrics.get("accuracy", 0),
        "candidate_accuracy": candidate_metrics.get("accuracy", 0),
        "accuracy_delta": candidate_metrics.get("accuracy", 0) - baseline_metrics.get("accuracy", 0),
        "baseline_flip_rate": baseline_metrics.get("flip_rate_right_to_wrong", 0),
        "candidate_flip_rate": candidate_metrics.get("flip_rate_right_to_wrong", 0),
    }

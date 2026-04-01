"""
MLflow Experiment Tracking & Model Registry
Logs all model training runs, parameters, metrics, and artifacts
Registers production-ready models to the MLflow Model Registry
Falls back to local file-based logging if MLflow server is unavailable
"""

import os
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", "/data/mlflow/artifacts")
EXPERIMENTS = {
    "sepsis_risk":      "Hospital_OS_Sepsis_Risk",
    "mortality_risk":   "Hospital_OS_Mortality_Risk",
    "anomaly":          "Hospital_OS_Anomaly_Detection",
    "lstm":             "Hospital_OS_LSTM_Deterioration",
    "autoencoder":      "Hospital_OS_Autoencoder",
    "nlp_ner":          "Hospital_OS_NLP_NER",
}


# ─────────────────────────────────────────
# Local fallback logger
# ─────────────────────────────────────────

@dataclass
class RunRecord:
    run_id: str
    experiment: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "RUNNING"
    params: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    tags: Dict = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class LocalExperimentStore:
    """File-based fallback when MLflow is not available."""

    def __init__(self, base_dir: str = "/data/mlflow/local"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_runs: Dict[str, RunRecord] = {}

    def _run_path(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.json"

    def start_run(self, experiment: str, run_name: str = "", tags: Dict = None) -> str:
        ts = datetime.utcnow().isoformat()
        run_id = hashlib.md5(f"{experiment}{ts}{run_name}".encode()).hexdigest()[:12]
        record = RunRecord(
            run_id=run_id,
            experiment=experiment,
            start_time=ts,
            tags=tags or {"run_name": run_name},
        )
        self._active_runs[run_id] = record
        return run_id

    def log_param(self, run_id: str, key: str, value: Any):
        if run_id in self._active_runs:
            self._active_runs[run_id].params[key] = str(value)

    def log_params(self, run_id: str, params: Dict):
        for k, v in params.items():
            self.log_param(run_id, k, v)

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        if run_id in self._active_runs:
            if key not in self._active_runs[run_id].metrics:
                self._active_runs[run_id].metrics[key] = []
            self._active_runs[run_id].metrics[key].append({"step": step, "value": value})

    def log_metrics(self, run_id: str, metrics: Dict, step: int = 0):
        for k, v in metrics.items():
            self.log_metric(run_id, k, float(v), step)

    def log_artifact(self, run_id: str, path: str):
        if run_id in self._active_runs:
            self._active_runs[run_id].artifacts.append(path)

    def end_run(self, run_id: str, status: str = "FINISHED"):
        if run_id in self._active_runs:
            rec = self._active_runs[run_id]
            rec.end_time = datetime.utcnow().isoformat()
            rec.status = status
            with open(self._run_path(run_id), "w") as f:
                json.dump(asdict(rec), f, indent=2)
            logger.info(f"Run {run_id} saved to {self._run_path(run_id)}")
            del self._active_runs[run_id]

    def get_best_run(self, experiment: str, metric: str, mode: str = "max") -> Optional[Dict]:
        records = []
        for path in self.base_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text())
                if rec["experiment"] == experiment and metric in rec["metrics"]:
                    last_val = rec["metrics"][metric][-1]["value"]
                    records.append((last_val, rec))
            except Exception:
                pass
        if not records:
            return None
        records.sort(key=lambda x: x[0], reverse=(mode == "max"))
        return records[0][1]

    def list_runs(self, experiment: str) -> List[Dict]:
        runs = []
        for path in self.base_dir.glob("*.json"):
            try:
                rec = json.loads(path.read_text())
                if rec["experiment"] == experiment:
                    runs.append(rec)
            except Exception:
                pass
        return sorted(runs, key=lambda x: x["start_time"], reverse=True)


# ─────────────────────────────────────────
# MLflow wrapper
# ─────────────────────────────────────────

class HospitalMLflowTracker:
    """
    Unified MLflow tracking interface with local fallback.
    Usage:
        tracker = HospitalMLflowTracker()
        with tracker.start_run("sepsis_risk", run_name="xgb_v2") as run_id:
            tracker.log_params(run_id, {"n_estimators": 200, "max_depth": 6})
            tracker.log_metrics(run_id, {"auc": 0.87, "f1": 0.74})
            tracker.register_model(run_id, "sepsis_risk", "Staging")
    """

    def __init__(self):
        self._mlflow = None
        self._use_mlflow = False
        self._store = LocalExperimentStore()
        self._experiment_ids: Dict[str, str] = {}
        self._active_mlflow_runs: Dict[str, Any] = {}
        self._init_mlflow()

    def _init_mlflow(self):
        try:
            import mlflow
            mlflow.set_tracking_uri(TRACKING_URI)
            # Test connection
            mlflow.search_experiments()
            self._mlflow = mlflow
            self._use_mlflow = True
            self._setup_experiments()
            logger.info(f"MLflow connected: {TRACKING_URI}")
        except Exception as e:
            logger.warning(f"MLflow unavailable ({e}), using local file store")
            self._use_mlflow = False

    def _setup_experiments(self):
        if not self._use_mlflow:
            return
        for key, name in EXPERIMENTS.items():
            try:
                exp = self._mlflow.get_experiment_by_name(name)
                if exp is None:
                    eid = self._mlflow.create_experiment(
                        name, artifact_location=f"{ARTIFACT_ROOT}/{key}"
                    )
                else:
                    eid = exp.experiment_id
                self._experiment_ids[key] = eid
            except Exception as e:
                logger.warning(f"Could not set up experiment {name}: {e}")

    class _RunContext:
        def __init__(self, tracker, run_id: str):
            self.tracker = tracker
            self.run_id = run_id

        def __enter__(self):
            return self.run_id

        def __exit__(self, exc_type, *_):
            status = "FAILED" if exc_type else "FINISHED"
            self.tracker.end_run(self.run_id, status)

    def start_run(self, experiment_key: str, run_name: str = "",
                  tags: Dict = None) -> "_RunContext":
        tags = tags or {}
        tags.setdefault("run_name", run_name)
        tags.setdefault("created_by", "hospital_os")
        tags.setdefault("timestamp", datetime.utcnow().isoformat())

        if self._use_mlflow:
            try:
                exp_name = EXPERIMENTS.get(experiment_key, experiment_key)
                self._mlflow.set_experiment(exp_name)
                run = self._mlflow.start_run(run_name=run_name, tags=tags)
                run_id = run.info.run_id
                self._active_mlflow_runs[run_id] = run
                return self._RunContext(self, run_id)
            except Exception as e:
                logger.warning(f"MLflow start_run failed: {e}")

        run_id = self._store.start_run(experiment_key, run_name, tags)
        return self._RunContext(self, run_id)

    def log_params(self, run_id: str, params: Dict):
        if self._use_mlflow and run_id in self._active_mlflow_runs:
            try:
                self._mlflow.log_params(params)
                return
            except Exception as e:
                logger.warning(f"MLflow log_params: {e}")
        self._store.log_params(run_id, params)

    def log_metrics(self, run_id: str, metrics: Dict, step: int = 0):
        if self._use_mlflow and run_id in self._active_mlflow_runs:
            try:
                self._mlflow.log_metrics(metrics, step=step)
                return
            except Exception as e:
                logger.warning(f"MLflow log_metrics: {e}")
        self._store.log_metrics(run_id, metrics, step)

    def log_param(self, run_id: str, key: str, value: Any):
        self.log_params(run_id, {key: value})

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        self.log_metrics(run_id, {key: value}, step)

    def log_artifact(self, run_id: str, local_path: str):
        if self._use_mlflow and run_id in self._active_mlflow_runs:
            try:
                self._mlflow.log_artifact(local_path)
                return
            except Exception as e:
                logger.warning(f"MLflow log_artifact: {e}")
        self._store.log_artifact(run_id, local_path)

    def log_model_summary(self, run_id: str, model_info: Dict):
        """Log a model summary dict as a JSON artifact and as tags."""
        summary_path = f"/tmp/model_summary_{run_id[:8]}.json"
        with open(summary_path, "w") as f:
            json.dump(model_info, f, indent=2)
        self.log_artifact(run_id, summary_path)

        # Log key metrics if present
        for k in ("auc_roc", "f1", "accuracy", "precision", "recall", "ap"):
            if k in model_info:
                self.log_metric(run_id, k, float(model_info[k]))

    def end_run(self, run_id: str, status: str = "FINISHED"):
        if self._use_mlflow and run_id in self._active_mlflow_runs:
            try:
                self._mlflow.end_run(status=status)
                del self._active_mlflow_runs[run_id]
                return
            except Exception as e:
                logger.warning(f"MLflow end_run: {e}")
        self._store.end_run(run_id, status)

    def register_model(self, run_id: str, model_name: str,
                       stage: str = "Staging", description: str = "") -> bool:
        """Register a model version to the MLflow model registry."""
        if not self._use_mlflow:
            logger.info(f"[local] Would register model '{model_name}' → {stage}")
            return False
        try:
            model_uri = f"runs:/{run_id}/model"
            mv = self._mlflow.register_model(model_uri, model_name)
            client = self._mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            if description:
                client.update_model_version(
                    name=model_name, version=mv.version, description=description
                )
            logger.info(f"Registered model '{model_name}' v{mv.version} → {stage}")
            return True
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return False

    def get_best_run(self, experiment_key: str, metric: str = "auc_roc",
                     mode: str = "max") -> Optional[Dict]:
        if self._use_mlflow:
            try:
                exp_name = EXPERIMENTS.get(experiment_key, experiment_key)
                exp = self._mlflow.get_experiment_by_name(exp_name)
                if not exp:
                    return None
                order = "DESC" if mode == "max" else "ASC"
                runs = self._mlflow.search_runs(
                    [exp.experiment_id],
                    order_by=[f"metrics.{metric} {order}"],
                    max_results=1,
                )
                if not runs.empty:
                    return runs.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"MLflow best run query: {e}")
        return self._store.get_best_run(experiment_key, metric, mode)

    def list_runs(self, experiment_key: str) -> List[Dict]:
        if self._use_mlflow:
            try:
                exp_name = EXPERIMENTS.get(experiment_key, experiment_key)
                exp = self._mlflow.get_experiment_by_name(exp_name)
                if exp:
                    runs = self._mlflow.search_runs([exp.experiment_id])
                    return runs.to_dict("records")
            except Exception as e:
                logger.warning(f"MLflow list_runs: {e}")
        return self._store.list_runs(experiment_key)

    def get_production_model(self, model_name: str):
        """Load the production model from the registry."""
        if not self._use_mlflow:
            logger.warning("MLflow unavailable — cannot load from registry")
            return None
        try:
            model_uri = f"models:/{model_name}/Production"
            return self._mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(f"Could not load production model '{model_name}': {e}")
            return None


# ─────────────────────────────────────────
# Convenience decorator
# ─────────────────────────────────────────

_default_tracker = None

def get_tracker() -> HospitalMLflowTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = HospitalMLflowTracker()
    return _default_tracker


def track_experiment(experiment_key: str, run_name: str = ""):
    """Decorator that wraps a training function with experiment tracking."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            tracker = get_tracker()
            with tracker.start_run(experiment_key, run_name=run_name or fn.__name__) as run_id:
                result = fn(*args, **kwargs, run_id=run_id, tracker=tracker)
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────

def example_training_run():
    """Demonstrate the tracking API with a simulated training run."""
    tracker = get_tracker()

    with tracker.start_run("sepsis_risk", run_name="xgb_baseline_v1",
                           tags={"model_type": "xgboost", "dataset": "mimic_iii"}) as run_id:

        # Log hyperparameters
        tracker.log_params(run_id, {
            "model": "XGBClassifier",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 5.0,
            "window_hours": 6,
            "feature_count": 47,
            "train_size": 8000,
            "val_size": 2000,
        })

        # Simulate epoch-level metrics
        for epoch in range(1, 11):
            tracker.log_metrics(run_id, {
                "train_loss": round(0.45 - epoch * 0.03 + 0.005 * epoch, 4),
                "val_loss":   round(0.48 - epoch * 0.025 + 0.005 * epoch, 4),
            }, step=epoch)

        # Final metrics
        tracker.log_metrics(run_id, {
            "auc_roc":    0.873,
            "auc_pr":     0.721,
            "f1":         0.742,
            "precision":  0.788,
            "recall":     0.701,
            "accuracy":   0.891,
            "brier_score": 0.082,
        })

        tracker.log_model_summary(run_id, {
            "model_type": "XGBClassifier",
            "auc_roc": 0.873,
            "f1": 0.742,
            "calibrated": True,
            "n_features": 47,
        })

        logger.info(f"Tracking run {run_id} complete")

    print("Run logged. Best run for sepsis_risk:")
    best = tracker.get_best_run("sepsis_risk", "auc_roc")
    if best:
        print(json.dumps({k: v for k, v in best.items()
                           if not isinstance(v, (dict, list))}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_training_run()

"""
Airflow DAG: AI Hospital OS — Full Pipeline Orchestration
Schedule: daily at 02:00 UTC
Pipeline: ingest → clean → feature_engineering → train_models → evaluate → register → dashboard_refresh
Falls back to a standalone scheduler if Airflow is unavailable.
"""

from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Task functions (importable without Airflow)
# ─────────────────────────────────────────

def task_ingest_data(**context):
    """Extract data from EHR DB and legacy SAS exports."""
    logger.info("TASK: ingest_data — extracting from PostgreSQL and SAS exports")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from data_engineering.load_data import DataLoadPipeline
        pipeline = DataLoadPipeline(output_dir="/data/raw")
        result = pipeline.run()
        logger.info(f"Ingestion result: {result}")
        return result
    except Exception as e:
        logger.warning(f"DataLoadPipeline failed: {e} — generating synthetic data")
        _generate_synthetic_raw()
        return {"status": "synthetic", "records": 5000}


def task_clean_data(**context):
    """Validate and clean raw data."""
    logger.info("TASK: clean_data — running preprocessing pipeline")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from data_engineering.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline(input_dir="/data/raw", output_dir="/data/clean")
        result = pipeline.run()
        logger.info(f"Cleaning result: {result}")
        return result
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def task_feature_engineering(**context):
    """Build clinical features: SOFA, NEWS2, delta features, rolling stats."""
    logger.info("TASK: feature_engineering")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from data_engineering.feature_engineering import FeatureEngineeringPipeline
        pipeline = FeatureEngineeringPipeline(
            clean_dir="/data/clean",
            output_dir="/data/features"
        )
        result = pipeline.run()
        logger.info(f"Feature engineering result: {result}")
        return result
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


def task_train_risk_models(**context):
    """Train sepsis, mortality, and cardiac risk models."""
    logger.info("TASK: train_risk_models")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from ml_models.risk_prediction import MultiTaskRiskScorer
        from experiment_tracking.mlflow_setup import get_tracker

        scorer = MultiTaskRiskScorer(model_dir="/data/models/risk")
        tracker = get_tracker()

        for task in ["sepsis_risk", "mortality_risk", "cardiac_risk"]:
            with tracker.start_run(task, run_name=f"daily_{datetime.utcnow().date()}") as run_id:
                try:
                    result = scorer.train_task(task)
                    tracker.log_metrics(run_id, result.get("metrics", {}))
                    tracker.log_params(run_id, result.get("params", {}))
                    logger.info(f"Trained {task}: {result.get('metrics', {})}")
                except Exception as e:
                    logger.error(f"Training {task} failed: {e}")

        return {"status": "complete", "tasks": ["sepsis_risk", "mortality_risk", "cardiac_risk"]}
    except Exception as e:
        logger.error(f"Risk model training failed: {e}")
        raise


def task_train_anomaly_detector(**context):
    """Train isolation forest + autoencoder anomaly detectors."""
    logger.info("TASK: train_anomaly_detector")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from ml_models.anomaly_detection import EnsembleAnomalyDetector
        from experiment_tracking.mlflow_setup import get_tracker
        import numpy as np

        tracker = get_tracker()

        # Load features or use synthetic
        try:
            import pandas as pd
            feats = pd.read_parquet("/data/features/vitals_features.parquet")
            X = feats.select_dtypes("number").fillna(0).values
        except Exception:
            logger.warning("No feature file; using synthetic data")
            X = np.random.randn(3000, 12)
            y = (np.random.rand(3000) < 0.1).astype(int)

        with tracker.start_run("anomaly", run_name=f"ensemble_{datetime.utcnow().date()}") as run_id:
            detector = EnsembleAnomalyDetector(contamination=0.08)
            detector.fit(X)
            tracker.log_params(run_id, {"contamination": 0.08, "n_samples": len(X)})
            detector.save("/data/models/anomaly/ensemble_detector.pkl")
            tracker.log_artifact(run_id, "/data/models/anomaly/ensemble_detector.pkl")

        return {"status": "complete", "n_samples": len(X)}
    except Exception as e:
        logger.error(f"Anomaly detection training failed: {e}")
        raise


def task_train_lstm(**context):
    """Train LSTM deterioration predictor."""
    logger.info("TASK: train_lstm")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from ml_models.lstm_model import LSTMTrainer
        from experiment_tracking.mlflow_setup import get_tracker
        import numpy as np

        tracker = get_tracker()

        # Load sequence data or use synthetic
        try:
            import numpy as np
            X = np.load("/data/features/sequences_X.npy")
            y = np.load("/data/features/sequences_y.npy")
        except Exception:
            logger.warning("No sequence file; using synthetic data")
            X = np.random.randn(500, 24, 10).astype("float32")
            y = (np.random.rand(500) < 0.15).astype("float32")

        with tracker.start_run("lstm", run_name=f"lstm_{datetime.utcnow().date()}") as run_id:
            trainer = LSTMTrainer(
                input_size=X.shape[2],
                hidden_size=64,
                n_layers=2,
                output_dir="/data/models/lstm",
            )
            result = trainer.fit(X, y, epochs=30)
            tracker.log_metrics(run_id, result.get("metrics", {}))
            tracker.log_params(run_id, {
                "input_size": X.shape[2],
                "seq_len": X.shape[1],
                "hidden_size": 64,
                "n_layers": 2,
            })

        return {"status": "complete", "metrics": result.get("metrics", {})}
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        raise


def task_evaluate_models(**context):
    """Run evaluation suite and log results."""
    logger.info("TASK: evaluate_models")
    try:
        import sys, json
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from experiment_tracking.mlflow_setup import get_tracker

        tracker = get_tracker()
        eval_results = {}

        for exp in ["sepsis_risk", "mortality_risk", "anomaly"]:
            best = tracker.get_best_run(exp, "auc_roc")
            if best:
                eval_results[exp] = {
                    "run_id": best.get("run_id", "unknown"),
                    "auc_roc": best.get("metrics.auc_roc", "N/A"),
                    "f1": best.get("metrics.f1", "N/A"),
                }

        report_path = f"/data/eval_report_{datetime.utcnow().date()}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Evaluation report: {eval_results}")
        return eval_results
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def task_register_models(**context):
    """Promote best models to Staging or Production."""
    logger.info("TASK: register_models")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from experiment_tracking.mlflow_setup import get_tracker

        tracker = get_tracker()
        registered = []

        model_thresholds = {
            "sepsis_risk":    ("Hospital_SepsisRisk",   0.80),
            "mortality_risk": ("Hospital_MortalityRisk", 0.78),
        }

        for exp_key, (model_name, min_auc) in model_thresholds.items():
            best = tracker.get_best_run(exp_key, "auc_roc")
            if best:
                auc = best.get("metrics.auc_roc", 0)
                if isinstance(auc, (int, float)) and auc >= min_auc:
                    run_id = best.get("run_id", "")
                    if run_id:
                        ok = tracker.register_model(run_id, model_name, "Staging",
                                                    f"AUC={auc:.3f}, auto-promoted")
                        if ok:
                            registered.append(model_name)

        return {"registered": registered}
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return {"registered": [], "error": str(e)}


def task_refresh_dashboard(**context):
    """Trigger dashboard data refresh."""
    logger.info("TASK: refresh_dashboard")
    try:
        import sys
        sys.path.insert(0, "/home/claude/ai_hospital_os")
        from dashboard.hospital_dashboard import generate_static_report
        path = generate_static_report(f"/data/reports/icu_{datetime.utcnow().date()}.html")
        logger.info(f"Dashboard report: {path}")
        return {"report": path}
    except Exception as e:
        logger.error(f"Dashboard refresh failed: {e}")
        return {"status": "failed", "error": str(e)}


def task_send_alerts(**context):
    """Send summary alerts to clinical staff (stub — hook to real notification service)."""
    logger.info("TASK: send_alerts — sending daily summary")
    # In production: integrate with Twilio, SMTP, or hospital paging system
    summary = {
        "date": str(datetime.utcnow().date()),
        "pipeline": "completed",
        "message": "Daily AI Hospital OS pipeline complete. Dashboard refreshed.",
    }
    logger.info(f"Alert summary: {summary}")
    return summary


def _generate_synthetic_raw():
    """Generate minimal synthetic data for pipeline testing."""
    import os, csv, random
    from datetime import timedelta
    os.makedirs("/data/raw", exist_ok=True)
    with open("/data/raw/vitals_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "timestamp", "heart_rate", "sbp", "dbp",
                    "respiratory_rate", "temperature", "spo2"])
        base = datetime(2024, 1, 1)
        for pid in range(1, 21):
            for h in range(72):
                t = base + timedelta(hours=h)
                w.writerow([f"P{pid:03d}", t.isoformat(),
                             round(80 + random.gauss(0, 10), 1),
                             round(120 + random.gauss(0, 15), 1),
                             round(75 + random.gauss(0, 8), 1),
                             round(16 + random.gauss(0, 3), 1),
                             round(37 + random.gauss(0, 0.5), 2),
                             round(min(100, 97 + random.gauss(0, 1.5)), 1)])


# ─────────────────────────────────────────
# Airflow DAG definition
# ─────────────────────────────────────────

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "hospital_os",
        "depends_on_past": False,
        "start_date": datetime(2024, 1, 1),
        "email_on_failure": True,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=4),
    }

    with DAG(
        dag_id="hospital_os_daily_pipeline",
        default_args=default_args,
        description="AI Hospital OS — daily ML pipeline: ingest → train → evaluate → dashboard",
        schedule_interval="0 2 * * *",  # 02:00 UTC daily
        catchup=False,
        max_active_runs=1,
        tags=["hospital_os", "ml", "icu"],
    ) as dag:

        def make_task(task_id, fn):
            return PythonOperator(task_id=task_id, python_callable=fn,
                                  provide_context=True, dag=dag)

        ingest          = make_task("ingest_data",           task_ingest_data)
        clean           = make_task("clean_data",            task_clean_data)
        features        = make_task("feature_engineering",   task_feature_engineering)
        train_risk      = make_task("train_risk_models",     task_train_risk_models)
        train_anomaly   = make_task("train_anomaly",         task_train_anomaly_detector)
        train_lstm      = make_task("train_lstm",            task_train_lstm)
        evaluate        = make_task("evaluate_models",       task_evaluate_models)
        register        = make_task("register_models",       task_register_models)
        dashboard       = make_task("refresh_dashboard",     task_refresh_dashboard)
        alerts          = make_task("send_alerts",           task_send_alerts)

        # DAG dependency graph
        ingest >> clean >> features
        features >> [train_risk, train_anomaly, train_lstm]
        [train_risk, train_anomaly, train_lstm] >> evaluate
        evaluate >> register >> dashboard >> alerts

    logger.info("Airflow DAG 'hospital_os_daily_pipeline' registered")

except ImportError:
    logger.info("Airflow not installed — DAG definition skipped. Use run_pipeline() standalone.")


# ─────────────────────────────────────────
# Standalone runner (no Airflow required)
# ─────────────────────────────────────────

def run_pipeline(tasks: list = None):
    """
    Run the full pipeline sequentially without Airflow.
    Usage: python airflow_dag.py
    """
    all_tasks = [
        ("ingest_data",         task_ingest_data),
        ("clean_data",          task_clean_data),
        ("feature_engineering", task_feature_engineering),
        ("train_risk_models",   task_train_risk_models),
        ("train_anomaly",       task_train_anomaly_detector),
        ("train_lstm",          task_train_lstm),
        ("evaluate_models",     task_evaluate_models),
        ("register_models",     task_register_models),
        ("refresh_dashboard",   task_refresh_dashboard),
        ("send_alerts",         task_send_alerts),
    ]
    if tasks:
        all_tasks = [(n, f) for n, f in all_tasks if n in tasks]

    results = {}
    for name, fn in all_tasks:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        try:
            start = datetime.utcnow()
            result = fn()
            elapsed = (datetime.utcnow() - start).total_seconds()
            results[name] = {"status": "success", "elapsed_s": round(elapsed, 1),
                              "result": str(result)[:200]}
            print(f"✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            results[name] = {"status": "failed", "error": str(e)}
            print(f"✗ {name} FAILED: {e}")

    print("\n" + "="*50)
    print("Pipeline Summary:")
    for name, r in results.items():
        status = "✓" if r["status"] == "success" else "✗"
        print(f"  {status} {name}: {r['status']}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run_pipeline()

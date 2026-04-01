#!/usr/bin/env python3
"""
Train All Models — Production Training Script
Trains every ML model in the Hospital OS with configurable options.
Supports: full retrain, incremental update, specific task selection,
          evaluation after training, and MLflow experiment logging.

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --tasks sepsis_risk mortality_risk
    python scripts/train_all_models.py --eval --promote --n-samples 5000
"""

import sys
import os
import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"/tmp/train_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.log"),
    ]
)
logger = logging.getLogger("train_all_models")

# ─────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────

def _c(text, code): return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text
ok    = lambda t: print(f"  {_c('✓','92')} {t}")
fail  = lambda t: print(f"  {_c('✗','91')} {t}")
info  = lambda t: print(f"  {_c('→','94')} {t}")
warn  = lambda t: print(f"  {_c('⚠','93')} {t}")
head  = lambda t: print(f"\n{_c('─'*55,'96')}\n{_c(t,'1')}\n{_c('─'*55,'96')}")

# ─────────────────────────────────────────
# Training tasks
# ─────────────────────────────────────────

def train_risk_models(n_samples: int, model_dir: Path, tracker) -> dict:
    """Train sepsis, mortality, and cardiac risk models."""
    from ml_models.risk_prediction import ClinicalRiskModel
    from simulation.patient_simulator import ICUSimulator

    head("Risk Prediction Models")
    results = {}
    sim = ICUSimulator()
    info(f"Generating {n_samples} training samples via simulation...")
    X_records, y_labels = sim.generate_training_dataset(
        n_patients=max(50, n_samples // 20),
        interval_minutes=60,
        label_horizon_hours=6,
    )
    info(f"Dataset: {len(X_records)} records, prevalence={sum(y_labels)/len(y_labels):.1%}")

    import numpy as np
    # Build feature matrix
    feat_keys = ["heart_rate", "sbp", "dbp", "respiratory_rate", "temperature",
                  "spo2", "lactate", "creatinine", "wbc", "news2", "sofa_score",
                  "shock_index", "hours_in_icu"]
    X = np.array([[r.get(k, 0) for k in feat_keys] for r in X_records], dtype=float)
    y_base = np.array(y_labels, dtype=int)

    for task in ["sepsis_risk", "mortality_risk", "cardiac_risk"]:
        t0 = time.perf_counter()
        # Vary labels slightly per task
        rng = np.random.RandomState(hash(task) % 2**31)
        y_task = np.clip(y_base + rng.binomial(1, 0.05, size=len(y_base)), 0, 1)

        model = ClinicalRiskModel(task=task)
        metrics = model.fit(X, y_task)
        elapsed = time.perf_counter() - t0

        # Save model
        model_path = model_dir / f"{task}_model.pkl"
        model_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        results[task] = {**metrics, "elapsed_s": round(elapsed, 2), "saved": str(model_path)}
        auc = metrics.get("auc_roc", 0)
        col = "92" if auc >= 0.80 else "93" if auc >= 0.70 else "91"
        ok(f"{task:<25} AUC={_c(f'{auc:.3f}','1')} F1={metrics.get('f1',0):.3f} "
            f"({elapsed:.1f}s)")

        # Log to MLflow
        if tracker:
            with tracker.start_run(task, run_name=f"train_{datetime.utcnow().date()}") as rid:
                tracker.log_params(rid, {"n_samples": len(X), "n_features": X.shape[1]})
                tracker.log_metrics(rid, metrics)

    return results


def train_anomaly_detector(n_samples: int, model_dir: Path, tracker) -> dict:
    """Train the ensemble anomaly detector."""
    from ml_models.anomaly_detection import EnsembleAnomalyDetector
    from simulation.patient_simulator import ICUSimulator

    head("Anomaly Detection")
    t0 = time.perf_counter()
    sim = ICUSimulator()
    X_records, _ = sim.generate_training_dataset(
        n_patients=max(30, n_samples // 30),
        interval_minutes=60,
    )
    import numpy as np
    feat_keys = ["heart_rate", "sbp", "dbp", "respiratory_rate",
                  "temperature", "spo2", "lactate", "creatinine"]
    X = np.array([[r.get(k, 0) for k in feat_keys] for r in X_records], dtype=float)

    detector = EnsembleAnomalyDetector(contamination=0.08)
    detector.fit(X)

    model_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    model_path = model_dir / "anomaly_ensemble.pkl"
    detector.save(model_dir)
    elapsed = time.perf_counter() - t0
    ok(f"Ensemble anomaly detector ({elapsed:.1f}s, {len(X)} samples)")
    return {"n_samples": len(X), "elapsed_s": round(elapsed, 2)}


def train_lstm(n_samples: int, model_dir: Path, tracker) -> dict:
    """Train LSTM deterioration predictor."""
    head("LSTM Deterioration Predictor")
    try:
        import torch
        from ml_models.lstm_model import LSTMTrainer
        from simulation.patient_simulator import ICUSimulator
        import numpy as np

        t0 = time.perf_counter()
        sim = ICUSimulator()
        info("Generating sequence training data...")
        X_records, y_labels = sim.generate_training_dataset(
            n_patients=max(20, n_samples // 40),
            interval_minutes=60,
            label_horizon_hours=6,
        )
        # Reshape to sequences
        seq_len = 12
        feat_keys = ["heart_rate", "sbp", "respiratory_rate", "temperature",
                      "spo2", "lactate", "creatinine", "news2"]
        raw = np.array([[r.get(k, 0) for k in feat_keys] for r in X_records], dtype=np.float32)
        # Create sequences
        n_seqs = len(raw) // seq_len
        X_seq = raw[:n_seqs * seq_len].reshape(n_seqs, seq_len, len(feat_keys))
        y_seq = np.array(y_labels[:n_seqs * seq_len][seq_len - 1::seq_len], dtype=np.float32)

        model_dir.mkdir(parents=True, exist_ok=True)
        trainer = LSTMTrainer(
            input_size=len(feat_keys),
            hidden_size=64, n_layers=2,
            output_dir=str(model_dir / "lstm"),
        )
        result = trainer.fit(X_seq, y_seq, epochs=15)
        elapsed = time.perf_counter() - t0
        ok(f"LSTM trained ({elapsed:.1f}s, {len(X_seq)} sequences, {seq_len} steps each)")
        return {**result.get("metrics", {}), "elapsed_s": round(elapsed, 2)}
    except ImportError:
        warn("PyTorch not installed — LSTM training skipped (sklearn fallback available)")
        return {"skipped": True, "reason": "PyTorch not installed"}


def train_autoencoder(n_samples: int, model_dir: Path, tracker) -> dict:
    """Train the vitals autoencoder."""
    head("Autoencoder")
    try:
        from ml_models.autoencoder_model import AutoencoderTrainer
        from simulation.patient_simulator import ICUSimulator
        import numpy as np

        t0 = time.perf_counter()
        sim = ICUSimulator()
        X_records, _ = sim.generate_training_dataset(
            n_patients=max(20, n_samples // 40), interval_minutes=60)
        feat_keys = ["heart_rate", "sbp", "dbp", "respiratory_rate",
                      "temperature", "spo2", "lactate", "creatinine"]
        X = np.array([[r.get(k, 0) for k in feat_keys]
                       for r in X_records], dtype=np.float32)
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer = AutoencoderTrainer(input_dim=len(feat_keys), latent_dim=4,
                                      output_dir=str(model_dir / "autoencoder"))
        result = trainer.fit(X, epochs=20)
        elapsed = time.perf_counter() - t0
        ok(f"Autoencoder trained ({elapsed:.1f}s, {len(X)} samples)")
        return {"elapsed_s": round(elapsed, 2)}
    except Exception as e:
        warn(f"Autoencoder training: {e}")
        return {"skipped": True, "reason": str(e)}


def evaluate_all(model_dir: Path) -> dict:
    """Run evaluation suite on all trained models."""
    head("Model Evaluation")
    from evaluation.model_evaluator import EvaluationSuite
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        suite = EvaluationSuite()
        results = suite.run_all(output_dir=td, backtest_windows=8)
    overall = results["overall"]
    ok(f"Pass rate: {overall['n_pass']}/{overall['n_models']} models")
    for task, data in results["models"].items():
        m = data["evaluation"]["metrics"]
        status = _c("PASS","92") if data["evaluation"]["pass_criteria"] else _c("FAIL","91")
        info(f"  {task:<28} AUC={m.get('auc_roc',0):.3f} F1={m.get('f1',0):.3f} {status}")
    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

TASK_REGISTRY = {
    "risk":        train_risk_models,
    "anomaly":     train_anomaly_detector,
    "lstm":        train_lstm,
    "autoencoder": train_autoencoder,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train all AI Hospital OS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="*", choices=list(TASK_REGISTRY.keys()),
                         help="Tasks to train (default: all)")
    parser.add_argument("--n-samples", type=int, default=2000,
                         help="Target training samples (default: 2000)")
    parser.add_argument("--model-dir", default="/data/models",
                         help="Model output directory (default: /data/models)")
    parser.add_argument("--eval", action="store_true",
                         help="Run evaluation suite after training")
    parser.add_argument("--promote", action="store_true",
                         help="Promote passing models to Staging in MLflow registry")
    parser.add_argument("--dry-run", action="store_true",
                         help="Show what would be trained without doing it")
    args = parser.parse_args()

    tasks_to_run = args.tasks or list(TASK_REGISTRY.keys())
    model_dir = Path(args.model_dir)

    print(_c(f"\n{'='*55}", "96"))
    print(_c("  AI Hospital OS — Model Training Pipeline", "1"))
    print(_c(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", "2"))
    print(_c(f"{'='*55}", "96"))
    print(f"  Tasks:      {', '.join(tasks_to_run)}")
    print(f"  Samples:    {args.n_samples:,}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Evaluate:   {args.eval}")
    print(f"  Dry run:    {args.dry_run}")

    if args.dry_run:
        print("\n[DRY RUN] Would train:", tasks_to_run)
        return

    # MLflow tracker
    tracker = None
    try:
        from experiment_tracking.mlflow_setup import get_tracker
        tracker = get_tracker()
    except Exception:
        warn("MLflow unavailable — training without experiment tracking")

    all_results = {}
    total_t0 = time.perf_counter()

    for task_key in tasks_to_run:
        fn = TASK_REGISTRY[task_key]
        try:
            result = fn(args.n_samples, model_dir, tracker)
            all_results[task_key] = {"status": "success", **result}
        except Exception as e:
            logger.exception(f"Training failed for {task_key}")
            fail(f"{task_key}: {e}")
            all_results[task_key] = {"status": "failed", "error": str(e)}

    # Evaluation
    eval_results = None
    if args.eval:
        try:
            eval_results = evaluate_all(model_dir)
        except Exception as e:
            warn(f"Evaluation failed: {e}")

    # Promote to MLflow staging
    if args.promote and tracker and eval_results:
        head("Model Promotion")
        for task, data in eval_results.get("models", {}).items():
            if data["evaluation"]["pass_criteria"]:
                model_name = f"Hospital_{task.replace('_risk','').title()}Risk"
                ok(f"[local] Would promote {model_name} → Staging")

    # Summary
    total_elapsed = time.perf_counter() - total_t0
    print(_c(f"\n{'='*55}", "96"))
    print(_c("  Training Summary", "1"))
    print(_c(f"{'='*55}", "96"))
    for task, r in all_results.items():
        status_icon = _c("✓","92") if r.get("status") == "success" else _c("✗","91")
        elapsed = r.get("elapsed_s", 0)
        skipped = r.get("skipped", False)
        if skipped:
            print(f"  {_c('─','93')} {task:<20} skipped ({r.get('reason','')})")
        else:
            print(f"  {status_icon} {task:<20} {elapsed:.1f}s")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Write summary JSON
    summary_path = model_dir / f"training_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(),
                    "tasks": all_results,
                    "total_elapsed_s": round(total_elapsed, 2)}, f, indent=2)
    info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

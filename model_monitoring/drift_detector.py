"""
Model Drift Detection & Continuous Monitoring
Detects when production ML models are drifting from their training distribution.
Implements: PSI (Population Stability Index), KS test, chi-squared test,
            prediction distribution drift, label drift (outcome shift).
Triggers retraining alerts when drift exceeds configurable thresholds.
"""

import math
import logging
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Drift test results
# ─────────────────────────────────────────

@dataclass
class DriftResult:
    feature: str
    test_name: str          # PSI | KS | Chi2 | prediction_drift | label_drift
    statistic: float
    threshold: float
    drift_detected: bool
    severity: str           # none | minor | moderate | severe
    reference_mean: float
    current_mean: float
    pct_change: float

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "test": self.test_name,
            "statistic": round(self.statistic, 4),
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(self.current_mean, 4),
            "pct_change": round(self.pct_change, 2),
        }


@dataclass
class DriftReport:
    model_name: str
    task: str
    check_timestamp: str
    reference_window: str
    current_window: str
    n_reference: int
    n_current: int
    drift_results: List[DriftResult] = field(default_factory=list)
    overall_drift_detected: bool = False
    drift_severity: str = "none"
    retrain_recommended: bool = False
    action: str = ""

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "check_timestamp": self.check_timestamp,
            "reference_window": self.reference_window,
            "current_window": self.current_window,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "overall_drift_detected": self.overall_drift_detected,
            "drift_severity": self.drift_severity,
            "retrain_recommended": self.retrain_recommended,
            "action": self.action,
            "n_drifted_features": sum(1 for r in self.drift_results if r.drift_detected),
            "drift_results": [r.to_dict() for r in self.drift_results],
        }

    def summary(self) -> str:
        n_drift = sum(1 for r in self.drift_results if r.drift_detected)
        return (f"[{self.model_name}/{self.task}] {self.drift_severity.upper()} "
                f"drift: {n_drift}/{len(self.drift_results)} features drifted "
                f"{'→ RETRAIN' if self.retrain_recommended else ''}")


# ─────────────────────────────────────────
# Statistical tests (stdlib-only)
# ─────────────────────────────────────────

def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index. <0.10 stable, 0.10-0.25 minor, >0.25 significant."""
    all_vals = np.concatenate([reference, current])
    lo, hi = np.min(all_vals), np.max(all_vals)
    if hi == lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    ref_pct = np.maximum(ref_counts / len(reference), 1e-6)
    cur_pct = np.maximum(cur_counts / len(current), 1e-6)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and approximate p-value."""
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    n1, n2 = len(ref_sorted), len(cur_sorted)
    all_vals = np.sort(np.concatenate([ref_sorted, cur_sorted]))

    cdf1 = np.searchsorted(ref_sorted, all_vals, side="right") / n1
    cdf2 = np.searchsorted(cur_sorted, all_vals, side="right") / n2
    D = float(np.max(np.abs(cdf1 - cdf2)))

    # Approximate p-value via Kolmogorov distribution
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p_val = 2 * math.exp(-2 * (D * en) ** 2)
    return round(D, 4), round(min(1.0, max(0.0, p_val)), 4)


def wasserstein_distance(ref: np.ndarray, cur: np.ndarray) -> float:
    """Earth Mover's Distance (Wasserstein-1) between two distributions."""
    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)
    n = max(len(ref_sorted), len(cur_sorted))
    # Interpolate to same length
    ref_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ref_sorted)), ref_sorted)
    cur_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(cur_sorted)), cur_sorted)
    return float(np.mean(np.abs(ref_interp - cur_interp)))


def _severity_from_psi(psi_val: float) -> str:
    if psi_val < 0.10:  return "none"
    if psi_val < 0.20:  return "minor"
    if psi_val < 0.25:  return "moderate"
    return "severe"


# ─────────────────────────────────────────
# Feature drift detector
# ─────────────────────────────────────────

class FeatureDriftDetector:
    """
    Detects drift in model input features between a reference and current window.
    """

    PSI_THRESHOLDS = {"minor": 0.10, "moderate": 0.20, "severe": 0.25}

    def __init__(self, feature_names: List[str],
                  psi_threshold: float = 0.20,
                  ks_alpha: float = 0.05):
        self.feature_names = feature_names
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self._reference: Optional[np.ndarray] = None

    def set_reference(self, X: np.ndarray):
        """Set the reference distribution (training/baseline window)."""
        self._reference = np.asarray(X, dtype=float)
        logger.info(f"FeatureDriftDetector: reference set ({len(X)} samples)")

    def check(self, X_current: np.ndarray) -> List[DriftResult]:
        """Check current window against reference."""
        if self._reference is None:
            raise RuntimeError("Call set_reference() first")
        X_cur = np.asarray(X_current, dtype=float)
        results = []
        for i, feat in enumerate(self.feature_names):
            if i >= self._reference.shape[1] or i >= X_cur.shape[1]:
                continue
            ref_col = self._reference[:, i]
            cur_col = X_cur[:, i]

            psi_val = psi(ref_col, cur_col)
            ks_stat, ks_p = ks_statistic(ref_col, cur_col)
            drift_detected = psi_val >= self.psi_threshold or ks_p < self.ks_alpha
            severity = _severity_from_psi(psi_val)

            ref_mean = float(np.mean(ref_col))
            cur_mean = float(np.mean(cur_col))
            pct_change = (cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

            results.append(DriftResult(
                feature=feat,
                test_name="PSI+KS",
                statistic=psi_val,
                threshold=self.psi_threshold,
                drift_detected=drift_detected,
                severity=severity,
                reference_mean=ref_mean,
                current_mean=cur_mean,
                pct_change=pct_change,
            ))
        return results


# ─────────────────────────────────────────
# Prediction drift detector
# ─────────────────────────────────────────

class PredictionDriftDetector:
    """Detects shifts in the model's output distribution."""

    def __init__(self, psi_threshold: float = 0.15,
                  mean_shift_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.mean_shift_threshold = mean_shift_threshold
        self._reference_preds: Optional[np.ndarray] = None

    def set_reference(self, predictions: np.ndarray):
        self._reference_preds = np.asarray(predictions, dtype=float)

    def check(self, predictions: np.ndarray) -> DriftResult:
        if self._reference_preds is None:
            raise RuntimeError("Call set_reference() first")
        cur = np.asarray(predictions, dtype=float)
        psi_val = psi(self._reference_preds, cur, n_bins=10)
        ref_mean = float(np.mean(self._reference_preds))
        cur_mean = float(np.mean(cur))
        mean_shift = abs(cur_mean - ref_mean)
        drift_detected = psi_val >= self.psi_threshold or mean_shift >= self.mean_shift_threshold
        return DriftResult(
            feature="prediction_score",
            test_name="PSI",
            statistic=psi_val,
            threshold=self.psi_threshold,
            drift_detected=drift_detected,
            severity=_severity_from_psi(psi_val),
            reference_mean=ref_mean,
            current_mean=cur_mean,
            pct_change=(cur_mean - ref_mean) / (ref_mean + 1e-9) * 100,
        )


# ─────────────────────────────────────────
# Label drift detector (outcome shift)
# ─────────────────────────────────────────

class LabelDriftDetector:
    """Detects changes in observed outcome prevalence (label shift)."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self._reference_rate: Optional[float] = None

    def set_reference(self, labels: List[int]):
        self._reference_rate = sum(labels) / max(len(labels), 1)

    def check(self, labels: List[int]) -> DriftResult:
        if self._reference_rate is None:
            raise RuntimeError("Call set_reference() first")
        cur_rate = sum(labels) / max(len(labels), 1)
        shift = abs(cur_rate - self._reference_rate)
        drift = shift >= self.threshold
        return DriftResult(
            feature="outcome_label",
            test_name="prevalence_shift",
            statistic=shift,
            threshold=self.threshold,
            drift_detected=drift,
            severity="severe" if shift > 0.10 else "moderate" if shift > 0.05 else "none",
            reference_mean=self._reference_rate,
            current_mean=cur_rate,
            pct_change=(cur_rate - self._reference_rate) / (self._reference_rate + 1e-9) * 100,
        )


# ─────────────────────────────────────────
# Model monitor (orchestrator)
# ─────────────────────────────────────────

class ModelMonitor:
    """
    Orchestrates all drift checks for a production model.
    Generates structured DriftReports and recommends actions.
    """

    RETRAIN_SEVERITIES = {"severe", "moderate"}

    def __init__(self, model_name: str, task: str,
                  feature_names: List[str],
                  psi_threshold: float = 0.20,
                  prediction_psi_threshold: float = 0.15,
                  label_shift_threshold: float = 0.05):
        self.model_name = model_name
        self.task = task
        self.feature_detector   = FeatureDriftDetector(feature_names, psi_threshold)
        self.prediction_detector = PredictionDriftDetector(prediction_psi_threshold)
        self.label_detector      = LabelDriftDetector(label_shift_threshold)
        self._reports: List[DriftReport] = []

    def set_reference(self, X_ref: np.ndarray,
                       predictions_ref: np.ndarray,
                       labels_ref: List[int]):
        """Establish reference distributions from training/validation data."""
        self.feature_detector.set_reference(X_ref)
        self.prediction_detector.set_reference(predictions_ref)
        self.label_detector.set_reference(labels_ref)
        logger.info(f"ModelMonitor [{self.model_name}/{self.task}]: "
                    f"reference set ({len(X_ref)} samples)")

    def check(self, X_current: np.ndarray,
               predictions_current: np.ndarray,
               labels_current: Optional[List[int]] = None,
               window_label: str = "") -> DriftReport:
        """Run all drift checks for the current production window."""
        now = datetime.utcnow()
        feature_drifts = self.feature_detector.check(X_current)
        pred_drift = self.prediction_detector.check(predictions_current)
        all_drifts = feature_drifts + [pred_drift]

        if labels_current is not None:
            label_drift = self.label_detector.check(labels_current)
            all_drifts.append(label_drift)

        # Aggregate severity
        severities = [r.severity for r in all_drifts if r.drift_detected]
        if "severe" in severities:
            overall_severity = "severe"
        elif "moderate" in severities:
            overall_severity = "moderate"
        elif "minor" in severities:
            overall_severity = "minor"
        else:
            overall_severity = "none"

        overall_drift = any(r.drift_detected for r in all_drifts)
        retrain = overall_severity in self.RETRAIN_SEVERITIES

        action = {
            "severe":   "Immediate model retraining required. Stop serving predictions.",
            "moderate": "Model retraining recommended within 48 hours.",
            "minor":    "Monitor closely. Schedule retraining in next cycle.",
            "none":     "No action required.",
        }[overall_severity]

        report = DriftReport(
            model_name=self.model_name,
            task=self.task,
            check_timestamp=now.isoformat(),
            reference_window="training",
            current_window=window_label or now.strftime("%Y-%m-%d"),
            n_reference=len(self.feature_detector._reference),
            n_current=len(X_current),
            drift_results=all_drifts,
            overall_drift_detected=overall_drift,
            drift_severity=overall_severity,
            retrain_recommended=retrain,
            action=action,
        )
        self._reports.append(report)
        if overall_drift:
            logger.warning(f"DRIFT: {report.summary()}")
        return report

    def get_report_history(self) -> List[Dict]:
        return [r.to_dict() for r in self._reports]

    def trend_analysis(self) -> Dict:
        """Analyse drift trend over time."""
        if not self._reports:
            return {}
        severities = [r.drift_severity for r in self._reports]
        n_retrain = sum(1 for r in self._reports if r.retrain_recommended)
        return {
            "n_checks": len(self._reports),
            "n_retrain_recommended": n_retrain,
            "severity_trend": severities[-10:],
            "latest_severity": severities[-1] if severities else "none",
            "deteriorating": (len(severities) >= 3 and
                               severities[-1] != "none" and
                               severities[-2] != "none"),
        }


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.RandomState(42)
    feature_names = ["heart_rate", "sbp", "lactate", "creatinine",
                      "news2", "sofa_score", "wbc", "procalcitonin"]
    n_ref = 500

    # Reference (training) distribution — stable ICU population
    X_ref = rng.randn(n_ref, len(feature_names))
    pred_ref = rng.beta(2, 8, n_ref)
    labels_ref = (pred_ref > 0.5).astype(int).tolist()

    monitor = ModelMonitor("SepsisRiskModel", "sepsis_risk", feature_names)
    monitor.set_reference(X_ref, pred_ref, labels_ref)

    print("=== Model Drift Detection Demo ===\n")

    # Scenario 1: stable window (no drift)
    X_stable = X_ref + rng.randn(200, len(feature_names)) * 0.05
    pred_stable = pred_ref[:200] + rng.randn(200) * 0.02
    report1 = monitor.check(X_stable, pred_stable, labels_ref[:200], "week_1")
    print(f"Week 1 (stable): {report1.summary()}")

    # Scenario 2: moderate drift (population shift)
    X_drift = X_ref[:300] + 1.2 + rng.randn(300, len(feature_names)) * 0.5
    pred_drift = np.clip(pred_ref[:300] + 0.15, 0, 1)
    labels_drift = (rng.rand(300) < 0.25).astype(int).tolist()
    report2 = monitor.check(X_drift, pred_drift, labels_drift, "week_4")
    print(f"Week 4 (drift):  {report2.summary()}")
    if report2.drift_results:
        top = sorted(report2.drift_results, key=lambda r: -r.statistic)[:3]
        for r in top:
            if r.drift_detected:
                print(f"  ⚠ {r.feature:<20} PSI={r.statistic:.3f} "
                      f"mean: {r.reference_mean:.2f}→{r.current_mean:.2f} "
                      f"({r.pct_change:+.0f}%)")

    # Scenario 3: severe drift
    X_severe = X_ref[:150] + 3.0 + rng.randn(150, len(feature_names))
    pred_severe = np.clip(pred_ref[:150] + 0.35, 0, 1)
    report3 = monitor.check(X_severe, pred_severe, window_label="week_8")
    print(f"Week 8 (severe): {report3.summary()}")
    print(f"  → Action: {report3.action}")

    trend = monitor.trend_analysis()
    print(f"\nTrend: {trend['severity_trend']} | Retrain recommended {trend['n_retrain_recommended']}x")

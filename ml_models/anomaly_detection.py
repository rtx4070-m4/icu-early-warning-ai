"""
AI Hospital Operating System
ML Models – Anomaly Detection
================================
Isolation Forest + statistical methods for detecting
physiological anomalies in ICU vital sign streams.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

logger = logging.getLogger("anomaly_detection")

VITAL_FEATURES = [
    "heart_rate", "sbp", "dbp", "map",
    "temperature", "spo2", "resp_rate", "gcs_total",
]
DERIVED_FEATURES = [
    "shock_index", "pulse_pressure", "news2",
    "delta_heart_rate", "delta_sbp", "delta_spo2",
]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Isolation Forest Detector
# ─────────────────────────────────────────────
class IsolationForestDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    Learns 'normal' ICU physiology and flags deviations.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.random_state  = random_state

        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self.fitted = False

    def _get_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        available = [c for c in VITAL_FEATURES + DERIVED_FEATURES if c in df.columns]
        return df[available].fillna(df[available].median()), available

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        X, self.feature_cols = self._get_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        logger.info(
            "IsolationForest fitted on %d samples, %d features",
            len(X), len(self.feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with:
          anomaly_score  : continuous score (negative = more anomalous)
          anomaly_flag   : 1 = anomaly, 0 = normal
          anomaly_prob   : normalised [0,1] anomaly probability
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first")

        df = df.copy()
        X, _ = self._get_features(df)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest returns -1 (anomaly) or 1 (normal)
        raw_pred  = self.model.predict(X_scaled)
        raw_score = self.model.score_samples(X_scaled)  # lower = more anomalous

        df["anomaly_flag"]  = (raw_pred == -1).astype(int)
        df["anomaly_score"] = raw_score

        # Normalize score to [0, 1] probability-like measure
        min_s, max_s = raw_score.min(), raw_score.max()
        if max_s > min_s:
            df["anomaly_prob"] = 1 - (raw_score - min_s) / (max_s - min_s)
        else:
            df["anomaly_prob"] = 0.5

        n_anomalies = df["anomaly_flag"].sum()
        logger.info(
            "Anomaly detection: %d/%d flagged (%.1f%%)",
            n_anomalies, len(df), 100 * n_anomalies / max(len(df), 1),
        )
        return df

    def evaluate(
        self, df: pd.DataFrame, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate against ground-truth labels (1 = anomaly)."""
        df_pred = self.predict(df)
        pred_labels = df_pred["anomaly_flag"].values
        pred_probs  = df_pred["anomaly_prob"].values

        metrics = {
            "roc_auc":   roc_auc_score(true_labels, pred_probs),
            "precision": precision_score(true_labels, pred_labels, zero_division=0),
            "recall":    recall_score(true_labels, pred_labels, zero_division=0),
            "f1":        f1_score(true_labels, pred_labels, zero_division=0),
        }
        cm = confusion_matrix(true_labels, pred_labels)
        metrics["confusion_matrix"] = cm.tolist()

        logger.info(
            "Evaluation – AUC: %.3f | Precision: %.3f | Recall: %.3f | F1: %.3f",
            metrics["roc_auc"], metrics["precision"],
            metrics["recall"], metrics["f1"],
        )
        return metrics

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded: %s", path)
        return obj


# ─────────────────────────────────────────────
# Statistical Anomaly Detector (Z-score / IQR)
# ─────────────────────────────────────────────
class StatisticalAnomalyDetector:
    """
    Rule-based and statistical anomaly detection.
    Combines clinical thresholds with Z-score analysis.
    Useful as a complementary / interpretable detector.
    """

    # Hard clinical alarm thresholds
    CLINICAL_THRESHOLDS = {
        "heart_rate":  {"lo": 40,  "hi": 150},
        "sbp":         {"lo": 70,  "hi": 200},
        "spo2":        {"lo": 88,  "hi": 101},
        "resp_rate":   {"lo": 8,   "hi": 35},
        "map":         {"lo": 55,  "hi": 130},
        "temperature": {"lo": 35,  "hi": 40},
        "gcs_total":   {"lo": 7,   "hi": 16},
    }

    def __init__(self, z_threshold: float = 3.5):
        self.z_threshold = z_threshold
        self.pop_stats: Dict[str, Dict] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "StatisticalAnomalyDetector":
        """Compute population-level statistics for Z-score detection."""
        for col in VITAL_FEATURES:
            if col in df.columns:
                series = df[col].dropna()
                self.pop_stats[col] = {
                    "mean": series.mean(),
                    "std":  series.std(),
                    "q1":   series.quantile(0.25),
                    "q3":   series.quantile(0.75),
                }
        self.fitted = True
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame with per-vital anomaly flags and composite score.
        """
        df = df.copy()
        flag_cols = []

        for col, thresholds in self.CLINICAL_THRESHOLDS.items():
            if col not in df.columns:
                continue
            flag_col = f"alert_{col}"

            # Clinical threshold breach
            clinical_flag = (
                (df[col] < thresholds["lo"]) |
                (df[col] > thresholds["hi"])
            ).fillna(False)

            # Z-score outlier
            if col in self.pop_stats and self.fitted:
                mean = self.pop_stats[col]["mean"]
                std  = self.pop_stats[col]["std"]
                if std > 0:
                    z_score = (df[col] - mean) / std
                    zscore_flag = z_score.abs() > self.z_threshold
                else:
                    zscore_flag = pd.Series(False, index=df.index)
            else:
                zscore_flag = pd.Series(False, index=df.index)

            df[flag_col] = (clinical_flag | zscore_flag).astype(int)
            flag_cols.append(flag_col)

        # Composite severity score
        if flag_cols:
            df["clinical_alert_score"] = df[flag_cols].sum(axis=1)
            df["has_clinical_alert"]   = (df["clinical_alert_score"] > 0).astype(int)

        return df

    def classify_severity(self, alert_score: int) -> str:
        """Translate alert score to severity label."""
        if alert_score == 0:   return "normal"
        elif alert_score == 1: return "mild"
        elif alert_score == 2: return "moderate"
        else:                  return "critical"


# ─────────────────────────────────────────────
# Ensemble Anomaly Detector
# ─────────────────────────────────────────────
class EnsembleAnomalyDetector:
    """
    Combines IsolationForest and StatisticalAnomalyDetector
    for robust, low false-positive detection.
    """

    def __init__(
        self,
        if_weight: float = 0.6,
        stat_weight: float = 0.4,
        contamination: float = 0.05,
    ):
        self.if_weight   = if_weight
        self.stat_weight = stat_weight
        self.if_detector   = IsolationForestDetector(contamination=contamination)
        self.stat_detector = StatisticalAnomalyDetector()

    def fit(self, df: pd.DataFrame) -> "EnsembleAnomalyDetector":
        self.if_detector.fit(df)
        self.stat_detector.fit(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # IsolationForest predictions
        if_df   = self.if_detector.predict(df)
        # Statistical predictions
        stat_df = self.stat_detector.detect(df)

        df = df.copy()
        df["if_anomaly_prob"]  = if_df["anomaly_prob"]
        df["stat_alert_score"] = stat_df.get("clinical_alert_score", 0)

        # Normalise stat score to [0,1]
        max_stat = max(df["stat_alert_score"].max(), 1)
        df["stat_anomaly_prob"] = df["stat_alert_score"] / max_stat

        # Weighted ensemble score
        df["ensemble_score"] = (
            self.if_weight   * df["if_anomaly_prob"] +
            self.stat_weight * df["stat_anomaly_prob"]
        )
        df["ensemble_flag"] = (df["ensemble_score"] > 0.5).astype(int)
        df["severity"] = df["ensemble_score"].apply(
            lambda s: "critical" if s > 0.8
                      else "warning" if s > 0.5
                      else "info"    if s > 0.3
                      else "normal"
        )

        return df

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.if_detector.save(directory / "if_detector.pkl")
        with open(directory / "stat_detector.pkl", "wb") as f:
            pickle.dump(self.stat_detector, f)
        logger.info("Ensemble model saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "EnsembleAnomalyDetector":
        obj = cls()
        obj.if_detector   = IsolationForestDetector.load(directory / "if_detector.pkl")
        with open(directory / "stat_detector.pkl", "rb") as f:
            obj.stat_detector = pickle.load(f)
        return obj


# ─────────────────────────────────────────────
# Real-time stream interface
# ─────────────────────────────────────────────
class RealTimeAnomalyScorer:
    """
    Thin wrapper for scoring single observations in real-time.
    Pre-loads a fitted ensemble model.
    """

    def __init__(self, model: Optional[EnsembleAnomalyDetector] = None):
        if model is None:
            model = EnsembleAnomalyDetector()
        self.model = model

    def score_observation(self, obs: Dict) -> Dict:
        """Score a single vital observation dict."""
        df = pd.DataFrame([obs])
        result = self.model.predict(df)
        row = result.iloc[0]
        return {
            "anomaly_score":  float(row.get("ensemble_score", 0)),
            "is_anomaly":     bool(row.get("ensemble_flag", 0)),
            "ensemble_score": float(row.get("ensemble_score", 0)),
            "ensemble_flag":  int(row.get("ensemble_flag", 0)),
            "severity":       row.get("severity", "normal"),
        }

    def score(self, obs) -> Dict:
        """Score a single observation. Accepts dict, numpy array, or list."""
        if isinstance(obs, np.ndarray):
            df = _ndarray_to_df(obs.reshape(1, -1))
        elif isinstance(obs, dict):
            df = pd.DataFrame([obs])
        else:
            df = _ndarray_to_df(np.asarray(obs).reshape(1, -1))
        result = self.model.predict(df)
        row = result.iloc[0]
        return {
            "anomaly_score": float(row.get("ensemble_score", 0)),
            "is_anomaly":    bool(row.get("ensemble_flag", 0)),
            "severity":      row.get("severity", "normal"),
        }


# ─────────────────────────────────────────────
# Entry point / demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_engineering.load_data import SyntheticDataGenerator
    from data_engineering.feature_engineering import FeatureEngineeringPipeline

    gen   = SyntheticDataGenerator(seed=7)
    vitals = gen.generate_vitals(n_patients=30, hours=48, deterioration_prob=0.3)

    pipe     = FeatureEngineeringPipeline()
    features = pipe.run(vitals)

    # Split train/test
    pids = features["patient_id"].unique()
    train_pids = pids[:20]
    test_pids  = pids[20:]

    train = features[features["patient_id"].isin(train_pids)]
    test  = features[features["patient_id"].isin(test_pids)]

    # Train ensemble
    detector = EnsembleAnomalyDetector(contamination=0.1)
    detector.fit(train)

    # Score test
    results = detector.predict(test)
    print("\n=== Anomaly Detection Results ===")
    print(results[["patient_id","ensemble_score","severity"]].head(20).to_string())
    print(f"\nAlerts: {results['ensemble_flag'].sum()} / {len(results)}")
    print(results["severity"].value_counts())


# ── Numpy-compatible wrappers ────────────────────────────────────────────────

def _ndarray_to_df(X) -> pd.DataFrame:
    """Convert numpy array or list to DataFrame, mapping to known feature names."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    n_cols = arr.shape[1] if arr.ndim == 2 else 1
    cols = (VITAL_FEATURES + DERIVED_FEATURES)[:n_cols]
    if len(cols) < n_cols:
        cols = list(cols) + [f"feat_{i}" for i in range(len(cols), n_cols)]
    return pd.DataFrame(arr, columns=cols)


# ── IsolationForestDetector: numpy support + score_samples ──────────────────

_if_fit_orig = IsolationForestDetector.fit
def _if_fit(self, X, y=None):
    return _if_fit_orig(self, _ndarray_to_df(X))
IsolationForestDetector.fit = _if_fit

_if_predict_orig = IsolationForestDetector.predict
def _if_predict(self, X):
    if isinstance(X, np.ndarray):
        df = _ndarray_to_df(X)
        result = _if_predict_orig(self, df)
        # Tests expect a 1-D binary array, not a DataFrame
        return result["anomaly_flag"].values
    return _if_predict_orig(self, X)
IsolationForestDetector.predict = _if_predict

def _if_score_samples(self, X):
    result = self.predict(_ndarray_to_df(X))
    return result["anomaly_prob"].values
IsolationForestDetector.score_samples = _if_score_samples


# ── StatisticalAnomalyDetector: numpy support + score_samples ───────────────

_stat_fit_orig = StatisticalAnomalyDetector.fit
def _stat_fit(self, X, y=None):
    return _stat_fit_orig(self, _ndarray_to_df(X))
StatisticalAnomalyDetector.fit = _stat_fit

def _stat_score_samples(self, X):
    result = self.detect(_ndarray_to_df(X))
    col = "anomaly_score" if "anomaly_score" in result.columns else result.columns[-1]
    return result[col].values
StatisticalAnomalyDetector.score_samples = _stat_score_samples


# ── EnsembleAnomalyDetector: numpy support + score_samples ──────────────────

_ens_fit_orig = EnsembleAnomalyDetector.fit
def _ens_fit(self, X, y=None):
    return _ens_fit_orig(self, _ndarray_to_df(X))
EnsembleAnomalyDetector.fit = _ens_fit

def _ens_score_samples(self, X):
    result = self.predict(_ndarray_to_df(X))
    return result["ensemble_score"].values
EnsembleAnomalyDetector.score_samples = _ens_score_samples

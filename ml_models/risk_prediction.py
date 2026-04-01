"""
AI Hospital Operating System
ML Models – Gradient Boosting Risk Prediction
===============================================
XGBoost / LightGBM ensemble for:
  - Sepsis risk
  - Cardiac arrest risk
  - ICU mortality risk
  - Abnormal lab prediction

Includes SHAP explanations for clinical interpretability.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    brier_score_loss, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("risk_prediction")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Feature sets per prediction task
# ─────────────────────────────────────────────
SEPSIS_FEATURES = [
    "heart_rate", "sbp", "dbp", "map", "temperature",
    "spo2", "resp_rate", "gcs_total",
    "sofa_total", "qsofa", "news2",
    "shock_index", "pulse_pressure",
    "delta_heart_rate", "delta_sbp",
    "tachycardia_flag", "tachypnea_flag", "fever_flag",
    "Creatinine", "WBC", "Lactate", "Platelets",
    "sirs_count",
]

CARDIAC_FEATURES = [
    "heart_rate", "sbp", "dbp", "map",
    "spo2", "resp_rate", "gcs_total",
    "shock_index", "pulse_pressure",
    "delta_heart_rate", "delta_sbp",
    "Troponin", "BUN", "Potassium",
    "news2", "sofa_cardio",
]

MORTALITY_FEATURES = [
    "heart_rate", "sbp", "spo2", "resp_rate", "gcs_total",
    "temperature", "map",
    "sofa_total", "news2", "qsofa", "apsiii",
    "shock_index",
    "Creatinine", "Lactate", "WBC", "Hemoglobin",
    "hypotension_flag", "severe_hypoxia",
    "age",
]

LAB_ABNORMAL_FEATURES = [
    "heart_rate", "sbp", "temperature", "resp_rate",
    "sofa_total", "news2",
    "Creatinine", "WBC", "Hemoglobin", "Platelets",
    "Sodium", "Potassium", "Glucose",
]


# ─────────────────────────────────────────────
# Base Risk Model
# ─────────────────────────────────────────────
class ClinicalRiskModel:
    """
    Gradient boosting classifier for clinical risk prediction.
    Supports XGBoost, LightGBM, and sklearn GBM.
    """

    def __init__(
        self,
        task: str = "sepsis_risk",
        backend: str = "auto",
        threshold: float = 0.4,
        n_cv_folds: int = 5,
    ):
        self.task      = task
        self.threshold = threshold
        self.n_cv_folds = n_cv_folds
        self.scaler    = StandardScaler()
        self.fitted    = False
        self.feature_importance_: Optional[pd.Series] = None
        self.cv_metrics_: Dict = {}
        self.model     = None

        # Select backend
        if backend == "auto":
            if XGB_AVAILABLE:   self.backend = "xgboost"
            elif LGB_AVAILABLE: self.backend = "lightgbm"
            else:               self.backend = "sklearn"
        else:
            self.backend = backend

        # Feature set per task
        self._feature_map = {
            "sepsis_risk":       SEPSIS_FEATURES,
            "cardiac_risk":      CARDIAC_FEATURES,
            "mortality_risk":    MORTALITY_FEATURES,
            "lab_abnormal_risk": LAB_ABNORMAL_FEATURES,
        }
        self.feature_cols: List[str] = self._feature_map.get(task, SEPSIS_FEATURES)

    def _build_estimator(self, scale_pos_weight: float = 1.0):
        if self.backend == "xgboost" and XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif self.backend == "lightgbm" and LGB_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Select features and fill NaNs with median."""
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            raise ValueError(f"No feature columns found in df for task={self.task}")
        self.used_features_ = available
        X = df[available].copy()
        X.fillna(X.median(), inplace=True)
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def fit(
        self, df, label_col: str = "label", y=None
    ) -> Dict:
        """Accept either (DataFrame, label_col) or (X_array, y_array) sklearn-style."""
        # ── Numpy / sklearn-style call: fit(X, y) ─────────────────────────────
        if isinstance(df, np.ndarray) or (y is not None and not isinstance(y, str)):
            X_arr = np.asarray(df, dtype=float)
            # When y is passed positionally as 2nd arg, label_col holds it
            y_arr = np.asarray(y if y is not None else label_col)
            self.scaler = StandardScaler()
            X_arr_scaled = self.scaler.fit_transform(X_arr)
            n_pos = y_arr.sum()
            n_neg = len(y_arr) - n_pos
            scale_pos = n_neg / max(n_pos, 1)
            estimator = self._build_estimator(scale_pos_weight=scale_pos)
            self.model = CalibratedClassifierCV(estimator, cv=3, method="isotonic")
            self.model.fit(X_arr_scaled, y_arr)
            self.fitted = True
            self.cv_metrics_ = {"cv_auc_mean": 0.0, "cv_auc_std": 0.0}
            self.used_features_ = [f"f{i}" for i in range(X_arr.shape[1])]
            # Compute quick metrics to return
            probs = self.model.predict_proba(X_arr_scaled)[:, 1]
            preds = (probs >= self.threshold).astype(int)
            from sklearn.metrics import roc_auc_score, f1_score
            try:
                auc = float(roc_auc_score(y_arr, probs)) if len(set(y_arr)) > 1 else 0.5
            except Exception:
                auc = 0.5
            f1 = float(f1_score(y_arr, preds, zero_division=0))
            return {"auc_roc": auc, "f1": f1, **self.cv_metrics_}

        # ── DataFrame path (original) ──────────────────────────────────────────
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if label_col not in df.columns:
            df = df.copy()
            df[label_col] = df.iloc[:, -1]
        y = df[label_col].values
        X = self._prepare_X(df, fit=True)

        # Class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        logger.info(
            "Task: %s | samples=%d | pos=%d (%.1f%%) | backend=%s",
            self.task, len(y), int(n_pos), 100 * n_pos / len(y), self.backend,
        )

        estimator = self._build_estimator(scale_pos_weight=scale_pos)

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        cv_auc = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        self.cv_metrics_ = {
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std":  float(cv_auc.std()),
        }
        logger.info(
            "CV AUC: %.3f ± %.3f", cv_auc.mean(), cv_auc.std()
        )

        # Fit with probability calibration
        self.model = CalibratedClassifierCV(estimator, cv=3, method="isotonic")
        self.model.fit(X, y)
        self.fitted = True

        # Feature importance (best effort)
        try:
            base = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base, "feature_importances_"):
                self.feature_importance_ = pd.Series(
                    base.feature_importances_, index=self.used_features_
                ).sort_values(ascending=False)
        except Exception:
            pass

        return self

    def predict_proba(self, df) -> np.ndarray:
        """Return probability of positive class. Accepts DataFrame or numpy array."""
        if isinstance(df, np.ndarray):
            X = self.scaler.transform(df.astype(float))
        else:
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            X = self._prepare_X(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(df)
        return pd.DataFrame({
            f"{self.task}_score": probs,
            f"{self.task}_flag":  (probs >= self.threshold).astype(int),
            f"{self.task}_level": pd.cut(
                probs,
                bins=[-0.01, 0.2, 0.4, 0.7, 1.01],
                labels=["low", "moderate", "high", "critical"],
            ),
        })

    def evaluate(self, df: pd.DataFrame, label_col: str = "label") -> Dict:
        y_true = df[label_col].values
        probs  = self.predict_proba(df)
        preds  = (probs >= self.threshold).astype(int)

        metrics = {
            "roc_auc":      roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else 0.5,
            "avg_precision": average_precision_score(y_true, probs) if len(set(y_true)) > 1 else 0.5,
            "brier_score":  brier_score_loss(y_true, probs),
            "f1":           f1_score(y_true, preds, zero_division=0),
            "report":       classification_report(y_true, preds),
            **self.cv_metrics_,
        }
        logger.info(
            "%s evaluation – AUC: %.3f | AP: %.3f | Brier: %.3f",
            self.task, metrics["roc_auc"],
            metrics["avg_precision"], metrics["brier_score"],
        )
        return metrics

    def explain_shap(
        self, df: pd.DataFrame, n_samples: int = 100
    ) -> Optional[pd.DataFrame]:
        """Generate SHAP feature importance explanations."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed – skipping explanation")
            return None
        X = self._prepare_X(df)[:n_samples]
        try:
            base = self.model.calibrated_classifiers_[0].estimator
            if XGB_AVAILABLE and isinstance(base, xgb.XGBClassifier):
                explainer = shap.TreeExplainer(base)
            else:
                explainer = shap.KernelExplainer(
                    base.predict_proba, X[:50]
                )
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            importance = pd.DataFrame(
                np.abs(shap_values).mean(axis=0)[np.newaxis, :],
                columns=self.used_features_,
            ).T.rename(columns={0: "mean_shap"})
            return importance.sort_values("mean_shap", ascending=False)
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)
            return None

    def top_risk_features(self, obs: Dict, top_n: int = 5) -> List[Dict]:
        """Return top contributing features for a single observation."""
        if self.feature_importance_ is None:
            return []
        top = self.feature_importance_.head(top_n)
        return [
            {
                "feature": feat,
                "importance": float(imp),
                "value": obs.get(feat, "N/A"),
            }
            for feat, imp in top.items()
        ]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "ClinicalRiskModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ─────────────────────────────────────────────
# Multi-Task Risk Scorer
# ─────────────────────────────────────────────
class MultiTaskRiskScorer:
    """
    Runs all four risk models and returns a unified risk report.
    """

    TASKS = ["sepsis_risk", "cardiac_risk", "mortality_risk", "lab_abnormal_risk"]

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.models: Dict[str, ClinicalRiskModel] = {}

    def train_all(
        self,
        df: pd.DataFrame,
        label_cols: Optional[Dict[str, str]] = None,
    ) -> "MultiTaskRiskScorer":
        """
        Train all risk models.
        label_cols: maps task name → label column in df.
        """
        if label_cols is None:
            # Try to derive labels from clinical features
            df = self._derive_synthetic_labels(df)
            label_cols = {t: f"label_{t}" for t in self.TASKS}

        for task in self.TASKS:
            label = label_cols.get(task)
            if not label or label not in df.columns:
                logger.warning("Skipping %s – label '%s' not found", task, label)
                continue
            model = ClinicalRiskModel(task=task)
            model.fit(df.copy(), label_col=label)
            self.models[task] = model
            model.save(self.model_dir / f"{task}.pkl")

        return self

    def _derive_synthetic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create plausible binary labels from clinical features for demo."""
        df = df.copy()

        # Sepsis: SOFA ≥ 2 + suspected infection proxy
        if "sofa_total" in df.columns:
            df["label_sepsis_risk"] = (
                (df["sofa_total"] >= 2) &
                (df.get("fever_flag", 0) == 1)
            ).astype(int)
        else:
            df["label_sepsis_risk"] = (df.get("heart_rate", 80) > 100).astype(int)

        # Cardiac: high shock index or extreme HR
        df["label_cardiac_risk"] = (
            (df.get("shock_index", 0) > 1.0) |
            (df.get("heart_rate", 80) > 140)
        ).astype(int)

        # Mortality: high NEWS2 score
        df["label_mortality_risk"] = (
            df.get("news2", 0) >= 7
        ).astype(int)

        # Lab abnormal: any critical flag proxy
        df["label_lab_abnormal_risk"] = (
            (df.get("Creatinine", 1.0) > 2.0) |
            (df.get("Lactate", 1.0) > 2.5)
        ).astype(int)

        return df

    def score_patient(self, obs: Dict) -> Dict:
        """Score a single patient observation across all tasks."""
        df = pd.DataFrame([obs])
        results = {"patient_id": obs.get("patient_id", "unknown")}

        for task, model in self.models.items():
            try:
                prob = float(model.predict_proba(df)[0])
                results[f"{task}_score"] = round(prob, 4)
                results[f"{task}_flag"]  = int(prob >= model.threshold)
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", task, exc)
                results[f"{task}_score"] = 0.0
                results[f"{task}_flag"]  = 0

        # Composite "deterioration" score (max across tasks)
        risk_scores = [results.get(f"{t}_score", 0) for t in self.TASKS]
        results["composite_risk"] = round(max(risk_scores), 4)
        results["max_risk_task"]  = self.TASKS[int(np.argmax(risk_scores))]
        return results

    def score_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all patients in a dataframe."""
        all_results = []
        for task, model in self.models.items():
            pred = model.predict(df)
            all_results.append(pred)
        if all_results:
            return pd.concat([df[["patient_id"]].reset_index(drop=True)]
                             + all_results, axis=1)
        return df

    def load_all(self) -> "MultiTaskRiskScorer":
        for task in self.TASKS:
            path = self.model_dir / f"{task}.pkl"
            if path.exists():
                self.models[task] = ClinicalRiskModel.load(path)
                logger.info("Loaded model: %s", task)
        return self


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_engineering.load_data import SyntheticDataGenerator
    from data_engineering.feature_engineering import FeatureEngineeringPipeline

    gen    = SyntheticDataGenerator(seed=1)
    vitals = gen.generate_vitals(n_patients=100, hours=24, deterioration_prob=0.25)
    labs   = gen.generate_labs(n_patients=100)

    labs_wide = labs.pivot_table(
        index="patient_id", columns="lab_name", values="value", aggfunc="mean"
    ).reset_index()

    pipe     = FeatureEngineeringPipeline()
    features = pipe.run(vitals, labs_wide)
    features["age"] = 65

    scorer = MultiTaskRiskScorer()
    scorer.train_all(features)

    scored = scorer.score_cohort(features)
    print("\n=== Multi-Task Risk Scores ===")
    print(scored[[c for c in scored.columns if "score" in c or "flag" in c]].head(10))

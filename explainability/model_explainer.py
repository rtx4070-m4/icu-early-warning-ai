"""
Model Explainability Module
Provides human-readable explanations for AI risk predictions.
Implements: feature attribution (SHAP-style permutation), counterfactual reasoning,
natural-language explanation generation, and contribution waterfall charts.
Works without sklearn/SHAP when unavailable — uses built-in permutation importance.
"""

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────

@dataclass
class FeatureContribution:
    feature: str
    value: float            # Actual value for this patient
    contribution: float     # SHAP-style contribution to prediction
    direction: str          # "increases_risk" | "decreases_risk" | "neutral"
    importance_rank: int
    reference_value: float  # Population mean for this feature
    clinical_note: str = ""

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "value": round(self.value, 3),
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
            "importance_rank": self.importance_rank,
            "reference_value": round(self.reference_value, 3),
            "clinical_note": self.clinical_note,
        }


@dataclass
class PredictionExplanation:
    patient_id: str
    model_name: str
    task: str
    prediction: float           # Raw probability
    prediction_label: str       # "High risk" | "Moderate risk" | "Low risk"
    base_rate: float            # Population prevalence
    top_features: List[FeatureContribution] = field(default_factory=list)
    counterfactuals: List[Dict] = field(default_factory=list)
    natural_language: str = ""
    confidence_interval: Tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "model_name": self.model_name,
            "task": self.task,
            "prediction": round(self.prediction, 4),
            "prediction_label": self.prediction_label,
            "base_rate": round(self.base_rate, 4),
            "top_features": [f.to_dict() for f in self.top_features],
            "counterfactuals": self.counterfactuals,
            "natural_language": self.natural_language,
            "confidence_interval": [round(x, 4) for x in self.confidence_interval],
        }

    def summary(self) -> str:
        top = self.top_features[:3] if self.top_features else []
        drivers = ", ".join(f"{f.feature}={f.value:.1f}" for f in top
                             if f.direction == "increases_risk")
        return (f"{self.task}: {self.prediction:.0%} ({self.prediction_label}) | "
                f"Drivers: {drivers or 'none identified'}")


# ─────────────────────────────────────────
# Clinical reference values (population means)
# ─────────────────────────────────────────

POPULATION_REFERENCE = {
    "heart_rate": 78.0, "sbp": 122.0, "dbp": 78.0, "map": 93.0,
    "respiratory_rate": 16.0, "temperature": 37.0, "spo2": 97.5, "gcs": 15.0,
    "lactate": 1.1, "creatinine": 1.0, "wbc": 8.5, "potassium": 4.1,
    "sodium": 139.0, "glucose": 105.0, "procalcitonin": 0.12,
    "troponin": 0.01, "bnp": 55.0, "inr": 1.1, "hemoglobin": 12.5,
    "platelet": 220.0, "news2": 2.0, "shock_index": 0.65,
    "sofa_score": 1.5, "hr_delta_1h": 0.0, "sbp_delta_1h": 0.0,
    "spo2_delta_1h": 0.0, "lactate_delta_6h": 0.0,
    "hours_in_icu": 24.0, "age": 65.0,
}

FEATURE_CLINICAL_NOTES = {
    "lactate": "Lactate reflects tissue perfusion; elevation indicates anaerobic metabolism",
    "heart_rate": "Tachycardia is a compensatory response to reduced cardiac output or sepsis",
    "sbp": "Hypotension signals haemodynamic compromise and end-organ hypoperfusion",
    "spo2": "SpO2 <94% indicates hypoxaemia and potential respiratory failure",
    "news2": "NEWS2 is a validated composite deterioration score (normal <3)",
    "respiratory_rate": "Tachypnoea is an early sign of sepsis, metabolic acidosis, or respiratory distress",
    "temperature": "Fever or hypothermia both indicate systemic inflammatory response",
    "creatinine": "Rising creatinine signals acute kidney injury",
    "wbc": "Leukocytosis or leukopenia both indicate active infection or immune dysregulation",
    "procalcitonin": "Procalcitonin is a sensitive marker of bacterial infection",
    "troponin": "Troponin elevation indicates myocardial injury",
    "shock_index": "Shock index (HR/SBP) >1 indicates haemodynamic instability",
    "gcs": "GCS <15 indicates altered neurological status",
    "potassium": "Hyperkalaemia is life-threatening and common in AKI",
    "sofa_score": "SOFA score quantifies multi-organ dysfunction",
    "hr_delta_1h": "Rapidly rising heart rate is an early deterioration signal",
    "sbp_delta_1h": "Rapidly falling blood pressure indicates acute haemodynamic compromise",
    "hours_in_icu": "Prolonged ICU stay is associated with increased complication risk",
}

# Risk direction: positive coefficient = increases risk
FEATURE_RISK_DIRECTION = {
    "lactate": 1, "heart_rate": 1, "respiratory_rate": 1, "temperature": 1,
    "wbc": 1, "news2": 1, "shock_index": 1, "procalcitonin": 1,
    "troponin": 1, "hours_in_icu": 1, "hr_delta_1h": 1, "sofa_score": 1,
    "sbp": -1, "spo2": -1, "gcs": -1, "map": -1,  # lower = worse
    "hemoglobin": -1, "sbp_delta_1h": -1, "spo2_delta_1h": -1,
}


# ─────────────────────────────────────────
# Permutation importance (no sklearn needed)
# ─────────────────────────────────────────

def permutation_importance(predict_fn: Callable,
                             X: np.ndarray,
                             feature_names: List[str],
                             n_repeats: int = 5,
                             seed: int = 42) -> Dict[str, float]:
    """
    Compute permutation feature importance.
    Returns dict: feature_name → mean importance score.
    """
    rng = np.random.RandomState(seed)
    baseline_preds = predict_fn(X)
    baseline_score = float(np.mean(baseline_preds))

    importances: Dict[str, List[float]] = {f: [] for f in feature_names}

    for feat_idx, feat_name in enumerate(feature_names):
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
            perm_preds = predict_fn(X_perm)
            perm_score = float(np.mean(perm_preds))
            # Importance = how much worse performance gets when feature is shuffled
            importances[feat_name].append(abs(baseline_score - perm_score))

    return {f: float(np.mean(v)) for f, v in importances.items()}


# ─────────────────────────────────────────
# SHAP-style single-observation attribution
# ─────────────────────────────────────────

def local_attribution(predict_fn: Callable,
                        observation: np.ndarray,
                        background: np.ndarray,
                        feature_names: List[str],
                        n_samples: int = 50,
                        seed: int = 42) -> Dict[str, float]:
    """
    Kernel SHAP approximation for a single observation.
    For each feature: contribution ≈ E[f(x) | x_i=v_i] - E[f(x) | x_i=background]
    """
    rng = np.random.RandomState(seed)
    n_features = len(feature_names)
    pred_baseline = float(np.mean(predict_fn(background)))
    pred_target = float(predict_fn(observation.reshape(1, -1))[0])
    total_effect = pred_target - pred_baseline

    contributions = {}
    raw_effects = {}

    for i, feat in enumerate(feature_names):
        # Sample background rows, replace feature i with observation value
        idx = rng.choice(len(background), size=min(n_samples, len(background)), replace=False)
        bg_sample = background[idx].copy()
        bg_sample_replaced = bg_sample.copy()
        bg_sample_replaced[:, i] = observation[i]
        effect = float(np.mean(predict_fn(bg_sample_replaced))) - float(np.mean(predict_fn(bg_sample)))
        raw_effects[feat] = effect

    # Normalise so contributions sum to total_effect
    total_raw = sum(abs(v) for v in raw_effects.values())
    for feat, effect in raw_effects.items():
        if total_raw > 0:
            contributions[feat] = effect * total_effect / total_raw
        else:
            contributions[feat] = 0.0

    return contributions


# ─────────────────────────────────────────
# Clinical explainer
# ─────────────────────────────────────────

class ClinicalExplainer:
    """
    Produces human-readable explanations for ML risk predictions.
    Works with any sklearn-compatible model (or a predict_proba callable).
    """

    LABEL_THRESHOLDS = {
        "sepsis_risk":    [(0.6, "High risk"), (0.35, "Moderate risk"), (0, "Low risk")],
        "mortality_risk": [(0.5, "High risk"), (0.25, "Moderate risk"), (0, "Low risk")],
        "default":        [(0.5, "High risk"), (0.25, "Moderate risk"), (0, "Low risk")],
    }

    BASE_RATES = {
        "sepsis_risk": 0.15, "mortality_risk": 0.10,
        "cardiac_risk": 0.12, "anomaly": 0.08,
    }

    def __init__(self, model=None, feature_names: Optional[List[str]] = None,
                  task: str = "sepsis_risk"):
        self.model = model
        self.feature_names = feature_names or list(POPULATION_REFERENCE.keys())
        self.task = task
        self._background: Optional[np.ndarray] = None

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            # Fallback: deterministic risk scorer
            return np.array([self._heuristic_score(x) for x in X])
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return self.model.predict(X).astype(float)

    def _heuristic_score(self, x: np.ndarray) -> float:
        """Rule-based risk score for demo when no model is available."""
        score = 0.15
        feat_vals = dict(zip(self.feature_names, x))
        # Weighted heuristic
        weights = {
            "lactate":           0.12, "news2":      0.15,
            "shock_index":       0.10, "heart_rate": 0.06,
            "respiratory_rate":  0.07, "temperature": 0.05,
            "wbc":               0.05, "procalcitonin": 0.08,
        }
        negative_weights = {"sbp": -0.08, "spo2": -0.10, "gcs": -0.06, "map": -0.06}

        for feat, w in weights.items():
            val = feat_vals.get(feat, POPULATION_REFERENCE.get(feat, 0))
            ref = POPULATION_REFERENCE.get(feat, val)
            if ref != 0:
                score += w * (val - ref) / ref * 0.5

        for feat, w in negative_weights.items():
            val = feat_vals.get(feat, POPULATION_REFERENCE.get(feat, 100))
            ref = POPULATION_REFERENCE.get(feat, val)
            if ref != 0:
                score += w * (val - ref) / ref * 0.5

        return max(0.01, min(0.99, score))

    def set_background(self, X_background: np.ndarray):
        """Provide background dataset for SHAP-style attribution."""
        self._background = X_background

    def _get_label(self, prediction: float) -> str:
        thresholds = self.LABEL_THRESHOLDS.get(self.task, self.LABEL_THRESHOLDS["default"])
        for thresh, label in thresholds:
            if prediction >= thresh:
                return label
        return "Low risk"

    def explain(self, patient_data: Dict, patient_id: str = "UNKNOWN",
                 top_k: int = 8) -> PredictionExplanation:
        """
        Generate a full explanation for one patient observation.
        patient_data: dict of feature_name → value
        """
        # Build observation vector
        obs = np.array([patient_data.get(f, POPULATION_REFERENCE.get(f, 0))
                         for f in self.feature_names], dtype=float)

        # Generate background if not set
        if self._background is None:
            self._background = self._synthetic_background(200)

        prediction = float(self._predict_fn(obs.reshape(1, -1))[0])

        # Local attribution
        attributions = local_attribution(
            self._predict_fn, obs, self._background, self.feature_names,
            n_samples=100,
        )

        # Build feature contributions
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val = obs[i]
            ref = POPULATION_REFERENCE.get(feat, val)
            contrib = attributions.get(feat, 0.0)
            natural_dir = FEATURE_RISK_DIRECTION.get(feat, 1)
            actual_dir = "increases_risk" if contrib > 0.002 else \
                         "decreases_risk" if contrib < -0.002 else "neutral"
            contributions.append(FeatureContribution(
                feature=feat,
                value=float(val),
                contribution=float(contrib),
                direction=actual_dir,
                importance_rank=0,
                reference_value=float(ref),
                clinical_note=FEATURE_CLINICAL_NOTES.get(feat, ""),
            ))

        # Rank by |contribution|
        contributions.sort(key=lambda c: -abs(c.contribution))
        for rank, c in enumerate(contributions):
            c.importance_rank = rank + 1
        top_features = contributions[:top_k]

        # Counterfactual reasoning
        counterfactuals = self._generate_counterfactuals(obs, prediction, attributions)

        # Confidence interval (bootstrap approximation)
        ci = self._bootstrap_ci(obs)

        # Natural language explanation
        nl = self._natural_language(prediction, top_features, patient_id)

        return PredictionExplanation(
            patient_id=patient_id,
            model_name=type(self.model).__name__ if self.model else "HeuristicModel",
            task=self.task,
            prediction=prediction,
            prediction_label=self._get_label(prediction),
            base_rate=self.BASE_RATES.get(self.task, 0.15),
            top_features=top_features,
            counterfactuals=counterfactuals,
            natural_language=nl,
            confidence_interval=ci,
        )

    def _synthetic_background(self, n: int = 200, seed: int = 42) -> np.ndarray:
        """Generate a synthetic background population."""
        rng = np.random.RandomState(seed)
        rows = []
        for _ in range(n):
            row = []
            for feat in self.feature_names:
                ref = POPULATION_REFERENCE.get(feat, 1.0)
                std = ref * 0.15 + 0.01
                val = max(0, rng.normal(ref, std))
                row.append(val)
            rows.append(row)
        return np.array(rows)

    def _generate_counterfactuals(self, obs: np.ndarray, prediction: float,
                                   attributions: Dict[str, float]) -> List[Dict]:
        """Identify what changes would most reduce risk."""
        counterfactuals = []
        if prediction < 0.3:
            return counterfactuals  # already low risk

        # Top risk-increasing features
        risk_drivers = [(f, v) for f, v in attributions.items() if v > 0.01]
        risk_drivers.sort(key=lambda x: -x[1])

        for feat, contrib in risk_drivers[:4]:
            ref = POPULATION_REFERENCE.get(feat, None)
            if ref is None:
                continue
            feat_idx = self.feature_names.index(feat) if feat in self.feature_names else -1
            if feat_idx < 0:
                continue

            natural_dir = FEATURE_RISK_DIRECTION.get(feat, 1)
            if natural_dir == 1:
                target = ref * 0.85  # reducing toward normal
                change_desc = f"Reduce {feat} toward {target:.1f} (population mean: {ref:.1f})"
            else:
                target = ref * 1.10
                change_desc = f"Increase {feat} toward {target:.1f} (population mean: {ref:.1f})"

            # Estimate new prediction
            obs_cf = obs.copy()
            obs_cf[feat_idx] = target
            new_pred = float(self._predict_fn(obs_cf.reshape(1, -1))[0])
            delta = prediction - new_pred

            if delta > 0.02:
                counterfactuals.append({
                    "feature": feat,
                    "current_value": round(float(obs[feat_idx]), 2),
                    "target_value": round(target, 2),
                    "change": change_desc,
                    "predicted_risk_reduction": round(delta, 4),
                    "new_predicted_risk": round(new_pred, 4),
                    "clinical_action": self._clinical_action(feat, target),
                })

        return counterfactuals[:3]

    def _clinical_action(self, feature: str, target: float) -> str:
        actions = {
            "lactate": "Aggressive fluid resuscitation and vasopressor support",
            "heart_rate": "Rate control with beta-blocker or treat underlying cause",
            "sbp": "Vasopressor therapy or fluid bolus to restore perfusion pressure",
            "spo2": "Supplemental oxygen or escalation of respiratory support",
            "news2": "Clinical review and escalation per hospital deterioration protocol",
            "respiratory_rate": "Treat underlying cause; consider NIV if tiring",
            "temperature": "Antipyretics; source control if infection suspected",
            "wbc": "Broad-spectrum antibiotics after cultures if infection confirmed",
            "creatinine": "Optimize fluid balance; avoid nephrotoxins; nephrology consult",
            "procalcitonin": "Initiate antibiotic therapy guided by sensitivities",
            "shock_index": "Urgent haemodynamic resuscitation; identify shock aetiology",
        }
        return actions.get(feature, f"Target {feature} normalisation")

    def _bootstrap_ci(self, obs: np.ndarray, n_boot: int = 50,
                        seed: int = 7) -> Tuple[float, float]:
        """Approximate 95% CI via bootstrap resampling of background."""
        rng = np.random.RandomState(seed)
        preds = []
        bg = self._background
        for _ in range(n_boot):
            idx = rng.choice(len(bg), size=1)[0]
            mixed = obs.copy()
            # Randomly zero out 20% of features using background
            mask = rng.random(len(obs)) < 0.20
            mixed[mask] = bg[idx][mask]
            preds.append(float(self._predict_fn(mixed.reshape(1, -1))[0]))
        preds.sort()
        lo = preds[int(0.025 * n_boot)]
        hi = preds[int(0.975 * n_boot)]
        return (round(lo, 4), round(hi, 4))

    def _natural_language(self, prediction: float,
                           top_features: List[FeatureContribution],
                           patient_id: str) -> str:
        label = self._get_label(prediction)
        base = self.BASE_RATES.get(self.task, 0.15)
        task_display = self.task.replace("_", " ").title()

        # Drivers and protectors
        risk_up = [f for f in top_features if f.direction == "increases_risk"][:3]
        risk_down = [f for f in top_features if f.direction == "decreases_risk"][:2]

        lines = [
            f"Patient {patient_id} has a predicted {task_display} probability of "
            f"{prediction:.0%} ({label}).",
            f"This is {'above' if prediction > base else 'near'} the population baseline "
            f"of {base:.0%}.",
        ]

        if risk_up:
            driver_str = "; ".join(
                f"{f.feature.replace('_',' ')} = {f.value:.1f} "
                f"(↑ from reference {f.reference_value:.1f})"
                if FEATURE_RISK_DIRECTION.get(f.feature, 1) == 1
                else f"{f.feature.replace('_',' ')} = {f.value:.1f} "
                     f"(↓ from reference {f.reference_value:.1f})"
                for f in risk_up
            )
            lines.append(f"Key risk drivers: {driver_str}.")

        if risk_down:
            protect_str = ", ".join(
                f"{f.feature.replace('_',' ')} ({f.value:.1f})" for f in risk_down
            )
            lines.append(f"Protective factors: {protect_str}.")

        lines.append(
            "This prediction is generated by an AI model and should be interpreted "
            "in the clinical context by the treating team."
        )
        return " ".join(lines)

    def explain_batch(self, patient_records: List[Dict],
                       patient_ids: Optional[List[str]] = None,
                       top_k: int = 5) -> List[PredictionExplanation]:
        """Explain multiple patients at once."""
        ids = patient_ids or [f"P{i+1:03d}" for i in range(len(patient_records))]
        if self._background is None:
            self._background = self._synthetic_background(200)
        return [self.explain(rec, pid, top_k) for rec, pid in zip(patient_records, ids)]

    def global_importance(self, n_patients: int = 200) -> Dict[str, float]:
        """Compute global feature importance over a synthetic population."""
        background = self._synthetic_background(n_patients)
        return permutation_importance(self._predict_fn, background, self.feature_names, n_repeats=3)


# ─────────────────────────────────────────
# Waterfall chart (text-based)
# ─────────────────────────────────────────

def print_waterfall(explanation: PredictionExplanation, width: int = 50):
    """Print a text-based feature contribution waterfall."""
    print(f"\n{'='*60}")
    print(f"Explanation: {explanation.task} for {explanation.patient_id}")
    print(f"Prediction: {explanation.prediction:.0%} ({explanation.prediction_label})")
    print(f"Base rate:  {explanation.base_rate:.0%}")
    print(f"{'='*60}")
    print(f"{'Feature':<22} {'Value':>8}  {'Contribution'}")
    print(f"{'-'*60}")

    running = explanation.base_rate
    for fc in explanation.top_features[:8]:
        bar_len = int(abs(fc.contribution) / 0.02)
        bar_len = min(bar_len, 20)
        if fc.contribution > 0:
            bar = "▓" * bar_len
            sign = "+"
            col = "↑"
        elif fc.contribution < 0:
            bar = "░" * bar_len
            sign = ""
            col = "↓"
        else:
            bar = ""
            sign = ""
            col = "–"
        running += fc.contribution
        print(f"{fc.feature:<22} {fc.value:>8.2f}  "
              f"{col} {sign}{fc.contribution:+.4f}  {bar}")

    print(f"{'-'*60}")
    print(f"{'Predicted':>32}: {explanation.prediction:.4f}")
    if explanation.counterfactuals:
        print(f"\nTop counterfactual:")
        cf = explanation.counterfactuals[0]
        print(f"  If {cf['feature']} → {cf['target_value']:.1f}: "
              f"risk reduces to {cf['new_predicted_risk']:.0%} "
              f"(Δ {cf['predicted_risk_reduction']:.0%})")
    print()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    explainer = ClinicalExplainer(task="sepsis_risk")
    background = explainer._synthetic_background(200)
    explainer.set_background(background)

    # Deteriorating patient
    sick_patient = {
        "heart_rate": 122, "sbp": 88, "dbp": 55, "map": 66,
        "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14,
        "lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "procalcitonin": 12.0,
        "news2": 9, "shock_index": 1.39, "troponin": 0.02,
        "hr_delta_1h": 15.0, "sbp_delta_1h": -18.0, "hours_in_icu": 8,
    }

    explanation = explainer.explain(sick_patient, patient_id="P_SICK", top_k=8)
    print_waterfall(explanation)
    print("Natural language explanation:")
    print(f"  {explanation.natural_language}")

    # Stable patient
    print("\n--- Stable Patient ---")
    stable_patient = {
        "heart_rate": 76, "sbp": 124, "dbp": 78, "map": 93,
        "respiratory_rate": 15, "temperature": 37.0, "spo2": 98, "gcs": 15,
        "lactate": 0.9, "creatinine": 0.9, "wbc": 8.0, "procalcitonin": 0.1,
        "news2": 1, "shock_index": 0.61, "troponin": 0.008,
        "hr_delta_1h": 0.0, "sbp_delta_1h": 1.0, "hours_in_icu": 6,
    }
    exp2 = explainer.explain(stable_patient, patient_id="P_STABLE")
    print(f"  {exp2.summary()}")

    # Global importance
    print("\nGlobal feature importance (top 8):")
    gi = explainer.global_importance(n_patients=100)
    top_gi = sorted(gi.items(), key=lambda x: -x[1])[:8]
    for feat, imp in top_gi:
        bar = "█" * int(imp * 500)
        print(f"  {feat:<24} {imp:.4f}  {bar}")

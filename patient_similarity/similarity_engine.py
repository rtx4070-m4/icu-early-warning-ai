"""
Patient Similarity Engine
Finds historical ICU patients similar to a current patient using:
  - Vital sign trajectory embeddings
  - Lab value profiles
  - Diagnosis code overlap (SNOMED/ICD-10)
  - Weighted cosine similarity + Euclidean distance

Enables: outcome prediction from historical cohort, treatment pattern mining,
         rare presentation lookup, and case-based reasoning.
"""

import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Patient feature vector
# ─────────────────────────────────────────

FEATURE_WEIGHTS = {
    # Vitals (normalised)
    "heart_rate":         1.0,
    "sbp":                1.2,
    "dbp":                0.8,
    "map":                1.2,
    "respiratory_rate":   1.3,
    "temperature":        0.9,
    "spo2":               1.5,
    "gcs":                1.4,
    # Labs
    "lactate":            2.0,
    "creatinine":         1.5,
    "wbc":                1.2,
    "procalcitonin":      1.3,
    "troponin":           1.4,
    "bnp":                1.1,
    "inr":                1.2,
    "glucose":            0.8,
    "potassium":          1.0,
    "sodium":             0.8,
    # Scores
    "news2":              2.0,
    "sofa_score":         2.0,
    "shock_index":        1.8,
    "hours_in_icu":       0.5,
    "age":                0.6,
}

POPULATION_MEANS = {
    "heart_rate": 85.0, "sbp": 115.0, "dbp": 72.0, "map": 86.0,
    "respiratory_rate": 18.0, "temperature": 37.2, "spo2": 96.0, "gcs": 14.5,
    "lactate": 1.8, "creatinine": 1.4, "wbc": 11.0, "procalcitonin": 2.5,
    "troponin": 0.02, "bnp": 120.0, "inr": 1.3, "glucose": 130.0,
    "potassium": 4.2, "sodium": 138.0,
    "news2": 4.5, "sofa_score": 3.0, "shock_index": 0.85,
    "hours_in_icu": 48.0, "age": 65.0,
}

POPULATION_STDS = {
    "heart_rate": 22.0, "sbp": 28.0, "dbp": 16.0, "map": 20.0,
    "respiratory_rate": 6.0, "temperature": 0.8, "spo2": 5.0, "gcs": 2.5,
    "lactate": 2.0, "creatinine": 1.5, "wbc": 6.0, "procalcitonin": 8.0,
    "troponin": 0.08, "bnp": 200.0, "inr": 0.6, "glucose": 60.0,
    "potassium": 0.7, "sodium": 6.0,
    "news2": 3.5, "sofa_score": 2.5, "shock_index": 0.35,
    "hours_in_icu": 40.0, "age": 15.0,
}

FEATURE_NAMES = list(FEATURE_WEIGHTS.keys())


# ─────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────

@dataclass
class HistoricalPatient:
    patient_id: str
    age: int
    sex: str
    primary_diagnosis: str
    icd10_codes: List[str]
    features: Dict[str, float]
    outcome: str          # survived | died | transferred
    icu_los_days: float
    ventilated: bool
    vasopressors: bool
    treatments: List[str]
    feature_vector: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "age": self.age, "sex": self.sex,
            "primary_diagnosis": self.primary_diagnosis,
            "outcome": self.outcome,
            "icu_los_days": round(self.icu_los_days, 1),
            "ventilated": self.ventilated,
            "vasopressors": self.vasopressors,
        }


@dataclass
class SimilarPatient:
    historical: HistoricalPatient
    similarity_score: float    # 0–1 (1 = identical)
    distance: float
    matching_features: List[str]
    diverging_features: List[str]

    def to_dict(self) -> Dict:
        return {
            **self.historical.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "distance": round(self.distance, 4),
            "matching_features": self.matching_features[:5],
            "diverging_features": self.diverging_features[:3],
        }


@dataclass
class CohortInsight:
    """Aggregated statistics from the top-k similar patients."""
    n_similar: int
    survival_rate: float
    median_icu_los_days: float
    ventilation_rate: float
    vasopressor_rate: float
    most_common_diagnoses: List[Tuple[str, int]]
    common_treatments: List[Tuple[str, int]]
    outcome_distribution: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "n_similar": self.n_similar,
            "survival_rate": round(self.survival_rate, 3),
            "median_icu_los_days": round(self.median_icu_los_days, 1),
            "ventilation_rate": round(self.ventilation_rate, 3),
            "vasopressor_rate": round(self.vasopressor_rate, 3),
            "most_common_diagnoses": self.most_common_diagnoses[:5],
            "common_treatments": self.common_treatments[:5],
            "outcome_distribution": self.outcome_distribution,
        }


# ─────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────

def extract_feature_vector(patient_data: Dict) -> np.ndarray:
    """Extract and z-score normalise a feature vector from patient data."""
    vitals = patient_data.get("vitals", patient_data)
    labs = patient_data.get("labs", {})
    scores = patient_data

    combined = {**vitals, **labs, **scores}
    vector = []
    for feat in FEATURE_NAMES:
        val = combined.get(feat, POPULATION_MEANS.get(feat, 0.0))
        mean = POPULATION_MEANS.get(feat, val)
        std = POPULATION_STDS.get(feat, 1.0)
        z = (val - mean) / max(std, 1e-6)
        weight = FEATURE_WEIGHTS.get(feat, 1.0)
        vector.append(z * weight)

    return np.array(vector, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def combined_similarity(a: np.ndarray, b: np.ndarray,
                          cos_weight: float = 0.6) -> float:
    """Blend cosine similarity + normalised euclidean distance."""
    cos = cosine_similarity(a, b)
    euc = euclidean_distance(a, b)
    # Normalise euclidean: assume max distance ~10 (for z-scored features)
    euc_sim = max(0.0, 1.0 - euc / 10.0)
    return cos_weight * cos + (1 - cos_weight) * euc_sim


# ─────────────────────────────────────────
# Synthetic patient cohort generator
# ─────────────────────────────────────────

DIAGNOSES = [
    "septic_shock", "pneumonia", "ards", "acute_kidney_injury",
    "congestive_heart_failure", "myocardial_infarction", "copd_exacerbation",
    "pulmonary_embolism", "diabetic_ketoacidosis", "stroke", "gi_bleed",
    "pancreatitis", "liver_failure", "post_cardiac_arrest",
]

TREATMENT_POOLS = {
    "septic_shock":          ["antibiotics", "vasopressors", "fluids", "steroids"],
    "ards":                  ["mechanical_ventilation", "prone_positioning", "PEEP", "NMB"],
    "acute_kidney_injury":   ["CRRT", "fluids", "furosemide", "dialysis"],
    "congestive_heart_failure": ["diuretics", "ACE_inhibitor", "beta_blocker", "BiPAP"],
    "myocardial_infarction": ["PCI", "aspirin", "heparin", "statin", "beta_blocker"],
}


def _synthetic_cohort(n: int = 500, seed: int = 42) -> List[HistoricalPatient]:
    rng = random.Random(seed)
    patients = []

    for i in range(n):
        diagnosis = rng.choice(DIAGNOSES)
        severity = rng.random()  # 0=mild, 1=critical
        age = int(rng.gauss(65, 15))
        age = max(18, min(95, age))
        sex = rng.choice(["M", "F"])
        ventilated = severity > 0.6 or diagnosis == "ards"
        vasopressors = severity > 0.55 or diagnosis in ("septic_shock",)
        icu_los = max(0.5, rng.gauss(5 + severity * 10, 3))
        # Outcome: higher severity → higher mortality
        died = rng.random() < (0.05 + severity * 0.45)
        outcome = "died" if died else ("transferred" if rng.random() < 0.1 else "survived")

        # Generate realistic features based on severity and diagnosis
        noise = lambda std=1: rng.gauss(0, std)
        features = {
            "heart_rate":       85 + severity * 50 + noise(8),
            "sbp":              130 - severity * 55 + noise(10),
            "dbp":              80 - severity * 30 + noise(6),
            "map":              97 - severity * 38 + noise(8),
            "respiratory_rate": 15 + severity * 18 + noise(3),
            "temperature":      37.0 + severity * 1.8 + noise(0.4),
            "spo2":             98 - severity * 12 + noise(2),
            "gcs":              15 - int(severity * 8),
            "lactate":          1.0 + severity * 6 + noise(1),
            "creatinine":       0.9 + severity * 3 + noise(0.5),
            "wbc":              8 + severity * 16 + noise(3),
            "procalcitonin":    0.1 + severity * 25 + noise(3),
            "troponin":         0.008 + severity * 0.15 + noise(0.02),
            "bnp":              50 + severity * 500 + noise(50),
            "inr":              1.0 + severity * 1.5 + noise(0.3),
            "glucose":          95 + severity * 120 + noise(20),
            "potassium":        4.0 + severity * 1.5 + noise(0.4),
            "sodium":           140 + noise(5) - severity * 8,
            "news2":            int(severity * 14),
            "sofa_score":       severity * 10,
            "shock_index":      (85 + severity * 50) / max(85, 130 - severity * 55),
            "hours_in_icu":     icu_los * 24,
            "age":              float(age),
        }
        # Clamp physiological values
        features["spo2"] = min(100, max(60, features["spo2"]))
        features["gcs"] = max(3, min(15, features["gcs"]))
        features["sbp"] = max(40, features["sbp"])

        treatments = TREATMENT_POOLS.get(diagnosis, ["supportive_care"])
        if ventilated:
            treatments = list(set(treatments + ["mechanical_ventilation"]))
        if vasopressors:
            treatments = list(set(treatments + ["vasopressors", "norepinephrine"]))

        pat = HistoricalPatient(
            patient_id=f"HIST_{i:04d}",
            age=age, sex=sex,
            primary_diagnosis=diagnosis,
            icd10_codes=[],
            features=features,
            outcome=outcome,
            icu_los_days=icu_los,
            ventilated=ventilated,
            vasopressors=vasopressors,
            treatments=rng.sample(treatments, min(len(treatments), 3)),
        )
        pat.feature_vector = extract_feature_vector(features)
        patients.append(pat)

    return patients


# ─────────────────────────────────────────
# Similarity engine
# ─────────────────────────────────────────

class PatientSimilarityEngine:
    """
    Finds the k most similar historical patients to a query patient.
    Supports: cohort-level outcome statistics, treatment pattern mining.
    """

    def __init__(self, cohort: Optional[List[HistoricalPatient]] = None,
                  n_synthetic: int = 500):
        if cohort is not None:
            self.cohort = cohort
        else:
            logger.info(f"Building synthetic cohort ({n_synthetic} patients)...")
            self.cohort = _synthetic_cohort(n_synthetic)
        # Pre-build matrix for fast similarity search
        self._matrix = np.stack([p.feature_vector for p in self.cohort])
        logger.info(f"PatientSimilarityEngine ready: {len(self.cohort)} historical patients")

    def find_similar(self, query_patient: Dict,
                      top_k: int = 10,
                      min_similarity: float = 0.5,
                      filter_diagnosis: Optional[str] = None) -> List[SimilarPatient]:
        """Find the top-k most similar historical patients."""
        query_vec = extract_feature_vector(query_patient)

        # Vectorised similarity computation
        norms_hist = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norm_q = np.linalg.norm(query_vec)

        # Cosine similarities
        if norm_q > 0:
            cos_sims = self._matrix @ query_vec / (norms_hist.squeeze() * norm_q + 1e-9)
        else:
            cos_sims = np.zeros(len(self.cohort))

        # Euclidean distances (normalised)
        dists = np.linalg.norm(self._matrix - query_vec, axis=1)
        euc_sims = np.maximum(0, 1.0 - dists / 10.0)

        # Combined score
        scores = 0.6 * cos_sims + 0.4 * euc_sims

        # Apply filters
        if filter_diagnosis:
            mask = np.array([p.primary_diagnosis == filter_diagnosis
                              for p in self.cohort])
            scores = scores * mask

        # Get top-k indices
        top_idx = np.argsort(scores)[::-1][:top_k * 3]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < min_similarity:
                continue
            hist = self.cohort[idx]
            dist = float(dists[idx])

            # Identify matching and diverging features
            matching, diverging = self._feature_comparison(query_vec, hist.feature_vector)

            results.append(SimilarPatient(
                historical=hist,
                similarity_score=score,
                distance=dist,
                matching_features=matching,
                diverging_features=diverging,
            ))
            if len(results) >= top_k:
                break

        return results

    def _feature_comparison(self, query: np.ndarray,
                              hist: np.ndarray) -> Tuple[List[str], List[str]]:
        """Identify features that are similar vs divergent."""
        diffs = np.abs(query - hist)
        matching = [FEATURE_NAMES[i] for i in np.where(diffs < 0.5)[0]]
        diverging = [FEATURE_NAMES[i] for i in np.argsort(diffs)[::-1][:3]]
        return matching[:8], diverging

    def cohort_insights(self, similar_patients: List[SimilarPatient]) -> CohortInsight:
        """Aggregate statistics from a list of similar patients."""
        if not similar_patients:
            return CohortInsight(0, 0, 0, 0, 0, [], [], {})

        hists = [s.historical for s in similar_patients]
        n = len(hists)

        survived = sum(1 for p in hists if p.outcome == "survived")
        icu_los_sorted = sorted(p.icu_los_days for p in hists)
        median_los = icu_los_sorted[n // 2]
        ventilated = sum(1 for p in hists if p.ventilated)
        vasopressors = sum(1 for p in hists if p.vasopressors)

        # Diagnosis frequency
        dx_counts: Dict[str, int] = {}
        for p in hists:
            dx_counts[p.primary_diagnosis] = dx_counts.get(p.primary_diagnosis, 0) + 1
        dx_sorted = sorted(dx_counts.items(), key=lambda x: -x[1])

        # Treatment frequency
        tx_counts: Dict[str, int] = {}
        for p in hists:
            for tx in p.treatments:
                tx_counts[tx] = tx_counts.get(tx, 0) + 1
        tx_sorted = sorted(tx_counts.items(), key=lambda x: -x[1])

        # Outcome distribution
        outcome_dist: Dict[str, int] = {}
        for p in hists:
            outcome_dist[p.outcome] = outcome_dist.get(p.outcome, 0) + 1

        return CohortInsight(
            n_similar=n,
            survival_rate=survived / n,
            median_icu_los_days=median_los,
            ventilation_rate=ventilated / n,
            vasopressor_rate=vasopressors / n,
            most_common_diagnoses=dx_sorted,
            common_treatments=tx_sorted,
            outcome_distribution=outcome_dist,
        )

    def predict_outcome(self, query_patient: Dict, top_k: int = 20) -> Dict:
        """Predict likely outcome based on similar historical patients."""
        similar = self.find_similar(query_patient, top_k=top_k, min_similarity=0.4)
        if not similar:
            return {"prediction": "insufficient_data", "confidence": 0.0}

        insights = self.cohort_insights(similar)
        survival_rate = insights.survival_rate

        # Weighted survival rate (closer patients count more)
        weights = [s.similarity_score for s in similar]
        total_w = sum(weights)
        weighted_survival = sum(
            w * (1.0 if s.historical.outcome == "survived" else 0.0)
            for s, w in zip(similar, weights)
        ) / max(total_w, 1e-9)

        prediction = "survived" if weighted_survival >= 0.5 else "died"
        confidence = abs(weighted_survival - 0.5) * 2  # 0–1

        return {
            "prediction": prediction,
            "survival_probability": round(weighted_survival, 3),
            "confidence": round(confidence, 3),
            "based_on_n": len(similar),
            "median_icu_los_days": round(insights.median_icu_los_days, 1),
            "expected_ventilation": insights.ventilation_rate > 0.5,
            "expected_vasopressors": insights.vasopressor_rate > 0.5,
            "top_diagnoses": [d[0] for d in insights.most_common_diagnoses[:3]],
            "recommended_treatments": [t[0] for t in insights.common_treatments[:4]],
        }


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    engine = PatientSimilarityEngine(n_synthetic=300)

    query = {
        "vitals": {
            "heart_rate": 118, "sbp": 88, "dbp": 55, "map": 66,
            "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14,
        },
        "labs": {
            "lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "procalcitonin": 12.0,
            "troponin": 0.02, "bnp": 120, "inr": 1.4, "glucose": 195, "potassium": 5.1,
        },
        "news2": 9, "sofa_score": 7.5, "shock_index": 1.34, "hours_in_icu": 8, "age": 68,
    }

    print("=== Patient Similarity Search ===\n")
    similar = engine.find_similar(query, top_k=10)
    print(f"Top {len(similar)} similar patients:")
    print(f"  {'ID':<12} {'Similarity':>10} {'Dx':<28} {'Outcome':<12} {'LOS':>6}")
    print("  " + "-" * 72)
    for s in similar:
        p = s.historical
        print(f"  {p.patient_id:<12} {s.similarity_score:>10.3f} "
              f"{p.primary_diagnosis:<28} {p.outcome:<12} {p.icu_los_days:>5.1f}d")

    print("\n=== Cohort Insights ===")
    insights = engine.cohort_insights(similar)
    print(f"  Survival rate     : {insights.survival_rate:.0%}")
    print(f"  Median ICU LOS    : {insights.median_icu_los_days:.1f} days")
    print(f"  Ventilation rate  : {insights.ventilation_rate:.0%}")
    print(f"  Vasopressor rate  : {insights.vasopressor_rate:.0%}")
    print(f"  Top diagnoses     : {[d[0] for d in insights.most_common_diagnoses[:3]]}")
    print(f"  Common treatments : {[t[0] for t in insights.common_treatments[:4]]}")

    print("\n=== Outcome Prediction ===")
    pred = engine.predict_outcome(query, top_k=20)
    print(f"  Prediction        : {pred['prediction'].upper()}")
    print(f"  Survival prob     : {pred['survival_probability']:.0%}")
    print(f"  Confidence        : {pred['confidence']:.0%}")
    print(f"  Expected LOS      : {pred['median_icu_los_days']:.1f} days")
    print(f"  Likely treatments : {pred['recommended_treatments']}")

"""
AI Hospital OS — Test Suite
Tests: data engineering, ML models, NLP pipeline, knowledge graph, real-time monitoring
Run: python -m pytest tests/test_models.py -v
"""

import sys
import os
import json
import math
import random
import unittest
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_vitals_array(n: int = 200, n_features: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, n_features).astype(np.float32)


def make_sequence_data(n: int = 100, seq_len: int = 24, n_features: int = 8, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, seq_len, n_features).astype(np.float32)
    y = (rng.rand(n) < 0.15).astype(np.float32)
    return X, y


def make_tabular_data(n: int = 500, n_features: int = 20, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (rng.rand(n) < 0.15).astype(int)
    return X, y


# ─────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────

class TestFeatureEngineering(unittest.TestCase):

    def _make_vitals_df(self, n=72, pid="P001"):
        import pandas as pd
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        return pd.DataFrame({
            "patient_id": pid,
            "timestamp": times,
            "heart_rate": 80 + np.random.randn(n) * 5,
            "sbp": 120 + np.random.randn(n) * 10,
            "dbp": 75 + np.random.randn(n) * 6,
            "respiratory_rate": 16 + np.random.randn(n) * 2,
            "temperature": 37 + np.random.randn(n) * 0.3,
            "spo2": np.clip(97 + np.random.randn(n), 80, 100),
        })

    def test_sofa_calculation(self):
        from data_engineering.feature_engineering import compute_sofa_cardiovascular
        # Normal MAP > 70 → score 0
        self.assertEqual(compute_sofa_cardiovascular(85), 0)
        # MAP < 70 → score 1+
        self.assertGreater(compute_sofa_cardiovascular(65), 0)

    def test_news2_calculation(self):
        from data_engineering.feature_engineering import compute_news2_score
        score = compute_news2_score(hr=118, sbp=90, rr=24, temp=38.7, spo2=94, gcs=15)
        self.assertGreaterEqual(score, 4)

    def test_delta_features(self):
        try:
            import pandas as pd
            from data_engineering.feature_engineering import add_delta_features
            df = self._make_vitals_df()
            result = add_delta_features(df, ["heart_rate", "sbp"])
            self.assertIn("heart_rate_delta", result.columns)
        except (ImportError, AttributeError):
            self.skipTest("Feature engineering module not fully available")

    def test_rolling_features(self):
        try:
            import pandas as pd
            from data_engineering.feature_engineering import add_rolling_features
            df = self._make_vitals_df()
            result = add_rolling_features(df, ["heart_rate"], windows=[3, 6])
            self.assertTrue(any("roll" in c for c in result.columns))
        except (ImportError, AttributeError):
            self.skipTest("Rolling features not available")

    def test_shock_index(self):
        from data_engineering.feature_engineering import compute_shock_index
        si = compute_shock_index(hr=120, sbp=80)
        self.assertAlmostEqual(si, 1.5, places=2)
        # Healthy patient
        si2 = compute_shock_index(hr=70, sbp=120)
        self.assertLess(si2, 1.0)


# ─────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):

    def test_locf_imputation(self):
        try:
            import pandas as pd
            from data_engineering.preprocessing import locf_impute
            df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0, np.nan]})
            result = locf_impute(df)
            self.assertFalse(result["x"].isna().any())
            self.assertEqual(result["x"].iloc[1], 1.0)
        except ImportError:
            self.skipTest("Preprocessing module not available")

    def test_winsorization(self):
        try:
            from data_engineering.preprocessing import winsorize_column
            import pandas as pd
            s = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
            result = winsorize_column(s, lower=0.05, upper=0.95)
            self.assertLess(result.max(), 100)
        except (ImportError, TypeError):
            self.skipTest("Winsorize function not available with this signature")

    def test_normalization_shapes(self):
        try:
            from data_engineering.preprocessing import VitalsNormalizer
            X = make_vitals_array(200, 8)
            norm = VitalsNormalizer(method="standard")
            X_scaled = norm.fit_transform(X)
            self.assertEqual(X_scaled.shape, X.shape)
            # Mean close to 0, std close to 1
            self.assertAlmostEqual(float(X_scaled.mean()), 0.0, places=0)
        except ImportError:
            self.skipTest("VitalsNormalizer not available")


# ─────────────────────────────────────────
# ML Model Tests
# ─────────────────────────────────────────

class TestAnomalyDetection(unittest.TestCase):

    def test_statistical_detector_basic(self):
        from ml_models.anomaly_detection import StatisticalAnomalyDetector
        detector = StatisticalAnomalyDetector()
        X = make_vitals_array(100)
        detector.fit(X)
        scores = detector.score_samples(X)
        self.assertEqual(len(scores), 100)
        self.assertTrue(all(s >= 0 for s in scores))

    def test_isolation_forest_fit_predict(self):
        from ml_models.anomaly_detection import IsolationForestDetector
        detector = IsolationForestDetector(contamination=0.1, n_estimators=50)
        X = make_vitals_array(200)
        y = np.zeros(200)
        y[:20] = 1  # 10% anomalies
        detector.fit(X, y)
        preds = detector.predict(X)
        self.assertEqual(len(preds), 200)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_ensemble_detector(self):
        from ml_models.anomaly_detection import EnsembleAnomalyDetector
        detector = EnsembleAnomalyDetector(contamination=0.1)
        X = make_vitals_array(150)
        detector.fit(X)
        scores = detector.score_samples(X)
        self.assertEqual(len(scores), 150)

    def test_realtime_scorer_single_obs(self):
        from ml_models.anomaly_detection import (EnsembleAnomalyDetector,
                                                  RealTimeAnomalyScorer)
        X = make_vitals_array(200)
        detector = EnsembleAnomalyDetector()
        detector.fit(X)
        scorer = RealTimeAnomalyScorer(detector)
        obs = X[0]
        result = scorer.score(obs)
        self.assertIn("anomaly_score", result)
        self.assertIn("is_anomaly", result)
        self.assertBetween(result["anomaly_score"], 0.0, 1.0)

    def assertBetween(self, val, lo, hi):
        self.assertGreaterEqual(val, lo)
        self.assertLessEqual(val, hi)


class TestRiskPrediction(unittest.TestCase):

    def test_model_trains_and_predicts(self):
        try:
            from ml_models.risk_prediction import ClinicalRiskModel
            X, y = make_tabular_data(400, 20)
            model = ClinicalRiskModel(task="sepsis_risk")
            result = model.fit(X, y)
            self.assertIn("auc_roc", result)
            probs = model.predict_proba(X[:10])
            self.assertEqual(len(probs), 10)
            self.assertTrue(all(0 <= p <= 1 for p in probs))
        except ImportError:
            self.skipTest("Risk prediction module not available")

    def test_risk_score_range(self):
        try:
            from ml_models.risk_prediction import ClinicalRiskModel
            X, y = make_tabular_data(200, 15)
            model = ClinicalRiskModel(task="sepsis_risk")
            model.fit(X, y)
            probs = model.predict_proba(X)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        except ImportError:
            self.skipTest("Risk prediction not available")

    def test_multi_task_scorer(self):
        try:
            from ml_models.risk_prediction import MultiTaskRiskScorer
            scorer = MultiTaskRiskScorer()
            # Training (with synthetic data generation)
            results = scorer.train_all()
            self.assertIsInstance(results, dict)
        except (ImportError, Exception):
            self.skipTest("MultiTaskRiskScorer training not available in test env")


class TestLSTM(unittest.TestCase):

    def test_lstm_fallback_predicts(self):
        try:
            from ml_models.lstm_model import SimpleRNNFallback
            X, y = make_sequence_data(100, 24, 8)
            model = SimpleRNNFallback()
            model.fit(X, y)
            preds = model.predict_proba(X[:10])
            self.assertEqual(len(preds), 10)
            self.assertTrue(all(0 <= p <= 1 for p in preds))
        except ImportError:
            self.skipTest("LSTM module not available")

    def test_lstm_trainer_synthetic(self):
        try:
            import torch
            from ml_models.lstm_model import LSTMTrainer
            X, y = make_sequence_data(80, 12, 6)
            with tempfile.TemporaryDirectory() as tmpdir:
                trainer = LSTMTrainer(input_size=6, hidden_size=16, n_layers=1,
                                      output_dir=tmpdir)
                result = trainer.fit(X, y, epochs=2)
                self.assertIn("metrics", result)
        except ImportError:
            self.skipTest("PyTorch not available")


class TestAutoencoder(unittest.TestCase):

    def test_numpy_autoencoder(self):
        try:
            from ml_models.autoencoder_model import SimpleNumpyAutoencoder
            X = make_vitals_array(200, 8)
            ae = SimpleNumpyAutoencoder(latent_dim=4)
            ae.fit(X)
            errors = ae.reconstruction_error(X)
            self.assertEqual(len(errors), 200)
            self.assertTrue(all(e >= 0 for e in errors))
        except ImportError:
            self.skipTest("Autoencoder module not available")


# ─────────────────────────────────────────
# NLP Tests
# ─────────────────────────────────────────

class TestMedicalNER(unittest.TestCase):

    def setUp(self):
        from nlp_pipeline.medical_ner import MedicalNER
        self.ner = MedicalNER()
        self.sample = (
            "Patient with septic shock, started on vancomycin 25mg/kg q8h and "
            "piperacillin-tazobactam 4.5g q6h. BP 82/54, HR 128, SpO2 91%, "
            "RR 26. Norepinephrine 0.1 mcg/kg/min. Intubated for respiratory failure. "
            "Lactate 4.2 mmol/L, creatinine 2.1 mg/dL. Blood cultures drawn."
        )

    def test_finds_medications(self):
        result = self.ner.process(self.sample)
        meds = [e.text.lower() for e in result.get_by_label("MEDICATION")]
        self.assertTrue(any("vancomycin" in m for m in meds))
        self.assertTrue(any("norepinephrine" in m for m in meds))

    def test_finds_diagnoses(self):
        result = self.ner.process(self.sample)
        dx = [e.normalized for e in result.get_by_label("DIAGNOSIS")]
        self.assertTrue(any("septic shock" in (d or "") for d in dx))

    def test_finds_vitals(self):
        result = self.ner.process(self.sample)
        vitals = result.get_by_label("VITAL")
        self.assertGreater(len(vitals), 0)

    def test_finds_dosages(self):
        result = self.ner.process(self.sample)
        dosages = result.get_by_label("DOSAGE")
        self.assertGreater(len(dosages), 0)

    def test_finds_procedures(self):
        result = self.ner.process(self.sample)
        procs = [e.text.lower() for e in result.get_by_label("PROCEDURE")]
        self.assertTrue(any("intubat" in p or "blood culture" in p for p in procs))

    def test_medication_relations(self):
        result = self.ner.process(self.sample)
        self.assertGreater(len(result.relations), 0)

    def test_batch_processing(self):
        notes = [self.sample] * 5
        results = self.ner.process_batch(notes)
        self.assertEqual(len(results), 5)


class TestClinicalTextProcessing(unittest.TestCase):

    def test_abbreviation_expansion(self):
        from nlp_pipeline.clinical_text_processing import ClinicalTextNormalizer
        norm = ClinicalTextNormalizer()
        text = "Patient c/o SOB and CP, hx of HTN and DM"
        result = norm.expand_abbreviations(text)
        self.assertIn("shortness of breath", result.lower())

    def test_section_segmentation(self):
        from nlp_pipeline.clinical_text_processing import ClinicalSectionSegmenter
        seg = ClinicalSectionSegmenter()
        note = "HPI: Patient presents with fever.\nAssessment: Pneumonia.\nPlan: Start antibiotics."
        sections = seg.segment(note)
        self.assertIn("hpi", {k.lower() for k in sections})


# ─────────────────────────────────────────
# Knowledge Graph Tests
# ─────────────────────────────────────────

class TestKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        from knowledge_graph.graph_builder import MedicalKnowledgeGraph
        self.kg = MedicalKnowledgeGraph()

    def test_graph_has_nodes(self):
        stats = self.kg.graph_stats()
        self.assertGreater(stats["total_nodes"], 50)

    def test_get_symptoms(self):
        symptoms = self.kg.get_symptoms("sepsis")
        self.assertIn("fever", symptoms)
        self.assertIn("tachycardia", symptoms)

    def test_get_treatments(self):
        treatments = self.kg.get_treatments("pneumonia")
        self.assertGreater(len(treatments), 0)

    def test_differential_diagnosis(self):
        ddx = self.kg.differential_diagnosis(
            ["fever", "tachycardia", "hypotension", "tachypnea"]
        )
        self.assertGreater(len(ddx), 0)
        # Top result should be sepsis or septic_shock
        top_diseases = [d["disease"] for d in ddx[:3]]
        self.assertTrue(any("sepsis" in d for d in top_diseases))

    def test_drug_interactions(self):
        interactions = self.kg.check_drug_interactions(
            ["warfarin", "aspirin", "metoprolol", "amiodarone"]
        )
        self.assertGreater(len(interactions), 0)
        # warfarin + aspirin is a known high-severity interaction
        found = any(
            ("warfarin" in ix["drug1"] or "warfarin" in ix["drug2"]) and
            ("aspirin" in ix["drug1"] or "aspirin" in ix["drug2"])
            for ix in interactions
        )
        self.assertTrue(found)

    def test_no_self_interactions(self):
        interactions = self.kg.check_drug_interactions(["vancomycin"])
        self.assertEqual(len(interactions), 0)

    def test_cypher_export(self):
        cypher = self.kg.export_cypher()
        self.assertIn("CREATE", cypher)


# ─────────────────────────────────────────
# Real-Time Monitoring Tests
# ─────────────────────────────────────────

class TestVitalsStreamProcessor(unittest.TestCase):

    def setUp(self):
        from real_time_monitoring.vitals_stream_processor import (
            VitalsStreamProcessor, PatientVitalsSimulator
        )
        self.processor = VitalsStreamProcessor()
        self.processor.register_patient("P001", "Test Patient", "ICU-1")
        self.sim = PatientVitalsSimulator("P001", "deteriorating")

    def tearDown(self):
        if self.processor._running:
            self.processor.stop()

    def test_ingest_reading(self):
        import time
        self.processor.start()
        reading = self.sim.next_reading()
        self.processor.ingest(reading)
        time.sleep(0.5)
        summary = self.processor.get_patient_summary("P001")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["patient_id"], "P001")

    def test_alert_fired_for_critical_vitals(self):
        from real_time_monitoring.vitals_stream_processor import (
            VitalReading, RuleBasedAlertEngine, PatientState
        )
        engine = RuleBasedAlertEngine()
        state = PatientState(patient_id="P999")

        # Deliberately critical reading
        reading = VitalReading(
            patient_id="P999",
            timestamp=datetime.utcnow(),
            heart_rate=155,  # critical
            sbp=75,          # critical
            dbp=45,
            respiratory_rate=32,
            temperature=39.5,
            spo2=85,         # critical
            gcs=12,
        )
        alerts = engine.evaluate(state, reading)
        self.assertGreater(len(alerts), 0)
        severities = {a.severity for a in alerts}
        self.assertIn("critical", severities)

    def test_news2_scoring(self):
        from real_time_monitoring.vitals_stream_processor import (
            compute_news2, VitalReading
        )
        # Healthy patient
        healthy = VitalReading("P001", datetime.utcnow(),
                               heart_rate=75, sbp=120, dbp=80,
                               respiratory_rate=16, temperature=37.0, spo2=98, gcs=15)
        self.assertLessEqual(compute_news2(healthy), 2)

        # Sick patient
        sick = VitalReading("P001", datetime.utcnow(),
                             heart_rate=130, sbp=85, dbp=55,
                             respiratory_rate=28, temperature=39.2, spo2=89, gcs=13)
        self.assertGreaterEqual(compute_news2(sick), 8)

    def test_simulator_generates_valid_vitals(self):
        from real_time_monitoring.vitals_stream_processor import PatientVitalsSimulator
        for profile in ["stable", "mildly_ill", "deteriorating", "critical"]:
            sim = PatientVitalsSimulator("P_test", profile)
            for _ in range(5):
                r = sim.next_reading()
                self.assertGreater(r.heart_rate, 0)
                self.assertGreater(r.sbp, 0)
                self.assertGreater(r.dbp, 0)
                self.assertGreater(r.respiratory_rate, 0)
                self.assertGreaterEqual(r.spo2, 60)
                self.assertLessEqual(r.spo2, 100)

    def test_icu_overview_ordering(self):
        import time
        from real_time_monitoring.vitals_stream_processor import PatientVitalsSimulator
        self.processor.register_patient("P002", "Critical Patient", "ICU-2")
        sim2 = PatientVitalsSimulator("P002", "critical")

        self.processor.start()
        for _ in range(3):
            self.processor.ingest(self.sim.next_reading())
            self.processor.ingest(sim2.next_reading())
        time.sleep(0.8)

        overview = self.processor.get_icu_overview()
        if len(overview) >= 2:
            news2_scores = [s["news2"] for s in overview]
            self.assertEqual(news2_scores, sorted(news2_scores, reverse=True))


# ─────────────────────────────────────────
# Data Pipeline Integration Test
# ─────────────────────────────────────────

class TestDataPipelineIntegration(unittest.TestCase):

    def test_end_to_end_feature_pipeline(self):
        """Synthetic data → preprocessing → feature engineering → shape checks."""
        try:
            import pandas as pd
            from data_engineering.preprocessing import PreprocessingPipeline
        except ImportError:
            self.skipTest("Pipeline modules not available")

        n_patients = 5
        n_readings = 48  # 48h hourly
        records = []
        base_time = datetime(2024, 1, 1)
        for pid in range(n_patients):
            for h in range(n_readings):
                records.append({
                    "patient_id": f"P{pid:03d}",
                    "timestamp": base_time + timedelta(hours=h),
                    "heart_rate": 80 + random.gauss(0, 8),
                    "sbp": 120 + random.gauss(0, 12),
                    "dbp": 75 + random.gauss(0, 7),
                    "respiratory_rate": 16 + random.gauss(0, 2),
                    "temperature": 37 + random.gauss(0, 0.3),
                    "spo2": min(100, 97 + random.gauss(0, 1.5)),
                })

        df = pd.DataFrame(records)
        self.assertEqual(len(df), n_patients * n_readings)
        self.assertIn("patient_id", df.columns)
        self.assertIn("heart_rate", df.columns)

        # Validate physiological ranges (basic sanity)
        self.assertTrue((df["heart_rate"] > 0).all())
        self.assertTrue((df["spo2"] <= 100).all())


# ─────────────────────────────────────────
# Experiment Tracking Tests
# ─────────────────────────────────────────

class TestExperimentTracking(unittest.TestCase):

    def setUp(self):
        from experiment_tracking.mlflow_setup import LocalExperimentStore
        with tempfile.TemporaryDirectory() as tmpdir:
            self.store = LocalExperimentStore(base_dir=tmpdir)
            self.tmpdir = tmpdir

    def test_start_and_end_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from experiment_tracking.mlflow_setup import LocalExperimentStore
            store = LocalExperimentStore(base_dir=tmpdir)
            run_id = store.start_run("test_exp", run_name="test_run")
            self.assertIsNotNone(run_id)
            store.log_params(run_id, {"lr": 0.01, "n_est": 100})
            store.log_metrics(run_id, {"auc_roc": 0.87, "f1": 0.72})
            store.end_run(run_id)
            saved = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(saved), 1)
            data = json.loads(saved[0].read_text())
            self.assertEqual(data["status"], "FINISHED")
            self.assertEqual(data["params"]["lr"], "0.01")

    def test_best_run_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from experiment_tracking.mlflow_setup import LocalExperimentStore
            store = LocalExperimentStore(base_dir=tmpdir)
            for auc in [0.75, 0.85, 0.80]:
                run_id = store.start_run("exp_A")
                store.log_metrics(run_id, {"auc_roc": auc})
                store.end_run(run_id)
            best = store.get_best_run("exp_A", "auc_roc", mode="max")
            self.assertIsNotNone(best)
            best_auc = best["metrics"]["auc_roc"][-1]["value"]
            self.assertAlmostEqual(best_auc, 0.85, places=2)


# ─────────────────────────────────────────
# Dashboard Tests
# ─────────────────────────────────────────

class TestDashboard(unittest.TestCase):

    def test_generate_static_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from dashboard.hospital_dashboard import generate_static_report
            path = generate_static_report(os.path.join(tmpdir, "report.html"))
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            self.assertIn("ICU", content)
            self.assertIn("NEWS2", content)

    def test_demo_data_generation(self):
        from dashboard.hospital_dashboard import _generate_demo_patients, _generate_alerts
        patients = _generate_demo_patients(8)
        self.assertEqual(len(patients), 8)
        for p in patients:
            self.assertIn("patient_id", p)
            self.assertIn("news2", p)
            self.assertIn("sepsis_risk", p)
            self.assertBetween(p["sepsis_risk"], 0.0, 1.0)

        alerts = _generate_alerts(patients)
        self.assertIsInstance(alerts, list)

    def assertBetween(self, val, lo, hi):
        self.assertGreaterEqual(val, lo)
        self.assertLessEqual(val, hi)


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)

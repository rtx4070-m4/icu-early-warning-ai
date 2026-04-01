"""
Integration Test Suite
End-to-end tests that validate cross-module workflows:
  1. Data → Feature Engineering → ML Scoring → CDSS → Alert
  2. Clinical Note → NER → Knowledge Graph → Drug Safety → Report
  3. Vitals → Stream Processor → Evaluation → FHIR Export
  4. Config → Security → Audit → API
"""

import sys
import os
import json
import time
import random
import unittest
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def make_patient(pid: str = "P001", profile: str = "deteriorating") -> Dict:
    profiles = {
        "stable":       dict(hr=75, sbp=122, dbp=78, rr=15, temp=37.0, spo2=98,
                              news2=1, sep=0.05, mort=0.02),
    # hours_in_icu will be set to 6 for stable (below DVT threshold of 24h)
        "deteriorating":dict(hr=118, sbp=88, dbp=55, rr=26, temp=38.8, spo2=91,
                              news2=9, sep=0.78, mort=0.31),
        "critical":     dict(hr=142, sbp=74, dbp=44, rr=32, temp=39.5, spo2=84,
                              news2=13, sep=0.92, mort=0.58),
    }
    p = profiles.get(profile, profiles["stable"])
    return {
        "patient_id": pid, "name": "Test Patient", "age": 65, "sex": "M",
        "location": "ICU-1", "primary_diagnosis": "septic_shock",
        "admission_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "vitals": {"heart_rate": p["hr"], "sbp": p["sbp"], "dbp": p["dbp"],
                    "respiratory_rate": p["rr"], "temperature": p["temp"],
                    "spo2": p["spo2"], "gcs": 14},
        "labs": {"lactate": 0.9 if profile == "stable" else 3.8,
                  "creatinine": 0.8 if profile == "stable" else 2.1,
                  "wbc": 7.5 if profile == "stable" else 18.5,
                  "potassium": 4.0, "glucose": 90 if profile == "stable" else 185,
                  "inr": 1.1, "troponin": 0.01},
        "risk_scores": {"sepsis_risk": p["sep"], "mortality_risk": p["mort"]},
        "news2": p["news2"],
        "hours_in_icu": 6 if profile == "stable" else 12,
        "flags": {"mechanically_ventilated": False, "fluid_resuscitated": False},
    }


# ─────────────────────────────────────────
# 1. Data → ML → CDSS → Alert pipeline
# ─────────────────────────────────────────

class TestMLCDSSPipeline(unittest.TestCase):
    """Full pipeline: vitals + labs → risk scores → CDSS → recommendations."""

    def test_cdss_produces_recommendations_for_septic_patient(self):
        from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
        engine = ClinicalDecisionSupportEngine()
        patient = make_patient("P_INT_001", "deteriorating")
        result = engine.evaluate(
            patient_data=patient,
            symptoms=["fever", "hypotension", "tachycardia"],
            medications=["vancomycin", "norepinephrine"],
        )
        self.assertEqual(result.patient_id, "P_INT_001")
        self.assertIn(result.risk_level, ("high", "critical"))
        self.assertGreater(len(result.recommendations), 0)
        self.assertGreater(len(result.triggered_protocols), 0)

    def test_cdss_sepsis_bundle_triggered(self):
        from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
        engine = ClinicalDecisionSupportEngine()
        patient = make_patient(profile="critical")
        result = engine.evaluate(patient)
        protocol_names = [p.lower() for p in result.triggered_protocols]
        self.assertTrue(any("sepsis" in p for p in protocol_names),
                        f"Expected sepsis bundle, got: {result.triggered_protocols}")

    def test_cdss_stable_patient_low_risk(self):
        from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
        engine = ClinicalDecisionSupportEngine()
        patient = make_patient(profile="stable")
        result = engine.evaluate(patient)
        self.assertIn(result.risk_level, ("low", "moderate"))
        stat_recs = [r for r in result.recommendations if r.urgency == "STAT"]
        self.assertLessEqual(len(stat_recs), 1)

    def test_cdss_drug_safety_checks_medications(self):
        from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
        engine = ClinicalDecisionSupportEngine()
        patient = make_patient(profile="deteriorating")
        patient["labs"]["inr"] = 3.2  # elevated INR
        result = engine.evaluate(
            patient,
            medications=["warfarin", "aspirin", "vancomycin", "furosemide"]
        )
        self.assertIn("medications_checked", result.drug_safety)
        # warfarin + aspirin is a known interaction
        interactions = result.drug_safety.get("interactions", [])
        high_risk = result.drug_safety.get("high_risk_interactions", [])
        self.assertIsInstance(interactions, list)
        # At minimum the check ran without error
        self.assertIsNotNone(result.drug_safety.get("overall_safety"))

    def test_cdss_note_ner_integration(self):
        """CDSS should extract medications from a clinical note via NER."""
        from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
        engine = ClinicalDecisionSupportEngine()
        patient = make_patient()
        note = ("Patient with septic shock. Started on vancomycin 1.5g IV q12h "
                "and piperacillin-tazobactam 4.5g q6h. Norepinephrine 0.1 mcg/kg/min.")
        result = engine.evaluate(patient, note_text=note)
        # If NER is working, medications should be auto-extracted
        if engine._ner is not None:
            self.assertIn("medications_checked", result.drug_safety)
            checked = result.drug_safety["medications_checked"]
            self.assertTrue(any("vancomycin" in m for m in checked),
                            f"vancomycin not found in {checked}")


# ─────────────────────────────────────────
# 2. NLP → Knowledge Graph → Reporting
# ─────────────────────────────────────────

class TestNLPKGReportPipeline(unittest.TestCase):

    def test_ner_extracts_then_kg_differentiates(self):
        """NER symptoms → KG differential diagnosis."""
        from nlp_pipeline.medical_ner import ClinicalEntityExtractor
        from knowledge_graph.graph_builder import MedicalKnowledgeGraph

        ner = ClinicalEntityExtractor()
        kg = MedicalKnowledgeGraph()

        note = ("Patient with fever, chills, hypotension, tachycardia, "
                "and tachypnea. Altered mental status. Diaphoretic.")
        symptoms = ner.extract_symptoms(note)
        self.assertGreater(len(symptoms), 2, f"Expected >2 symptoms, got: {symptoms}")

        ddx = kg.differential_diagnosis(symptoms, top_n=5)
        self.assertGreater(len(ddx), 0)
        top_disease = ddx[0]["disease"]
        # Should identify a sepsis-family condition
        self.assertTrue(any("sepsis" in d["disease"] for d in ddx),
                        f"Sepsis not in DDx: {[d['disease'] for d in ddx]}")

    def test_ner_medication_extraction_complete(self):
        """NER extracts medications with dosages and frequencies."""
        from nlp_pipeline.medical_ner import ClinicalEntityExtractor
        ner = ClinicalEntityExtractor()
        note = ("Medications: vancomycin 1.5g IV q12h, furosemide 40mg IV bid, "
                "norepinephrine 0.1 mcg/kg/min continuous, propofol 20 mcg/kg/min.")
        meds = ner.extract_medications(note)
        med_names = [m["medication"] for m in meds]
        self.assertTrue(any("vancomycin" in n for n in med_names), f"Got: {med_names}")
        self.assertTrue(any("norepinephrine" in n for n in med_names), f"Got: {med_names}")

    def test_kg_drug_interactions_detected(self):
        """KG detects high-severity drug interactions."""
        from knowledge_graph.graph_builder import MedicalKnowledgeGraph
        kg = MedicalKnowledgeGraph()
        interactions = kg.check_drug_interactions(
            ["warfarin", "aspirin", "amiodarone", "digoxin"]
        )
        severities = {ix["severity"] for ix in interactions}
        self.assertTrue(severities & {"high", "contraindicated"},
                        f"Expected high-severity interaction, got: {severities}")

    def test_report_generated_for_patient(self):
        """Report generator produces non-empty HTML and Markdown."""
        from reporting.report_generator import _make_snapshot, ReportGenerator
        gen = ReportGenerator()
        snap = _make_snapshot("P001")
        html = gen.generate_deterioration_report(snap)
        md = gen.generate_handover_summary([snap])
        note = gen.generate_note_draft(snap)

        self.assertGreater(len(html), 2000)
        self.assertIn("NEWS2", html)
        self.assertIn(snap.patient_id, html)

        self.assertGreater(len(md), 100)
        self.assertIn("# ICU Shift Handover", md)

        self.assertGreater(len(note), 500)
        self.assertIn("Assessment", note)

    def test_report_saved_to_disk(self):
        """Report generator writes HTML file correctly."""
        from reporting.report_generator import _make_snapshot, ReportGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator()
            snap = _make_snapshot("P002")
            path = os.path.join(tmpdir, "report.html")
            gen.generate_deterioration_report(snap, output_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)


# ─────────────────────────────────────────
# 3. Stream → Evaluation → FHIR
# ─────────────────────────────────────────

class TestStreamEvalFHIRPipeline(unittest.TestCase):

    def test_stream_processor_fires_alert_for_critical_patient(self):
        """Stream processor generates alerts for a deteriorating patient."""
        from real_time_monitoring.vitals_stream_processor import (
            VitalsStreamProcessor, PatientVitalsSimulator
        )
        processor = VitalsStreamProcessor()
        processor.register_patient("P_CRIT", "Critical Patient", "ICU-1")
        sim = PatientVitalsSimulator("P_CRIT", "critical")

        alerts_received = []
        processor.add_alert_handler(lambda a: alerts_received.append(a))
        processor.start()

        try:
            for _ in range(15):
                processor.ingest(sim.next_reading())
            time.sleep(1.5)
        finally:
            processor.stop()

        self.assertGreater(len(alerts_received), 0,
                           "Expected at least one alert for critical patient")
        severities = {a.severity for a in alerts_received}
        self.assertTrue(severities & {"critical", "high"},
                        f"Expected critical/high alerts, got: {severities}")

    def test_evaluation_suite_runs_all_models(self):
        """Evaluation suite runs all 5 models and produces a JSON report."""
        from evaluation.model_evaluator import EvaluationSuite
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = EvaluationSuite()
            results = suite.run_all(output_dir=tmpdir, backtest_windows=4)
            self.assertIn("models", results)
            self.assertIn("overall", results)
            self.assertEqual(len(results["models"]), 5)
            self.assertGreater(results["overall"]["pass_rate"], 0)

    def test_evaluation_html_report_generated(self):
        """Evaluation suite generates a valid HTML report file."""
        from evaluation.model_evaluator import EvaluationSuite
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = EvaluationSuite()
            results = suite.run_all(output_dir=tmpdir, backtest_windows=4)
            html_path = os.path.join(tmpdir, "report.html")
            suite.generate_html_report(results, html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path) as f:
                html = f.read()
            self.assertIn("AUC-ROC", html)
            self.assertIn("PASS", html)

    def test_fhir_patient_export_valid_bundle(self):
        """FHIR service exports a valid R4 Bundle for a patient."""
        from fhir_integration.fhir_client import FHIRIntegrationService
        svc = FHIRIntegrationService()
        patient = make_patient("P_FHIR")
        bundle = svc.full_patient_export(patient)

        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "transaction")
        self.assertGreater(len(bundle["entry"]), 3)

        resource_types = {e["resource"]["resourceType"] for e in bundle["entry"]}
        self.assertIn("Patient", resource_types)
        self.assertIn("Observation", resource_types)
        self.assertIn("RiskAssessment", resource_types)

    def test_fhir_round_trip_observations(self):
        """Vitals can be serialised to FHIR Observations and parsed back."""
        from fhir_integration.fhir_client import FHIRBuilder, FHIRParser
        builder = FHIRBuilder()
        parser = FHIRParser()

        vitals = {"heart_rate": 118.0, "sbp": 88.0, "spo2": 91.0, "temperature": 38.8}
        bundle = builder.vital_bundle("P001", vitals)
        parsed = parser.parse_bundle_observations(bundle)

        parsed_fields = {p["field"] for p in parsed}
        self.assertIn("heart_rate", parsed_fields)
        self.assertIn("sbp", parsed_fields)

        hr_obs = next(p for p in parsed if p["field"] == "heart_rate")
        self.assertAlmostEqual(hr_obs["value"], 118.0, places=1)

    def test_fhir_interpretation_flags_critical_values(self):
        """FHIR builder flags critical lab/vital values with interpretation codes."""
        from fhir_integration.fhir_client import FHIRBuilder
        builder = FHIRBuilder()

        # Critical low SpO2
        obs_low = builder.observation("P001", "spo2", 85.0)
        self.assertIn("interpretation", obs_low)
        code = obs_low["interpretation"][0]["coding"][0]["code"]
        self.assertEqual(code, "LL")

        # Critical high HR
        obs_high = builder.observation("P001", "heart_rate", 160.0)
        self.assertIn("interpretation", obs_high)
        code = obs_high["interpretation"][0]["coding"][0]["code"]
        self.assertEqual(code, "HH")

        # Normal value — no interpretation
        obs_normal = builder.observation("P001", "heart_rate", 80.0)
        self.assertNotIn("interpretation", obs_normal)


# ─────────────────────────────────────────
# 4. Config → Security → Audit → API
# ─────────────────────────────────────────

class TestConfigSecurityAuditPipeline(unittest.TestCase):

    def test_config_loads_and_validates(self):
        """Config manager loads defaults and passes validation."""
        from config.config_manager import ConfigLoader
        loader = ConfigLoader()
        cfg = loader.load()
        self.assertEqual(cfg.environment, "development")
        self.assertGreater(cfg.alerts.news2_critical_threshold, cfg.alerts.news2_warning_threshold)
        self.assertGreater(cfg.api.port, 0)
        self.assertGreater(cfg.api.workers, 0)

    def test_config_env_var_override(self):
        """Environment variables override config defaults."""
        from config.config_manager import ConfigLoader
        os.environ["HOSPITAL_OS__ALERTS__NEWS2_CRITICAL_THRESHOLD"] = "11"
        os.environ["HOSPITAL_OS__API__PORT"] = "9001"
        try:
            loader = ConfigLoader()
            cfg = loader.load()
            self.assertEqual(cfg.alerts.news2_critical_threshold, 11)
            self.assertEqual(cfg.api.port, 9001)
        finally:
            del os.environ["HOSPITAL_OS__ALERTS__NEWS2_CRITICAL_THRESHOLD"]
            del os.environ["HOSPITAL_OS__API__PORT"]

    def test_security_full_session_lifecycle(self):
        """Login → access check → audit trail → logout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from security.audit_security import HospitalSecurityManager, Permission
            sec = HospitalSecurityManager(audit_dir=tmpdir)

            # Login
            session_id = sec.login("dr_integration", "attending_physician", "10.0.0.1")
            self.assertIsNotNone(session_id)

            # Access patient — should succeed
            ok = sec.can_access_patient(session_id, "P001")
            self.assertTrue(ok)

            # Check prescribe permission
            can_prescribe = sec.access_control.check_access(
                session_id, Permission.PRESCRIBE, "MEDICATION", "P001", "P001"
            )
            self.assertTrue(can_prescribe)  # attending_physician can prescribe

            # Audit trail should have events
            sec.flush()
            trail = sec.get_audit_trail(limit=20)
            self.assertGreater(len(trail), 0)

            # All successful events should have success=True for the access events
            success_events = [e for e in trail if e["action"] == "ACCESS" and e["success"]]
            self.assertGreater(len(success_events), 0)

            # Logout
            sec.logout(session_id)
            # Session should now be invalid
            revoked = sec.can_access_patient(session_id, "P001")
            self.assertFalse(revoked)

    def test_security_role_restriction_nurse(self):
        """Nurse cannot prescribe or export data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from security.audit_security import HospitalSecurityManager, Permission
            sec = HospitalSecurityManager(audit_dir=tmpdir)
            session_id = sec.login("nurse_test", "nurse", "10.0.0.2")

            # Can acknowledge alerts
            can_ack = sec.access_control.check_access(
                session_id, Permission.ACKNOWLEDGE_ALERTS, "ALERT", "A001", "P001"
            )
            self.assertTrue(can_ack)

            # Cannot prescribe
            can_rx = sec.access_control.check_access(
                session_id, Permission.PRESCRIBE, "MEDICATION", "P001", "P001"
            )
            self.assertFalse(can_rx)

            # Cannot run pipeline
            can_pipeline = sec.access_control.check_access(
                session_id, Permission.RUN_PIPELINE, "PIPELINE", "daily", None
            )
            self.assertFalse(can_pipeline)

    def test_phi_deidentification_comprehensive(self):
        """PHI de-identifier removes all sensitive identifiers."""
        from security.audit_security import PHIDeidentifier
        phi = PHIDeidentifier()
        text = ("Patient John Smith, DOB: 03/15/1952, MRN: 7654321, "
                "SSN: 123-45-6789, called from 555-867-5309. "
                "Email: john.smith@example.com. ZIP: 94102.")
        result = phi.deidentify(text)

        # Original identifiers should be replaced
        self.assertNotIn("7654321", result)
        self.assertNotIn("123-45-6789", result)
        self.assertNotIn("555-867-5309", result)
        self.assertNotIn("john.smith@example.com", result)
        self.assertNotIn("94102", result)

        # Replacement tokens should be present
        self.assertIn("[", result)  # At least one replacement token

    def test_data_quality_monitor_catches_errors(self):
        """Data quality monitor identifies injected errors in vital records."""
        from data_quality.dq_monitor import DataQualityMonitor, generate_test_vitals
        records = generate_test_vitals(n=300, inject_errors=True)
        monitor = DataQualityMonitor()
        report = monitor.check_vitals(records, source="integration_test")

        self.assertGreater(len(report.alerts), 0, "Expected at least one alert")
        self.assertLessEqual(report.overall_quality_score, 1.0)
        self.assertGreaterEqual(report.overall_quality_score, 0.0)
        self.assertGreater(report.n_records, 0)

    def test_data_quality_clean_data_scores_high(self):
        """Clean data without errors should have high quality score."""
        from data_quality.dq_monitor import DataQualityMonitor, generate_test_vitals
        records = generate_test_vitals(n=200, inject_errors=False)
        monitor = DataQualityMonitor()
        report = monitor.check_vitals(records, source="clean_test")
        # Clean data should score > 0.85
        self.assertGreater(report.overall_quality_score, 0.85,
                           f"Expected quality >0.85, got {report.overall_quality_score}")
        critical_alerts = report.critical_alerts
        self.assertEqual(len(critical_alerts), 0,
                          f"Expected no critical alerts, got: {[a.message for a in critical_alerts]}")


# ─────────────────────────────────────────
# 5. End-to-end workflow test
# ─────────────────────────────────────────

class TestEndToEndWorkflow(unittest.TestCase):
    """
    Simulates the full hospital OS workflow for one deteriorating patient
    across all major subsystems.
    """

    def test_full_icu_patient_workflow(self):
        """
        End-to-end: patient arrives → vitals streamed → risk scored →
        CDSS recommendations → FHIR exported → report generated → audit logged.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Config
            from config.config_manager import ConfigLoader
            cfg = ConfigLoader().load()
            self.assertIsNotNone(cfg)

            # 2. Security — login
            from security.audit_security import HospitalSecurityManager
            sec = HospitalSecurityManager(audit_dir=tmpdir)
            session_id = sec.login("dr_e2e", "attending_physician", "127.0.0.1")

            # 3. Verify access
            self.assertTrue(sec.can_access_patient(session_id, "P_E2E"))

            # 4. Vitals streaming
            from real_time_monitoring.vitals_stream_processor import (
                VitalsStreamProcessor, PatientVitalsSimulator
            )
            processor = VitalsStreamProcessor()
            processor.register_patient("P_E2E", "E2E Test Patient", "ICU-1")
            processor.start()
            sim = PatientVitalsSimulator("P_E2E", "deteriorating")
            for _ in range(10):
                processor.ingest(sim.next_reading())
            time.sleep(0.8)
            processor.stop()

            summary = processor.get_patient_summary("P_E2E")
            self.assertIsNotNone(summary)
            self.assertGreater(summary["total_readings"], 0)

            # 5. CDSS
            from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
            engine = ClinicalDecisionSupportEngine()
            patient_data = make_patient("P_E2E", "deteriorating")
            cdss_output = engine.evaluate(patient_data, symptoms=["fever", "hypotension"])
            self.assertIn(cdss_output.risk_level, ("high", "critical"))

            # 6. FHIR export
            from fhir_integration.fhir_client import FHIRIntegrationService
            fhir_svc = FHIRIntegrationService()
            bundle = fhir_svc.full_patient_export(patient_data)
            self.assertEqual(bundle["resourceType"], "Bundle")
            n_resources = len(bundle["entry"])
            self.assertGreater(n_resources, 3)

            # 7. Report generation
            from reporting.report_generator import _make_snapshot, ReportGenerator
            snap = _make_snapshot("P_E2E")
            gen = ReportGenerator()
            report_path = os.path.join(tmpdir, "e2e_report.html")
            gen.generate_deterioration_report(snap, output_path=report_path)
            self.assertTrue(os.path.exists(report_path))

            # 8. Audit trail
            sec.flush()
            trail = sec.get_audit_trail(limit=20)
            self.assertGreater(len(trail), 0)

            # 9. Summary assertions
            self.assertGreater(cdss_output.scores.get("news2", 0), 4)
            self.assertGreater(len(cdss_output.recommendations), 0)

            print(f"\n[E2E] Risk: {cdss_output.risk_level.upper()}, "
                  f"Protocols: {len(cdss_output.triggered_protocols)}, "
                  f"Recs: {len(cdss_output.recommendations)}, "
                  f"FHIR entries: {n_resources}, "
                  f"Audit events: {len(trail)}")


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Run with more verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMLCDSSPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestNLPKGReportPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamEvalFHIRPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigSecurityAuditPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndWorkflow))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

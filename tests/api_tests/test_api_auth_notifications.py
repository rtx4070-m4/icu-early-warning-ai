"""
API Test Suite
Tests for the REST API server, auth middleware, and rate limiter.
Runs without a live server — tests logic directly via imports.
"""

import sys
import os
import json
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestJWTAuth(unittest.TestCase):
    """JWT token creation and verification."""

    def setUp(self):
        from api.auth import create_token, verify_token, extract_token
        self.create_token = create_token
        self.verify_token = verify_token
        self.extract_token = extract_token

    def test_create_and_verify_token(self):
        token = self.create_token("dr_smith", "attending_physician")
        payload = self.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "dr_smith")
        self.assertEqual(payload["role"], "attending_physician")

    def test_expired_token_rejected(self):
        # Create token with -1 minute expiry (already expired)
        token = self.create_token("user", "nurse", expiry_minutes=-1)
        result = self.verify_token(token)
        self.assertIsNone(result)

    def test_tampered_token_rejected(self):
        token = self.create_token("user", "nurse")
        parts = token.split(".")
        # Tamper with payload
        parts[1] = parts[1][:-2] + "XX"
        tampered = ".".join(parts)
        result = self.verify_token(tampered)
        self.assertIsNone(result)

    def test_malformed_token_rejected(self):
        self.assertIsNone(self.verify_token("not.a.valid.jwt.at.all"))
        self.assertIsNone(self.verify_token(""))
        self.assertIsNone(self.verify_token("only_one_part"))

    def test_extract_bearer_token(self):
        self.assertEqual(self.extract_token("Bearer mytoken123"), "mytoken123")
        self.assertIsNone(self.extract_token(None))
        self.assertIsNone(self.extract_token("Basic abc123"))
        self.assertIsNone(self.extract_token(""))

    def test_all_roles_create_valid_tokens(self):
        roles = ["attending_physician", "resident", "nurse", "data_scientist", "admin"]
        for role in roles:
            token = self.create_token(f"user_{role}", role)
            payload = self.verify_token(token)
            self.assertIsNotNone(payload, f"Token invalid for role: {role}")
            self.assertEqual(payload["role"], role)

    def test_token_contains_expiry(self):
        token = self.create_token("user", "nurse", expiry_minutes=60)
        payload = self.verify_token(token)
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)
        self.assertGreater(payload["exp"], payload["iat"])


class TestRateLimiter(unittest.TestCase):
    """Token bucket rate limiter."""

    def setUp(self):
        from api.auth import RateLimiter
        self.RateLimiter = RateLimiter

    def test_allows_requests_under_limit(self):
        rl = self.RateLimiter(requests_per_minute=10)
        for _ in range(10):
            self.assertTrue(rl.is_allowed("192.168.1.1"))

    def test_blocks_requests_over_limit(self):
        rl = self.RateLimiter(requests_per_minute=3)
        # Drain the bucket
        for _ in range(3):
            rl.is_allowed("10.0.0.1")
        # Should now be blocked
        self.assertFalse(rl.is_allowed("10.0.0.1"))

    def test_different_ips_independent_buckets(self):
        rl = self.RateLimiter(requests_per_minute=2)
        # Drain IP1
        rl.is_allowed("1.1.1.1")
        rl.is_allowed("1.1.1.1")
        self.assertFalse(rl.is_allowed("1.1.1.1"))
        # IP2 should still have tokens
        self.assertTrue(rl.is_allowed("2.2.2.2"))

    def test_remaining_tokens_tracked(self):
        rl = self.RateLimiter(requests_per_minute=10)
        self.assertEqual(rl.remaining("new.ip"), 10)
        rl.is_allowed("new.ip")
        self.assertEqual(rl.remaining("new.ip"), 9)


class TestNotificationService(unittest.TestCase):
    """Alert notification service (stub mode)."""

    def setUp(self):
        from notifications.alert_service import (
            NotificationService, ClinicalAlert
        )
        self.svc = NotificationService(cooldown_minutes=0)
        self.ClinicalAlert = ClinicalAlert
        import uuid
        self.uuid = uuid

    def _make_alert(self, severity="high", location="ICU-1"):
        return self.ClinicalAlert(
            alert_id=str(self.uuid.uuid4()),
            patient_id="P_TEST",
            patient_name="Test Patient",
            location=location,
            severity=severity,
            alert_type="DETERIORATION",
            message="Test alert message",
            news2=9,
        )

    def test_critical_alert_sent_to_all_channels(self):
        alert = self._make_alert("critical")
        results = self.svc.notify(alert, channels=["email", "sms", "slack", "pager"])
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.success for r in results))

    def test_stub_mode_always_succeeds(self):
        alert = self._make_alert("moderate")
        results = self.svc.notify(alert, channels=["email", "sms", "slack"])
        for r in results:
            self.assertTrue(r.success, f"Channel {r.channel} should succeed in stub mode")

    def test_cooldown_suppresses_duplicate_non_critical(self):
        svc = self.__class__.__bases__[0].__dict__["setUp"](self)
        from notifications.alert_service import NotificationService
        svc = NotificationService(cooldown_minutes=60)  # 60 min cooldown
        alert = self._make_alert("high")
        # First send
        r1 = svc.notify(alert, channels=["slack"])
        # Second send (should be suppressed)
        r2 = svc.notify(alert, channels=["slack"])
        self.assertGreater(len(r1), 0)
        self.assertEqual(len(r2), 0)  # suppressed by cooldown

    def test_notification_log_recorded(self):
        alert = self._make_alert("high")
        self.svc.notify(alert, channels=["slack"])
        log = self.svc.get_notification_log()
        self.assertGreater(len(log), 0)
        self.assertEqual(log[-1]["patient_id"], "P_TEST")

    def test_stats_tracked(self):
        for sev in ["critical", "high", "moderate"]:
            alert = self._make_alert(sev)
            alert.patient_id = f"P_{sev}"
            self.svc.notify(alert, channels=["slack"])
        stats = self.svc.get_stats()
        self.assertIn("total_notifications", stats)
        self.assertGreaterEqual(stats["total_notifications"], 3)

    def test_alert_html_body_contains_patient(self):
        alert = self._make_alert("critical")
        html = alert.html_body()
        self.assertIn("P_TEST", html)
        self.assertIn("CRITICAL", html)
        self.assertIn("Test Patient", html)


class TestSurvivalAnalysis(unittest.TestCase):
    """Survival analysis module."""

    def setUp(self):
        from survival_analysis.survival_model import (
            generate_synthetic_cohort, KaplanMeier, SurvivalAnalyzer,
            log_rank_test, CoxPH
        )
        self.cohort = generate_synthetic_cohort(n=150, seed=42)
        self.KaplanMeier = KaplanMeier
        self.SurvivalAnalyzer = SurvivalAnalyzer
        self.log_rank_test = log_rank_test
        self.CoxPH = CoxPH

    def test_km_curve_starts_at_one(self):
        km = self.KaplanMeier(self.cohort).fit()
        self.assertAlmostEqual(km.curve[0].survival, 1.0, places=5)

    def test_km_survival_non_increasing(self):
        km = self.KaplanMeier(self.cohort).fit()
        survivals = [p.survival for p in km.curve]
        for i in range(1, len(survivals)):
            self.assertLessEqual(survivals[i], survivals[i-1] + 1e-9)

    def test_km_survival_at_zero_is_one(self):
        km = self.KaplanMeier(self.cohort)
        self.assertAlmostEqual(km.survival_at(0), 1.0, places=5)

    def test_km_survival_decreases_over_time(self):
        km = self.KaplanMeier(self.cohort)
        s7 = km.survival_at(7)
        s30 = km.survival_at(30)
        self.assertGreater(s7, s30)

    def test_km_to_result(self):
        km = self.KaplanMeier(self.cohort)
        result = km.to_result("all")
        self.assertIn("survival_at_7d", result.to_dict())
        self.assertGreaterEqual(result.survival_at_7d, 0)
        self.assertLessEqual(result.survival_at_7d, 1)

    def test_log_rank_detects_difference(self):
        from survival_analysis.survival_model import SurvivalObservation
        # Create two clearly different groups
        good = [SurvivalObservation(f"G{i}", 25.0 + i * 0.5, 1, "discharged") for i in range(50)]
        bad  = [SurvivalObservation(f"B{i}", 5.0 + i * 0.1,  1, "died") for i in range(50)]
        chi2, p = self.log_rank_test(good, bad)
        self.assertGreater(chi2, 0)
        self.assertLess(p, 0.05)  # significant difference

    def test_cox_ph_fits_and_returns_hrs(self):
        cox = self.CoxPH(["age", "news2", "lactate"], lr=0.05, max_iter=50)
        cox.fit(self.cohort)
        self.assertTrue(cox.fitted)
        self.assertIn("age", cox.hazard_ratios)
        for hr in cox.hazard_ratios.values():
            self.assertGreater(hr, 0)

    def test_analyzer_full_report(self):
        analyzer = self.SurvivalAnalyzer(self.cohort)
        report = analyzer.full_report()
        self.assertIn("overall", report)
        self.assertIn("competing_risks", report)
        self.assertIn("cox_model", report)
        self.assertGreater(report["n_total"], 0)

    def test_analyzer_predict_patient(self):
        analyzer = self.SurvivalAnalyzer(self.cohort)
        pred = analyzer.predict_patient({
            "age": 72, "news2": 9, "lactate": 3.9,
            "sofa_score": 7, "ventilated": 1, "vasopressors": 1,
        })
        self.assertIn("predicted_survival", pred)
        for t, s in pred["predicted_survival"].items():
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 1)


class TestModelDriftDetector(unittest.TestCase):
    """Model drift detection."""

    def setUp(self):
        import numpy as np
        from model_monitoring.drift_detector import (
            ModelMonitor, FeatureDriftDetector, psi, ks_statistic
        )
        self.np = np
        self.ModelMonitor = ModelMonitor
        self.FeatureDriftDetector = FeatureDriftDetector
        self.psi = psi
        self.ks_statistic = ks_statistic
        self.rng = np.random.RandomState(42)

    def test_psi_zero_for_identical_distributions(self):
        x = self.rng.randn(500)
        score = self.psi(x, x)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_psi_large_for_shifted_distribution(self):
        ref = self.rng.randn(500)
        cur = self.rng.randn(500) + 5  # large shift
        score = self.psi(ref, cur)
        self.assertGreater(score, 0.25)

    def test_ks_small_for_same_distribution(self):
        x = self.rng.randn(300)
        y = self.rng.randn(300)
        D, p = self.ks_statistic(x, y)
        self.assertGreater(p, 0.05)  # should not be significant

    def test_ks_significant_for_different_distribution(self):
        ref = self.rng.randn(200)
        cur = self.rng.randn(200) + 3  # clear shift
        D, p = self.ks_statistic(ref, cur)
        self.assertLess(p, 0.05)

    def test_monitor_no_drift_stable_data(self):
        feature_names = ["hr", "sbp", "lactate", "news2"]
        X_ref = self.rng.randn(300, 4)
        preds_ref = self.rng.beta(2, 8, 300)
        labels_ref = (preds_ref > 0.5).astype(int).tolist()

        monitor = self.ModelMonitor("TestModel", "sepsis_risk", feature_names,
                                      psi_threshold=0.25, prediction_psi_threshold=0.25,
                                      label_shift_threshold=0.10)
        monitor.set_reference(X_ref, preds_ref, labels_ref)

        X_cur = X_ref[:100] + self.rng.randn(100, 4) * 0.02
        preds_cur = preds_ref[:100] + self.rng.randn(100) * 0.005
        report = monitor.check(X_cur, preds_cur, labels_ref[:100], "stable")
        # Stable data should not trigger severe drift
        self.assertNotEqual(report.drift_severity, "severe")

    def test_monitor_detects_severe_drift(self):
        feature_names = ["hr", "sbp", "lactate"]
        X_ref = self.rng.randn(300, 3)
        preds_ref = self.rng.beta(2, 8, 300)

        monitor = self.ModelMonitor("TestModel", "sepsis_risk", feature_names)
        monitor.set_reference(X_ref, preds_ref, [0] * 300)

        # Severely drifted data
        X_drift = self.rng.randn(150, 3) + 4.0
        preds_drift = self.rng.beta(8, 2, 150)  # completely flipped distribution
        report = monitor.check(X_drift, preds_drift, window_label="severe_drift")
        self.assertTrue(report.overall_drift_detected)
        self.assertIn(report.drift_severity, ("moderate", "severe"))


class TestMigrationScript(unittest.TestCase):
    """DB migration script (logic-only tests, no DB required)."""

    def test_parse_version_valid(self):
        from scripts.db_migrate import _parse_version
        result = _parse_version("V001__add_patient_flags.sql")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], "add patient flags")

    def test_parse_version_down_file(self):
        from scripts.db_migrate import _parse_version
        result = _parse_version("V003__create_indexes.down.sql")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 3)

    def test_parse_version_invalid(self):
        from scripts.db_migrate import _parse_version
        self.assertIsNone(_parse_version("not_a_migration.sql"))
        self.assertIsNone(_parse_version("schema.sql"))
        self.assertIsNone(_parse_version("README.md"))

    def test_migration_file_exists(self):
        import pathlib
        mig_dir = pathlib.Path(__file__).parent.parent.parent / "database" / "migrations"
        v001 = mig_dir / "V001__add_patient_flags_and_model_registry.sql"
        self.assertTrue(v001.exists(), f"V001 migration file not found at {v001}")

    def test_down_file_exists(self):
        import pathlib
        mig_dir = pathlib.Path(__file__).parent.parent.parent / "database" / "migrations"
        v001_down = mig_dir / "V001__add_patient_flags_and_model_registry.down.sql"
        self.assertTrue(v001_down.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

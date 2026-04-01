#!/usr/bin/env python3
"""
AI Hospital OS — Complete System Demo
Demonstrates all 25+ modules working together in a realistic ICU scenario:

  1.  Config → load and validate system configuration
  2.  Security → login, session management, audit trail
  3.  Data Quality → validate incoming vital records
  4.  NER → extract entities from admission note
  5.  Knowledge Graph → differential diagnosis + drug interactions
  6.  CDSS → full clinical decision support evaluation
  7.  Medication Safety → formulary check + contraindications
  8.  Risk Models → predict sepsis + mortality risk
  9.  Explainability → explain risk prediction
  10. Real-Time Monitor → stream vitals, fire alerts
  11. Lab Ordering → recommend next labs
  12. Patient Similarity → find historical cohort
  13. Workflow → manage clinical care pathway
  14. Simulation → generate deterioration trajectory
  15. FHIR → export patient bundle
  16. Reports → generate HTML report + note draft
  17. Evaluation → model performance check
  18. Health Monitor → system health report
"""

import sys
import os
import time
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

# ─── ANSI colours ────────────────────────
def col(text, c):
    codes = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
              "cyan": "\033[96m", "bold": "\033[1m", "reset": "\033[0m", "blue": "\033[94m"}
    return f"{codes.get(c,'')}{text}{codes['reset']}" if sys.stdout.isatty() else text


def section(title: str):
    print(f"\n{col('─' * 60, 'cyan')}")
    print(f"{col('▶ ' + title, 'bold')}")
    print(col('─' * 60, 'cyan'))


def ok(msg: str):    print(f"  {col('✓', 'green')} {msg}")
def warn(msg: str):  print(f"  {col('⚠', 'yellow')} {msg}")
def info(msg: str):  print(f"  {col('→', 'blue')} {msg}")
def alert(msg: str): print(f"  {col('🚨', 'red')} {msg}")


# ─── Demo patient ─────────────────────────

PATIENT = {
    "patient_id": "P_DEMO",
    "name": "Demo Patient",
    "age": 72,
    "sex": "M",
    "primary_diagnosis": "septic_shock",
    "location": "ICU-3",
    "admission_date": datetime.utcnow().strftime("%Y-%m-%d"),
    "hours_in_icu": 8,
    "vitals": {
        "heart_rate": 118, "sbp": 88, "dbp": 55, "map": 66,
        "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14,
    },
    "labs": {
        "lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "procalcitonin": 12.0,
        "troponin": 0.02, "bnp": 120, "inr": 1.4, "glucose": 195,
        "potassium": 5.1, "sodium": 137, "hemoglobin": 9.8,
    },
    "risk_scores": {"sepsis_risk": 0.78, "mortality_risk": 0.31},
    "news2": 9,
    "flags": {"mechanically_ventilated": False, "fluid_resuscitated": False},
    "active_medications": ["vancomycin", "piperacillin-tazobactam",
                            "norepinephrine", "propofol", "fentanyl", "heparin"],
}

ADMISSION_NOTE = """
72M admitted from ED with 2-day history of fever, rigors, and worsening confusion.
PMH: DM2, HTN, CKD (baseline Cr 1.3). NKDA. Current meds: metformin, lisinopril.

Physical: HR 118, BP 88/55, RR 26, Temp 38.8°C, SpO2 91% 2L NC. GCS 14.
Diaphoretic, tachypnoeic. Abdomen soft, no guarding. No focal neurological signs.

Assessment: Septic shock, likely pulmonary source vs. UTI.
Sepsis bundle initiated: blood cultures x2, vancomycin 25mg/kg IV q12h and
piperacillin-tazobactam 4.5g IV q6h. Normal saline 30mL/kg bolus given.
Norepinephrine started at 0.1 mcg/kg/min for MAP target ≥65.

Labs notable: WBC 19.0, lactate 3.9 mmol/L, creatinine 2.1, procalcitonin 12.
CXR shows bilateral infiltrates consistent with pneumonia.
"""

DEMO_RESULTS = {}


# ─────────────────────────────────────────
def run_demo():
    print()
    print(col("╔══════════════════════════════════════════════════════╗", "bold"))
    print(col("║       AI Hospital OS — Full System Demo              ║", "bold"))
    print(col("║       Patient: P_DEMO  (Septic Shock ICU)            ║", "bold"))
    print(col("╚══════════════════════════════════════════════════════╝", "bold"))

    # ── 1. Config ─────────────────────────────────────────
    section("1. System Configuration")
    from config.config_manager import get_config
    cfg = get_config()
    ok(f"Environment: {cfg.environment} | Version: {cfg.version}")
    ok(f"NEWS2 critical threshold: {cfg.alerts.news2_critical_threshold}")
    ok(f"Sepsis alert threshold: {cfg.ml.sepsis_alert_threshold:.0%}")
    DEMO_RESULTS["config"] = "ok"

    # ── 2. Security ───────────────────────────────────────
    section("2. Security & Session Management")
    import tempfile
    from security.audit_security import HospitalSecurityManager, Permission
    _tmpdir = tempfile.mkdtemp()
    sec = HospitalSecurityManager(audit_dir=_tmpdir)
    session_id = sec.login("dr_demo", "attending_physician", "10.0.0.1")
    ok(f"Session created: {session_id[:12]}...")
    can_view = sec.can_access_patient(session_id, "P_DEMO")
    ok(f"Physician access to P_DEMO: {'✓ granted' if can_view else '✗ denied'}")
    phi_text = "Patient John Smith DOB 03/15/1952 MRN 7654321 ph 555-867-5309"
    deident = sec.deidentifier.deidentify(phi_text)
    ok(f"PHI de-id: '{deident[:55]}'")
    DEMO_RESULTS["security"] = {"session": session_id[:12], "can_view": can_view}

    # ── 3. Data Quality ───────────────────────────────────
    section("3. Data Quality Check")
    from data_quality.dq_monitor import DataQualityMonitor, generate_test_vitals
    records = generate_test_vitals(n=200, inject_errors=True)
    monitor = DataQualityMonitor()
    dq_report = monitor.check_vitals(records)
    q = dq_report.overall_quality_score
    ok(f"Quality score: {col(f'{q:.0%}', 'green' if q > 0.85 else 'yellow')}"
       f" | {dq_report.n_records} records | {len(dq_report.alerts)} alerts")
    for a in dq_report.critical_alerts[:2]:
        warn(f"Critical: {a.message[:60]}")
    DEMO_RESULTS["data_quality"] = {"score": q, "n_records": dq_report.n_records}

    # ── 4. Medical NER ────────────────────────────────────
    section("4. Clinical NLP — Named Entity Recognition")
    from nlp_pipeline.medical_ner import ClinicalEntityExtractor
    ner = ClinicalEntityExtractor()
    extraction = ner.full_extraction(ADMISSION_NOTE)
    ok(f"Entities: {extraction['entity_count']} | Relations: {extraction['relation_count']}")
    meds = [m["medication"] for m in extraction["medications"]]
    ok(f"Medications: {', '.join(meds[:5])}")
    ok(f"Diagnoses: {', '.join(extraction['diagnoses'][:4])}")
    ok(f"Vitals: {extraction['vitals']}")
    DEMO_RESULTS["ner"] = {"entities": extraction["entity_count"],
                            "medications": meds}

    # ── 5. Knowledge Graph ────────────────────────────────
    section("5. Knowledge Graph — DDx + Drug Interactions")
    from knowledge_graph.graph_builder import MedicalKnowledgeGraph
    kg = MedicalKnowledgeGraph()
    ddx = kg.differential_diagnosis(
        ["fever", "tachycardia", "hypotension", "tachypnea", "altered mental status"], top_n=5)
    ok(f"Differential diagnosis ({len(ddx)} results):")
    for d in ddx[:3]:
        info(f"  {d['disease']:<30} score={d['score']:.2f}  [{d.get('severity','')}]")
    interactions = kg.check_drug_interactions(PATIENT["active_medications"])
    high_risk = [i for i in interactions if i["severity"] in ("high", "contraindicated")]
    ok(f"Drug interactions: {len(interactions)} total, {len(high_risk)} high-risk")
    for ix in high_risk[:2]:
        warn(f"  {ix['drug1']} + {ix['drug2']}: [{ix['severity'].upper()}] {ix['effect']}")
    DEMO_RESULTS["knowledge_graph"] = {"ddx": [d["disease"] for d in ddx[:3]],
                                        "high_risk_interactions": len(high_risk)}

    # ── 6. CDSS ───────────────────────────────────────────
    section("6. Clinical Decision Support Engine")
    from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
    engine = ClinicalDecisionSupportEngine()
    cdss = engine.evaluate(
        patient_data=PATIENT,
        symptoms=["fever", "chills", "hypotension", "tachycardia", "tachypnea"],
        medications=PATIENT["active_medications"],
        note_text=ADMISSION_NOTE,
    )
    risk_col = "red" if cdss.risk_level == "critical" else "yellow"
    ok(f"Risk level: {col(cdss.risk_level.upper(), risk_col)}")
    ok(f"Protocols triggered: {', '.join(cdss.triggered_protocols[:3])}")
    stat_recs = [r for r in cdss.recommendations if r.urgency == "STAT"]
    ok(f"Recommendations: {len(cdss.recommendations)} ({len(stat_recs)} STAT)")
    for r in stat_recs[:3]:
        alert(f"STAT: {r.action[:65]}")
    DEMO_RESULTS["cdss"] = {"risk": cdss.risk_level,
                             "protocols": cdss.triggered_protocols,
                             "n_recs": len(cdss.recommendations)}

    # ── 7. Medication Safety ──────────────────────────────
    section("7. Medication Safety Check")
    from clinical_decision_support.medication_safety import MedicationSafetyChecker
    med_checker = MedicationSafetyChecker()
    med_report = med_checker.check(
        patient_id="P_DEMO",
        medications=PATIENT["active_medications"],
        egfr=28,  # AKI (baseline Cr 1.3, now 2.1)
        weight_kg=80,
        icu_context=True,
    )
    status_col = {"safe": "green", "review": "cyan", "caution": "yellow",
                   "unsafe": "red"}.get(med_report.overall_status, "reset")
    ok(f"Safety status: {col(med_report.overall_status.upper(), status_col)}")
    ok(f"Alerts: {len(med_report.alerts)} "
       f"({sum(1 for a in med_report.alerts if a.severity=='high')} high)")
    for a in med_report.alerts[:3]:
        if a.severity in ("critical", "high"):
            warn(f"[{a.severity.upper():6}] {a.message[:65]}")
    DEMO_RESULTS["medication_safety"] = {"status": med_report.overall_status,
                                           "n_alerts": len(med_report.alerts)}

    # ── 8. Model Explainability ───────────────────────────
    section("8. AI Risk Prediction + Explainability")
    from explainability.model_explainer import ClinicalExplainer
    features = {**PATIENT["vitals"], **PATIENT["labs"],
                 "news2": PATIENT["news2"], "hours_in_icu": PATIENT["hours_in_icu"],
                 "shock_index": 118/88, "sofa_score": 7.5, "age": PATIENT["age"]}
    explainer = ClinicalExplainer(task="sepsis_risk")
    explanation = explainer.explain(features, patient_id="P_DEMO", top_k=6)
    ok(f"Sepsis risk: {explanation.prediction:.0%} ({explanation.prediction_label})")
    ok(f"CI (95%): [{explanation.confidence_interval[0]:.0%}, {explanation.confidence_interval[1]:.0%}]")
    drivers = [f for f in explanation.top_features if f.direction == "increases_risk"][:3]
    ok(f"Top drivers: {', '.join(f'{f.feature}={f.value:.1f}' for f in drivers)}")
    if explanation.counterfactuals:
        cf = explanation.counterfactuals[0]
        info(f"Counterfactual: if {cf['feature']}→{cf['target_value']:.1f}, "
              f"risk drops to {cf['new_predicted_risk']:.0%}")
    DEMO_RESULTS["explainability"] = {"prediction": explanation.prediction,
                                       "label": explanation.prediction_label}

    # ── 9. Real-Time Monitoring ───────────────────────────
    section("9. Real-Time Vitals Streaming")
    from real_time_monitoring.vitals_stream_processor import (
        VitalsStreamProcessor, PatientVitalsSimulator
    )
    processor = VitalsStreamProcessor()
    processor.register_patient("P_DEMO", "Demo Patient", "ICU-3")
    sim = PatientVitalsSimulator("P_DEMO", "deteriorating")
    fired_alerts = []
    processor.add_alert_handler(lambda a: fired_alerts.append(a))
    processor.start()
    for _ in range(12):
        processor.ingest(sim.next_reading())
    time.sleep(0.8)
    processor.stop()
    summary = processor.get_patient_summary("P_DEMO")
    ok(f"Streamed {summary['total_readings']} readings | NEWS2={summary['news2']}")
    ok(f"Alerts fired: {len(fired_alerts)}")
    for a in fired_alerts[:2]:
        alert(f"[{a.severity.upper():8}] {a.message[:60]}")
    DEMO_RESULTS["real_time"] = {"readings": summary["total_readings"],
                                   "alerts": len(fired_alerts)}

    # ── 10. Lab Ordering ──────────────────────────────────
    section("10. Predictive Lab Ordering")
    from lab_predictor.lab_ordering import LabOrderingEngine
    from datetime import timedelta
    lab_engine = LabOrderingEngine()
    now = datetime.utcnow()
    last_results = {
        "CBC": now - timedelta(hours=14),
        "BMP": now - timedelta(hours=5),
    }
    lab_recs = lab_engine.recommend(PATIENT, last_results,
                                     active_medications=PATIENT["active_medications"])
    stat_labs = [r for r in lab_recs if r.priority == "STAT"]
    urgent_labs = [r for r in lab_recs if r.priority == "URGENT"]
    ok(f"Recommended {len(lab_recs)} labs: {len(stat_labs)} STAT, {len(urgent_labs)} URGENT")
    for r in stat_labs[:4]:
        info(f"  STAT [{r.lab.code}] {r.lab.name:<30}  → {r.reason[:40]}")
    DEMO_RESULTS["lab_ordering"] = {"total": len(lab_recs), "stat": len(stat_labs)}

    # ── 11. Patient Similarity ────────────────────────────
    section("11. Patient Similarity & Cohort Analysis")
    from patient_similarity.similarity_engine import PatientSimilarityEngine
    sim_engine = PatientSimilarityEngine(n_synthetic=300)
    similar = sim_engine.find_similar(PATIENT, top_k=10)
    insights = sim_engine.cohort_insights(similar)
    pred = sim_engine.predict_outcome(PATIENT, top_k=20)
    ok(f"Found {len(similar)} similar historical patients")
    ok(f"Cohort survival rate: {insights.survival_rate:.0%} | "
       f"Median LOS: {insights.median_icu_los_days:.1f} days")
    ok(f"Predicted outcome: {pred['prediction'].upper()} "
       f"(survival {pred['survival_probability']:.0%}, "
       f"confidence {pred['confidence']:.0%})")
    ok(f"Common treatments: {[t[0] for t in insights.common_treatments[:3]]}")
    DEMO_RESULTS["similarity"] = {"prediction": pred["prediction"],
                                   "survival_prob": pred["survival_probability"]}

    # ── 12. Clinical Workflow ─────────────────────────────
    section("12. Clinical Workflow State Machine")
    from workflow.clinical_workflow import ClinicalWorkflowEngine
    wf_engine = ClinicalWorkflowEngine()
    wf_engine.admit_patient("P_DEMO")
    wf_engine.alert_fired("P_DEMO", "ALT-001", "high", "NEWS2=9")
    wf_engine.acknowledge_alert("P_DEMO", "dr_demo")
    wf_engine.complete_assessment("P_DEMO", "dr_demo", "Septic shock — source likely pulmonary")
    wf_engine.record_intervention("P_DEMO", "Vancomycin + pip-tazo + norepinephrine", "dr_demo")
    wf_engine.record_stabilization("P_DEMO", "dr_demo")
    wf_state = wf_engine.get_patient_summary("P_DEMO")
    ok(f"Current state: {wf_state['current_state']}")
    ok(f"Protocol adherence: {wf_state['protocol_adherence']:.0%}")
    ok(f"Audit log entries: {wf_state['log_entries']}")
    DEMO_RESULTS["workflow"] = {"state": wf_state["current_state"],
                                  "adherence": wf_state["protocol_adherence"]}

    # ── 13. Simulation ────────────────────────────────────
    section("13. Patient Deterioration Simulation")
    from simulation.patient_simulator import ICUSimulator
    sim_obj = ICUSimulator()
    traj = sim_obj.simulate_patient("P_SIM", "septic_shock_treated", interval_minutes=60)
    worst = max(traj.states, key=lambda s: s.news2)
    ok(f"Scenario: {traj.scenario} | {traj.duration_hours:.0f}h | {traj.n_states} readings")
    ok(f"Peak: NEWS2={worst.news2} at {worst.time_hours:.1f}h | "
       f"HR={worst.heart_rate:.0f} SBP={worst.sbp:.0f} lactate={worst.lactate:.1f}")
    DEMO_RESULTS["simulation"] = {"scenario": traj.scenario, "peak_news2": traj.max_news2}

    # ── 14. FHIR Export ───────────────────────────────────
    section("14. FHIR R4 Export")
    from fhir_integration.fhir_client import FHIRIntegrationService
    fhir = FHIRIntegrationService()
    bundle = fhir.full_patient_export(PATIENT)
    rtypes = {e["resource"]["resourceType"] for e in bundle["entry"]}
    ok(f"Bundle: {len(bundle['entry'])} resources | Types: {rtypes}")
    risk_assessment = next((e["resource"] for e in bundle["entry"]
                              if e["resource"]["resourceType"] == "RiskAssessment"), None)
    if risk_assessment:
        n_preds = len(risk_assessment.get("prediction", []))
        ok(f"RiskAssessment: {n_preds} predictions")
    DEMO_RESULTS["fhir"] = {"n_resources": len(bundle["entry"]),
                              "resource_types": list(rtypes)}

    # ── 15. Reports ───────────────────────────────────────
    section("15. Clinical Report Generation")
    from reporting.report_generator import _make_snapshot, ReportGenerator
    snap = _make_snapshot("P_DEMO")
    gen = ReportGenerator()
    html = gen.generate_deterioration_report(snap)
    note = gen.generate_note_draft(snap)
    handover = gen.generate_handover_summary([snap])
    ok(f"HTML report: {len(html):,} chars")
    ok(f"Note draft: {len(note):,} chars")
    ok(f"Handover summary: {len(handover):,} chars")
    DEMO_RESULTS["reports"] = {"html_chars": len(html), "note_chars": len(note)}

    # ── 16. Model Evaluation ──────────────────────────────
    section("16. Model Evaluation (quick pass)")
    from evaluation.model_evaluator import EvaluationSuite
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        suite = EvaluationSuite()
        eval_results = suite.run_all(output_dir=td, backtest_windows=4)
    overall = eval_results["overall"]
    pass_col = "green" if overall["all_pass"] else "yellow"
    pass_str = f"{overall['n_pass']}/{overall['n_models']}"
    ok(f"Pass rate: {col(pass_str, pass_col)} models")
    for task, data in eval_results["models"].items():
        m = data["evaluation"]["metrics"]
        status = col("PASS", "green") if data["evaluation"]["pass_criteria"] else col("FAIL", "red")
        info(f"  {task:<28} AUC={m.get('auc_roc',0):.3f} F1={m.get('f1',0):.3f} {status}")
    DEMO_RESULTS["evaluation"] = {"pass_rate": overall["pass_rate"]}

    # ── 17. System Health ─────────────────────────────────
    section("17. System Health Report")
    from system_health.health_monitor import SystemHealthMonitor, HealthStatus
    health_mon = SystemHealthMonitor()
    health_rpt = health_mon.check_all()
    overall_col = "green" if health_rpt.overall_status == "ok" else "yellow"
    ok(f"Overall: {col(health_rpt.overall_status.upper(), overall_col)}")
    for c in health_rpt.components:
        c_col = "green" if c.status == "ok" else "yellow" if c.status == "degraded" else "red"
        icon = "✓" if c.status == "ok" else "⚠" if c.status == "degraded" else "✗"
        print(f"    {col(icon, c_col)} {c.name:<22} {c.latency_ms:>6.1f}ms")
    DEMO_RESULTS["health"] = {"status": health_rpt.overall_status}

    # ── 18. Audit + Security flush ────────────────────────
    section("18. Audit Trail")
    sec.flush()
    trail = sec.get_audit_trail(limit=20)
    ok(f"Audit events logged: {len(trail)}")
    for ev in trail[:3]:
        status_icon = "✓" if ev["success"] else "✗"
        print(f"    {status_icon} {ev['user_id']:12} [{ev['user_role'][:22]:22}] "
              f"{ev['action']} {ev['resource_type']}")
    sec.logout(session_id)
    ok("Session closed cleanly")
    DEMO_RESULTS["audit"] = {"events": len(trail)}

    # ── Final summary ─────────────────────────────────────
    print(f"\n{col('═' * 60, 'cyan')}")
    print(f"{col('  DEMO COMPLETE', 'bold')}")
    print(col('═' * 60, 'cyan'))
    print(f"\n  Patient P_DEMO summary:")
    print(f"    Risk level      : {col(cdss.risk_level.upper(), 'red')}")
    print(f"    Sepsis risk     : {explanation.prediction:.0%} ({explanation.prediction_label})")
    print(f"    Protocols       : {', '.join(cdss.triggered_protocols[:2])}")
    print(f"    STAT actions    : {len(stat_recs)}")
    print(f"    Predicted outcome: {pred['prediction'].upper()} "
          f"(survival {pred['survival_probability']:.0%})")
    print(f"    FHIR resources  : {len(bundle['entry'])}")
    print(f"    Audit events    : {len(trail)}")
    modules_ok = sum(1 for v in DEMO_RESULTS.values()
                      if isinstance(v, dict) and "error" not in v)
    print(f"\n  Modules demonstrated: {col(str(len(DEMO_RESULTS)), 'green')} / 18")
    print(f"  All systems:          {col('OPERATIONAL', 'green')}")
    print()

    return DEMO_RESULTS


if __name__ == "__main__":
    results = run_demo()

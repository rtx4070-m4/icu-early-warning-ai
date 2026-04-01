#!/usr/bin/env python3
"""
AI Hospital OS — Command Line Interface
Provides quick access to all system functions from the terminal.

Usage:
    python cli/hospital_cli.py <command> [options]
    
Commands:
    patient  <id>             Get patient summary
    icu                       Show ICU census
    alerts   [--severity=X]   List active alerts
    risk     <id>             Show risk scores for a patient
    ner      <text|file>      Run NER on clinical text
    ddx      <symptoms>       Differential diagnosis
    drugs    <med1,med2,...>  Check drug interactions
    medsafe  <meds> [--egfr=X] [--weight=X]  Full medication safety check
    eval                      Run model evaluation suite
    quality  <file.csv>       Run data quality check on a file
    simulate <scenario>       Run patient simulation
    explain  <features.json>  Explain a risk prediction
    health                    System health report
    config                    Show system configuration
    report   <id>             Generate patient report
    version                   Show version info
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

VERSION = "1.0.0"
BANNER = f"""
╔══════════════════════════════════════════╗
║   AI Hospital Operating System v{VERSION}   ║
║   Clinical Decision Support Platform    ║
╚══════════════════════════════════════════╝
"""


# ─────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────

def _print_json(data, indent: int = 2):
    print(json.dumps(data, indent=indent, default=str))


def _print_table(rows: List[dict], columns: List[str] = None):
    if not rows:
        print("  (no data)")
        return
    cols = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  " + "  ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def _color(text: str, color: str) -> str:
    colors = {"red": "\033[91m", "yellow": "\033[93m", "green": "\033[92m",
               "blue": "\033[94m", "cyan": "\033[96m", "bold": "\033[1m", "reset": "\033[0m"}
    if not sys.stdout.isatty():
        return text
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def _severity_color(sev: str) -> str:
    return {"critical": "red", "high": "yellow", "moderate": "cyan",
             "low": "green", "safe": "green", "unsafe": "red", "caution": "yellow"}.get(sev, "reset")


# ─────────────────────────────────────────
# Command implementations
# ─────────────────────────────────────────

def cmd_patient(args):
    """Show patient summary."""
    pid = args.id
    print(f"\nPatient Summary: {_color(pid, 'bold')}")
    try:
        from ehr_system.patient_management import EHRService
        svc = EHRService()
        summary = svc.get_patient_summary(pid)
    except Exception:
        from reporting.report_generator import _make_snapshot
        snap = _make_snapshot(pid)
        summary = {
            "patient_id": snap.patient_id, "name": snap.name,
            "location": snap.location, "diagnosis": snap.primary_diagnosis,
            "hours_in_icu": snap.hours_in_icu, "news2": snap.news2,
            "vitals": snap.vitals, "labs": snap.labs,
            "risk_scores": snap.risk_scores, "active_alerts": snap.active_alerts,
        }
    if not summary:
        print(f"  Patient {pid} not found")
        return

    # Vitals
    vitals = summary.get("vitals", {})
    print(f"\n  Name    : {summary.get('name', 'Unknown')}")
    print(f"  Location: {summary.get('location', '?')}")
    print(f"  Dx      : {summary.get('primary_diagnosis', summary.get('diagnosis', '?'))}")
    print(f"  In ICU  : {summary.get('hours_in_icu', '?')} hours")
    news2 = summary.get("news2", 0)
    news2_col = "red" if news2 >= 7 else "yellow" if news2 >= 5 else "green"
    print(f"  NEWS2   : {_color(str(news2), news2_col)}")

    if vitals:
        print(f"\n  Vitals:")
        print(f"    HR={vitals.get('heart_rate', '?'):.0f}  "
              f"BP={vitals.get('sbp', '?'):.0f}/{vitals.get('dbp', '?'):.0f}  "
              f"SpO2={vitals.get('spo2', '?'):.0f}%  "
              f"RR={vitals.get('respiratory_rate', '?'):.0f}  "
              f"T={vitals.get('temperature', '?'):.1f}°C")

    risk = summary.get("risk_scores", {})
    if risk:
        print(f"\n  Risk scores:")
        for k, v in risk.items():
            col = "red" if v > 0.6 else "yellow" if v > 0.3 else "green"
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"    {k:<20} {_color(f'{v:.0%}', col)} {bar}")

    alerts = summary.get("active_alerts", 0)
    if alerts:
        print(f"\n  {_color(f'Active alerts: {alerts}', 'red')}")


def cmd_icu(args):
    """Show ICU census."""
    print(f"\n{_color('ICU Census', 'bold')}")
    try:
        from ehr_system.patient_management import EHRService
        census = EHRService().get_icu_census()
    except Exception:
        from dashboard.hospital_dashboard import _generate_demo_patients
        census = _generate_demo_patients(8)

    cols = ["patient_id", "name", "location", "news2", "sepsis_risk", "active_alerts"]
    rows = []
    for p in sorted(census, key=lambda x: -x.get("news2", 0)):
        news2 = p.get("news2", 0)
        rows.append({
            "patient_id": p.get("patient_id", "?"),
            "name": p.get("name", "?")[:18],
            "location": p.get("location", "?"),
            "news2": _color(str(news2), "red" if news2 >= 7 else "yellow" if news2 >= 5 else "green"),
            "sepsis_risk": f"{p.get('sepsis_risk', 0):.0%}",
            "active_alerts": str(p.get("active_alerts", 0)),
        })

    _print_table(rows, cols)
    print(f"\n  Total: {len(census)} patients | "
          f"Critical (NEWS2≥7): {sum(1 for p in census if p.get('news2',0) >= 7)}")


def cmd_alerts(args):
    """List active alerts."""
    severity = getattr(args, "severity", None)
    print(f"\n{_color('Active Alerts', 'bold')}"
          + (f" [{severity}]" if severity else ""))

    from dashboard.hospital_dashboard import _generate_demo_patients, _generate_alerts
    patients = _generate_demo_patients(8)
    alerts = _generate_alerts(patients)
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]

    if not alerts:
        print("  No alerts found.")
        return

    for a in alerts:
        col = _severity_color(a.get("severity", "info"))
        icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡"}.get(a.get("severity"), "⚪")
        print(f"\n  {icon} {_color(a.get('severity','').upper(), col):<12} "
              f"{a.get('patient_id')} | {a.get('location')} | {a.get('time')}")
        print(f"     {a.get('message')}")


def cmd_risk(args):
    """Show risk scores for a patient."""
    pid = args.id
    from explainability.model_explainer import ClinicalExplainer
    from reporting.report_generator import _make_snapshot

    snap = _make_snapshot(pid)
    features = {**snap.vitals, **snap.labs,
                 "news2": snap.news2, "hours_in_icu": snap.hours_in_icu}

    explainer = ClinicalExplainer(task="sepsis_risk")
    explanation = explainer.explain(features, patient_id=pid, top_k=6)

    print(f"\n{_color('Risk Scores', 'bold')}: {pid}")
    print(f"\n  Sepsis risk   : {_color(f'{explanation.prediction:.0%}', _severity_color('critical' if explanation.prediction > 0.6 else 'moderate'))} ({explanation.prediction_label})")
    print(f"  CI (95%)      : [{explanation.confidence_interval[0]:.0%}, {explanation.confidence_interval[1]:.0%}]")
    print(f"  Base rate     : {explanation.base_rate:.0%}")

    print(f"\n  Top contributing features:")
    for fc in explanation.top_features:
        dir_icon = "↑" if fc.direction == "increases_risk" else "↓" if fc.direction == "decreases_risk" else "–"
        col = "red" if fc.direction == "increases_risk" else "green" if fc.direction == "decreases_risk" else "reset"
        print(f"    {dir_icon} {fc.feature:<22} value={fc.value:.2f}  "
              f"ref={fc.reference_value:.2f}  "
              f"contrib={_color(f'{fc.contribution:+.4f}', col)}")

    if explanation.counterfactuals:
        print(f"\n  Top counterfactual:")
        cf = explanation.counterfactuals[0]
        print(f"    If {cf['feature']} → {cf['target_value']:.1f}: "
              f"risk reduces to {cf['new_predicted_risk']:.0%} "
              f"(saves {cf['predicted_risk_reduction']:.0%})")


def cmd_ner(args):
    """Run NER on clinical text."""
    text = args.text
    if os.path.exists(text):
        with open(text) as f:
            text = f.read()

    from nlp_pipeline.medical_ner import ClinicalEntityExtractor
    extractor = ClinicalEntityExtractor()
    extraction = extractor.full_extraction(text)

    print(f"\n{_color('Medical NER Results', 'bold')}")
    print(f"  Text length: {len(text)} chars | Entities: {extraction['entity_count']}\n")

    for label in ["medications", "diagnoses", "symptoms", "procedures"]:
        items = extraction.get(label, [])
        if items:
            if isinstance(items[0], dict):
                display = [m.get("medication", str(m)) for m in items]
            else:
                display = items
            print(f"  {label.title():<15} ({len(display)}): {', '.join(display[:8])}")

    vitals = extraction.get("vitals", {})
    if vitals:
        print(f"\n  Vitals: " + " | ".join(f"{k}={v}" for k, v in vitals.items()))
    labs = extraction.get("labs", {})
    if labs:
        print(f"  Labs:   " + " | ".join(f"{k}={v}" for k, v in labs.items()))


def cmd_ddx(args):
    """Differential diagnosis from symptoms."""
    symptoms = [s.strip() for s in args.symptoms.split(",")]
    from knowledge_graph.graph_builder import MedicalKnowledgeGraph
    kg = MedicalKnowledgeGraph()
    ddx = kg.differential_diagnosis(symptoms, top_n=getattr(args, "top", 8))

    print(f"\n{_color('Differential Diagnosis', 'bold')}")
    print(f"  Symptoms: {', '.join(symptoms)}\n")
    for i, d in enumerate(ddx, 1):
        score_bar = "█" * int(d["score"] * 20) + "░" * (20 - int(d["score"] * 20))
        sev_col = {"critical": "red", "high": "yellow", "moderate": "cyan"}.get(d.get("severity", ""), "reset")
        print(f"  {i:2}. {d['disease'].replace('_',' ').title():<30} "
              f"score={d['score']:.2f} {score_bar} "
              f"{_color(d.get('severity',''), sev_col)}")


def cmd_drugs(args):
    """Check drug interactions."""
    meds = [m.strip().lower() for m in args.medications.split(",")]
    from knowledge_graph.graph_builder import MedicalKnowledgeGraph
    kg = MedicalKnowledgeGraph()
    interactions = kg.check_drug_interactions(meds)

    print(f"\n{_color('Drug Interactions', 'bold')}")
    print(f"  Medications: {', '.join(meds)}\n")
    if not interactions:
        print(f"  {_color('No known interactions found.', 'green')}")
        return
    for ix in interactions:
        col = _severity_color(ix["severity"])
        print(f"  {_color('[' + ix['severity'].upper() + ']', col):22} "
              f"{ix['drug1']} + {ix['drug2']}")
        print(f"    → {ix.get('effect', '')}")


def cmd_medsafe(args):
    """Full medication safety check."""
    meds = [m.strip().lower() for m in args.medications.split(",")]
    egfr = float(args.egfr) if hasattr(args, "egfr") and args.egfr else None
    weight = float(args.weight) if hasattr(args, "weight") and args.weight else None
    allergies = [a.strip().lower() for a in args.allergies.split(",")] if hasattr(args, "allergies") and args.allergies else []

    from clinical_decision_support.medication_safety import MedicationSafetyChecker
    checker = MedicationSafetyChecker()
    report = checker.check(
        patient_id=getattr(args, "patient", "CLI"),
        medications=meds, egfr=egfr, weight_kg=weight, allergies=allergies,
    )

    print(f"\n{_color('Medication Safety Report', 'bold')}")
    status_col = _severity_color(report.overall_status)
    print(f"  Status: {_color(report.overall_status.upper(), status_col)}")
    print(f"  Medications checked: {', '.join(report.medications_checked)}")
    if report.formulary_misses:
        print(f"  Not in formulary: {', '.join(report.formulary_misses)}")

    if report.alerts:
        print(f"\n  Alerts ({len(report.alerts)}):")
        for a in report.alerts:
            icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡"}.get(a.severity, "⚪")
            col = _severity_color(a.severity)
            print(f"    {icon} {_color(a.severity.upper(), col):<12} {a.message}")
            print(f"       → {a.recommendation}")
    else:
        print(f"\n  {_color('No interactions or contraindications found.', 'green')}")

    if report.monitoring_plan:
        print(f"\n  Monitoring plan:")
        for drug, params in list(report.monitoring_plan.items())[:5]:
            print(f"    {drug}: {', '.join(params[:3])}")


def cmd_simulate(args):
    """Run patient simulation."""
    scenario = args.scenario
    from simulation.patient_simulator import ICUSimulator, SCENARIOS

    if scenario == "list":
        print(f"\nAvailable scenarios:")
        for name, phases in SCENARIOS.items():
            hours = sum(p.duration_hours for p in phases)
            print(f"  {name:<32} {len(phases)} phases, {hours:.0f}h")
        return

    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    sim = ICUSimulator()
    interval = int(getattr(args, "interval", 60))
    result = sim.simulate_patient("P_SIM", scenario, interval_minutes=interval)

    print(f"\n{_color('Patient Simulation', 'bold')}: {scenario}")
    print(f"  Duration: {result.duration_hours:.0f}h | Readings: {result.n_states}")
    print(f"  Max NEWS2: {result.max_news2} | Min SBP: {result.min_sbp:.0f} mmHg | Max lactate: {result.max_lactate:.1f} mmol/L")
    print(f"  Peak phase: {result.peak_phase}")

    print(f"\n  Timeline (every 4th reading):")
    print(f"  {'Hour':>5}  {'HR':>5}  {'SBP':>5}  {'SpO2':>5}  {'Lac':>5}  {'NEWS2':>6}")
    print(f"  " + "-" * 42)
    for state in result.states[::max(1, len(result.states)//12)]:
        news2 = state.news2
        col = "red" if news2 >= 7 else "yellow" if news2 >= 5 else "reset"
        print(f"  {state.time_hours:>5.1f}  {state.heart_rate:>5.0f}  {state.sbp:>5.0f}  "
              f"{state.spo2:>5.0f}%  {state.lactate:>5.2f}  "
              f"{_color(str(news2), col):>6}")


def cmd_eval(args):
    """Run model evaluation suite."""
    from evaluation.model_evaluator import EvaluationSuite
    import tempfile

    print(f"\n{_color('Model Evaluation Suite', 'bold')}")
    print("  Running evaluations...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        suite = EvaluationSuite()
        results = suite.run_all(output_dir=tmpdir, backtest_windows=6)
        overall = results["overall"]

        pass_msg = f"{overall['n_pass']}/{overall['n_models']} models PASS"
        col = "green" if overall["all_pass"] else "yellow"
        print(f"\n  Overall: {_color(pass_msg, col)}")

        print(f"\n  {'Model':<28} {'AUC':>6} {'F1':>6} {'Recall':>7} {'ECE':>6}  {'Status'}")
        print(f"  " + "-" * 65)
        for task, data in results["models"].items():
            m = data["evaluation"]["metrics"]
            status = _color("PASS", "green") if data["evaluation"]["pass_criteria"] else _color("FAIL", "red")
            print(f"  {task:<28} {m.get('auc_roc',0):.3f}  {m.get('f1',0):.3f}  "
                  f"{m.get('recall',0):.3f}  {data['evaluation']['calibration_ece']:.3f}  {status}")


def cmd_health(args):
    """System health report."""
    from system_health.health_monitor import SystemHealthMonitor, HealthStatus

    print(f"\n{_color('System Health Report', 'bold')}")
    monitor = SystemHealthMonitor()
    report = monitor.check_all()

    overall_col = "green" if report.overall_status == "ok" else "yellow" if report.overall_status == "degraded" else "red"
    print(f"  Overall: {_color(report.overall_status.upper(), overall_col)}")
    print(f"  Uptime : {report.uptime_seconds:.0f}s | Version: {report.version}\n")

    status_icons = {HealthStatus.OK: "✓", HealthStatus.DEGRADED: "⚠",
                     HealthStatus.DOWN: "✗", HealthStatus.UNKNOWN: "?"}
    for c in report.components:
        col = "green" if c.status == "ok" else "yellow" if c.status == "degraded" else "red"
        icon = status_icons.get(c.status, "?")
        print(f"  {_color(icon, col)} {c.name:<22} {_color(c.status.upper(), col):<12} "
              f"{c.latency_ms:>6.0f}ms  {c.message[:45]}")

    ready = monitor.readiness_probe()
    print(f"\n  Readiness: {_color('READY', 'green') if ready else _color('NOT READY', 'red')}")


def cmd_config(args):
    """Show system configuration."""
    from config.config_manager import get_config
    cfg = get_config()

    print(f"\n{_color('System Configuration', 'bold')}")
    print(f"  Environment : {cfg.environment}")
    print(f"  Version     : {cfg.version}")
    print(f"  Log level   : {cfg.log_level}")
    print(f"\n  Database    : {cfg.database.host}:{cfg.database.port}/{cfg.database.name}")
    print(f"  MLflow URI  : {cfg.mlflow.tracking_uri}")
    print(f"  Dashboard   : {cfg.dashboard.host}:{cfg.dashboard.port}")
    print(f"  API server  : {cfg.api.host}:{cfg.api.port} ({cfg.api.workers} workers)")
    print(f"\n  Alerts:")
    print(f"    NEWS2 warning  : {cfg.alerts.news2_warning_threshold}")
    print(f"    NEWS2 critical : {cfg.alerts.news2_critical_threshold}")
    print(f"    Sepsis thresh  : {cfg.ml.sepsis_alert_threshold:.0%}")
    print(f"  Security:")
    print(f"    Session timeout: {cfg.security.session_timeout_minutes}min")
    print(f"    PHI de-id on export: {cfg.security.phi_deidentify_exports}")


def cmd_report(args):
    """Generate patient report."""
    pid = args.id
    import tempfile
    from reporting.report_generator import _make_snapshot, ReportGenerator

    snap = _make_snapshot(pid)
    gen = ReportGenerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, f"report_{pid}.html")
        gen.generate_deterioration_report(snap, output_path=html_path)
        size = os.path.getsize(html_path)

        note = gen.generate_note_draft(snap)
        note_path = os.path.join(tmpdir, f"note_{pid}.md")
        with open(note_path, "w") as f:
            f.write(note)

        outdir = getattr(args, "output", "/tmp")
        import shutil
        final_html = os.path.join(outdir, f"report_{pid}.html")
        final_note = os.path.join(outdir, f"note_{pid}.md")
        shutil.copy(html_path, final_html)
        shutil.copy(note_path, final_note)

    print(f"\n{_color('Reports Generated', 'bold')} for {pid}:")
    print(f"  HTML report   : {final_html} ({size:,} bytes)")
    print(f"  Note draft    : {final_note}")
    print(f"\n  Patient: {snap.name} | NEWS2={snap.news2} | "
          f"Sepsis {snap.risk_scores.get('sepsis_risk',0):.0%}")


def cmd_version(args):
    print(BANNER)
    print(f"  Version  : {VERSION}")
    print(f"  Python   : {sys.version.split()[0]}")
    import platform
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Root     : {PROJECT_ROOT}")


# ─────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hospital_cli",
        description="AI Hospital OS — Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    sub = parser.add_subparsers(dest="command", metavar="command")

    # patient
    p = sub.add_parser("patient", help="Get patient summary")
    p.add_argument("id")

    # icu
    sub.add_parser("icu", help="Show ICU census")

    # alerts
    p = sub.add_parser("alerts", help="List active alerts")
    p.add_argument("--severity", choices=["critical", "high", "moderate", "low"])

    # risk
    p = sub.add_parser("risk", help="Risk scores + explanation")
    p.add_argument("id")

    # ner
    p = sub.add_parser("ner", help="Medical NER on clinical text")
    p.add_argument("text", help="Text string or path to file")

    # ddx
    p = sub.add_parser("ddx", help="Differential diagnosis")
    p.add_argument("symptoms", help="Comma-separated symptoms")
    p.add_argument("--top", type=int, default=8)

    # drugs
    p = sub.add_parser("drugs", help="Drug interaction checker")
    p.add_argument("medications", help="Comma-separated drug names")

    # medsafe
    p = sub.add_parser("medsafe", help="Medication safety check")
    p.add_argument("medications", help="Comma-separated drug names")
    p.add_argument("--egfr", type=float, help="eGFR mL/min/1.73m²")
    p.add_argument("--weight", type=float, help="Weight in kg")
    p.add_argument("--allergies", help="Comma-separated drug allergies")
    p.add_argument("--patient", default="CLI")

    # simulate
    p = sub.add_parser("simulate", help="Patient deterioration simulation")
    p.add_argument("scenario", help="Scenario name or 'list'")
    p.add_argument("--interval", type=int, default=60, help="Sampling interval (minutes)")

    # eval
    sub.add_parser("eval", help="Run model evaluation suite")

    # health
    sub.add_parser("health", help="System health report")

    # config
    sub.add_parser("config", help="Show configuration")

    # report
    p = sub.add_parser("report", help="Generate patient report")
    p.add_argument("id")
    p.add_argument("--output", default="/tmp", help="Output directory")

    # version
    sub.add_parser("version", help="Show version info")

    return parser


COMMANDS = {
    "patient":  cmd_patient,
    "icu":      cmd_icu,
    "alerts":   cmd_alerts,
    "risk":     cmd_risk,
    "ner":      cmd_ner,
    "ddx":      cmd_ddx,
    "drugs":    cmd_drugs,
    "medsafe":  cmd_medsafe,
    "simulate": cmd_simulate,
    "eval":     cmd_eval,
    "health":   cmd_health,
    "config":   cmd_config,
    "report":   cmd_report,
    "version":  cmd_version,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    fn = COMMANDS.get(args.command)
    if not fn:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

    try:
        fn(args)
        print()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except Exception as e:
        print(f"\n  {_color('Error', 'red')}: {e}")
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

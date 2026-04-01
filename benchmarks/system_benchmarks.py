"""
System Benchmark Suite
Measures throughput, latency, and accuracy of all Hospital OS subsystems.
Reports p50/p95/p99 latencies and items/second throughput.
"""

import sys
import os
import time
import statistics
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Benchmark result
# ─────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    n_iterations: int
    total_seconds: float
    latencies_ms: List[float]
    throughput_per_sec: float
    errors: int = 0
    notes: str = ""

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        idx = int(0.95 * len(self.latencies_ms))
        return sorted(self.latencies_ms)[idx]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        idx = int(0.99 * len(self.latencies_ms))
        return sorted(self.latencies_ms)[idx]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    def summary_line(self) -> str:
        return (f"{self.name:<40} n={self.n_iterations:<5} "
                f"p50={self.p50_ms:>7.1f}ms "
                f"p95={self.p95_ms:>7.1f}ms "
                f"p99={self.p99_ms:>7.1f}ms "
                f"tput={self.throughput_per_sec:>8.1f}/s "
                + (f"errors={self.errors}" if self.errors else ""))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_iterations": self.n_iterations,
            "total_seconds": round(self.total_seconds, 3),
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "throughput_per_sec": round(self.throughput_per_sec, 2),
            "errors": self.errors,
            "notes": self.notes,
        }


# ─────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────

def run_benchmark(name: str, fn: Callable, n: int = 100,
                   warmup: int = 5) -> BenchmarkResult:
    """Run a function n times and collect latency statistics."""
    # Warmup
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass

    latencies = []
    errors = 0
    t_start = time.perf_counter()

    for _ in range(n):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            errors += 1
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    total = time.perf_counter() - t_start
    tput = n / total if total > 0 else 0

    return BenchmarkResult(
        name=name,
        n_iterations=n,
        total_seconds=total,
        latencies_ms=latencies,
        throughput_per_sec=tput,
        errors=errors,
    )


# ─────────────────────────────────────────
# Individual benchmarks
# ─────────────────────────────────────────

def bench_medical_ner():
    from nlp_pipeline.medical_ner import MedicalNER
    ner = MedicalNER()
    short_note = "Patient with septic shock. Started vancomycin 25mg/kg q8h. HR 118, BP 82/54, SpO2 88%."
    long_note = short_note * 5

    results = []
    results.append(run_benchmark("NER / short note (80 chars)", lambda: ner.process(short_note), n=200))
    results.append(run_benchmark("NER / long note (400 chars)", lambda: ner.process(long_note), n=100))
    return results


def bench_knowledge_graph():
    from knowledge_graph.graph_builder import MedicalKnowledgeGraph
    kg = MedicalKnowledgeGraph()
    symptoms = ["fever", "tachycardia", "hypotension", "tachypnea", "altered mental status"]
    meds = ["vancomycin", "piperacillin-tazobactam", "heparin", "furosemide", "amiodarone"]

    results = []
    results.append(run_benchmark("KG / DDx (5 symptoms)",
                                  lambda: kg.differential_diagnosis(symptoms), n=200))
    results.append(run_benchmark("KG / drug interactions (5 drugs)",
                                  lambda: kg.check_drug_interactions(meds), n=200))
    results.append(run_benchmark("KG / disease info lookup",
                                  lambda: kg.get_disease_info("sepsis"), n=500))
    return results


def bench_cdss():
    from clinical_decision_support.cdss_engine import ClinicalDecisionSupportEngine
    engine = ClinicalDecisionSupportEngine()
    patient = {
        "patient_id": "BENCH",
        "vitals": {"heart_rate": 118, "sbp": 88, "dbp": 55, "respiratory_rate": 26,
                    "temperature": 38.8, "spo2": 91, "gcs": 14},
        "labs": {"lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "glucose": 185},
        "risk_scores": {"sepsis_risk": 0.78, "mortality_risk": 0.31},
        "news2": 9, "hours_in_icu": 8, "flags": {},
    }

    results = [run_benchmark("CDSS / full evaluation",
                               lambda: engine.evaluate(patient,
                                                       symptoms=["fever", "hypotension"],
                                                       medications=["vancomycin"]), n=50)]
    return results


def bench_explainability():
    from explainability.model_explainer import ClinicalExplainer
    explainer = ClinicalExplainer(task="sepsis_risk")
    bg = explainer._synthetic_background(200)
    explainer.set_background(bg)
    patient = {
        "heart_rate": 118, "sbp": 88, "map": 66, "respiratory_rate": 26,
        "temperature": 38.8, "spo2": 91, "gcs": 14, "lactate": 3.9,
        "creatinine": 2.1, "wbc": 19.0, "news2": 9, "shock_index": 1.34,
        "procalcitonin": 12.0, "hours_in_icu": 8, "hr_delta_1h": 15.0,
    }

    results = [run_benchmark("Explainability / local attribution",
                               lambda: explainer.explain(patient, top_k=6), n=30)]
    return results


def bench_real_time_monitor():
    from real_time_monitoring.vitals_stream_processor import (
        PatientVitalsSimulator, compute_news2, RuleBasedAlertEngine, PatientState
    )
    sim = PatientVitalsSimulator("BENCH", "deteriorating")
    engine = RuleBasedAlertEngine()
    state = PatientState(patient_id="BENCH")

    results = []
    results.append(run_benchmark("Monitor / vital generation",
                                  lambda: sim.next_reading(), n=1000))
    reading = sim.next_reading()
    results.append(run_benchmark("Monitor / NEWS2 computation",
                                  lambda: compute_news2(reading), n=5000))
    results.append(run_benchmark("Monitor / rule alert evaluation",
                                  lambda: engine.evaluate(state, reading), n=500))
    return results


def bench_simulation():
    from simulation.patient_simulator import ICUSimulator
    sim = ICUSimulator()

    results = []
    results.append(run_benchmark("Simulation / single patient (60min intervals)",
                                  lambda: sim.simulate_patient("BENCH", "septic_shock_treated",
                                                                interval_minutes=60, seed=7), n=20))
    results.append(run_benchmark("Simulation / ward (8 patients)",
                                  lambda: sim.simulate_ward(n_patients=8, interval_minutes=120), n=5))
    return results


def bench_similarity_engine():
    from patient_similarity.similarity_engine import PatientSimilarityEngine
    engine = PatientSimilarityEngine(n_synthetic=300)
    query = {
        "vitals": {"heart_rate": 118, "sbp": 88, "dbp": 55, "map": 66,
                    "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14},
        "labs":   {"lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "procalcitonin": 12.0},
        "news2": 9, "sofa_score": 7.5, "shock_index": 1.34, "hours_in_icu": 8, "age": 68,
    }

    results = []
    results.append(run_benchmark("Similarity / find top-10 (300 cohort)",
                                  lambda: engine.find_similar(query, top_k=10), n=50))
    results.append(run_benchmark("Similarity / outcome prediction",
                                  lambda: engine.predict_outcome(query, top_k=20), n=30))
    return results


def bench_lab_ordering():
    from lab_predictor.lab_ordering import LabOrderingEngine
    from datetime import timedelta
    engine = LabOrderingEngine()
    patient = {
        "vitals": {"heart_rate": 118, "sbp": 88, "dbp": 55, "map": 66,
                    "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14},
        "labs":   {"lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "troponin": 0.02},
        "risk_scores": {"sepsis_risk": 0.78},
        "news2": 9, "flags": {},
    }
    from datetime import datetime
    now = datetime.utcnow()
    last_results = {"CBC": now - timedelta(hours=14), "BMP": now - timedelta(hours=6)}

    results = [run_benchmark("Lab ordering / recommendations",
                               lambda: engine.recommend(patient, last_results,
                                                        ["vancomycin"]), n=200)]
    return results


def bench_medication_safety():
    from clinical_decision_support.medication_safety import MedicationSafetyChecker
    checker = MedicationSafetyChecker()
    meds = ["vancomycin", "piperacillin-tazobactam", "norepinephrine",
             "propofol", "fentanyl", "heparin", "furosemide", "insulin"]

    results = [run_benchmark("Medication safety / 8-drug check",
                               lambda: checker.check("BENCH", meds, egfr=28, weight_kg=72), n=200)]
    return results


def bench_data_quality():
    from data_quality.dq_monitor import DataQualityMonitor, generate_test_vitals
    records = generate_test_vitals(n=200, inject_errors=True)
    monitor = DataQualityMonitor()

    results = [run_benchmark("Data quality / 200 vitals records",
                               lambda: monitor.check_vitals(records), n=20)]
    return results


def bench_fhir():
    from fhir_integration.fhir_client import FHIRBuilder, FHIRParser
    builder = FHIRBuilder()
    parser = FHIRParser()
    patient = {"patient_id": "BENCH", "name": "Bench Test", "age": 65, "sex": "M"}
    vitals = {"heart_rate": 88, "sbp": 122, "dbp": 76, "spo2": 97, "respiratory_rate": 15}

    results = []
    results.append(run_benchmark("FHIR / build patient resource",
                                  lambda: builder.patient(patient), n=1000))
    results.append(run_benchmark("FHIR / build vital bundle (5 observations)",
                                  lambda: builder.vital_bundle("BENCH", vitals), n=500))
    bundle = builder.vital_bundle("BENCH", vitals)
    results.append(run_benchmark("FHIR / parse observation bundle",
                                  lambda: parser.parse_bundle_observations(bundle), n=1000))
    return results


# ─────────────────────────────────────────
# Full suite runner
# ─────────────────────────────────────────

SUITE = [
    ("Medical NER",        bench_medical_ner),
    ("Knowledge Graph",    bench_knowledge_graph),
    ("CDSS Engine",        bench_cdss),
    ("Explainability",     bench_explainability),
    ("Real-Time Monitor",  bench_real_time_monitor),
    ("Patient Simulation", bench_simulation),
    ("Similarity Engine",  bench_similarity_engine),
    ("Lab Ordering",       bench_lab_ordering),
    ("Medication Safety",  bench_medication_safety),
    ("Data Quality",       bench_data_quality),
    ("FHIR Integration",   bench_fhir),
]


def run_full_suite(suites: Optional[List[str]] = None,
                    output_json: bool = False) -> Dict:
    """Run all benchmarks and return structured results."""
    import json

    print("\n" + "=" * 85)
    print("  AI Hospital OS — System Benchmark Suite")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 85)
    print(f"  {'Benchmark':<40} {'n':>5}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'tput':>10}")
    print("  " + "-" * 82)

    all_results = {}
    for suite_name, bench_fn in SUITE:
        if suites and suite_name not in suites:
            continue
        print(f"\n  [{suite_name}]")
        try:
            results = bench_fn()
            all_results[suite_name] = [r.to_dict() for r in results]
            for r in results:
                print(f"    {r.summary_line()}")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[suite_name] = [{"error": str(e)}]

    print("\n" + "=" * 85)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version.split()[0],
        "results": all_results,
    }

    if output_json:
        print(json.dumps(report, indent=2))

    return report


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hospital OS Benchmarks")
    parser.add_argument("--suite", nargs="*", help="Run specific suites only")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    report = run_full_suite(suites=args.suite, output_json=args.json)
    total = sum(len(v) for v in report["results"].values()
                if isinstance(v, list) and v and "error" not in v[0])
    print(f"  {total} benchmark results collected across {len(report['results'])} suites")

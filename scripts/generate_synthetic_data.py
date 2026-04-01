#!/usr/bin/env python3
"""
Generate Synthetic ICU Data
Creates realistic synthetic patient datasets for development, testing,
and model training — no real patient data required.

Outputs CSV files to /data/raw/ (or --output-dir):
  - vitals_synthetic.csv      (hourly vital signs)
  - labs_synthetic.csv        (lab results)
  - medications_synthetic.csv (medication administrations)
  - patients_synthetic.csv    (patient demographics)
  - diagnoses_synthetic.csv   (ICD-10 diagnoses)

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --patients 100 --hours 72
    python scripts/generate_synthetic_data.py --scenario septic_shock_treated
"""

import sys
import os
import csv
import random
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────
# Demographics helpers
# ─────────────────────────────────────────

FIRST_NAMES_M = ["James","Robert","John","Michael","David","William","Richard",
                  "Joseph","Thomas","Charles","Raj","Wei","Carlos","Ahmed","Yuki"]
FIRST_NAMES_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan",
                  "Jessica","Sarah","Karen","Priya","Mei","Sofia","Fatima","Aiko"]
LAST_NAMES     = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
                   "Davis","Martinez","Anderson","Taylor","Wilson","Moore","Patel","Kim"]

DIAGNOSES_ICD10 = [
    ("A41.9",  "Sepsis, unspecified"),
    ("J18.9",  "Pneumonia, unspecified"),
    ("J80",    "Acute respiratory distress syndrome"),
    ("N17.9",  "Acute kidney failure, unspecified"),
    ("I50.9",  "Heart failure, unspecified"),
    ("I21.9",  "Acute myocardial infarction, unspecified"),
    ("I48.91", "Longstanding persistent atrial fibrillation"),
    ("E10.10", "Type 1 diabetes mellitus with ketoacidosis without coma"),
    ("I26.99", "Other pulmonary embolism"),
    ("J44.1",  "COPD with acute exacerbation"),
    ("K85.9",  "Acute pancreatitis, unspecified"),
    ("K92.1",  "Melena"),
    ("G35",    "Multiple sclerosis"),
]

MEDICATIONS = [
    ("vancomycin",           "mg",    1500, "IV",  "q12h"),
    ("piperacillin-tazobactam", "mg", 4500, "IV",  "q6h"),
    ("norepinephrine",       "mcg/kg/min", 0.1, "IV", "continuous"),
    ("propofol",             "mcg/kg/min", 20,  "IV", "continuous"),
    ("fentanyl",             "mcg/h", 50,   "IV",  "continuous"),
    ("heparin",              "units/h", 1000, "IV", "continuous"),
    ("furosemide",           "mg",    40,   "IV",  "q8h"),
    ("pantoprazole",         "mg",    40,   "IV",  "daily"),
    ("insulin",              "units/h", 2,   "IV", "continuous"),
    ("metoprolol",           "mg",    25,   "PO",  "bid"),
    ("atorvastatin",         "mg",    40,   "PO",  "daily"),
    ("enoxaparin",           "mg",    40,   "SC",  "daily"),
    ("azithromycin",         "mg",    500,  "IV",  "daily"),
    ("meropenem",            "mg",    1000, "IV",  "q8h"),
    ("dexamethasone",        "mg",    6,    "IV",  "daily"),
]

LAB_TESTS = [
    ("WBC",          "k/uL",   4.0, 12.0,  8.5),
    ("Hemoglobin",   "g/dL",   7.0, 16.0,  11.5),
    ("Platelet",     "k/uL",   50,  400,   220),
    ("Creatinine",   "mg/dL",  0.3, 8.0,   1.8),
    ("BUN",          "mg/dL",  5,   80,    28),
    ("Sodium",       "mEq/L",  128, 150,   138),
    ("Potassium",    "mEq/L",  2.5, 6.5,   4.2),
    ("Glucose",      "mg/dL",  50,  400,   145),
    ("Lactate",      "mmol/L", 0.5, 12.0,  2.4),
    ("Troponin",     "ng/mL",  0.0, 2.0,   0.08),
    ("BNP",          "pg/mL",  10,  2000,  280),
    ("Procalcitonin","ng/mL",  0.0, 50.0,  3.5),
    ("INR",          "",       0.9, 5.0,   1.6),
    ("ALT",          "U/L",    10,  500,   55),
    ("AST",          "U/L",    10,  800,   68),
]


def _name(sex: str) -> str:
    first = random.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
    return f"{first} {random.choice(LAST_NAMES)}"


def _jitter(val: float, pct: float = 0.08) -> float:
    return max(0.0, val * (1 + random.gauss(0, pct)))


# ─────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────

def generate_all(n_patients: int = 50,
                  hours: int = 72,
                  output_dir: str = "/data/raw",
                  scenario: str = "mixed",
                  seed: int = 42) -> dict:

    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2024, 1, 1)

    patients_rows = []
    vitals_rows = []
    labs_rows = []
    meds_rows = []
    dx_rows = []

    # Try to use the simulation engine for realistic trajectories
    try:
        from simulation.patient_simulator import ICUSimulator, SCENARIOS
        sim = ICUSimulator()
        use_sim = True
        scenario_list = list(SCENARIOS.keys())
    except Exception:
        use_sim = False
        scenario_list = ["mixed"]

    for i in range(n_patients):
        pid = f"P{str(i+1).zfill(4)}"
        sex = rng.choice(["M", "F"])
        age = int(rng.gauss(65, 14))
        age = max(18, min(95, age))
        name = _name(sex)
        admit_time = base_time + timedelta(days=rng.randint(0, 90),
                                             hours=rng.randint(0, 23))
        dx = rng.choice(DIAGNOSES_ICD10)

        patients_rows.append({
            "patient_id": pid, "name": name, "age": age, "sex": sex,
            "dob": (admit_time - timedelta(days=age*365)).strftime("%Y-%m-%d"),
            "admission_date": admit_time.strftime("%Y-%m-%d %H:%M"),
            "location": f"ICU-{rng.randint(1,8)}",
            "primary_icd10": dx[0],
            "primary_diagnosis": dx[1],
        })

        dx_rows.append({
            "patient_id": pid,
            "icd10_code": dx[0],
            "description": dx[1],
            "diagnosis_type": "primary",
            "recorded_at": admit_time.strftime("%Y-%m-%d %H:%M"),
        })
        # Secondary diagnosis sometimes
        if rng.random() < 0.6:
            sdx = rng.choice(DIAGNOSES_ICD10)
            dx_rows.append({
                "patient_id": pid, "icd10_code": sdx[0],
                "description": sdx[1], "diagnosis_type": "secondary",
                "recorded_at": admit_time.strftime("%Y-%m-%d %H:%M"),
            })

        # Vitals — use simulator or fallback
        if use_sim and scenario != "manual":
            sc = scenario if scenario in scenario_list else rng.choice(scenario_list)
            try:
                result = sim.simulate_patient(pid, sc, interval_minutes=60, seed=i)
                for state in result.states:
                    ts = admit_time + timedelta(hours=state.time_hours)
                    vitals_rows.append({
                        "patient_id": pid,
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "heart_rate": round(state.heart_rate, 1),
                        "sbp": round(state.sbp, 1),
                        "dbp": round(state.dbp, 1),
                        "map": round(state.map, 1),
                        "respiratory_rate": round(state.respiratory_rate, 1),
                        "temperature": round(state.temperature, 2),
                        "spo2": round(state.spo2, 1),
                        "gcs": state.gcs,
                        "news2": state.news2,
                    })
                # Labs every 6 hours from simulation
                for h in range(0, min(hours, int(result.duration_hours)), 6):
                    ts = admit_time + timedelta(hours=h)
                    # Pick state closest to h
                    closest = min(result.states,
                                   key=lambda s: abs(s.time_hours - h))
                    for test, unit, lo, hi, mean in LAB_TESTS:
                        val = mean + rng.gauss(0, (hi - lo) * 0.1)
                        # Adjust based on clinical state
                        if test == "Lactate":
                            val = closest.lactate + rng.gauss(0, 0.2)
                        elif test == "Creatinine":
                            val = closest.creatinine + rng.gauss(0, 0.1)
                        val = max(lo * 0.8, min(hi * 1.1, val))
                        labs_rows.append({
                            "patient_id": pid, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "test_name": test, "value": round(val, 3), "unit": unit,
                            "reference_low": lo, "reference_high": hi,
                            "is_critical": val < lo or val > hi,
                        })
                continue
            except Exception:
                pass  # Fallback to manual generation

        # Manual vital generation (no simulator)
        severity = rng.random()
        base_hr = 80 + severity * 40
        base_sbp = 130 - severity * 40
        for h in range(min(hours, 72)):
            ts = admit_time + timedelta(hours=h)
            drift = 1 + (rng.random() - 0.5) * 0.02
            vitals_rows.append({
                "patient_id": pid,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "heart_rate": round(_jitter(base_hr * drift), 1),
                "sbp": round(_jitter(base_sbp / drift), 1),
                "dbp": round(_jitter(base_sbp / drift * 0.62), 1),
                "map": round(_jitter((base_sbp / drift) * 0.78), 1),
                "respiratory_rate": round(_jitter(16 + severity * 8), 1),
                "temperature": round(37.0 + severity * 1.5 + rng.gauss(0, 0.2), 2),
                "spo2": round(min(100, 98 - severity * 8 + rng.gauss(0, 1)), 1),
                "gcs": max(3, 15 - int(severity * 5)),
                "news2": min(14, int(severity * 10)),
            })

        # Labs
        for h in range(0, min(hours, 72), 6):
            ts = admit_time + timedelta(hours=h)
            for test, unit, lo, hi, mean in LAB_TESTS:
                val = max(lo * 0.8, min(hi * 1.1, mean + rng.gauss(0, (hi - lo) * 0.15)))
                labs_rows.append({
                    "patient_id": pid, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_name": test, "value": round(val, 3), "unit": unit,
                    "reference_low": lo, "reference_high": hi,
                    "is_critical": val < lo or val > hi,
                })

        # Medications (3-6 per patient)
        n_meds = rng.randint(3, 6)
        selected_meds = rng.sample(MEDICATIONS, n_meds)
        for med_name, unit, dose, route, freq in selected_meds:
            start_ts = admit_time + timedelta(hours=rng.randint(0, 4))
            meds_rows.append({
                "patient_id": pid,
                "medication": med_name,
                "dose": round(dose * _jitter(1.0, 0.1), 2),
                "unit": unit,
                "route": route,
                "frequency": freq,
                "start_time": start_ts.strftime("%Y-%m-%d %H:%M"),
                "status": "active",
            })

    # Write CSVs
    files_written = {}
    for name, rows, fieldnames in [
        ("patients_synthetic.csv",    patients_rows,   list(patients_rows[0].keys()) if patients_rows else []),
        ("vitals_synthetic.csv",      vitals_rows,     ["patient_id","timestamp","heart_rate","sbp","dbp","map","respiratory_rate","temperature","spo2","gcs","news2"]),
        ("labs_synthetic.csv",        labs_rows,       ["patient_id","timestamp","test_name","value","unit","reference_low","reference_high","is_critical"]),
        ("medications_synthetic.csv", meds_rows,       ["patient_id","medication","dose","unit","route","frequency","start_time","status"]),
        ("diagnoses_synthetic.csv",   dx_rows,         ["patient_id","icd10_code","description","diagnosis_type","recorded_at"]),
    ]:
        path = out / name
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        files_written[name] = {"path": str(path), "rows": len(rows)}

    # Write metadata
    meta = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_patients": n_patients,
        "hours_per_patient": hours,
        "scenario": scenario,
        "seed": seed,
        "files": files_written,
    }
    meta_path = out / "synthetic_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ICU data")
    parser.add_argument("--patients",   type=int, default=50,        help="Number of patients")
    parser.add_argument("--hours",      type=int, default=72,         help="Hours per patient")
    parser.add_argument("--output-dir", default="/data/raw",          help="Output directory")
    parser.add_argument("--scenario",   default="mixed",              help="ICU scenario or 'mixed'")
    parser.add_argument("--seed",       type=int, default=42,         help="Random seed")
    args = parser.parse_args()

    print(f"Generating synthetic ICU data...")
    print(f"  Patients: {args.patients}")
    print(f"  Hours:    {args.hours}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Output:   {args.output_dir}")

    meta = generate_all(
        n_patients=args.patients,
        hours=args.hours,
        output_dir=args.output_dir,
        scenario=args.scenario,
        seed=args.seed,
    )

    print(f"\nFiles generated:")
    for fname, info in meta["files"].items():
        print(f"  {fname:<40} {info['rows']:>6,} rows  →  {info['path']}")
    print(f"\nMetadata: {args.output_dir}/synthetic_metadata.json")

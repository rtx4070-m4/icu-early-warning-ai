"""
Predictive Lab Ordering System
Recommends which lab tests to order next based on:
  - Current clinical state and vital signs
  - Time since last result
  - Clinical deterioration signals
  - Protocol requirements (sepsis bundle, AKI monitoring, etc.)
  - Cost-effectiveness and patient burden minimisation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Lab catalogue
# ─────────────────────────────────────────

@dataclass
class LabTest:
    code: str
    name: str
    category: str           # CHEMISTRY | HAEMATOLOGY | COAGULATION | MICROBIOLOGY | ABG | CARDIAC
    turnaround_minutes: int
    cost_units: int         # Relative cost (1=cheap, 5=expensive)
    critical_for: List[str]  # Conditions where this is essential
    repeat_interval_hours: float  # Minimum repeat interval under normal monitoring
    urgent_interval_hours: float  # Repeat interval during active deterioration
    loinc_code: str = ""


LAB_CATALOGUE: Dict[str, LabTest] = {
    "cbc":          LabTest("CBC",  "Complete Blood Count",    "HAEMATOLOGY",  60,  1, ["sepsis","haematological"], 24, 6),
    "bmp":          LabTest("BMP",  "Basic Metabolic Panel",   "CHEMISTRY",    60,  1, ["aki","electrolytes"],       12, 4),
    "cmp":          LabTest("CMP",  "Comprehensive Metabolic Panel","CHEMISTRY",60,  2, ["liver_failure","hepatic"], 24, 8),
    "lactate":      LabTest("LAC",  "Lactate",                 "CHEMISTRY",    30,  1, ["sepsis","shock","perfusion"],2, 1),
    "blood_culture":LabTest("BCX",  "Blood Cultures x2",       "MICROBIOLOGY",240,  2, ["sepsis","bacteraemia"],    72,24),
    "ua_culture":   LabTest("UAC",  "Urine Analysis + Culture","MICROBIOLOGY",240,  1, ["uti","sepsis"],             48,24),
    "abg":          LabTest("ABG",  "Arterial Blood Gas",      "ABG",           15,  1, ["respiratory","acidosis","vent"],4,1),
    "troponin":     LabTest("TROP", "Troponin I",              "CARDIAC",       60,  2, ["acs","mi","cardiac"],       6,2),
    "bnp":          LabTest("BNP",  "BNP",                     "CARDIAC",       60,  2, ["chf","cardiac"],            24,6),
    "procalcitonin":LabTest("PCT",  "Procalcitonin",           "CHEMISTRY",     60,  3, ["sepsis","infection"],       24,8),
    "crp":          LabTest("CRP",  "C-Reactive Protein",      "CHEMISTRY",     60,  1, ["infection","inflammation"],  24,8),
    "coagulation":  LabTest("COAG", "PT/INR/aPTT",             "COAGULATION",   60,  1, ["bleeding","anticoagulation"],12,4),
    "fibrinogen":   LabTest("FIB",  "Fibrinogen",              "COAGULATION",   60,  2, ["dic","bleeding"],            12,6),
    "d_dimer":      LabTest("DDM",  "D-Dimer",                 "COAGULATION",   60,  2, ["pe","dvt","dic"],            24,8),
    "lipase":       LabTest("LIP",  "Lipase",                  "CHEMISTRY",     60,  1, ["pancreatitis"],              24,8),
    "ammonia":      LabTest("AMM",  "Ammonia",                 "CHEMISTRY",     60,  2, ["liver_failure","hepatic_encephalopathy"],8,4),
    "vancomycin_level": LabTest("VAN","Vancomycin AUC/Trough", "CHEMISTRY",    60,  2, ["vancomycin_monitoring"],      12,6),
    "ferritin":     LabTest("FER",  "Ferritin",                "CHEMISTRY",     60,  2, ["haemophagocytic","sepsis"],  48,24),
    "cortisol":     LabTest("CORT", "Random Cortisol",         "CHEMISTRY",     60,  3, ["adrenal_insufficiency"],    72,24),
    "thyroid_panel":LabTest("TFT",  "TSH + Free T4",           "CHEMISTRY",    120,  2, ["thyroid","atrial_fibrillation"],72,48),
    "echo_doppler": LabTest("ECHO", "Point-of-Care Echo",      "CARDIAC",       30,  4, ["cardiac","haemodynamic"],   24,8),
    "chest_xr":     LabTest("CXR",  "Portable Chest X-Ray",    "IMAGING",       30,  1, ["respiratory","ards","pneumonia"],24,6),
}


# ─────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────

@dataclass
class LabRecommendation:
    lab: LabTest
    priority: str         # STAT | URGENT | ROUTINE
    reason: str
    due_in_hours: float
    triggered_by: str     # sepsis_bundle | aki_monitoring | clinical_change | protocol | routine

    def to_dict(self) -> Dict:
        return {
            "code": self.lab.code,
            "name": self.lab.name,
            "category": self.lab.category,
            "priority": self.priority,
            "reason": self.reason,
            "due_in_hours": round(self.due_in_hours, 1),
            "triggered_by": self.triggered_by,
            "turnaround_minutes": self.lab.turnaround_minutes,
        }


# ─────────────────────────────────────────
# Ordering engine
# ─────────────────────────────────────────

class LabOrderingEngine:
    """
    Recommends next labs to order based on clinical state,
    last-result timing, and active protocols.
    """

    def recommend(self,
                   patient_data: Dict,
                   last_results: Optional[Dict[str, datetime]] = None,
                   active_medications: Optional[List[str]] = None) -> List[LabRecommendation]:
        """
        patient_data: dict with vitals, labs, risk_scores, flags
        last_results: dict of lab_code → datetime of last result
        active_medications: list of current drugs (for monitoring labs)
        """
        vitals = patient_data.get("vitals", patient_data)
        labs = patient_data.get("labs", {})
        flags = patient_data.get("flags", {})
        risk = patient_data.get("risk_scores", {})
        news2 = patient_data.get("news2", 0)
        last_results = last_results or {}
        active_medications = [m.lower() for m in (active_medications or [])]

        recs: List[LabRecommendation] = []
        now = datetime.utcnow()

        def hours_since(lab_code: str) -> float:
            last = last_results.get(lab_code)
            if last is None:
                return 9999
            return (now - last).total_seconds() / 3600

        def add(lab_code: str, priority: str, reason: str, due: float, trigger: str):
            if lab_code in LAB_CATALOGUE:
                # Skip if recently done
                interval = LAB_CATALOGUE[lab_code].urgent_interval_hours if priority == "STAT" \
                            else LAB_CATALOGUE[lab_code].repeat_interval_hours
                if hours_since(lab_code) < interval * 0.8:
                    return
                recs.append(LabRecommendation(
                    lab=LAB_CATALOGUE[lab_code],
                    priority=priority,
                    reason=reason,
                    due_in_hours=due,
                    triggered_by=trigger,
                ))

        # ── Sepsis bundle (qSOFA / NEWS2) ──────────────
        hr = vitals.get("heart_rate", 80)
        sbp = vitals.get("sbp", 120)
        rr = vitals.get("respiratory_rate", 16)
        temp = vitals.get("temperature", 37.0)
        spo2 = vitals.get("spo2", 98)
        map_ = vitals.get("map", 93)
        gcs = vitals.get("gcs", 15)
        lactate = labs.get("lactate", 1.0)
        creatinine = labs.get("creatinine", 1.0)
        wbc = labs.get("wbc", 8.0)
        troponin = labs.get("troponin", 0.01)

        sepsis_risk = risk.get("sepsis_risk", 0)
        qsofa = int(sbp <= 100) + int(rr >= 22) + int(gcs < 15)

        if sepsis_risk > 0.4 or qsofa >= 2 or news2 >= 5:
            add("blood_culture", "STAT",   "Sepsis screen: blood cultures before antibiotics",  0, "sepsis_bundle")
            add("lactate",       "STAT",   "Sepsis: baseline lactate; repeat if >2 mmol/L",      0, "sepsis_bundle")
            add("cbc",           "STAT",   "Sepsis: complete blood count",                       0, "sepsis_bundle")
            add("bmp",           "STAT",   "Sepsis: electrolytes, creatinine, glucose",          0, "sepsis_bundle")
            add("procalcitonin", "URGENT", "Sepsis biomarker for antibiotic guidance",           1, "sepsis_bundle")
            add("coagulation",   "URGENT", "DIC screening in sepsis",                            2, "sepsis_bundle")
            add("ua_culture",    "URGENT", "Source control: urine culture if UTI suspected",     2, "sepsis_bundle")

        # ── Lactate monitoring ─────────────────────────
        if lactate > 2.0:
            add("lactate", "STAT",   f"Lactate {lactate:.1f} mmol/L → repeat in 2h to assess response",  0, "lactate_monitoring")
        elif lactate > 1.5:
            add("lactate", "URGENT", f"Lactate {lactate:.1f} — trending up; recheck in 4h",              2, "lactate_monitoring")

        # ── AKI monitoring ─────────────────────────────
        if creatinine > 1.5:
            add("bmp",       "URGENT", f"AKI monitoring: creatinine {creatinine:.1f} mg/dL",          2, "aki_monitoring")
            add("coagulation","URGENT","AKI: check coagulation status",                                4, "aki_monitoring")
        if creatinine > 3.0:
            add("bmp",       "STAT",   f"Severe AKI: creatinine {creatinine:.1f} mg/dL",              0, "aki_monitoring")

        # ── Haemodynamic instability ────────────────────
        if sbp < 90 or map_ < 65:
            add("abg",   "STAT",   f"Haemodynamic compromise: SBP={sbp:.0f}, MAP={map_:.0f}",      0, "clinical_change")
            add("lactate","STAT",  "Shock assessment: lactate",                                     0, "clinical_change")
            add("echo_doppler","URGENT","Point-of-care echo for cardiac function in shock",         1, "clinical_change")

        # ── Respiratory compromise ──────────────────────
        if spo2 < 92 or rr >= 25:
            add("abg",    "STAT",   f"Respiratory distress: SpO2={spo2:.0f}%, RR={rr:.0f}",        0, "clinical_change")
            add("chest_xr","STAT",  "Respiratory compromise: CXR",                                  0, "clinical_change")

        # ── Cardiac ────────────────────────────────────
        if troponin > 0.04 or flags.get("chest_pain"):
            add("troponin","STAT",   f"Troponin {troponin:.3f} ng/mL → serial troponin q3-6h",     0, "clinical_change")
            add("bnp",     "URGENT", "Cardiac biomarker: BNP",                                      1, "clinical_change")

        # ── Infection / fever ─────────────────────────
        if temp > 38.3:
            add("blood_culture","URGENT", f"New fever {temp:.1f}°C — blood cultures",              1, "clinical_change")
            add("procalcitonin","URGENT", "Fever workup: procalcitonin",                            2, "clinical_change")
            add("crp",          "ROUTINE","Inflammatory marker: CRP",                              4, "clinical_change")
            add("ua_culture",   "URGENT", "UTI screening: urine culture",                          2, "clinical_change")

        # ── Drug monitoring ────────────────────────────
        if any("vancomycin" in m for m in active_medications):
            add("vancomycin_level","URGENT","Vancomycin AUC monitoring",                           4, "drug_monitoring")
            add("bmp",             "URGENT","Vancomycin nephrotoxicity: renal function",            6, "drug_monitoring")

        if any("heparin" in m for m in active_medications):
            add("coagulation","URGENT","Heparin therapy: aPTT monitoring",                         6, "drug_monitoring")

        if any("amiodarone" in m for m in active_medications):
            add("thyroid_panel","ROUTINE","Amiodarone: thyroid function monitoring",               24, "drug_monitoring")

        # ── Routine ICU monitoring ─────────────────────
        if news2 >= 3:
            add("cbc",  "ROUTINE","Routine ICU monitoring: CBC",                                   8, "protocol")
            add("bmp",  "ROUTINE","Routine ICU monitoring: electrolytes",                          8, "protocol")
            add("abg",  "ROUTINE","Routine ICU monitoring: ABG",                                  12, "protocol")

        # Coagulation if mechanically ventilated
        if flags.get("mechanically_ventilated"):
            add("coagulation","ROUTINE","Ventilated patient: coagulation panel",                   8, "protocol")
            add("chest_xr",   "ROUTINE","Daily CXR in ventilated patient",                        12, "protocol")

        # Deduplicate by lab code (keep highest priority)
        priority_order = {"STAT": 0, "URGENT": 1, "ROUTINE": 2}
        seen: Dict[str, LabRecommendation] = {}
        for r in recs:
            code = r.lab.code
            if code not in seen or priority_order[r.priority] < priority_order[seen[code].priority]:
                seen[code] = r

        # Sort: STAT first, then URGENT, then ROUTINE; within tier by due_in_hours
        sorted_recs = sorted(seen.values(),
                              key=lambda r: (priority_order[r.priority], r.due_in_hours))
        return sorted_recs

    def format_order_set(self, recs: List[LabRecommendation]) -> str:
        """Format recommendations as a clinical order set text."""
        if not recs:
            return "No lab orders recommended at this time."

        lines = [f"Lab Order Recommendations — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC", ""]
        for priority in ["STAT", "URGENT", "ROUTINE"]:
            tier = [r for r in recs if r.priority == priority]
            if not tier:
                continue
            lines.append(f"{'=' * 40}")
            lines.append(f"  {priority} ORDERS")
            lines.append(f"{'=' * 40}")
            for r in tier:
                lines.append(f"  [{r.lab.code:4s}] {r.lab.name:<30}  "
                              f"due {r.due_in_hours:.0f}h  ({r.triggered_by})")
                lines.append(f"         Reason: {r.reason}")
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    engine = LabOrderingEngine()

    patient = {
        "vitals": {"heart_rate": 118, "sbp": 88, "dbp": 55, "map": 66,
                    "respiratory_rate": 26, "temperature": 38.8, "spo2": 91, "gcs": 14},
        "labs":   {"lactate": 3.9, "creatinine": 2.1, "wbc": 19.0, "troponin": 0.02},
        "risk_scores": {"sepsis_risk": 0.78},
        "news2": 9,
        "flags": {"mechanically_ventilated": False},
    }

    # Simulate that some labs were done recently
    now = datetime.utcnow()
    last_results = {
        "CBC":  now - timedelta(hours=14),
        "BMP":  now - timedelta(hours=6),
        "LAC":  now - timedelta(hours=3),
    }

    recs = engine.recommend(patient, last_results=last_results,
                             active_medications=["vancomycin", "norepinephrine"])

    print(engine.format_order_set(recs))
    print(f"Total recommendations: {len(recs)} "
          f"({sum(1 for r in recs if r.priority=='STAT')} STAT, "
          f"{sum(1 for r in recs if r.priority=='URGENT')} URGENT, "
          f"{sum(1 for r in recs if r.priority=='ROUTINE')} ROUTINE)")

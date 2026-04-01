"""
Medication Safety Checker
Cross-references prescriptions against:
  - Formulary (approved drugs, standard doses, routes)
  - Patient-specific contraindications (renal/hepatic/allergy)
  - Drug-drug interactions (severity-graded)
  - Weight-based dosing validation
  - ICU-specific alerts (nephrotoxins in AKI, QTc prolongers, etc.)

Returns structured SafetyReport with actionable recommendations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Formulary database
# ─────────────────────────────────────────

@dataclass
class FormularyEntry:
    generic_name: str
    brand_names: List[str]
    drug_class: str
    standard_doses: Dict[str, str]     # route → dose string
    max_dose_mg_kg_day: Optional[float]
    renal_adjustment_required: bool
    hepatic_adjustment_required: bool
    pregnancy_category: str            # A/B/C/D/X/N
    dialyzable: bool
    common_adverse_effects: List[str]
    monitoring_parameters: List[str]
    icu_alert: Optional[str] = None    # Special ICU warning


FORMULARY: Dict[str, FormularyEntry] = {
    "vancomycin": FormularyEntry(
        generic_name="vancomycin",
        brand_names=["Vancocin"],
        drug_class="Glycopeptide antibiotic",
        standard_doses={"IV": "15–20 mg/kg q8–12h (AUC-guided)", "PO": "125–500mg q6h (C.diff only)"},
        max_dose_mg_kg_day=60,
        renal_adjustment_required=True,
        hepatic_adjustment_required=False,
        pregnancy_category="C",
        dialyzable=True,
        common_adverse_effects=["nephrotoxicity", "ototoxicity", "red man syndrome"],
        monitoring_parameters=["AUC/MIC ratio", "SCr q48h", "trough (if AUC unavailable)"],
        icu_alert="Monitor vancomycin AUC/MIC; avoid with piperacillin-tazobactam if possible (↑AKI risk)",
    ),
    "piperacillin-tazobactam": FormularyEntry(
        generic_name="piperacillin-tazobactam",
        brand_names=["Zosyn"],
        drug_class="Beta-lactam/beta-lactamase inhibitor",
        standard_doses={"IV": "3.375g q6h or 4.5g q8h (extended infusion 4h preferred in ICU)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=True,
        hepatic_adjustment_required=False,
        pregnancy_category="B",
        dialyzable=True,
        common_adverse_effects=["diarrhoea", "hypokalaemia", "neurotoxicity (high dose/renal failure)"],
        monitoring_parameters=["renal function", "electrolytes", "CBC"],
        icu_alert="Consider extended infusion (4h) in ICU for pharmacokinetic advantage; avoid + vancomycin (↑AKI)",
    ),
    "meropenem": FormularyEntry(
        generic_name="meropenem",
        brand_names=["Merrem"],
        drug_class="Carbapenem",
        standard_doses={"IV": "1–2g q8h (2g q8h for CNS/resistant organisms)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=True,
        hepatic_adjustment_required=False,
        pregnancy_category="B",
        dialyzable=True,
        common_adverse_effects=["diarrhoea", "seizures (high dose/low seizure threshold)", "neurotoxicity"],
        monitoring_parameters=["renal function", "neurological status"],
    ),
    "norepinephrine": FormularyEntry(
        generic_name="norepinephrine",
        brand_names=["Levophed"],
        drug_class="Vasopressor / catecholamine",
        standard_doses={"IV": "0.01–3 mcg/kg/min continuous infusion (via central line)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=False,
        pregnancy_category="C",
        dialyzable=False,
        common_adverse_effects=["peripheral ischaemia", "tachycardia", "hyperglycaemia"],
        monitoring_parameters=["MAP q1h", "peripheral perfusion", "urine output"],
        icu_alert="Administer via central line only; titrate to MAP ≥65 mmHg; monitor for ischaemia",
    ),
    "heparin": FormularyEntry(
        generic_name="heparin",
        brand_names=["Heparin Sodium"],
        drug_class="Anticoagulant",
        standard_doses={"IV": "80 units/kg bolus then 18 units/kg/h (per nomogram)", "SC": "5000 units q8-12h (prophylaxis)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=True,
        pregnancy_category="C",
        dialyzable=False,
        common_adverse_effects=["bleeding", "HIT (thrombocytopenia)", "osteoporosis (long-term)"],
        monitoring_parameters=["aPTT q6h until therapeutic", "platelet count q2-3d", "anti-Xa level"],
        icu_alert="Monitor for HIT (platelet drop >50% after day 4); avoid in active bleeding",
    ),
    "furosemide": FormularyEntry(
        generic_name="furosemide",
        brand_names=["Lasix"],
        drug_class="Loop diuretic",
        standard_doses={"IV": "20–200mg q6-8h", "PO": "20–600mg daily"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=True,
        hepatic_adjustment_required=False,
        pregnancy_category="C",
        dialyzable=True,
        common_adverse_effects=["electrolyte disturbances", "ototoxicity", "dehydration", "azotaemia"],
        monitoring_parameters=["electrolytes q24h", "renal function", "urine output", "weight"],
    ),
    "propofol": FormularyEntry(
        generic_name="propofol",
        brand_names=["Diprivan"],
        drug_class="Sedative/anaesthetic",
        standard_doses={"IV": "5–50 mcg/kg/min continuous (ICU sedation)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=False,
        pregnancy_category="B",
        dialyzable=False,
        common_adverse_effects=["hypotension", "bradycardia", "propofol infusion syndrome (>4mg/kg/h >48h)"],
        monitoring_parameters=["triglycerides q48h", "arterial blood gas", "ECG if high dose", "RASS score"],
        icu_alert="PRIS risk if >4mg/kg/h for >48h; monitor triglycerides; limit to <4mg/kg/h where possible",
    ),
    "fentanyl": FormularyEntry(
        generic_name="fentanyl",
        brand_names=["Sublimaze"],
        drug_class="Opioid analgesic",
        standard_doses={"IV": "25–200 mcg/h infusion or 25–100 mcg PRN bolus"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=True,
        pregnancy_category="C",
        dialyzable=False,
        common_adverse_effects=["respiratory depression", "hypotension", "constipation", "chest wall rigidity (high dose)"],
        monitoring_parameters=["respiratory rate", "sedation score (CPOT/NRS)", "bowel sounds"],
    ),
    "insulin": FormularyEntry(
        generic_name="insulin (human regular)",
        brand_names=["Humulin R", "Novolin R"],
        drug_class="Antidiabetic / hormone",
        standard_doses={"IV": "0.05–0.1 units/kg/h infusion (DKA/ICU glycaemia protocol)"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=True,
        hepatic_adjustment_required=False,
        pregnancy_category="B",
        dialyzable=False,
        common_adverse_effects=["hypoglycaemia", "hypokalaemia"],
        monitoring_parameters=["glucose q1-2h", "potassium q4h", "anion gap (DKA)"],
        icu_alert="Target glucose 140–180 mg/dL in ICU (NICE-SUGAR); treat hypoglycaemia <70 immediately",
    ),
    "amiodarone": FormularyEntry(
        generic_name="amiodarone",
        brand_names=["Cordarone", "Pacerone"],
        drug_class="Class III antiarrhythmic",
        standard_doses={"IV": "150mg over 10min, then 1mg/min × 6h, then 0.5mg/min", "PO": "200–400mg daily maintenance"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=True,
        pregnancy_category="D",
        dialyzable=False,
        common_adverse_effects=["pulmonary toxicity", "thyroid dysfunction", "hepatotoxicity", "QTc prolongation", "photosensitivity"],
        monitoring_parameters=["TFTs q3-6mo", "CXR q6mo", "LFTs q6mo", "QTc", "ECG"],
        icu_alert="Multiple drug interactions (digoxin, warfarin, statins); QTc prolongation risk; phlebitis with peripheral IV",
    ),
    "metoprolol": FormularyEntry(
        generic_name="metoprolol",
        brand_names=["Lopressor", "Toprol-XL"],
        drug_class="Beta-1 selective adrenergic blocker",
        standard_doses={"IV": "2.5–5mg q5min (max 15mg IV)", "PO": "25–200mg daily"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=True,
        pregnancy_category="C",
        dialyzable=False,
        common_adverse_effects=["bradycardia", "hypotension", "bronchospasm", "heart block"],
        monitoring_parameters=["heart rate", "blood pressure", "PR interval"],
        icu_alert="Avoid in decompensated HF, cardiogenic shock, or severe bradycardia; use with caution in COPD",
    ),
    "warfarin": FormularyEntry(
        generic_name="warfarin",
        brand_names=["Coumadin"],
        drug_class="Vitamin K antagonist anticoagulant",
        standard_doses={"PO": "Individualised per INR target; usual 2–10mg daily"},
        max_dose_mg_kg_day=None,
        renal_adjustment_required=False,
        hepatic_adjustment_required=True,
        pregnancy_category="X",
        dialyzable=False,
        common_adverse_effects=["bleeding", "warfarin-induced skin necrosis", "purple toe syndrome"],
        monitoring_parameters=["INR at least weekly until stable, then monthly"],
        icu_alert="Many ICU drug interactions; hold before invasive procedures; reverse with Vit K/4F-PCC if bleeding",
    ),
}


# ─────────────────────────────────────────
# Drug-drug interaction database
# ─────────────────────────────────────────

# (drug1, drug2): (severity, mechanism, clinical_effect, management)
INTERACTIONS: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {
    ("vancomycin", "piperacillin-tazobactam"): (
        "moderate", "Additive nephrotoxicity", "↑ AKI risk vs either agent alone",
        "Monitor SCr q24h; use alternative if AKI present",
    ),
    ("amiodarone", "warfarin"): (
        "high", "CYP2C9 inhibition → ↓ warfarin metabolism",
        "INR may increase 30–50%; bleeding risk",
        "Reduce warfarin dose 30–50%; monitor INR closely",
    ),
    ("amiodarone", "digoxin"): (
        "high", "P-gp inhibition → ↑ digoxin levels",
        "Digoxin toxicity (bradycardia, AV block, arrhythmia)",
        "Reduce digoxin dose by 50%; monitor digoxin level",
    ),
    ("amiodarone", "metoprolol"): (
        "moderate", "Additive negative chronotropy",
        "Bradycardia, hypotension, AV block",
        "Monitor HR and BP closely; reduce doses if bradycardia",
    ),
    ("heparin", "warfarin"): (
        "moderate", "Additive anticoagulation",
        "Increased bleeding risk during transition period",
        "Overlap as per bridging protocol; monitor INR",
    ),
    ("furosemide", "vancomycin"): (
        "moderate", "Additive ototoxicity and nephrotoxicity",
        "↑ hearing loss and renal injury risk",
        "Avoid concurrent high-dose furosemide; monitor audiometry",
    ),
    ("propofol", "fentanyl"): (
        "moderate", "Additive CNS/respiratory depression",
        "Enhanced sedation, respiratory depression, hypotension",
        "Titrate both carefully; monitor respiratory rate; use RASS",
    ),
    ("insulin", "metoprolol"): (
        "low", "Beta-blockade masks hypoglycaemia signs",
        "Tachycardia response to hypoglycaemia blunted",
        "Monitor glucose more frequently; educate on alternative symptoms",
    ),
    ("norepinephrine", "amiodarone"): (
        "moderate", "Additive vasoconstriction and QTc effects",
        "Increased peripheral ischaemia; QTc prolongation",
        "Monitor ECG; assess peripheral perfusion regularly",
    ),
    ("warfarin", "metronidazole"): (
        "high", "CYP2C9 inhibition",
        "Marked INR elevation; major bleeding risk",
        "Avoid if possible; if required, reduce warfarin 25% and monitor INR q2-3d",
    ),
}

# Normalise to bidirectional
_INTERACTIONS_BIDIRECTIONAL: Dict[Tuple[str, str], Tuple] = {}
for (d1, d2), val in INTERACTIONS.items():
    _INTERACTIONS_BIDIRECTIONAL[(d1, d2)] = val
    _INTERACTIONS_BIDIRECTIONAL[(d2, d1)] = val


# ─────────────────────────────────────────
# Contraindication rules
# ─────────────────────────────────────────

def _renal_check(drug: str, entry: FormularyEntry, gfr: Optional[float]) -> Optional[str]:
    if not entry.renal_adjustment_required or gfr is None:
        return None
    if gfr < 15:
        if drug in ("metformin",):
            return f"CONTRAINDICATED in eGFR <30 (eGFR={gfr:.0f})"
        return f"DOSE ADJUSTMENT required: eGFR={gfr:.0f} mL/min/1.73m² (standard adjustment thresholds apply)"
    if gfr < 30:
        return f"SIGNIFICANT dose reduction required: eGFR={gfr:.0f} mL/min/1.73m²"
    if gfr < 50:
        return f"Dose adjustment may be needed: eGFR={gfr:.0f} mL/min/1.73m²"
    return None


def _hepatic_check(drug: str, entry: FormularyEntry,
                    child_pugh: Optional[str]) -> Optional[str]:
    if not entry.hepatic_adjustment_required or child_pugh is None:
        return None
    if child_pugh.upper() == "C":
        return f"Use with extreme caution or avoid in Child-Pugh C hepatic impairment"
    if child_pugh.upper() == "B":
        return f"Dose reduction required in Child-Pugh B hepatic impairment"
    return None


def _weight_dose_check(drug: str, dose_mg: Optional[float],
                        weight_kg: Optional[float], entry: FormularyEntry) -> Optional[str]:
    if dose_mg is None or weight_kg is None or entry.max_dose_mg_kg_day is None:
        return None
    actual_mgkgday = dose_mg / weight_kg
    if actual_mgkgday > entry.max_dose_mg_kg_day:
        return (f"Dose {dose_mg:.0f}mg ({actual_mgkgday:.1f} mg/kg) exceeds "
                f"maximum {entry.max_dose_mg_kg_day:.0f} mg/kg/day")
    return None


# ─────────────────────────────────────────
# Safety report
# ─────────────────────────────────────────

@dataclass
class MedicationSafetyAlert:
    drug: str
    alert_type: str        # INTERACTION | CONTRAINDICATION | DOSE | MONITORING | ICU_WARNING
    severity: str          # critical | high | moderate | low | info
    message: str
    recommendation: str
    reference: str = ""

    def to_dict(self) -> Dict:
        return {
            "drug": self.drug,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass
class SafetyReport:
    patient_id: str
    medications_checked: List[str]
    alerts: List[MedicationSafetyAlert] = field(default_factory=list)
    monitoring_plan: Dict[str, List[str]] = field(default_factory=dict)
    formulary_misses: List[str] = field(default_factory=list)
    overall_status: str = "safe"   # safe | caution | unsafe

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "medications_checked": self.medications_checked,
            "overall_status": self.overall_status,
            "n_alerts": len(self.alerts),
            "critical_alerts": [a.to_dict() for a in self.alerts if a.severity in ("critical", "high")],
            "all_alerts": [a.to_dict() for a in self.alerts],
            "monitoring_plan": self.monitoring_plan,
            "formulary_misses": self.formulary_misses,
        }

    def summary(self) -> str:
        critical = [a for a in self.alerts if a.severity in ("critical", "high")]
        return (f"MedSafety [{self.patient_id}] status={self.overall_status.upper()} "
                f"meds={len(self.medications_checked)} alerts={len(self.alerts)} "
                f"critical={len(critical)}")


# ─────────────────────────────────────────
# Checker
# ─────────────────────────────────────────

class MedicationSafetyChecker:
    """
    Checks a medication list against formulary, interactions, and patient-specific factors.
    """

    def check(self, patient_id: str,
               medications: List[str],
               weight_kg: Optional[float] = None,
               egfr: Optional[float] = None,
               child_pugh: Optional[str] = None,
               allergies: Optional[List[str]] = None,
               doses_mg: Optional[Dict[str, float]] = None,
               icu_context: bool = True) -> SafetyReport:

        medications_norm = [m.lower().strip() for m in medications]
        allergies_norm = [a.lower().strip() for a in (allergies or [])]
        doses_mg = doses_mg or {}

        report = SafetyReport(patient_id=patient_id,
                               medications_checked=medications_norm)
        alerts: List[MedicationSafetyAlert] = []

        # 1. Formulary lookup
        entries: Dict[str, FormularyEntry] = {}
        for med in medications_norm:
            if med in FORMULARY:
                entries[med] = FORMULARY[med]
            else:
                # Try brand name match
                match = next((k for k, v in FORMULARY.items()
                               if med in [b.lower() for b in v.brand_names]), None)
                if match:
                    entries[med] = FORMULARY[match]
                else:
                    report.formulary_misses.append(med)

        # 2. Allergy checks
        for med, entry in entries.items():
            if med in allergies_norm:
                alerts.append(MedicationSafetyAlert(
                    drug=med, alert_type="CONTRAINDICATION", severity="critical",
                    message=f"ALLERGY: {med} is listed as a patient allergy",
                    recommendation="Discontinue immediately and document allergy",
                ))

        # 3. Drug-drug interactions
        checked_pairs = set()
        for i, med1 in enumerate(medications_norm):
            for med2 in medications_norm[i+1:]:
                pair = tuple(sorted([med1, med2]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                interaction = _INTERACTIONS_BIDIRECTIONAL.get((med1, med2))
                if interaction:
                    sev, mechanism, effect, management = interaction
                    alerts.append(MedicationSafetyAlert(
                        drug=f"{med1} + {med2}",
                        alert_type="INTERACTION",
                        severity=sev,
                        message=f"{med1} + {med2}: {effect} ({mechanism})",
                        recommendation=management,
                    ))

        # 4. Renal adjustments
        if egfr is not None:
            for med, entry in entries.items():
                warning = _renal_check(med, entry, egfr)
                if warning:
                    severity = "high" if egfr < 30 else "moderate"
                    alerts.append(MedicationSafetyAlert(
                        drug=med, alert_type="DOSE", severity=severity,
                        message=f"{med}: {warning}",
                        recommendation=f"Consult renal dosing guidelines for eGFR={egfr:.0f}",
                    ))

        # 5. Hepatic adjustments
        if child_pugh:
            for med, entry in entries.items():
                warning = _hepatic_check(med, entry, child_pugh)
                if warning:
                    alerts.append(MedicationSafetyAlert(
                        drug=med, alert_type="CONTRAINDICATION", severity="high",
                        message=f"{med}: {warning}",
                        recommendation="Consult hepatic dosing guidelines; consider dose reduction",
                    ))

        # 6. Weight-based dose checks
        if weight_kg:
            for med, dose in doses_mg.items():
                med_norm = med.lower()
                entry = entries.get(med_norm)
                if entry:
                    warning = _weight_dose_check(med_norm, dose, weight_kg, entry)
                    if warning:
                        alerts.append(MedicationSafetyAlert(
                            drug=med_norm, alert_type="DOSE", severity="high",
                            message=f"{med_norm}: {warning}",
                            recommendation="Reduce dose to within recommended weight-based limit",
                        ))

        # 7. ICU-specific alerts
        if icu_context:
            for med, entry in entries.items():
                if entry.icu_alert:
                    alerts.append(MedicationSafetyAlert(
                        drug=med, alert_type="ICU_WARNING", severity="moderate",
                        message=f"{med}: {entry.icu_alert}",
                        recommendation="Review per ICU protocol",
                    ))

        # 8. Build monitoring plan
        monitoring_plan: Dict[str, List[str]] = {}
        for med, entry in entries.items():
            if entry.monitoring_parameters:
                monitoring_plan[med] = entry.monitoring_parameters

        report.alerts = sorted(alerts, key=lambda a: {
            "critical": 0, "high": 1, "moderate": 2, "low": 3, "info": 4
        }.get(a.severity, 5))
        report.monitoring_plan = monitoring_plan

        # Overall status
        severities = {a.severity for a in alerts}
        if "critical" in severities:
            report.overall_status = "unsafe"
        elif "high" in severities:
            report.overall_status = "caution"
        elif "moderate" in severities or "low" in severities:
            report.overall_status = "review"
        else:
            report.overall_status = "safe"

        return report


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    checker = MedicationSafetyChecker()

    print("=== Medication Safety Check — ICU Polypharmacy ===\n")

    report = checker.check(
        patient_id="P001",
        medications=["vancomycin", "piperacillin-tazobactam", "norepinephrine",
                     "propofol", "fentanyl", "heparin", "furosemide", "insulin",
                     "amiodarone", "metoprolol"],
        weight_kg=72,
        egfr=28,          # AKI
        child_pugh=None,
        allergies=["penicillin"],
        doses_mg={"vancomycin": 1800},  # 25 mg/kg for 72kg patient
        icu_context=True,
    )

    print(report.summary())
    print(f"\nFormulary misses: {report.formulary_misses or 'none'}")
    print(f"\nAlerts ({len(report.alerts)}):")
    for a in report.alerts:
        icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡",
                "low": "🔵", "info": "⚪"}.get(a.severity, "⚪")
        print(f"  {icon} [{a.severity.upper():8s}] [{a.alert_type:16s}] {a.message}")
        print(f"          → {a.recommendation}")

    print(f"\nMonitoring plan ({len(report.monitoring_plan)} drugs):")
    for drug, params in list(report.monitoring_plan.items())[:5]:
        print(f"  {drug}: {', '.join(params)}")

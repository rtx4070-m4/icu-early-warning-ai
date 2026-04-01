"""
Clinical Decision Support Engine (CDSS)
Synthesizes ML risk scores, knowledge graph, NER, and rule-based logic
into structured clinical recommendations for ICU clinicians.

Output:
  - Prioritized action list
  - Medication safety check
  - Differential diagnosis
  - Triggered clinical protocols
  - Reasoning chain (explainable)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────

@dataclass
class ClinicalRecommendation:
    category: str         # MEDICATION | MONITORING | PROCEDURE | CONSULT | PROTOCOL
    action: str           # What to do
    rationale: str        # Why
    urgency: str          # STAT | URGENT | ROUTINE
    evidence_source: str  # rule | ml_model | knowledge_graph | guideline
    confidence: float     # 0–1
    references: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "category": self.category,
            "action": self.action,
            "rationale": self.rationale,
            "urgency": self.urgency,
            "evidence_source": self.evidence_source,
            "confidence": round(self.confidence, 3),
            "references": self.references,
        }


@dataclass
class CDSSOutput:
    patient_id: str
    timestamp: str
    risk_level: str               # low | moderate | high | critical
    recommendations: List[ClinicalRecommendation]
    differential_diagnosis: List[Dict]
    drug_safety: Dict
    triggered_protocols: List[str]
    reasoning: List[str]          # Plain-language reasoning chain
    scores: Dict                  # Numeric scores summary

    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "differential_diagnosis": self.differential_diagnosis,
            "drug_safety": self.drug_safety,
            "triggered_protocols": self.triggered_protocols,
            "reasoning": self.reasoning,
            "scores": self.scores,
        }

    def summary(self) -> str:
        stat_recs = [r for r in self.recommendations if r.urgency == "STAT"]
        urgent_recs = [r for r in self.recommendations if r.urgency == "URGENT"]
        lines = [
            f"Patient {self.patient_id} | Risk: {self.risk_level.upper()}",
            f"Protocols: {', '.join(self.triggered_protocols) or 'None'}",
        ]
        if stat_recs:
            lines.append(f"STAT ({len(stat_recs)}): {stat_recs[0].action}")
        if urgent_recs:
            lines.append(f"URGENT ({len(urgent_recs)}): {urgent_recs[0].action}")
        if self.differential_diagnosis:
            top = self.differential_diagnosis[0]
            lines.append(f"Top DDx: {top['disease']} (score={top['score']:.2f})")
        return " | ".join(lines)


# ─────────────────────────────────────────
# Clinical protocols
# ─────────────────────────────────────────

PROTOCOLS = {
    "sepsis_bundle": {
        "name": "Sepsis Bundle (Hour-1)",
        "trigger": lambda ctx: ctx.get("sepsis_risk", 0) > 0.5 or ctx.get("qsofa", 0) >= 2,
        "actions": [
            ("PROCEDURE", "Draw blood cultures × 2 (peripheral + central)", "STAT"),
            ("MEDICATION", "Initiate broad-spectrum antibiotics within 1 hour", "STAT"),
            ("MEDICATION", "IV fluid resuscitation: 30 mL/kg crystalloid", "STAT"),
            ("MONITORING", "Measure lactate; repeat if >2 mmol/L", "STAT"),
            ("MONITORING", "Reassess hemodynamics q1h", "URGENT"),
        ],
        "guidelines": ["Surviving Sepsis Campaign 2021", "SCCM Hour-1 Bundle"],
    },
    "vasopressor_protocol": {
        "name": "Vasopressor Protocol",
        "trigger": lambda ctx: ctx.get("map", 100) < 65 and ctx.get("fluid_resuscitated", False),
        "actions": [
            ("MEDICATION", "Norepinephrine: start 0.1–0.2 mcg/kg/min, titrate to MAP ≥65", "STAT"),
            ("PROCEDURE", "Insert arterial line for continuous BP monitoring", "STAT"),
            ("PROCEDURE", "Insert central venous catheter if not present", "URGENT"),
            ("MONITORING", "Monitor CVP and ScvO2", "URGENT"),
        ],
        "guidelines": ["SCCM Vasopressor Guidelines 2022"],
    },
    "ventilator_protocol": {
        "name": "Lung-Protective Ventilation (ARDS)",
        "trigger": lambda ctx: ctx.get("spo2", 100) < 88 or ctx.get("pf_ratio", 300) < 200,
        "actions": [
            ("PROCEDURE", "Initiate mechanical ventilation: TV 6 mL/kg IBW", "STAT"),
            ("MONITORING", "Target plateau pressure <30 cmH2O", "URGENT"),
            ("MEDICATION", "Consider neuromuscular blockade if P/F <150", "URGENT"),
            ("PROCEDURE", "Prone positioning if P/F <150 despite optimisation", "URGENT"),
        ],
        "guidelines": ["ARDSNet Protocol", "PROSEVA Trial 2013"],
    },
    "aki_protocol": {
        "name": "AKI Management Protocol",
        "trigger": lambda ctx: ctx.get("creatinine", 1.0) > 2.0 or ctx.get("aki_stage", 0) >= 2,
        "actions": [
            ("MEDICATION", "Discontinue nephrotoxic agents (NSAIDs, aminoglycosides, contrast)", "STAT"),
            ("MONITORING", "Strict fluid balance: hourly urine output via Foley", "URGENT"),
            ("PROCEDURE", "Consider CRRT if oliguria >12h or volume overload", "URGENT"),
            ("MONITORING", "Monitor potassium q6h — treat if >6.0 mEq/L", "URGENT"),
            ("MEDICATION", "Furosemide challenge if euvolemic AKI", "ROUTINE"),
        ],
        "guidelines": ["KDIGO AKI Guidelines 2012"],
    },
    "dvt_prophylaxis": {
        "name": "DVT Prophylaxis Protocol",
        "trigger": lambda ctx: ctx.get("hours_in_icu", 0) >= 24,
        "actions": [
            ("MEDICATION", "Enoxaparin 40 mg SC daily (unless contraindicated)", "ROUTINE"),
            ("PROCEDURE", "Sequential compression devices if anticoagulation contraindicated", "ROUTINE"),
        ],
        "guidelines": ["ACCP VTE Prophylaxis Guidelines 2012"],
    },
    "glucose_management": {
        "name": "Glycemic Management Protocol",
        "trigger": lambda ctx: ctx.get("glucose", 100) > 180 or ctx.get("glucose", 100) < 70,
        "actions": [
            ("MEDICATION", "Insulin infusion: target glucose 140–180 mg/dL", "URGENT"),
            ("MONITORING", "Bedside glucose checks q1–2h until stable", "URGENT"),
            ("MONITORING", "Treat hypoglycemia <70 mg/dL immediately: D50W 25 mL IV", "STAT"),
        ],
        "guidelines": ["NICE-SUGAR Trial", "ADA ICU Glycemic Guidelines"],
    },
    "stress_ulcer_prophylaxis": {
        "name": "Stress Ulcer Prophylaxis",
        "trigger": lambda ctx: ctx.get("mechanically_ventilated", False) or
                               ctx.get("coagulopathy", False),
        "actions": [
            ("MEDICATION", "Pantoprazole 40 mg IV daily", "ROUTINE"),
        ],
        "guidelines": ["SCCM Stress Ulcer Prophylaxis Guidelines"],
    },
}

# Critical lab thresholds → immediate actions
CRITICAL_LAB_ACTIONS = {
    "potassium_high":    (6.0, "MEDICATION", "Treat hyperkalemia: calcium gluconate + kayexalate ± dialysis", "STAT"),
    "potassium_low":     (3.0, "MEDICATION", "Replace potassium: KCl 40 mEq IV over 4h, monitor ECG", "URGENT"),
    "sodium_low":        (125, "MEDICATION", "Hypertonic saline if symptomatic hyponatremia; correct ≤8 mEq/24h", "URGENT"),
    "glucose_high":      (400, "MEDICATION", "Insulin infusion for hyperglycemia; check for DKA/HHS", "STAT"),
    "glucose_low":       (60,  "MEDICATION", "D50W 25 mL IV now; recheck glucose in 15 min", "STAT"),
    "lactate_high":      (4.0, "MONITORING", "Lactate >4: initiate sepsis bundle, reassess perfusion", "STAT"),
    "inr_high":          (3.5, "MEDICATION", "Hold anticoagulation; consider FFP/Vitamin K if bleeding", "URGENT"),
    "troponin_elevated": (0.04,"PROCEDURE",  "Obtain 12-lead ECG; cardiology consult for ACS workup", "STAT"),
    "creatinine_high":   (4.0, "PROCEDURE",  "Nephrology consult; consider CRRT initiation", "URGENT"),
}


# ─────────────────────────────────────────
# CDSS Engine
# ─────────────────────────────────────────

class ClinicalDecisionSupportEngine:
    """
    Synthesizes patient data, ML scores, and clinical knowledge into
    structured, prioritised clinical recommendations.
    """

    def __init__(self):
        self._kg = None
        self._ner = None
        self._init_components()

    def _init_components(self):
        try:
            from knowledge_graph.graph_builder import MedicalKnowledgeGraph
            self._kg = MedicalKnowledgeGraph()
            logger.info("CDSS: Knowledge graph loaded")
        except Exception as e:
            logger.warning(f"CDSS: KG unavailable — {e}")

        try:
            from nlp_pipeline.medical_ner import ClinicalEntityExtractor
            self._ner = ClinicalEntityExtractor()
            logger.info("CDSS: NER loaded")
        except Exception as e:
            logger.warning(f"CDSS: NER unavailable — {e}")

    # ── Context building ──────────────────

    def _build_context(self, patient_data: Dict) -> Dict:
        """Flatten patient data into a context dict for rule evaluation."""
        vitals = patient_data.get("vitals", {})
        labs = patient_data.get("labs", {})
        flags = patient_data.get("flags", {})

        hr = vitals.get("heart_rate", 80)
        sbp = vitals.get("sbp", 120)
        dbp = vitals.get("dbp", 75)
        map_ = (sbp + 2 * dbp) / 3
        rr = vitals.get("respiratory_rate", 16)
        spo2 = vitals.get("spo2", 98)
        temp = vitals.get("temperature", 37.0)
        gcs = vitals.get("gcs", 15)

        qsofa = int(sbp <= 100) + int(rr >= 22) + int(gcs < 15)
        sirs = (int(temp > 38.3 or temp < 36) + int(hr > 90) +
                int(rr > 20) + int(labs.get("wbc", 8) > 12 or labs.get("wbc", 8) < 4))

        return {
            "heart_rate": hr, "sbp": sbp, "dbp": dbp, "map": map_,
            "respiratory_rate": rr, "spo2": spo2, "temperature": temp, "gcs": gcs,
            "qsofa": qsofa, "sirs": sirs,
            "lactate": labs.get("lactate", 1.0),
            "creatinine": labs.get("creatinine", 1.0),
            "potassium": labs.get("potassium", 4.0),
            "sodium": labs.get("sodium", 140),
            "glucose": labs.get("glucose", 100),
            "wbc": labs.get("wbc", 8.0),
            "inr": labs.get("inr", 1.0),
            "troponin": labs.get("troponin", 0.01),
            "bnp": labs.get("bnp", 50),
            "sepsis_risk": patient_data.get("risk_scores", {}).get("sepsis_risk", 0),
            "mortality_risk": patient_data.get("risk_scores", {}).get("mortality_risk", 0),
            "news2": patient_data.get("news2", 0),
            "hours_in_icu": patient_data.get("hours_in_icu", 0),
            "mechanically_ventilated": flags.get("mechanically_ventilated", False),
            "fluid_resuscitated": flags.get("fluid_resuscitated", False),
            "coagulopathy": labs.get("inr", 1.0) > 2.0,
            "aki_stage": _aki_stage(labs.get("creatinine", 1.0),
                                     labs.get("baseline_creatinine", 1.0)),
            "pf_ratio": _pf_ratio(spo2),
        }

    # ── Protocol evaluation ───────────────

    def _evaluate_protocols(self, ctx: Dict) -> Tuple[List[str], List[ClinicalRecommendation]]:
        triggered = []
        recs = []
        for key, protocol in PROTOCOLS.items():
            try:
                if protocol["trigger"](ctx):
                    triggered.append(protocol["name"])
                    for (cat, action, urgency) in protocol["actions"]:
                        recs.append(ClinicalRecommendation(
                            category=cat,
                            action=action,
                            rationale=f"Triggered by: {protocol['name']}",
                            urgency=urgency,
                            evidence_source="guideline",
                            confidence=0.90,
                            references=protocol.get("guidelines", []),
                        ))
            except Exception as e:
                logger.debug(f"Protocol {key} eval error: {e}")
        return triggered, recs

    # ── Critical lab actions ───────────────

    def _evaluate_critical_labs(self, ctx: Dict) -> List[ClinicalRecommendation]:
        recs = []
        checks = [
            ("potassium", "potassium_high", lambda v: v > 6.0),
            ("potassium", "potassium_low",  lambda v: v < 3.0),
            ("sodium",    "sodium_low",     lambda v: v < 125),
            ("glucose",   "glucose_high",   lambda v: v > 400),
            ("glucose",   "glucose_low",    lambda v: v < 60),
            ("lactate",   "lactate_high",   lambda v: v > 4.0),
            ("inr",       "inr_high",       lambda v: v > 3.5),
            ("troponin",  "troponin_elevated", lambda v: v > 0.04),
            ("creatinine","creatinine_high", lambda v: v > 4.0),
        ]
        for lab_key, check_key, condition in checks:
            val = ctx.get(lab_key)
            if val is not None and condition(val):
                _, cat, action, urgency = CRITICAL_LAB_ACTIONS[check_key]
                recs.append(ClinicalRecommendation(
                    category=cat,
                    action=action,
                    rationale=f"Critical lab value: {lab_key} = {val:.2f}",
                    urgency=urgency,
                    evidence_source="rule",
                    confidence=0.98,
                ))
        return recs

    # ── Vital sign alerts ──────────────────

    def _evaluate_vitals(self, ctx: Dict) -> List[ClinicalRecommendation]:
        recs = []
        hr, sbp, rr, spo2 = ctx["heart_rate"], ctx["sbp"], ctx["respiratory_rate"], ctx["spo2"]

        if spo2 < 88:
            recs.append(ClinicalRecommendation(
                category="PROCEDURE",
                action="Increase FiO2; assess for intubation if SpO2 <88% persists",
                rationale=f"Severe hypoxemia: SpO2 = {spo2:.0f}%",
                urgency="STAT", evidence_source="rule", confidence=0.97,
            ))
        elif spo2 < 92:
            recs.append(ClinicalRecommendation(
                category="MONITORING",
                action="Supplement oxygen; target SpO2 ≥94%. Assess cause of hypoxemia.",
                rationale=f"Hypoxemia: SpO2 = {spo2:.0f}%",
                urgency="URGENT", evidence_source="rule", confidence=0.93,
            ))

        if sbp < 80:
            recs.append(ClinicalRecommendation(
                category="MEDICATION",
                action="Fluid bolus 500 mL NS stat; initiate vasopressors if no response",
                rationale=f"Severe hypotension: SBP = {sbp:.0f} mmHg",
                urgency="STAT", evidence_source="rule", confidence=0.97,
            ))

        if hr > 150:
            recs.append(ClinicalRecommendation(
                category="PROCEDURE",
                action="Obtain 12-lead ECG; evaluate for SVT/AF — consider rate control",
                rationale=f"Severe tachycardia: HR = {hr:.0f} bpm",
                urgency="STAT", evidence_source="rule", confidence=0.92,
            ))

        if rr > 28:
            recs.append(ClinicalRecommendation(
                category="MONITORING",
                action="Assess work of breathing; prepare for intubation if tiring",
                rationale=f"Severe tachypnea: RR = {rr:.0f}",
                urgency="URGENT", evidence_source="rule", confidence=0.90,
            ))

        if ctx.get("news2", 0) >= 7:
            recs.append(ClinicalRecommendation(
                category="CONSULT",
                action=f"NEWS2 = {ctx['news2']}: Immediate senior clinician review required",
                rationale="NEWS2 ≥7 indicates high risk of deterioration",
                urgency="STAT", evidence_source="guideline", confidence=0.95,
                references=["RCP NEWS2 Guidelines 2017"],
            ))

        return recs

    # ── Drug safety ────────────────────────

    def _check_drug_safety(self, medications: List[str], ctx: Dict) -> Dict:
        interactions = []
        contraindications = []

        if self._kg:
            interactions = self._kg.check_drug_interactions(medications)

        # Rule-based contraindications
        creat = ctx.get("creatinine", 1.0)
        if creat > 3.0:
            nephrotoxic = {"vancomycin", "gentamicin", "tobramycin", "ibuprofen",
                           "metformin", "lisinopril"}
            for m in medications:
                if any(n in m.lower() for n in nephrotoxic):
                    contraindications.append({
                        "medication": m,
                        "reason": f"Nephrotoxic — creatinine = {creat:.1f} mg/dL",
                        "severity": "high",
                    })

        if ctx.get("inr", 1.0) > 2.0:
            for m in medications:
                if "aspirin" in m.lower() or "nsaid" in m.lower():
                    contraindications.append({
                        "medication": m,
                        "reason": "Bleeding risk — elevated INR",
                        "severity": "high",
                    })

        high_risk = [i for i in interactions if i.get("severity") in ("high", "contraindicated")]
        return {
            "medications_checked": medications,
            "interactions": interactions,
            "contraindications": contraindications,
            "high_risk_interactions": high_risk,
            "overall_safety": "unsafe" if (high_risk or contraindications) else "safe",
        }

    # ── Differential diagnosis ─────────────

    def _differential_diagnosis(self, ctx: Dict,
                                 symptoms: List[str],
                                 note_text: Optional[str] = None) -> List[Dict]:
        all_symptoms = list(symptoms)

        # Derive additional symptoms from vitals
        if ctx.get("temperature", 37) > 38.3:
            all_symptoms.append("fever")
        if ctx.get("heart_rate", 80) > 100:
            all_symptoms.append("tachycardia")
        if ctx.get("sbp", 120) < 90:
            all_symptoms.append("hypotension")
        if ctx.get("spo2", 98) < 94:
            all_symptoms.append("dyspnea")
        if ctx.get("respiratory_rate", 16) > 22:
            all_symptoms.append("tachypnea")

        # Also pull from NER if note provided
        if note_text and self._ner:
            extracted_sx = self._ner.extract_symptoms(note_text)
            all_symptoms += extracted_sx

        all_symptoms = list(set(all_symptoms))

        if self._kg and all_symptoms:
            return self._kg.differential_diagnosis(all_symptoms, top_n=5)
        return []

    # ── Reasoning chain ────────────────────

    def _build_reasoning(self, ctx: Dict, triggered: List[str],
                          recs: List[ClinicalRecommendation]) -> List[str]:
        reasoning = []

        news2 = ctx.get("news2", 0)
        sepsis_risk = ctx.get("sepsis_risk", 0)
        qsofa = ctx.get("qsofa", 0)

        reasoning.append(f"Patient assessment: NEWS2={news2}, qSOFA={qsofa}, "
                         f"Sepsis risk={sepsis_risk:.0%}")

        if news2 >= 7:
            reasoning.append(f"NEWS2 ≥7 ({news2}) → immediate escalation criteria met")
        elif news2 >= 5:
            reasoning.append(f"NEWS2 {news2} → urgent review criteria met")

        if ctx.get("lactate", 0) > 2.0:
            reasoning.append(f"Lactate {ctx['lactate']:.1f} mmol/L suggests tissue hypoperfusion")

        if ctx.get("map", 100) < 65:
            reasoning.append(f"MAP {ctx['map']:.0f} mmHg (<65) → hemodynamic compromise")

        if triggered:
            reasoning.append(f"Clinical protocols triggered: {', '.join(triggered)}")

        stat_count = sum(1 for r in recs if r.urgency == "STAT")
        urgent_count = sum(1 for r in recs if r.urgency == "URGENT")
        if stat_count:
            reasoning.append(f"{stat_count} STAT action(s) recommended")
        if urgent_count:
            reasoning.append(f"{urgent_count} URGENT action(s) recommended")

        return reasoning

    # ── Risk level ─────────────────────────

    def _compute_risk_level(self, ctx: Dict, recs: List[ClinicalRecommendation]) -> str:
        if any(r.urgency == "STAT" for r in recs):
            return "critical"
        news2 = ctx.get("news2", 0)
        sepsis = ctx.get("sepsis_risk", 0)
        if news2 >= 7 or sepsis >= 0.6:
            return "critical"
        if news2 >= 5 or sepsis >= 0.35:
            return "high"
        if news2 >= 3 or sepsis >= 0.15:
            return "moderate"
        return "low"

    # ── Main entry point ───────────────────

    def evaluate(self,
                 patient_data: Dict,
                 symptoms: Optional[List[str]] = None,
                 medications: Optional[List[str]] = None,
                 note_text: Optional[str] = None) -> CDSSOutput:
        """
        Run full CDSS evaluation for a patient.

        Args:
            patient_data: Dict with vitals, labs, risk_scores, flags, news2, hours_in_icu
            symptoms: Known symptoms (will be supplemented from vitals + NER)
            medications: Current medication list for safety checking
            note_text: Free-text clinical note for NER extraction

        Returns:
            CDSSOutput with recommendations, DDx, drug safety, protocols
        """
        patient_id = patient_data.get("patient_id", "UNKNOWN")
        symptoms = symptoms or []
        medications = medications or []

        # Extract medications from note if available
        if note_text and self._ner:
            note_meds = [m["medication"] for m in self._ner.extract_medications(note_text)]
            medications = list(set(medications + note_meds))

        ctx = self._build_context(patient_data)

        # Gather recommendations from all sources
        all_recs: List[ClinicalRecommendation] = []
        triggered_protocols, protocol_recs = self._evaluate_protocols(ctx)
        all_recs += protocol_recs
        all_recs += self._evaluate_critical_labs(ctx)
        all_recs += self._evaluate_vitals(ctx)

        # Deduplicate and sort by urgency
        urgency_order = {"STAT": 0, "URGENT": 1, "ROUTINE": 2}
        seen_actions = set()
        unique_recs = []
        for r in sorted(all_recs, key=lambda x: urgency_order.get(x.urgency, 3)):
            key = r.action[:60]
            if key not in seen_actions:
                seen_actions.add(key)
                unique_recs.append(r)

        ddx = self._differential_diagnosis(ctx, symptoms, note_text)
        drug_safety = self._check_drug_safety(medications, ctx)
        reasoning = self._build_reasoning(ctx, triggered_protocols, unique_recs)
        risk_level = self._compute_risk_level(ctx, unique_recs)

        scores = {
            "news2": ctx.get("news2"),
            "qsofa": ctx.get("qsofa"),
            "sirs": ctx.get("sirs"),
            "map": round(ctx.get("map", 0), 1),
            "shock_index": round(ctx.get("heart_rate", 0) /
                                  max(ctx.get("sbp", 1), 1), 3),
            "sepsis_risk": ctx.get("sepsis_risk"),
            "mortality_risk": patient_data.get("risk_scores", {}).get("mortality_risk"),
            "lactate": ctx.get("lactate"),
        }

        return CDSSOutput(
            patient_id=patient_id,
            timestamp=datetime.utcnow().isoformat(),
            risk_level=risk_level,
            recommendations=unique_recs,
            differential_diagnosis=ddx,
            drug_safety=drug_safety,
            triggered_protocols=triggered_protocols,
            reasoning=reasoning,
            scores=scores,
        )


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────

def _aki_stage(creatinine: float, baseline: float) -> int:
    if baseline <= 0:
        baseline = 1.0
    ratio = creatinine / baseline
    if ratio >= 3.0 or creatinine >= 4.0:
        return 3
    if ratio >= 2.0:
        return 2
    if ratio >= 1.5 or (creatinine - baseline) >= 0.3:
        return 1
    return 0


def _pf_ratio(spo2: float) -> float:
    """Estimate PaO2/FiO2 from SpO2 (assumes room air, FiO2=0.21)."""
    # SpO2 → PaO2 approximation (Ellis equation)
    spo2_clamped = max(0.01, min(0.999, spo2 / 100))
    try:
        import math
        pao2 = math.exp(0.0058 * (spo2_clamped * 100 - 100) + 4.285)
    except Exception:
        pao2 = spo2 * 1.2
    return pao2 / 0.21


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.WARNING)

    engine = ClinicalDecisionSupportEngine()

    # Simulate a deteriorating septic patient
    patient = {
        "patient_id": "P003",
        "vitals": {
            "heart_rate": 118, "sbp": 88, "dbp": 55,
            "respiratory_rate": 26, "temperature": 38.8,
            "spo2": 91, "gcs": 14,
        },
        "labs": {
            "lactate": 3.8, "creatinine": 2.2, "wbc": 18.5,
            "potassium": 5.2, "sodium": 138, "glucose": 195,
            "inr": 1.6, "troponin": 0.02,
        },
        "risk_scores": {"sepsis_risk": 0.78, "mortality_risk": 0.31},
        "news2": 9,
        "hours_in_icu": 6,
        "flags": {"mechanically_ventilated": False, "fluid_resuscitated": False},
    }

    note = """
    72M presents with fever, chills, hypotension. Started on vancomycin 25mg/kg q8h
    and piperacillin-tazobactam 4.5g q6h. Blood cultures drawn. Norepinephrine initiated.
    Assessment: septic shock, likely pulmonary source.
    """

    result = engine.evaluate(
        patient_data=patient,
        symptoms=["fever", "chills", "hypotension", "tachycardia"],
        medications=["vancomycin", "norepinephrine"],
        note_text=note,
    )

    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    print(f"\nRisk Level: {result.risk_level.upper()}")
    print(f"\nTriggered Protocols: {result.triggered_protocols}")
    print(f"\nTop Differential: {result.differential_diagnosis[:3]}")
    print(f"\nDrug Safety: {result.drug_safety['overall_safety']}")
    if result.drug_safety["interactions"]:
        for ix in result.drug_safety["interactions"]:
            print(f"  ⚠  {ix['drug1']} + {ix['drug2']}: [{ix['severity']}] {ix['effect']}")

    print(f"\nRecommendations ({len(result.recommendations)}):")
    for r in result.recommendations[:6]:
        print(f"  [{r.urgency}] {r.category}: {r.action}")

    print(f"\nReasoning chain:")
    for line in result.reasoning:
        print(f"  • {line}")

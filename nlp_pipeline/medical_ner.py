"""
Medical Named Entity Recognition (NER) Pipeline
Extracts clinical entities: medications, diagnoses, symptoms, vitals, procedures, dosages
Rule-based + pattern matching; optional spaCy/transformers integration
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Entity dataclasses
# ─────────────────────────────────────────

@dataclass
class Entity:
    text: str
    label: str          # MEDICATION, DIAGNOSIS, SYMPTOM, VITAL, PROCEDURE, DOSAGE, FREQUENCY, LAB
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    attributes: Dict = field(default_factory=dict)


@dataclass
class NERResult:
    text: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)  # medication-dosage pairs etc.

    def to_dict(self):
        return {
            "text": self.text,
            "entities": [asdict(e) for e in self.entities],
            "relations": self.relations
        }

    def get_by_label(self, label: str) -> List[Entity]:
        return [e for e in self.entities if e.label == label]


# ─────────────────────────────────────────
# Lexicons
# ─────────────────────────────────────────

MEDICATION_LEXICON = {
    # Antibiotics
    "vancomycin": "vancomycin", "vanco": "vancomycin",
    "piperacillin": "piperacillin-tazobactam", "pip-tazo": "piperacillin-tazobactam",
    "zosyn": "piperacillin-tazobactam", "meropenem": "meropenem",
    "ceftriaxone": "ceftriaxone", "rocephin": "ceftriaxone",
    "azithromycin": "azithromycin", "zithromax": "azithromycin",
    "ciprofloxacin": "ciprofloxacin", "cipro": "ciprofloxacin",
    "metronidazole": "metronidazole", "flagyl": "metronidazole",
    "levofloxacin": "levofloxacin", "levaquin": "levofloxacin",
    # Analgesics / Sedation
    "morphine": "morphine", "fentanyl": "fentanyl",
    "dilaudid": "hydromorphone", "hydromorphone": "hydromorphone",
    "propofol": "propofol", "diprivan": "propofol",
    "midazolam": "midazolam", "versed": "midazolam",
    "lorazepam": "lorazepam", "ativan": "lorazepam",
    "ketamine": "ketamine",
    # Vasopressors
    "norepinephrine": "norepinephrine", "levophed": "norepinephrine",
    "vasopressin": "vasopressin", "dopamine": "dopamine",
    "epinephrine": "epinephrine", "adrenaline": "epinephrine",
    "phenylephrine": "phenylephrine", "neosynephrine": "phenylephrine",
    # Anticoagulants
    "heparin": "heparin", "enoxaparin": "enoxaparin", "lovenox": "enoxaparin",
    "warfarin": "warfarin", "coumadin": "warfarin",
    "apixaban": "apixaban", "eliquis": "apixaban",
    # Cardiac
    "metoprolol": "metoprolol", "lopressor": "metoprolol",
    "amiodarone": "amiodarone", "cordarone": "amiodarone",
    "digoxin": "digoxin", "lanoxin": "digoxin",
    "furosemide": "furosemide", "lasix": "furosemide",
    "lisinopril": "lisinopril", "atorvastatin": "atorvastatin", "lipitor": "atorvastatin",
    # Diabetes
    "insulin": "insulin", "metformin": "metformin",
    "glipizide": "glipizide", "januvia": "sitagliptin",
    # Steroids
    "prednisone": "prednisone", "methylprednisolone": "methylprednisolone",
    "solumedrol": "methylprednisolone", "dexamethasone": "dexamethasone",
    "hydrocortisone": "hydrocortisone",
    # GI
    "pantoprazole": "pantoprazole", "protonix": "pantoprazole",
    "omeprazole": "omeprazole", "ondansetron": "ondansetron", "zofran": "ondansetron",
    # Other
    "acetaminophen": "acetaminophen", "tylenol": "acetaminophen",
    "ibuprofen": "ibuprofen", "aspirin": "aspirin",
    "albuterol": "albuterol", "ipratropium": "ipratropium",
    "sodium bicarbonate": "sodium bicarbonate", "bicarb": "sodium bicarbonate",
    "albumin": "albumin", "normal saline": "normal saline", "ns": "normal saline",
    "lactated ringers": "lactated ringer's", "lr": "lactated ringer's",
}

DIAGNOSIS_LEXICON = {
    "sepsis": "sepsis", "septic shock": "septic shock",
    "pneumonia": "pneumonia", "cap": "community-acquired pneumonia",
    "hap": "hospital-acquired pneumonia", "vap": "ventilator-associated pneumonia",
    "ards": "acute respiratory distress syndrome",
    "acute respiratory distress syndrome": "acute respiratory distress syndrome",
    "respiratory failure": "respiratory failure", "arf": "acute renal failure",
    "acute kidney injury": "acute kidney injury", "aki": "acute kidney injury",
    "ckd": "chronic kidney disease", "esrd": "end-stage renal disease",
    "copd": "COPD", "chf": "congestive heart failure",
    "congestive heart failure": "congestive heart failure",
    "heart failure": "heart failure", "mi": "myocardial infarction",
    "myocardial infarction": "myocardial infarction", "stemi": "STEMI",
    "nstemi": "NSTEMI", "acs": "acute coronary syndrome",
    "afib": "atrial fibrillation", "atrial fibrillation": "atrial fibrillation",
    "dvt": "deep vein thrombosis", "pe": "pulmonary embolism",
    "pulmonary embolism": "pulmonary embolism",
    "stroke": "stroke", "cva": "cerebrovascular accident",
    "ischemic stroke": "ischemic stroke", "hemorrhagic stroke": "hemorrhagic stroke",
    "tia": "transient ischemic attack",
    "diabetes": "diabetes mellitus", "dm": "diabetes mellitus",
    "dm1": "type 1 diabetes mellitus", "dm2": "type 2 diabetes mellitus",
    "dka": "diabetic ketoacidosis",
    "hypertension": "hypertension", "htn": "hypertension",
    "hypertensive urgency": "hypertensive urgency",
    "hypotension": "hypotension", "shock": "shock",
    "cirrhosis": "hepatic cirrhosis", "hepatitis": "hepatitis",
    "pancreatitis": "pancreatitis", "gi bleed": "gastrointestinal bleed",
    "ugib": "upper GI bleed", "lgib": "lower GI bleed",
    "urinary tract infection": "urinary tract infection", "uti": "UTI",
    "cellulitis": "cellulitis", "bacteremia": "bacteremia",
    "endocarditis": "endocarditis", "meningitis": "meningitis",
    "encephalopathy": "encephalopathy", "delirium": "delirium",
    "altered mental status": "altered mental status", "ams": "altered mental status",
    "covid": "COVID-19", "covid-19": "COVID-19",
    "influenza": "influenza", "flu": "influenza",
}

SYMPTOM_LEXICON = {
    "fever": "fever", "febrile": "fever",
    "chills": "chills", "rigors": "rigors",
    "tachycardia": "tachycardia", "bradycardia": "bradycardia",
    "hypotension": "hypotension", "hypertension": "hypertension",
    "tachypnea": "tachypnea", "bradypnea": "bradypnea",
    "dyspnea": "dyspnea", "shortness of breath": "dyspnea", "sob": "dyspnea",
    "chest pain": "chest pain", "chest tightness": "chest tightness",
    "palpitations": "palpitations",
    "nausea": "nausea", "vomiting": "vomiting", "n/v": "nausea/vomiting",
    "diarrhea": "diarrhea", "constipation": "constipation",
    "abdominal pain": "abdominal pain", "abd pain": "abdominal pain",
    "headache": "headache", "ha": "headache",
    "dizziness": "dizziness", "syncope": "syncope",
    "confusion": "confusion", "altered": "altered mental status",
    "lethargy": "lethargy", "fatigue": "fatigue",
    "weakness": "weakness", "malaise": "malaise",
    "edema": "edema", "swelling": "edema",
    "cough": "cough", "productive cough": "productive cough",
    "hemoptysis": "hemoptysis", "hematemesis": "hematemesis",
    "hematuria": "hematuria", "hematochezia": "hematochezia",
    "oliguria": "oliguria", "anuria": "anuria",
    "diaphoresis": "diaphoresis", "diaphoretic": "diaphoresis",
    "jaundice": "jaundice", "icteric": "jaundice",
    "rash": "rash", "erythema": "erythema",
    "pain": "pain", "tenderness": "tenderness",
}

PROCEDURE_LEXICON = {
    "intubation": "endotracheal intubation", "intubated": "endotracheal intubation",
    "mechanical ventilation": "mechanical ventilation", "vent": "mechanical ventilation",
    "extubation": "extubation", "extubated": "extubation",
    "central line": "central venous catheter", "cvc": "central venous catheter",
    "arterial line": "arterial line", "a-line": "arterial line",
    "foley": "foley catheter", "foley catheter": "foley catheter",
    "ng tube": "nasogastric tube", "ngt": "nasogastric tube",
    "chest tube": "chest tube",
    "bronchoscopy": "bronchoscopy", "bronch": "bronchoscopy",
    "colonoscopy": "colonoscopy", "endoscopy": "endoscopy",
    "echocardiogram": "echocardiogram", "echo": "echocardiogram",
    "ct scan": "CT scan", "ct": "CT scan",
    "mri": "MRI", "x-ray": "X-ray", "cxr": "chest X-ray",
    "dialysis": "dialysis", "crrt": "CRRT", "hemodialysis": "hemodialysis",
    "intubate": "endotracheal intubation",
    "thoracentesis": "thoracentesis", "paracentesis": "paracentesis",
    "lumbar puncture": "lumbar puncture", "lp": "lumbar puncture",
    "blood culture": "blood culture", "urine culture": "urine culture",
    "ecg": "ECG", "ekg": "ECG", "electrocardiogram": "ECG",
    "cvvh": "CVVH", "cvvhd": "CVVHD",
}

LAB_LEXICON = {
    "wbc": "WBC", "white blood cell": "WBC", "white count": "WBC",
    "hgb": "hemoglobin", "hemoglobin": "hemoglobin", "hematocrit": "hematocrit", "hct": "hematocrit",
    "platelet": "platelets", "plt": "platelets",
    "creatinine": "creatinine", "cr": "creatinine", "cre": "creatinine",
    "bun": "BUN", "blood urea nitrogen": "BUN",
    "sodium": "sodium", "na": "sodium", "potassium": "potassium", "k": "potassium",
    "chloride": "chloride", "cl": "chloride",
    "bicarbonate": "bicarbonate", "hco3": "bicarbonate",
    "glucose": "glucose", "gluc": "glucose",
    "lactate": "lactate", "lactic acid": "lactate",
    "troponin": "troponin", "trop": "troponin",
    "bnp": "BNP", "pro-bnp": "proBNP",
    "alt": "ALT", "ast": "AST", "alp": "ALP",
    "bilirubin": "bilirubin", "bili": "bilirubin",
    "albumin": "albumin", "alb": "albumin",
    "inr": "INR", "pt": "PT", "ptt": "PTT", "aptt": "aPTT",
    "ph": "pH", "pao2": "PaO2", "paco2": "PaCO2", "spo2": "SpO2", "o2 sat": "SpO2",
    "procalcitonin": "procalcitonin", "pct": "procalcitonin",
    "crp": "CRP", "c-reactive protein": "CRP",
    "esr": "ESR", "ferritin": "ferritin",
    "d-dimer": "D-dimer", "fibrinogen": "fibrinogen",
    "tsh": "TSH", "t4": "T4", "t3": "T3",
    "hba1c": "HbA1c", "a1c": "HbA1c",
    "lipase": "lipase", "amylase": "amylase",
    "ammonia": "ammonia", "cortisol": "cortisol",
}

VITAL_PATTERNS = {
    "heart rate": "HR", "hr": "HR", "pulse": "HR",
    "blood pressure": "BP", "bp": "BP", "sbp": "SBP", "dbp": "DBP",
    "respiratory rate": "RR", "rr": "RR", "resp rate": "RR",
    "temperature": "Temp", "temp": "Temp",
    "oxygen saturation": "SpO2", "spo2": "SpO2", "o2 sat": "SpO2",
    "mean arterial pressure": "MAP", "map": "MAP",
    "gcs": "GCS", "glasgow coma scale": "GCS",
}


# ─────────────────────────────────────────
# Pattern Utilities
# ─────────────────────────────────────────

DOSAGE_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|mL|L|units?|meq|mmol|mEq|%|IU)'
    r'(?:\s*/\s*(?:kg|hr|hour|day|min|dose|kg/hr|kg/min))?',
    re.IGNORECASE
)

FREQUENCY_PATTERN = re.compile(
    r'\b(q\d+h(?:rs?)?|every\s+\d+\s+hours?|bid|tid|qid|qd|daily|twice\s+daily|'
    r'three\s+times\s+daily|prn|as\s+needed|once\s+daily|qam|qpm|qhs|'
    r'continuous(?:ly)?|infusion|drip|bolus)\b',
    re.IGNORECASE
)

VITAL_VALUE_PATTERN = re.compile(
    r'\b(HR|BP|SBP|DBP|RR|Temp|SpO2|MAP|GCS|pulse|temperature|heart rate|'
    r'blood pressure|respiratory rate|oxygen saturation)\s*[:\s=of]*'
    r'(\d{2,3}(?:\.\d+)?(?:\s*/\s*\d{2,3})?)\s*'
    r'(%|bpm|mmHg|C|F|breaths?/min)?',
    re.IGNORECASE
)

LAB_VALUE_PATTERN = re.compile(
    r'\b(creatinine|lactate|troponin|hemoglobin|hgb|wbc|platelet|plt|sodium|potassium|'
    r'glucose|bun|inr|ph|pao2|procalcitonin|bnp|bilirubin|albumin|alt|ast)\s*'
    r'(?:of|was|is|:|=|level)?\s*'
    r'(\d+(?:\.\d+)?)\s*(mg/dL|g/dL|mmol/L|mEq/L|ng/mL|U/L|%|k/uL|x10[39]/L)?',
    re.IGNORECASE
)


# ─────────────────────────────────────────
# Core NER
# ─────────────────────────────────────────

class MedicalNER:
    """Rule-based medical NER with lexicon matching and regex patterns."""

    def __init__(self):
        self._build_compiled_patterns()

    def _build_compiled_patterns(self):
        """Build sorted (longest-first) compiled patterns from lexicons."""
        def _compile(lexicon: Dict) -> List[Tuple[re.Pattern, str, str]]:
            pairs = sorted(lexicon.items(), key=lambda x: -len(x[0]))
            return [(re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), k, v)
                    for k, v in pairs]

        self.med_patterns = _compile(MEDICATION_LEXICON)
        self.dx_patterns = _compile(DIAGNOSIS_LEXICON)
        self.sx_patterns = _compile(SYMPTOM_LEXICON)
        self.proc_patterns = _compile(PROCEDURE_LEXICON)
        self.lab_patterns = _compile(LAB_LEXICON)
        self.vital_name_patterns = _compile(VITAL_PATTERNS)

    def _find_lexicon_matches(self, text: str, patterns, label: str) -> List[Entity]:
        entities = []
        covered = set()
        for pattern, raw, normalized in patterns:
            for m in pattern.finditer(text):
                span = set(range(m.start(), m.end()))
                if span & covered:
                    continue
                covered |= span
                entities.append(Entity(
                    text=m.group(),
                    label=label,
                    start=m.start(),
                    end=m.end(),
                    normalized=normalized,
                    confidence=0.95,
                ))
        return entities

    def _find_dosages(self, text: str) -> List[Entity]:
        entities = []
        for m in DOSAGE_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(),
                label="DOSAGE",
                start=m.start(),
                end=m.end(),
                confidence=0.9,
                attributes={"value": m.group(1), "unit": m.group(2)},
            ))
        return entities

    def _find_frequencies(self, text: str) -> List[Entity]:
        entities = []
        for m in FREQUENCY_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(),
                label="FREQUENCY",
                start=m.start(),
                end=m.end(),
                confidence=0.9,
            ))
        return entities

    def _find_vital_values(self, text: str) -> List[Entity]:
        entities = []
        for m in VITAL_VALUE_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(),
                label="VITAL",
                start=m.start(),
                end=m.end(),
                normalized=VITAL_PATTERNS.get(m.group(1).lower(), m.group(1).upper()),
                confidence=0.92,
                attributes={"name": m.group(1), "value": m.group(2)},
            ))
        return entities

    def _find_lab_values(self, text: str) -> List[Entity]:
        entities = []
        for m in LAB_VALUE_PATTERN.finditer(text):
            entities.append(Entity(
                text=m.group(),
                label="LAB",
                start=m.start(),
                end=m.end(),
                normalized=LAB_LEXICON.get(m.group(1).lower(), m.group(1).upper()),
                confidence=0.92,
                attributes={"name": m.group(1), "value": m.group(2)},
            ))
        return entities

    def _merge_and_deduplicate(self, all_entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keep highest confidence."""
        sorted_ents = sorted(all_entities, key=lambda e: (e.start, -e.confidence))
        merged = []
        last_end = -1
        for e in sorted_ents:
            if e.start >= last_end:
                merged.append(e)
                last_end = e.end
        return sorted(merged, key=lambda e: e.start)

    def _extract_relations(self, entities: List[Entity]) -> List[Dict]:
        """Link medications to nearby dosages and frequencies."""
        relations = []
        meds = [e for e in entities if e.label == "MEDICATION"]
        dosages = [e for e in entities if e.label == "DOSAGE"]
        freqs = [e for e in entities if e.label == "FREQUENCY"]

        for med in meds:
            # Nearby = within 80 chars after medication
            nearby_dosages = [d for d in dosages if med.end <= d.start <= med.end + 80]
            nearby_freqs = [f for f in freqs if med.end <= f.start <= med.end + 100]

            if nearby_dosages or nearby_freqs:
                rel = {
                    "type": "MEDICATION_DOSING",
                    "medication": med.normalized or med.text,
                    "dosages": [d.text for d in nearby_dosages],
                    "frequencies": [f.text for f in nearby_freqs],
                }
                relations.append(rel)

        return relations

    def process(self, text: str) -> NERResult:
        """Run full NER pipeline on clinical text."""
        result = NERResult(text=text)

        all_entities = []
        all_entities += self._find_lexicon_matches(text, self.med_patterns, "MEDICATION")
        all_entities += self._find_lexicon_matches(text, self.dx_patterns, "DIAGNOSIS")
        all_entities += self._find_lexicon_matches(text, self.sx_patterns, "SYMPTOM")
        all_entities += self._find_lexicon_matches(text, self.proc_patterns, "PROCEDURE")
        all_entities += self._find_vital_values(text)
        all_entities += self._find_lab_values(text)
        all_entities += self._find_dosages(text)
        all_entities += self._find_frequencies(text)

        result.entities = self._merge_and_deduplicate(all_entities)
        result.relations = self._extract_relations(result.entities)
        return result

    def process_batch(self, texts: List[str]) -> List[NERResult]:
        return [self.process(t) for t in texts]


# ─────────────────────────────────────────
# Structured extraction
# ─────────────────────────────────────────

class ClinicalEntityExtractor:
    """Higher-level extractor that returns structured clinical summaries."""

    def __init__(self):
        self.ner = MedicalNER()

    def extract_medications(self, text: str) -> List[Dict]:
        result = self.ner.process(text)
        meds = {}
        for e in result.get_by_label("MEDICATION"):
            key = e.normalized or e.text.lower()
            if key not in meds:
                meds[key] = {"medication": key, "mentions": [], "dosages": [], "frequencies": []}
            meds[key]["mentions"].append(e.text)

        for rel in result.relations:
            key = rel["medication"]
            if key in meds:
                meds[key]["dosages"].extend(rel.get("dosages", []))
                meds[key]["frequencies"].extend(rel.get("frequencies", []))

        return list(meds.values())

    def extract_diagnoses(self, text: str) -> List[str]:
        result = self.ner.process(text)
        return list({e.normalized or e.text for e in result.get_by_label("DIAGNOSIS")})

    def extract_symptoms(self, text: str) -> List[str]:
        result = self.ner.process(text)
        return list({e.normalized or e.text for e in result.get_by_label("SYMPTOM")})

    def extract_vitals(self, text: str) -> Dict[str, str]:
        result = self.ner.process(text)
        vitals = {}
        for e in result.get_by_label("VITAL"):
            if "name" in e.attributes and "value" in e.attributes:
                key = e.normalized or e.attributes["name"].upper()
                vitals[key] = e.attributes["value"]
        return vitals

    def extract_labs(self, text: str) -> Dict[str, str]:
        result = self.ner.process(text)
        labs = {}
        for e in result.get_by_label("LAB"):
            if "name" in e.attributes and "value" in e.attributes:
                key = e.normalized or e.attributes["name"].upper()
                labs[key] = e.attributes["value"]
        return labs

    def full_extraction(self, text: str) -> Dict:
        result = self.ner.process(text)
        return {
            "medications": self.extract_medications(text),
            "diagnoses": self.extract_diagnoses(text),
            "symptoms": self.extract_symptoms(text),
            "vitals": self.extract_vitals(text),
            "labs": self.extract_labs(text),
            "procedures": list({e.normalized or e.text
                                 for e in result.get_by_label("PROCEDURE")}),
            "entity_count": len(result.entities),
            "relation_count": len(result.relations),
        }


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────

if __name__ == "__main__":
    sample_note = """
    72M with hx of CHF, DM2, HTN presents with SOB and fever.
    Vitals: HR 118, BP 88/52, RR 28, Temp 38.9C, SpO2 91% on 4L NC.
    Labs notable for WBC 18.2, lactate 4.1 mmol/L, creatinine 2.8 mg/dL, troponin 0.08 ng/mL.
    Impression: septic shock likely secondary to pneumonia vs UTI.
    Plan: Start vancomycin 25mg/kg q8h and piperacillin-tazobactam 4.5g q6h.
    NS 30mL/kg bolus over 3h. Norepinephrine 0.1 mcg/kg/min, titrate for MAP >65.
    Blood cultures x2 prior to abx. CXR obtained. Place central line and arterial line.
    Foley catheter placed. Patient intubated for hypoxic respiratory failure.
    Propofol 10mcg/kg/min for sedation, fentanyl 25mcg/hr for pain.
    """

    extractor = ClinicalEntityExtractor()
    extraction = extractor.full_extraction(sample_note)

    print("=== Medical NER Results ===")
    print(f"Medications ({len(extraction['medications'])}):")
    for m in extraction['medications']:
        dosage_str = ", ".join(m['dosages']) if m['dosages'] else "no dosage"
        freq_str = ", ".join(m['frequencies']) if m['frequencies'] else "no freq"
        print(f"  {m['medication']}: {dosage_str} | {freq_str}")

    print(f"\nDiagnoses: {extraction['diagnoses']}")
    print(f"Symptoms: {extraction['symptoms']}")
    print(f"Vitals: {extraction['vitals']}")
    print(f"Labs: {extraction['labs']}")
    print(f"Procedures: {extraction['procedures']}")
    print(f"\nTotal entities: {extraction['entity_count']}, Relations: {extraction['relation_count']}")

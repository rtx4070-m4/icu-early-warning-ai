"""
AI Hospital Operating System
NLP Pipeline – Clinical Text Processing
=========================================
Tokenization, section segmentation, keyword extraction,
diagnosis classification for clinical notes.
"""

import logging
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("clinical_nlp")

# ─────────────────────────────────────────────
# Clinical stopwords (extend standard ones)
# ─────────────────────────────────────────────
CLINICAL_STOPWORDS = {
    "patient", "pt", "the", "a", "an", "is", "was", "are", "were",
    "with", "for", "and", "of", "in", "to", "that", "this", "on",
    "at", "from", "no", "not", "without", "also", "as", "has",
    "been", "he", "she", "his", "her", "be", "or", "but", "if",
    "by", "it", "will", "can", "due", "per", "which", "who",
    "have", "had", "did", "do", "does", "so", "than", "then",
    "some", "such", "other", "may", "should", "would", "could",
}

# ─────────────────────────────────────────────
# Clinical section headers (for segmentation)
# ─────────────────────────────────────────────
SECTION_PATTERNS = {
    "chief_complaint":   r"(?i)chief\s*complaint[:\s]",
    "hpi":               r"(?i)history\s*of\s*present(?:ing)?\s*illness[:\s]|HPI[:\s]",
    "pmh":               r"(?i)past\s*medical\s*history[:\s]|PMH[:\s]",
    "medications":       r"(?i)medications?[:\s]|current\s*meds[:\s]",
    "allergies":         r"(?i)allergi(?:es|c)[:\s]|NKDA",
    "review_of_systems": r"(?i)review\s*of\s*systems[:\s]|ROS[:\s]",
    "physical_exam":     r"(?i)physical\s*exam(?:ination)?[:\s]|PE[:\s]",
    "vitals":            r"(?i)vital\s*signs?[:\s]|VS[:\s]",
    "labs":              r"(?i)laborator(?:y|ies)[:\s]|labs?[:\s]",
    "imaging":           r"(?i)imaging[:\s]|radiology[:\s]|CT[:\s]|MRI[:\s]|CXR[:\s]",
    "assessment":        r"(?i)assessment[:\s]|impression[:\s]",
    "plan":              r"(?i)\bplan[:\s]|\bmanagement[:\s]",
    "discharge_summary": r"(?i)discharge\s*summary[:\s]",
}

# ─────────────────────────────────────────────
# Clinical Abbreviation Expander
# ─────────────────────────────────────────────
ABBREVIATIONS = {
    r"\bSOB\b": "shortness of breath",
    r"\bCP\b":  "chest pain",
    r"\bDOE\b": "dyspnea on exertion",
    r"\bHTN\b": "hypertension",
    r"\bDM\b":  "diabetes mellitus",
    r"\bCAD\b": "coronary artery disease",
    r"\bCHF\b": "congestive heart failure",
    r"\bCOPD\b":"chronic obstructive pulmonary disease",
    r"\bAKI\b": "acute kidney injury",
    r"\bARDS\b":"acute respiratory distress syndrome",
    r"\bICU\b": "intensive care unit",
    r"\bBP\b":  "blood pressure",
    r"\bHR\b":  "heart rate",
    r"\bRR\b":  "respiratory rate",
    r"\bSpO2\b":"oxygen saturation",
    r"\bT\b(?=\s*[\d])": "temperature",
    r"\bTID\b": "three times daily",
    r"\bBID\b": "twice daily",
    r"\bQD\b":  "once daily",
    r"\bPRN\b": "as needed",
    r"\bIV\b":  "intravenous",
    r"\bPO\b":  "by mouth",
    r"\bSubQ\b":"subcutaneous",
    r"\bNPO\b": "nothing by mouth",
    r"\bc/o\b": "complaining of",
    r"\bh/o\b": "history of",
    r"\bw/\b":  "with",
    r"\bs/p\b": "status post",
    r"\bWNL\b": "within normal limits",
    r"\bNAD\b": "no acute distress",
    r"\bGCS\b": "glasgow coma scale",
    r"\bMAE\b": "moves all extremities",
}


class ClinicalTextNormalizer:
    """Clean and normalize raw clinical note text."""

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Expand abbreviations
        for pattern, replacement in ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Remove special chars (keep medical symbols: /, -)
        text = re.sub(r"[^\w\s\-\/\.\,]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def expand_abbreviations(self, text: str) -> str:
        for pattern, replacement in ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
class ClinicalTokenizer:
    """
    Clinical domain tokenizer that handles:
    - Medical abbreviations (MI, COPD, AKI…)
    - Numeric values with units (37.5°C, 120/80 mmHg)
    - De-identification placeholders
    """

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        self.normalizer       = ClinicalTextNormalizer()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalizer.normalize(text)
        # Split on whitespace and punctuation boundaries
        tokens = re.findall(r"[\w\-]+(?:\.[\w]+)*", text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in CLINICAL_STOPWORDS and len(t) > 1]
        return tokens

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using clinical-aware patterns."""
        # Split on period, newline, or numbered list items
        sentences = re.split(r"(?<=[.!?])\s+|\n+|(?=\d+\.\s)", text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in texts]


# ─────────────────────────────────────────────
# Section Segmenter
# ─────────────────────────────────────────────
class ClinicalSectionSegmenter:
    """Extract structured sections from unstructured clinical notes."""

    def segment(self, text: str) -> Dict[str, str]:
        """Return dict of {section_name: section_text}."""
        sections: Dict[str, str] = {}
        positions: List[Tuple[int, str]] = []

        for section, pattern in SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text):
                positions.append((match.start(), section, match.end()))

        positions.sort(key=lambda x: x[0])

        for i, (start, section, end) in enumerate(positions):
            next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections[section] = text[end:next_start].strip()

        # Anything before first section = header
        if positions:
            sections["header"] = text[:positions[0][0]].strip()
        else:
            sections["full_text"] = text

        return sections

    def extract_section(self, text: str, section: str) -> Optional[str]:
        return self.segment(text).get(section)


# ─────────────────────────────────────────────
# Medical Keyword Extractor (TF-IDF + domain dict)
# ─────────────────────────────────────────────
class MedicalKeywordExtractor:
    """
    Extract medically relevant keywords using:
    1. Domain-specific medical term dictionary
    2. TF-IDF scoring across a document collection
    3. Named pattern matching (diagnoses, medications, vitals)
    """

    # High-value medical terms (simplified domain lexicon)
    MEDICAL_TERMS = {
        "diagnoses": {
            "sepsis", "pneumonia", "myocardial infarction", "heart failure",
            "acute kidney injury", "respiratory failure", "stroke", "shock",
            "arrhythmia", "pulmonary embolism", "deep vein thrombosis",
            "ards", "copd", "diabetes", "hypertension", "cirrhosis",
        },
        "symptoms": {
            "chest pain", "shortness of breath", "dyspnea", "fever",
            "hypotension", "tachycardia", "bradycardia", "oliguria",
            "altered mental status", "confusion", "cough", "hemoptysis",
            "edema", "syncope", "diaphoresis", "nausea", "vomiting",
        },
        "procedures": {
            "intubation", "extubation", "bronchoscopy", "catheterization",
            "dialysis", "transfusion", "thoracentesis", "paracentesis",
            "central line", "arterial line", "tracheostomy",
        },
        "medications": {
            "antibiotics", "vasopressors", "insulin", "heparin", "warfarin",
            "diuretics", "steroids", "analgesics", "sedatives", "anticoagulants",
        },
        "labs": {
            "creatinine", "troponin", "lactate", "procalcitonin", "bun",
            "hemoglobin", "white blood cell", "platelets", "sodium", "potassium",
        },
    }

    # Vital pattern: "HR 112", "BP 88/52", "SpO2 91%"
    VITAL_PATTERN = re.compile(
        r"(?:HR|heart rate|BP|blood pressure|SpO2|O2 sat|"
        r"RR|resp rate|temp|temperature|GCS)\s*[:=]?\s*([\d\.\/]+)\s*(%|mmHg|bpm|°C|°F)?",
        re.IGNORECASE,
    )

    def extract_keywords(self, text: str, top_n: int = 20) -> Dict[str, List[str]]:
        """Extract keywords by category."""
        text_lower = text.lower()
        found: Dict[str, List[str]] = defaultdict(list)

        for category, terms in self.MEDICAL_TERMS.items():
            for term in terms:
                if term in text_lower:
                    found[category].append(term)

        return dict(found)

    def extract_vitals_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract vital sign values mentioned in clinical notes."""
        vitals = []
        for match in self.VITAL_PATTERN.finditer(text):
            vitals.append({
                "parameter": match.group(0).split()[0],
                "value":     match.group(1),
                "unit":      match.group(2) or "",
                "context":   text[max(0, match.start()-20):match.end()+20],
            })
        return vitals

    def extract_medications_from_text(self, text: str) -> List[str]:
        """Simple medication name extraction using suffix patterns."""
        # Common medication suffixes
        patterns = [
            r"\b\w+(?:cillin|mycin|cycline|floxacin|azole|prazole|"
            r"statin|olol|pril|sartan|dipine|mab|nib|tinib)\b",
            r"\b(?:norepinephrine|vasopressin|epinephrine|dopamine|"
            r"dobutamine|milrinone|phenylephrine|midazolam|propofol|"
            r"fentanyl|morphine|ketamine|rocuronium|vecuronium)\b",
        ]
        found = []
        for pattern in patterns:
            found.extend(re.findall(pattern, text, re.IGNORECASE))
        return list(set(m.lower() for m in found))

    def compute_tfidf_keywords(
        self, texts: List[str], top_n: int = 15
    ) -> List[List[Tuple[str, float]]]:
        """Compute TF-IDF scores and return top keywords per document."""
        tokenizer   = ClinicalTokenizer(remove_stopwords=True)
        token_lists = tokenizer.tokenize_batch(texts)

        # Compute IDF
        n_docs = len(texts)
        df_counts: Counter = Counter()
        for tokens in token_lists:
            for t in set(tokens):
                df_counts[t] += 1
        idf = {t: np.log(n_docs / (1 + count)) for t, count in df_counts.items()}

        # TF-IDF per document
        results = []
        for tokens in token_lists:
            tf = Counter(tokens)
            total = max(sum(tf.values()), 1)
            scored = {
                t: (count / total) * idf.get(t, 0)
                for t, count in tf.items()
            }
            top = sorted(scored.items(), key=lambda x: -x[1])[:top_n]
            results.append(top)
        return results


# ─────────────────────────────────────────────
# Diagnosis Classifier (rule-based + bow)
# ─────────────────────────────────────────────
class DiagnosisClassifier:
    """
    Rule-based ICD-category classifier from clinical text.
    Maps clinical note content to broad ICD-10 chapters.
    """

    # ICD-10 chapter → keyword signals
    CHAPTER_SIGNALS = {
        "Infectious Diseases (A00–B99)": [
            "infection", "sepsis", "bacteremia", "pneumonia", "uti",
            "cellulitis", "meningitis", "endocarditis", "abscess",
        ],
        "Neoplasms (C00–D49)": [
            "cancer", "tumor", "malignancy", "carcinoma", "lymphoma",
            "leukemia", "metastasis", "chemotherapy",
        ],
        "Cardiovascular (I00–I99)": [
            "myocardial infarction", "heart failure", "arrhythmia",
            "atrial fibrillation", "coronary artery disease", "stroke",
            "hypertension", "cardiac arrest", "angina",
        ],
        "Respiratory (J00–J99)": [
            "copd", "asthma", "pneumonia", "respiratory failure",
            "pulmonary embolism", "pleural effusion", "ards",
            "shortness of breath", "dyspnea",
        ],
        "Gastrointestinal (K00–K93)": [
            "cirrhosis", "hepatitis", "pancreatitis", "gi bleed",
            "bowel obstruction", "appendicitis", "cholecystitis",
        ],
        "Genitourinary (N00–N99)": [
            "acute kidney injury", "chronic kidney disease", "renal failure",
            "urinary tract infection", "nephrotic",
        ],
        "Endocrine (E00–E89)": [
            "diabetes", "dka", "hyperglycemia", "hypothyroid",
            "hyperthyroid", "adrenal", "electrolyte",
        ],
        "Neurological (G00–G99)": [
            "stroke", "seizure", "encephalopathy", "altered mental status",
            "coma", "meningitis", "neuropathy", "dementia",
        ],
        "Injuries/Poisoning (S00–T88)": [
            "trauma", "fracture", "overdose", "poisoning", "burn",
            "laceration", "hemorrhage",
        ],
    }

    def classify(self, text: str) -> List[Dict[str, float]]:
        """Return list of (chapter, confidence) sorted by confidence."""
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for chapter, signals in self.CHAPTER_SIGNALS.items():
            hits = sum(1 for s in signals if s in text_lower)
            if hits > 0:
                scores[chapter] = hits / len(signals)

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [{"chapter": ch, "confidence": round(sc, 4)} for ch, sc in sorted_scores]

    def top_diagnosis(self, text: str) -> Optional[str]:
        predictions = self.classify(text)
        return predictions[0]["chapter"] if predictions else None


# ─────────────────────────────────────────────
# Full NLP Pipeline
# ─────────────────────────────────────────────
class ClinicalNLPPipeline:
    """
    End-to-end NLP pipeline for processing clinical notes.
    Outputs structured information extracted from free text.
    """

    def __init__(self):
        self.tokenizer   = ClinicalTokenizer(remove_stopwords=True)
        self.segmenter   = ClinicalSectionSegmenter()
        self.extractor   = MedicalKeywordExtractor()
        self.classifier  = DiagnosisClassifier()
        self.normalizer  = ClinicalTextNormalizer()

    def process_note(self, text: str) -> Dict:
        """Process a single clinical note."""
        if not text or not text.strip():
            return {}

        # Normalize
        normalized = self.normalizer.normalize(text)

        # Segment sections
        sections = self.segmenter.segment(text)

        # Tokenize
        tokens = self.tokenizer.tokenize(text)

        # Keywords
        keywords = self.extractor.extract_keywords(text)

        # Vitals from text
        vitals_mentioned = self.extractor.extract_vitals_from_text(text)

        # Medications
        medications = self.extractor.extract_medications_from_text(text)

        # Diagnosis classification
        diagnosis_predictions = self.classifier.classify(text)

        return {
            "sections":             sections,
            "token_count":          len(tokens),
            "top_tokens":           Counter(tokens).most_common(20),
            "keywords":             keywords,
            "vitals_mentioned":     vitals_mentioned,
            "medications":          medications,
            "diagnosis_categories": diagnosis_predictions[:3],
            "top_diagnosis":        diagnosis_predictions[0]["chapter"] if diagnosis_predictions else None,
        }

    def process_batch(
        self, df: pd.DataFrame, text_col: str = "text"
    ) -> pd.DataFrame:
        """Process a DataFrame of clinical notes."""
        results = []
        for _, row in df.iterrows():
            text   = row.get(text_col, "")
            result = self.process_note(text)
            result["note_id"]   = row.get("note_id", "")
            result["patient_id"] = row.get("patient_id", "")
            result["category"]  = row.get("category", "")
            results.append(result)

        processed = pd.DataFrame(results)
        logger.info("Processed %d clinical notes", len(processed))
        return processed

    def summarize_patient_notes(
        self, notes: List[str]
    ) -> Dict:
        """Generate a structured summary from multiple notes."""
        all_keywords: Dict[str, List] = defaultdict(list)
        all_diagnoses: Counter = Counter()
        all_medications: List[str] = []
        all_vitals: List[Dict] = []

        for note in notes:
            result = self.process_note(note)
            for cat, kws in result.get("keywords", {}).items():
                all_keywords[cat].extend(kws)
            top_dx = result.get("top_diagnosis")
            if top_dx:
                all_diagnoses[top_dx] += 1
            all_medications.extend(result.get("medications", []))
            all_vitals.extend(result.get("vitals_mentioned", []))

        return {
            "note_count":        len(notes),
            "diagnosis_summary": dict(all_diagnoses.most_common(5)),
            "symptom_summary":   list(set(all_keywords.get("symptoms", []))),
            "medication_summary": list(set(all_medications)),
            "vitals_in_notes":   all_vitals[:10],
            "keyword_categories": {k: list(set(v)) for k, v in all_keywords.items()},
        }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    notes = [
        """
        Patient is a 75-year-old male with h/o COPD, CHF, and HTN admitted with
        acute respiratory failure. Chief complaint: SOB at rest for 2 days.
        On admission: HR 108, BP 88/52 mmHg, SpO2 82% on room air, RR 28, T 38.6°C.
        ABG: pH 7.28, pCO2 62, pO2 55. CXR: bilateral infiltrates.
        Assessment: Acute hypoxic respiratory failure secondary to ARDS vs community-
        acquired pneumonia. Sepsis suspected given fever, tachycardia, and hypotension.
        Plan: 1. Start broad-spectrum antibiotics - vancomycin + piperacillin-tazobactam.
        2. IV fluids resuscitation 30 mL/kg. 3. Norepinephrine for MAP >65.
        4. ICU admit. 5. Repeat cultures x2 before antibiotics.
        """,
        """
        ICU Day 3 progress note. Patient remains intubated on AC/VC mode.
        PEEP 10, FiO2 0.55, TV 380 mL. SpO2 94%. HR 92 bpm regular.
        BP 108/62 mmHg on norepinephrine 0.08 mcg/kg/min. Temp 37.2°C.
        Labs: WBC 16.2, Creatinine 2.4 (baseline 1.0), Lactate 1.8, Troponin 0.9.
        Assessment: Improving hemodynamics, vasopressor weaning in progress.
        Renal function worsening - possible AKI. Chest X-ray shows improving bilateral
        infiltrates. Blood cultures with gram-negative rods - adjusted antibiotics.
        Plan: Continue current ventilator settings. Wean norepinephrine as tolerated.
        Nephrology consult for AKI management. Daily SBT trial when ready.
        """,
    ]

    pipeline = ClinicalNLPPipeline()

    for i, note in enumerate(notes, 1):
        print(f"\n{'='*60}")
        print(f"NOTE {i}")
        print('='*60)
        result = pipeline.process_note(note)
        print(f"Sections found:     {list(result['sections'].keys())}")
        print(f"Top diagnosis:      {result['top_diagnosis']}")
        print(f"Diagnoses detected: {result['keywords'].get('diagnoses', [])}")
        print(f"Symptoms:           {result['keywords'].get('symptoms', [])}")
        print(f"Medications:        {result['medications']}")
        print(f"Vitals in text:     {result['vitals_mentioned']}")

    # Batch summary
    summary = pipeline.summarize_patient_notes(notes)
    print("\n=== PATIENT SUMMARY ACROSS NOTES ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

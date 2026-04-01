"""
HL7 / FHIR R4 Integration Layer
Converts internal Hospital OS data structures to/from FHIR R4 resources.
Supports: Patient, Observation (vitals + labs), Condition, MedicationRequest,
          AllergyIntolerance, DiagnosticReport, RiskAssessment, Alert (DetectedIssue)

Also provides a lightweight FHIR client for sending/receiving from an EHR FHIR endpoint.
"""

import json
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FHIR_VERSION = "4.0.1"
FHIR_BASE_URL = "http://hl7.org/fhir"

# LOINC codes for common vitals/labs
LOINC = {
    "heart_rate":         ("8867-4",  "Heart rate"),
    "sbp":                ("8480-6",  "Systolic blood pressure"),
    "dbp":                ("8462-4",  "Diastolic blood pressure"),
    "map":                ("8478-0",  "Mean blood pressure"),
    "respiratory_rate":   ("9279-1",  "Respiratory rate"),
    "temperature":        ("8310-5",  "Body temperature"),
    "spo2":               ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry"),
    "gcs":                ("9269-2",  "Glasgow coma scale total"),
    "wbc":                ("6690-2",  "Leukocytes [#/volume] in Blood"),
    "hemoglobin":         ("718-7",   "Hemoglobin [Mass/volume] in Blood"),
    "platelet":           ("777-3",   "Platelets [#/volume] in Blood"),
    "creatinine":         ("2160-0",  "Creatinine [Mass/volume] in Serum or Plasma"),
    "bun":                ("3094-0",  "Urea nitrogen [Mass/volume] in Serum or Plasma"),
    "sodium":             ("2951-2",  "Sodium [Moles/volume] in Serum or Plasma"),
    "potassium":          ("2823-3",  "Potassium [Moles/volume] in Serum or Plasma"),
    "glucose":            ("2345-7",  "Glucose [Mass/volume] in Serum or Plasma"),
    "lactate":            ("2524-7",  "Lactate [Moles/volume] in Serum or Plasma"),
    "troponin":           ("10839-9", "Troponin I.cardiac [Mass/volume] in Serum or Plasma"),
    "bnp":                ("42637-9", "Natriuretic peptide B [Mass/volume] in Serum or Plasma"),
    "inr":                ("6301-6",  "INR in Platelet poor plasma by Coagulation assay"),
    "procalcitonin":      ("33959-8", "Procalcitonin [Mass/volume] in Serum or Plasma"),
}

# SNOMED codes for common diagnoses
SNOMED_DX = {
    "sepsis":                  ("91302008",  "Sepsis (disorder)"),
    "septic_shock":            ("76571007",  "Septic shock (disorder)"),
    "pneumonia":               ("233604007", "Pneumonia (disorder)"),
    "ards":                    ("67782005",  "Acute respiratory distress syndrome (disorder)"),
    "acute_kidney_injury":     ("14669001",  "Acute renal failure syndrome (disorder)"),
    "congestive_heart_failure": ("42343007", "Congestive heart failure (disorder)"),
    "myocardial_infarction":   ("22298006",  "Myocardial infarction (disorder)"),
    "atrial_fibrillation":     ("49436004",  "Atrial fibrillation (disorder)"),
    "copd":                    ("13645005",  "Chronic obstructive lung disease (disorder)"),
    "pulmonary_embolism":      ("59282003",  "Pulmonary embolism (disorder)"),
    "diabetic_ketoacidosis":   ("44054006",  "Diabetic ketoacidosis (disorder)"),
}

# Unit systems
UCUM = {
    "heart_rate": "/min", "sbp": "mm[Hg]", "dbp": "mm[Hg]",
    "map": "mm[Hg]", "respiratory_rate": "/min",
    "temperature": "Cel", "spo2": "%", "gcs": "{score}",
    "wbc": "10*3/uL", "hemoglobin": "g/dL", "platelet": "10*3/uL",
    "creatinine": "mg/dL", "bun": "mg/dL", "sodium": "mEq/L",
    "potassium": "mEq/L", "glucose": "mg/dL", "lactate": "mmol/L",
    "troponin": "ng/mL", "bnp": "pg/mL", "inr": "{INR}",
    "procalcitonin": "ng/mL",
}


# ─────────────────────────────────────────
# FHIR resource builders
# ─────────────────────────────────────────

class FHIRBuilder:
    """Converts internal data dicts → FHIR R4 JSON resources."""

    # ── Patient ───────────────────────────

    def patient(self, patient_data: Dict) -> Dict:
        """Build a FHIR Patient resource."""
        pid = patient_data.get("patient_id", str(uuid.uuid4()))
        name_parts = patient_data.get("name", "Unknown Patient").split()
        family = name_parts[-1] if name_parts else "Unknown"
        given  = name_parts[:-1] if len(name_parts) > 1 else ["Unknown"]

        resource = {
            "resourceType": "Patient",
            "id": pid,
            "meta": {"versionId": "1",
                     "lastUpdated": datetime.utcnow().isoformat() + "Z",
                     "profile": [f"{FHIR_BASE_URL}/StructureDefinition/Patient"]},
            "identifier": [{
                "use": "usual",
                "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                     "code": "MR", "display": "Medical record number"}]},
                "value": pid,
            }],
            "active": True,
            "name": [{"use": "official", "family": family, "given": given}],
            "gender": {"M": "male", "F": "female"}.get(
                patient_data.get("sex", "U"), "unknown"),
        }
        if "date_of_birth" in patient_data:
            resource["birthDate"] = patient_data["date_of_birth"]
        if "age" in patient_data:
            # Approximate birth year from age
            birth_year = datetime.utcnow().year - int(patient_data["age"])
            resource["birthDate"] = f"{birth_year}-01-01"
        return resource

    # ── Observation (vital / lab) ─────────

    def observation(self, patient_id: str, field: str, value: float,
                     timestamp: Optional[str] = None,
                     status: str = "final",
                     category: str = "vital-signs") -> Dict:
        """Build a FHIR Observation resource."""
        loinc_code, loinc_display = LOINC.get(field, ("unknown", field.replace("_", " ")))
        unit = UCUM.get(field, "")
        ts = timestamp or datetime.utcnow().isoformat() + "Z"
        obs_id = str(uuid.uuid4())

        resource = {
            "resourceType": "Observation",
            "id": obs_id,
            "meta": {"lastUpdated": datetime.utcnow().isoformat() + "Z"},
            "status": status,
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": category,
                    "display": category.replace("-", " ").title(),
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": loinc_display,
                }],
                "text": loinc_display,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": ts,
            "issued": datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {
                "value": round(float(value), 3),
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit,
            },
        }

        # Add interpretation for critical values
        interp = self._interpretation(field, value)
        if interp:
            resource["interpretation"] = [interp]

        return resource

    def _interpretation(self, field: str, value: float) -> Optional[Dict]:
        critical_high = {"heart_rate": 150, "sbp": 200, "respiratory_rate": 30,
                          "temperature": 39.5, "wbc": 20, "lactate": 4.0, "creatinine": 4.0}
        critical_low  = {"sbp": 80, "spo2": 88, "hemoglobin": 7.0, "platelet": 50}
        if field in critical_high and value > critical_high[field]:
            code, display = "HH", "Critical High"
        elif field in critical_low and value < critical_low[field]:
            code, display = "LL", "Critical Low"
        else:
            return None
        return {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                             "code": code, "display": display}]}

    def vital_bundle(self, patient_id: str, vitals: Dict,
                      timestamp: Optional[str] = None) -> Dict:
        """Build a FHIR Bundle of Observation resources for a full vital-signs set."""
        entries = []
        for field, value in vitals.items():
            if not isinstance(value, (int, float)) or value is None:
                continue
            obs = self.observation(patient_id, field, float(value),
                                    timestamp=timestamp, category="vital-signs")
            entries.append({
                "fullUrl": f"urn:uuid:{obs['id']}",
                "resource": obs,
                "request": {"method": "POST", "url": "Observation"},
            })
        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "transaction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": entries,
        }

    # ── Condition (diagnosis) ──────────────

    def condition(self, patient_id: str, diagnosis: str,
                   severity: str = "moderate",
                   clinical_status: str = "active",
                   encounter_id: Optional[str] = None) -> Dict:
        """Build a FHIR Condition resource."""
        dx_key = diagnosis.lower().replace(" ", "_")
        snomed_code, snomed_display = SNOMED_DX.get(dx_key, ("unknown", diagnosis))
        severity_code = {
            "mild": ("255604002", "Mild"),
            "moderate": ("6736007", "Moderate"),
            "severe": ("24484000", "Severe"),
            "critical": ("442452003", "Life threatening severity"),
        }.get(severity, ("6736007", "Moderate"))

        resource = {
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                             "code": clinical_status}]
            },
            "verificationStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                             "code": "confirmed"}]
            },
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                       "code": "encounter-diagnosis"}]}],
            "severity": {"coding": [{"system": "http://snomed.info/sct",
                                      "code": severity_code[0],
                                      "display": severity_code[1]}]},
            "code": {"coding": [{"system": "http://snomed.info/sct",
                                  "code": snomed_code,
                                  "display": snomed_display}],
                     "text": diagnosis},
            "subject": {"reference": f"Patient/{patient_id}"},
            "onsetDateTime": datetime.utcnow().isoformat() + "Z",
            "recordedDate": datetime.utcnow().isoformat() + "Z",
        }
        if encounter_id:
            resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}
        return resource

    # ── MedicationRequest ─────────────────

    def medication_request(self, patient_id: str, medication: str,
                            dose: Optional[str] = None,
                            frequency: Optional[str] = None,
                            route: str = "IV",
                            status: str = "active") -> Dict:
        """Build a FHIR MedicationRequest resource."""
        resource = {
            "resourceType": "MedicationRequest",
            "id": str(uuid.uuid4()),
            "status": status,
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                             "display": medication}],
                "text": medication,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "authoredOn": datetime.utcnow().isoformat() + "Z",
        }
        dosage: Dict[str, Any] = {
            "route": {"coding": [{"system": "http://snomed.info/sct",
                                   "display": route}]},
        }
        if dose:
            dosage["text"] = dose
        if frequency:
            dosage["timing"] = {"repeat": {"boundsPeriod": {
                "start": datetime.utcnow().isoformat() + "Z"
            }}, "code": {"text": frequency}}
        resource["dosageInstruction"] = [dosage]
        return resource

    # ── RiskAssessment ────────────────────

    def risk_assessment(self, patient_id: str, risk_scores: Dict,
                         news2: int = 0) -> Dict:
        """Build a FHIR RiskAssessment resource for AI risk scores."""
        predictions = []
        risk_code_map = {
            "sepsis_risk":    ("91302008",  "Sepsis"),
            "mortality_risk": ("399166001", "Fatal outcome"),
            "cardiac_risk":   ("22298006",  "Myocardial infarction"),
        }
        for key, score in risk_scores.items():
            snomed_code, snomed_display = risk_code_map.get(key, ("unknown", key))
            predictions.append({
                "outcome": {"coding": [{"system": "http://snomed.info/sct",
                                        "code": snomed_code,
                                        "display": snomed_display}]},
                "probabilityDecimal": round(float(score), 4),
                "qualitativeRisk": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                        "code": "high" if score > 0.5 else "moderate" if score > 0.25 else "low",
                    }]
                },
            })

        return {
            "resourceType": "RiskAssessment",
            "id": str(uuid.uuid4()),
            "status": "final",
            "method": {"coding": [{"system": "http://snomed.info/sct",
                                    "code": "AIML",
                                    "display": "AI/ML Risk Model"}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": datetime.utcnow().isoformat() + "Z",
            "basis": [{"display": f"NEWS2 Score: {news2}"}],
            "prediction": predictions,
            "note": [{"text": "AI-generated risk assessment — requires clinical validation"}],
        }

    # ── DetectedIssue (Alert) ──────────────

    def detected_issue(self, patient_id: str, alert_type: str,
                        message: str, severity: str = "high") -> Dict:
        """Build a FHIR DetectedIssue resource for a clinical alert."""
        severity_map = {
            "critical": "high", "high": "high",
            "moderate": "moderate", "low": "low",
        }
        return {
            "resourceType": "DetectedIssue",
            "id": str(uuid.uuid4()),
            "status": "preliminary",
            "severity": severity_map.get(severity, "moderate"),
            "patient": {"reference": f"Patient/{patient_id}"},
            "identifiedDateTime": datetime.utcnow().isoformat() + "Z",
            "detail": message,
            "code": {"text": alert_type},
        }

    # ── DiagnosticReport ──────────────────

    def diagnostic_report(self, patient_id: str, title: str,
                           observation_ids: List[str],
                           conclusion: str = "") -> Dict:
        """Build a FHIR DiagnosticReport wrapping multiple observations."""
        return {
            "resourceType": "DiagnosticReport",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {"text": title},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "issued": datetime.utcnow().isoformat() + "Z",
            "result": [{"reference": f"Observation/{oid}"} for oid in observation_ids],
            "conclusion": conclusion,
        }


# ─────────────────────────────────────────
# FHIR parser (FHIR → internal)
# ─────────────────────────────────────────

class FHIRParser:
    """Parses FHIR R4 resources into internal Hospital OS dicts."""

    def parse_patient(self, resource: Dict) -> Dict:
        names = resource.get("name", [{}])
        name_obj = names[0] if names else {}
        given = " ".join(name_obj.get("given", []))
        family = name_obj.get("family", "")
        full_name = f"{given} {family}".strip()
        gender_map = {"male": "M", "female": "F", "unknown": "U", "other": "O"}
        return {
            "patient_id": resource.get("id"),
            "name": full_name,
            "sex": gender_map.get(resource.get("gender", "unknown"), "U"),
            "date_of_birth": resource.get("birthDate"),
            "active": resource.get("active", True),
        }

    def parse_observation(self, resource: Dict) -> Dict:
        code_obj = resource.get("code", {})
        codings = code_obj.get("coding", [{}])
        loinc_code = codings[0].get("code", "") if codings else ""

        # Reverse-lookup field from LOINC code
        field = next((k for k, (c, _) in LOINC.items() if c == loinc_code), loinc_code)

        value_q = resource.get("valueQuantity", {})
        return {
            "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "field": field,
            "value": value_q.get("value"),
            "unit": value_q.get("unit"),
            "timestamp": resource.get("effectiveDateTime"),
            "status": resource.get("status"),
        }

    def parse_bundle_observations(self, bundle: Dict) -> List[Dict]:
        """Extract all observations from a FHIR Bundle."""
        results = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            if res.get("resourceType") == "Observation":
                results.append(self.parse_observation(res))
        return results

    def parse_condition(self, resource: Dict) -> Dict:
        code_obj = resource.get("code", {})
        codings = code_obj.get("coding", [{}])
        return {
            "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "diagnosis": code_obj.get("text") or (codings[0].get("display") if codings else ""),
            "snomed_code": codings[0].get("code") if codings else "",
            "clinical_status": (resource.get("clinicalStatus", {})
                                .get("coding", [{}])[0].get("code", "")),
            "recorded_date": resource.get("recordedDate"),
        }

    def parse_risk_assessment(self, resource: Dict) -> Dict:
        scores = {}
        for pred in resource.get("prediction", []):
            outcome = (pred.get("outcome", {}).get("coding", [{}])[0].get("display", ""))
            prob = pred.get("probabilityDecimal", 0)
            scores[outcome] = prob
        return {
            "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "risk_scores": scores,
            "timestamp": resource.get("occurrenceDateTime"),
        }


# ─────────────────────────────────────────
# Lightweight FHIR client
# ─────────────────────────────────────────

class FHIRClient:
    """
    HTTP client for a FHIR R4 server.
    Falls back gracefully when no server is available.
    """

    def __init__(self, base_url: str = "http://localhost:8080/fhir",
                  token: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._available = self._check_server()

    def _headers(self) -> Dict:
        h = {"Content-Type": "application/fhir+json",
              "Accept": "application/fhir+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _check_server(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/metadata", headers=self._headers()
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            logger.info(f"FHIR server not available at {self.base_url}")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Optional[Dict]:
        import urllib.request
        url = f"{self.base_url}/{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error(f"FHIR {method} {url} failed: {e}")
            return None

    def create(self, resource: Dict) -> Optional[Dict]:
        rtype = resource.get("resourceType", "Resource")
        result = self._request("POST", rtype, resource)
        if result:
            logger.info(f"Created FHIR {rtype}/{result.get('id')}")
        return result

    def read(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        return self._request("GET", f"{resource_type}/{resource_id}")

    def update(self, resource: Dict) -> Optional[Dict]:
        rtype = resource.get("resourceType")
        rid = resource.get("id")
        return self._request("PUT", f"{rtype}/{rid}", resource)

    def search(self, resource_type: str, params: Dict) -> Optional[Dict]:
        import urllib.parse
        qs = urllib.parse.urlencode(params)
        return self._request("GET", f"{resource_type}?{qs}")

    def submit_bundle(self, bundle: Dict) -> Optional[Dict]:
        return self._request("POST", "", bundle)

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        return self.read("Patient", patient_id)

    def get_patient_observations(self, patient_id: str,
                                  category: str = "vital-signs") -> Optional[Dict]:
        return self.search("Observation",
                            {"subject": f"Patient/{patient_id}", "category": category})


# ─────────────────────────────────────────
# Integration service
# ─────────────────────────────────────────

class FHIRIntegrationService:
    """
    High-level service: converts Hospital OS data → FHIR and submits to server.
    Works in offline mode (no server) by returning the FHIR JSON locally.
    """

    def __init__(self, fhir_server_url: str = "http://localhost:8080/fhir",
                  token: Optional[str] = None):
        self.builder = FHIRBuilder()
        self.parser = FHIRParser()
        self.client = FHIRClient(fhir_server_url, token)
        logger.info(f"FHIRIntegrationService init — server {'available' if self.client.available else 'offline'}")

    def export_patient(self, patient_data: Dict) -> Dict:
        """Export a patient record to FHIR."""
        resource = self.builder.patient(patient_data)
        if self.client.available:
            result = self.client.create(resource)
            return result or resource
        return resource

    def export_vitals(self, patient_id: str, vitals: Dict,
                       timestamp: Optional[str] = None) -> Dict:
        """Export a vital-signs set to FHIR as a transaction Bundle."""
        bundle = self.builder.vital_bundle(patient_id, vitals, timestamp)
        if self.client.available:
            result = self.client.submit_bundle(bundle)
            return result or bundle
        return bundle

    def export_risk_scores(self, patient_id: str, risk_scores: Dict,
                            news2: int = 0) -> Dict:
        """Export AI risk scores as a FHIR RiskAssessment."""
        resource = self.builder.risk_assessment(patient_id, risk_scores, news2)
        if self.client.available:
            result = self.client.create(resource)
            return result or resource
        return resource

    def export_alert(self, patient_id: str, alert_type: str,
                      message: str, severity: str = "high") -> Dict:
        """Export a clinical alert as a FHIR DetectedIssue."""
        resource = self.builder.detected_issue(patient_id, alert_type, message, severity)
        if self.client.available:
            result = self.client.create(resource)
            return result or resource
        return resource

    def import_patient_bundle(self, bundle: Dict) -> List[Dict]:
        """Import a FHIR bundle of patients and observations."""
        results = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")
            if rtype == "Patient":
                results.append({"type": "patient", "data": self.parser.parse_patient(resource)})
            elif rtype == "Observation":
                results.append({"type": "observation", "data": self.parser.parse_observation(resource)})
            elif rtype == "Condition":
                results.append({"type": "condition", "data": self.parser.parse_condition(resource)})
        return results

    def full_patient_export(self, patient_data: Dict) -> Dict:
        """Build a complete FHIR Bundle for one patient (offline-safe)."""
        pid = patient_data.get("patient_id", "unknown")
        entries = []

        # Patient resource
        pt = self.builder.patient(patient_data)
        entries.append({"fullUrl": f"urn:uuid:{pid}", "resource": pt,
                         "request": {"method": "PUT", "url": f"Patient/{pid}"}})

        # Vitals
        if "vitals" in patient_data:
            bundle = self.builder.vital_bundle(pid, patient_data["vitals"])
            entries += bundle.get("entry", [])

        # Risk assessment
        if "risk_scores" in patient_data:
            ra = self.builder.risk_assessment(pid, patient_data["risk_scores"],
                                               patient_data.get("news2", 0))
            entries.append({"fullUrl": f"urn:uuid:{ra['id']}", "resource": ra,
                             "request": {"method": "POST", "url": "RiskAssessment"}})

        # Diagnosis
        if "primary_diagnosis" in patient_data:
            cond = self.builder.condition(pid, patient_data["primary_diagnosis"])
            entries.append({"fullUrl": f"urn:uuid:{cond['id']}", "resource": cond,
                             "request": {"method": "POST", "url": "Condition"}})

        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "transaction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": entries,
        }


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    svc = FHIRIntegrationService()
    print(f"FHIR server: {'online' if svc.client.available else 'offline (local mode)'}")

    patient = {
        "patient_id": "P001", "name": "Alice Chen", "age": 68, "sex": "F",
        "primary_diagnosis": "septic_shock",
        "vitals": {"heart_rate": 118, "sbp": 88, "dbp": 55,
                    "respiratory_rate": 26, "temperature": 38.8, "spo2": 91},
        "risk_scores": {"sepsis_risk": 0.78, "mortality_risk": 0.31},
        "news2": 9,
    }

    bundle = svc.full_patient_export(patient)
    n_entries = len(bundle["entry"])
    print(f"\nFHIR Bundle for P001:")
    print(f"  Total entries : {n_entries}")
    for e in bundle["entry"]:
        rtype = e["resource"]["resourceType"]
        rid = e["resource"].get("id", "?")[:8]
        print(f"  {rtype:25s} id={rid}...")

    # Parse back
    parsed = svc.import_patient_bundle(bundle)
    print(f"\nParsed back: {len(parsed)} resources")
    for p in parsed[:4]:
        print(f"  type={p['type']:12s} data={str(p['data'])[:60]}...")

    # Serialize a single vital observation
    builder = FHIRBuilder()
    obs = builder.observation("P001", "lactate", 3.8)
    print(f"\nSample Observation (lactate=3.8):")
    print(f"  LOINC: {obs['code']['coding'][0]['code']} — {obs['code']['coding'][0]['display']}")
    print(f"  Value: {obs['valueQuantity']['value']} {obs['valueQuantity']['unit']}")
    if "interpretation" in obs:
        print(f"  Flag:  {obs['interpretation'][0]['coding'][0]['display']}")

    # Validate round-trip
    parsed_obs = FHIRParser().parse_observation(obs)
    print(f"  Round-trip field: {parsed_obs['field']}, value: {parsed_obs['value']}")
    print("\nFHIR integration demo complete.")

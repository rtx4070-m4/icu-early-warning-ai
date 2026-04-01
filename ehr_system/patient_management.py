"""
AI Hospital Operating System
EHR Patient Management Module
==============================
Provides CRUD operations, search, and summary generation
for the Electronic Health Record system.
"""

import os
import uuid
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
import pandas as pd
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("ehr_system")

# ─────────────────────────────────────────────
# Database connection helper
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "hospital_os"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}


def get_connection():
    """Return a new psycopg2 connection using environment config."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


# ─────────────────────────────────────────────
# Data Classes (lightweight domain models)
# ─────────────────────────────────────────────
@dataclass
class Patient:
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    race: str = ""
    ethnicity: str = ""
    address: str = ""
    phone: str = ""
    emergency_contact: str = ""
    insurance_id: str = ""
    patient_id: Optional[str] = None

    @property
    def age(self) -> int:
        today = date.today()
        dob = self.date_of_birth
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class VitalSigns:
    patient_id: str
    charttime: datetime
    heart_rate: float
    sbp: float
    dbp: float
    map: float
    temperature: float
    spo2: float
    resp_rate: float
    gcs_total: int = 15
    icustay_id: Optional[str] = None
    vital_id: Optional[str] = None

    def is_critical(self) -> bool:
        """Return True if any vital is outside critical thresholds."""
        return (
            self.heart_rate < 40 or self.heart_rate > 150
            or self.sbp < 70 or self.sbp > 200
            or self.spo2 < 88
            or self.resp_rate > 35
            or self.temperature < 34 or self.temperature > 40
            or self.gcs_total < 8
        )


@dataclass
class LabResult:
    patient_id: str
    charttime: datetime
    lab_name: str
    value: float
    value_uom: str
    ref_range_lower: float
    ref_range_upper: float
    flag: str = "normal"
    icustay_id: Optional[str] = None


# ─────────────────────────────────────────────
# Patient Repository
# ─────────────────────────────────────────────
class PatientRepository:
    """CRUD layer for the patients table."""

    def create(self, patient: Patient) -> Patient:
        """Insert a new patient record and return with generated patient_id."""
        sql = """
            INSERT INTO patients
                (mrn, first_name, last_name, date_of_birth, gender, race,
                 ethnicity, address, phone, emergency_contact, insurance_id)
            VALUES
                (%(mrn)s, %(first_name)s, %(last_name)s, %(date_of_birth)s,
                 %(gender)s, %(race)s, %(ethnicity)s, %(address)s, %(phone)s,
                 %(emergency_contact)s, %(insurance_id)s)
            RETURNING patient_id
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, asdict(patient))
                row = cur.fetchone()
                conn.commit()
        patient.patient_id = str(row["patient_id"])
        logger.info("Created patient %s (MRN: %s)", patient.full_name, patient.mrn)
        return patient

    def get_by_id(self, patient_id: str) -> Optional[Dict]:
        sql = "SELECT * FROM patients WHERE patient_id = %s"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (patient_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def get_by_mrn(self, mrn: str) -> Optional[Dict]:
        sql = "SELECT * FROM patients WHERE mrn = %s"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (mrn,))
                row = cur.fetchone()
        return dict(row) if row else None

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """Full-text search across name, MRN, and diagnoses."""
        sql = """
            SELECT p.*,
                   DATE_PART('year', AGE(p.date_of_birth))::INT AS age
            FROM patients p
            WHERE to_tsvector('english',
                      p.first_name || ' ' || p.last_name || ' ' || p.mrn
                  ) @@ plainto_tsquery('english', %s)
               OR p.mrn ILIKE %s
               OR p.first_name ILIKE %s
               OR p.last_name  ILIKE %s
            LIMIT %s OFFSET %s
        """
        like_q = f"%{query}%"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (query, like_q, like_q, like_q, limit, offset))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def list_all(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        sql = """
            SELECT * FROM patient_summary
            ORDER BY last_admit DESC NULLS LAST
            LIMIT %s OFFSET %s
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=(limit, offset))

    def update(self, patient_id: str, updates: Dict[str, Any]) -> bool:
        allowed = {"first_name","last_name","address","phone",
                   "emergency_contact","insurance_id","updated_at"}
        safe_updates = {k: v for k, v in updates.items() if k in allowed}
        if not safe_updates:
            return False
        safe_updates["updated_at"] = datetime.utcnow()
        set_clause = ", ".join(f"{k} = %({k})s" for k in safe_updates)
        safe_updates["patient_id"] = patient_id
        sql = f"UPDATE patients SET {set_clause} WHERE patient_id = %(patient_id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, safe_updates)
                affected = cur.rowcount
            conn.commit()
        return affected > 0


# ─────────────────────────────────────────────
# Encounter Repository
# ─────────────────────────────────────────────
class EncounterRepository:

    def create(self, patient_id: str, encounter_type: str,
               admit_time: datetime, chief_complaint: str,
               attending_md: str) -> str:
        sql = """
            INSERT INTO encounters
                (patient_id, encounter_type, admit_time, chief_complaint, attending_md)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING encounter_id
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (patient_id, encounter_type, admit_time,
                                  chief_complaint, attending_md))
                enc_id = cur.fetchone()[0]
            conn.commit()
        logger.info("Created encounter %s for patient %s", enc_id, patient_id)
        return str(enc_id)

    def get_patient_encounters(self, patient_id: str) -> pd.DataFrame:
        sql = """
            SELECT e.*, a.admission_type, a.hospital_expire_flag
            FROM encounters e
            LEFT JOIN admissions a ON e.encounter_id = a.encounter_id
            WHERE e.patient_id = %s
            ORDER BY e.admit_time DESC
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=(patient_id,))

    def discharge(self, encounter_id: str, discharge_time: datetime) -> bool:
        sql = """
            UPDATE encounters
            SET discharge_time = %s
            WHERE encounter_id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (discharge_time, encounter_id))
                affected = cur.rowcount
            conn.commit()
        return affected > 0


# ─────────────────────────────────────────────
# Vitals Repository
# ─────────────────────────────────────────────
class VitalsRepository:

    def insert(self, vitals: VitalSigns) -> str:
        sql = """
            INSERT INTO vital_signs
                (patient_id, icustay_id, charttime, heart_rate, sbp, dbp, map,
                 temperature, spo2, resp_rate, gcs_total)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING vital_id
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    vitals.patient_id, vitals.icustay_id, vitals.charttime,
                    vitals.heart_rate, vitals.sbp, vitals.dbp, vitals.map,
                    vitals.temperature, vitals.spo2, vitals.resp_rate, vitals.gcs_total,
                ))
                vid = cur.fetchone()[0]
            conn.commit()

        if vitals.is_critical():
            logger.warning("CRITICAL vitals detected for patient %s at %s",
                           vitals.patient_id, vitals.charttime)
        return str(vid)

    def get_timeseries(
        self,
        patient_id: str,
        hours_back: int = 24,
    ) -> pd.DataFrame:
        sql = """
            SELECT charttime, heart_rate, sbp, dbp, map,
                   temperature, spo2, resp_rate, gcs_total
            FROM vital_signs
            WHERE patient_id = %s
              AND charttime >= NOW() - INTERVAL '%s hours'
            ORDER BY charttime
        """
        with get_connection() as conn:
            df = pd.read_sql(sql, conn, params=(patient_id, hours_back))
        return df

    def get_latest(self, patient_id: str) -> Optional[Dict]:
        sql = """
            SELECT * FROM icu_vitals_latest
            WHERE patient_id = %s
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (patient_id,))
                row = cur.fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────
# Lab Repository
# ─────────────────────────────────────────────
class LabRepository:

    def insert(self, lab: LabResult) -> str:
        sql = """
            INSERT INTO lab_results
                (patient_id, icustay_id, charttime, lab_name, value,
                 value_uom, ref_range_lower, ref_range_upper, flag)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING lab_id
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    lab.patient_id, lab.icustay_id, lab.charttime,
                    lab.lab_name, lab.value, lab.value_uom,
                    lab.ref_range_lower, lab.ref_range_upper, lab.flag,
                ))
                lid = cur.fetchone()[0]
            conn.commit()
        return str(lid)

    def get_labs(self, patient_id: str, lab_name: Optional[str] = None,
                 hours_back: int = 48) -> pd.DataFrame:
        params: list = [patient_id, hours_back]
        lab_filter = ""
        if lab_name:
            lab_filter = "AND lab_name = %s"
            params.append(lab_name)
        sql = f"""
            SELECT charttime, lab_name, value, value_uom, flag
            FROM lab_results
            WHERE patient_id = %s
              AND charttime >= NOW() - INTERVAL '%s hours'
              {lab_filter}
            ORDER BY charttime
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=tuple(params))

    def get_critical_labs(self) -> pd.DataFrame:
        """All unacknowledged critical labs across all active ICU patients."""
        sql = """
            SELECT lr.*, p.mrn, p.first_name || ' ' || p.last_name AS patient_name
            FROM lab_results lr
            JOIN patients p ON lr.patient_id = p.patient_id
            WHERE lr.flag IN ('critical_low','critical_high')
              AND lr.charttime >= NOW() - INTERVAL '12 hours'
            ORDER BY lr.charttime DESC
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn)


# ─────────────────────────────────────────────
# Clinical Note Repository
# ─────────────────────────────────────────────
class ClinicalNoteRepository:

    def insert(self, patient_id: str, encounter_id: str,
               category: str, description: str,
               text: str, cgid: str = "SYS") -> str:
        sql = """
            INSERT INTO clinical_notes
                (patient_id, encounter_id, charttime, category, description, text, cgid)
            VALUES (%s, %s, NOW(), %s, %s, %s, %s)
            RETURNING note_id
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (patient_id, encounter_id, category,
                                  description, text, cgid))
                nid = cur.fetchone()[0]
            conn.commit()
        return str(nid)

    def search_notes(self, query: str, patient_id: Optional[str] = None,
                     limit: int = 20) -> pd.DataFrame:
        params: list = [query]
        patient_filter = ""
        if patient_id:
            patient_filter = "AND n.patient_id = %s"
            params.append(patient_id)
        sql = f"""
            SELECT n.note_id, n.patient_id, n.charttime, n.category,
                   n.description,
                   ts_headline('english', n.text,
                       plainto_tsquery('english', %s),
                       'MaxWords=50, MinWords=20') AS excerpt,
                   p.mrn
            FROM clinical_notes n
            JOIN patients p ON n.patient_id = p.patient_id
            WHERE to_tsvector('english', n.text)
                  @@ plainto_tsquery('english', %s)
              {patient_filter}
            ORDER BY n.charttime DESC
            LIMIT %s
        """
        params = [query] + params + [limit]  # headline param + search param
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=tuple(params))

    def get_recent(self, patient_id: str, limit: int = 10) -> pd.DataFrame:
        sql = """
            SELECT note_id, charttime, category, description,
                   LEFT(text, 200) AS preview
            FROM clinical_notes
            WHERE patient_id = %s
            ORDER BY charttime DESC
            LIMIT %s
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=(patient_id, limit))


# ─────────────────────────────────────────────
# EHR Service (facade combining all repos)
# ─────────────────────────────────────────────
class EHRService:
    """
    High-level EHR service composing all repositories.
    This is the primary interface for other modules.
    """

    def __init__(self):
        self.patients   = PatientRepository()
        self.encounters = EncounterRepository()
        self.vitals     = VitalsRepository()
        self.labs       = LabRepository()
        self.notes      = ClinicalNoteRepository()

    # ── Patient Operations ──────────────────────
    def admit_patient(
        self,
        patient: Patient,
        encounter_type: str,
        chief_complaint: str,
        attending_md: str,
    ) -> Dict:
        """Register new patient and open encounter."""
        existing = self.patients.get_by_mrn(patient.mrn)
        if existing:
            patient_id = str(existing["patient_id"])
            logger.info("Patient %s already exists, opening new encounter", patient.mrn)
        else:
            patient = self.patients.create(patient)
            patient_id = patient.patient_id

        enc_id = self.encounters.create(
            patient_id, encounter_type,
            datetime.utcnow(), chief_complaint, attending_md,
        )
        return {"patient_id": patient_id, "encounter_id": enc_id}

    def get_patient_summary(self, patient_id: str) -> Dict:
        """Return a comprehensive patient summary."""
        patient = self.patients.get_by_id(patient_id)
        if not patient:
            return {}
        encounters  = self.encounters.get_patient_encounters(patient_id)
        latest_vitals = self.vitals.get_latest(patient_id)
        recent_labs   = self.labs.get_labs(patient_id, hours_back=24)
        recent_notes  = self.notes.get_recent(patient_id, limit=5)

        return {
            "patient":       patient,
            "encounters":    encounters.to_dict("records") if not encounters.empty else [],
            "latest_vitals": latest_vitals,
            "recent_labs":   recent_labs.to_dict("records") if not recent_labs.empty else [],
            "recent_notes":  recent_notes.to_dict("records") if not recent_notes.empty else [],
        }

    def get_icu_census(self) -> pd.DataFrame:
        """Active ICU census with latest vitals and risk scores."""
        sql = """
            SELECT
                p.mrn, p.first_name || ' ' || p.last_name AS name,
                DATE_PART('year', AGE(p.date_of_birth))::INT AS age,
                i.icu_unit, i.intime,
                ROUND(i.los::numeric, 1) AS los_days,
                v.heart_rate, v.sbp, v.spo2, v.resp_rate, v.temperature,
                mp.score AS sepsis_risk,
                mp.model_name
            FROM icustays i
            JOIN patients     p ON i.patient_id  = p.patient_id
            LEFT JOIN icu_vitals_latest v ON i.patient_id = v.patient_id
            LEFT JOIN LATERAL (
                SELECT score, model_name
                FROM ml_predictions
                WHERE patient_id = i.patient_id
                  AND prediction_type = 'sepsis_risk'
                ORDER BY pred_time DESC LIMIT 1
            ) mp ON TRUE
            WHERE i.outtime IS NULL
            ORDER BY mp.score DESC NULLS LAST
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn)

    def get_active_alerts(self) -> pd.DataFrame:
        sql = "SELECT * FROM active_alerts"
        with get_connection() as conn:
            return pd.read_sql(sql, conn)

    def acknowledge_alert(self, alert_id: str, clinician: str) -> bool:
        sql = """
            UPDATE ai_alerts
            SET acknowledged = TRUE, acknowledged_by = %s
            WHERE alert_id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (clinician, alert_id))
                affected = cur.rowcount
            conn.commit()
        return affected > 0

    def save_prediction(
        self,
        patient_id: str,
        icustay_id: str,
        model_name: str,
        model_version: str,
        prediction_type: str,
        score: float,
        threshold: float = 0.5,
        feature_json: Optional[Dict] = None,
    ) -> None:
        """Persist ML prediction result and create alert if threshold exceeded."""
        import json
        is_positive = score >= threshold
        sql = """
            INSERT INTO ml_predictions
                (patient_id, icustay_id, model_name, model_version,
                 prediction_type, score, threshold, is_positive, feature_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    patient_id, icustay_id, model_name, model_version,
                    prediction_type, score, threshold, is_positive,
                    json.dumps(feature_json) if feature_json else None,
                ))
            conn.commit()

        if is_positive:
            severity = "critical" if score > 0.8 else "warning"
            alert_sql = """
                INSERT INTO ai_alerts
                    (patient_id, icustay_id, alert_type, severity,
                     risk_score, model_name, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(alert_sql, (
                        patient_id, icustay_id, prediction_type, severity,
                        score, model_name,
                        f"{prediction_type} alert: score={score:.3f} (threshold={threshold})",
                    ))
                conn.commit()
            logger.warning("Alert created: %s for patient %s (score=%.3f)",
                           prediction_type, patient_id, score)


# ─────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    svc = EHRService()

    # Example: print ICU census
    try:
        census = svc.get_icu_census()
        print("\n=== ICU CENSUS ===")
        print(census.to_string(index=False))
    except Exception as exc:
        logger.error("Cannot connect to database: %s", exc)
        print("(Run with a live database to see ICU census)")

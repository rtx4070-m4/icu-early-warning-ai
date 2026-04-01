"""
AI Hospital Operating System
Data Engineering – Data Loader
================================
Handles extraction from PostgreSQL + CSV sources,
validation, and loading into the processing pipeline.
"""

import os
import io
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("data_loader")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_URL = (
    f"postgresql://"
    f"{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','postgres')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}"
    f"/{os.getenv('DB_NAME','hospital_os')}"
)

DATA_DIR   = Path(os.getenv("DATA_DIR",   "/data"))
RAW_DIR    = DATA_DIR / "raw"
CLEAN_DIR  = DATA_DIR / "clean"
EXPORT_DIR = DATA_DIR / "export"

for d in [RAW_DIR, CLEAN_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Database connector
# ─────────────────────────────────────────────
def get_engine():
    return create_engine(DB_URL, pool_size=5, max_overflow=10)


# ─────────────────────────────────────────────
# Data Validators
# ─────────────────────────────────────────────
VITAL_RANGES = {
    "heart_rate":   (20,   300),
    "sbp":          (40,   300),
    "dbp":          (10,   200),
    "map":          (20,   200),
    "temperature":  (25,   44),
    "spo2":         (50,   100),
    "resp_rate":    (1,    80),
    "gcs_total":    (3,    15),
}

LAB_RANGES = {
    "Creatinine":  (0.1,   20.0),
    "WBC":         (0.1,  100.0),
    "Hemoglobin":  (1.0,   25.0),
    "Platelets":   (5.0, 2000.0),
    "Sodium":     (100.0,  180.0),
    "Potassium":   (1.0,   10.0),
    "Glucose":    (20.0,  800.0),
    "BUN":         (1.0,  200.0),
    "Lactate":     (0.0,   15.0),
    "Troponin":    (0.0,   50.0),
}


def validate_vitals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate vital signs against physiological plausibility ranges.
    Returns (clean_df, rejected_df).
    """
    df = df.copy()
    rejection_mask = pd.Series(False, index=df.index)

    for col, (lo, hi) in VITAL_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = (df[col] < lo) | (df[col] > hi)
        # Nullify out-of-range values rather than rejecting entire row
        df.loc[out_of_range, col] = np.nan
        logger.debug("Nullified %d out-of-range %s values", out_of_range.sum(), col)

    # Reject rows missing more than 60% of vital fields
    vital_cols = [c for c in VITAL_RANGES if c in df.columns]
    missing_frac = df[vital_cols].isna().mean(axis=1)
    rejection_mask = missing_frac > 0.6

    clean = df[~rejection_mask].copy()
    rejected = df[rejection_mask].copy()

    logger.info(
        "Vitals validation: %d clean, %d rejected (%.1f%%)",
        len(clean), len(rejected),
        100 * len(rejected) / max(len(df), 1),
    )
    return clean, rejected


def validate_labs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate labs against plausibility ranges and flag each result.
    """
    df = df.copy()
    flags = []
    for _, row in df.iterrows():
        lab = row.get("lab_name", "")
        val = row.get("value", np.nan)
        lo, hi = LAB_RANGES.get(lab, (None, None))
        if lo is not None and not np.isnan(val):
            if val < lo or val > hi:
                flags.append("implausible")
                continue
        ref_lo = row.get("ref_range_lower", np.nan)
        ref_hi = row.get("ref_range_upper", np.nan)
        if not np.isnan(ref_lo) and not np.isnan(ref_hi):
            if val < ref_lo * 0.5:
                flags.append("critical_low")
            elif val > ref_hi * 3:
                flags.append("critical_high")
            elif val < ref_lo:
                flags.append("low")
            elif val > ref_hi:
                flags.append("high")
            else:
                flags.append("normal")
        else:
            flags.append("unknown")
    df["flag"] = flags
    return df[df["flag"] != "implausible"]


# ─────────────────────────────────────────────
# Extractors
# ─────────────────────────────────────────────
class DatabaseExtractor:
    """Pull datasets from PostgreSQL EHR."""

    def __init__(self):
        self.engine = get_engine()

    def extract_vitals(
        self,
        patient_ids: Optional[List[str]] = None,
        hours_back: int = 72,
    ) -> pd.DataFrame:
        """Extract recent vital signs."""
        params: dict = {"cutoff": datetime.utcnow() - timedelta(hours=hours_back)}
        where = "WHERE charttime >= :cutoff"
        if patient_ids:
            where += " AND patient_id = ANY(:pids)"
            params["pids"] = patient_ids

        query = f"""
            SELECT v.*,
                   p.mrn, p.first_name, p.last_name,
                   i.icu_unit
            FROM vital_signs v
            JOIN patients p ON v.patient_id = p.patient_id
            LEFT JOIN icustays i ON v.icustay_id = i.icustay_id
            {where}
            ORDER BY v.patient_id, v.charttime
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        logger.info("Extracted %d vital sign records (last %dh)", len(df), hours_back)
        return df

    def extract_labs(
        self,
        patient_ids: Optional[List[str]] = None,
        hours_back: int = 72,
    ) -> pd.DataFrame:
        params: dict = {"cutoff": datetime.utcnow() - timedelta(hours=hours_back)}
        where = "WHERE charttime >= :cutoff"
        if patient_ids:
            where += " AND patient_id = ANY(:pids)"
            params["pids"] = patient_ids

        query = f"""
            SELECT lr.*, p.mrn
            FROM lab_results lr
            JOIN patients p ON lr.patient_id = p.patient_id
            {where}
            ORDER BY lr.patient_id, lr.charttime
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        logger.info("Extracted %d lab records (last %dh)", len(df), hours_back)
        return df

    def extract_icu_patients(self) -> pd.DataFrame:
        """All currently active ICU patients with stay info."""
        query = """
            SELECT
                i.icustay_id, i.patient_id, i.icu_unit,
                i.intime,
                EXTRACT(EPOCH FROM (NOW() - i.intime)) / 3600 AS los_hours,
                p.mrn, p.first_name, p.last_name,
                DATE_PART('year', AGE(p.date_of_birth))::INT AS age,
                p.gender
            FROM icustays i
            JOIN patients p ON i.patient_id = p.patient_id
            WHERE i.outtime IS NULL
            ORDER BY i.intime
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info("Found %d active ICU patients", len(df))
        return df

    def extract_diagnoses(
        self, patient_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        where = ""
        params: dict = {}
        if patient_ids:
            where = "WHERE d.patient_id = ANY(:pids)"
            params["pids"] = patient_ids
        query = f"""
            SELECT d.*, p.mrn
            FROM diagnoses d
            JOIN patients p ON d.patient_id = p.patient_id
            {where}
            ORDER BY d.diag_priority
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def extract_medications(
        self, patient_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        where = ""
        params: dict = {}
        if patient_ids:
            where = "WHERE m.patient_id = ANY(:pids)"
            params["pids"] = patient_ids
        query = f"""
            SELECT m.*, p.mrn
            FROM medications m
            JOIN patients p ON m.patient_id = p.patient_id
            {where}
            ORDER BY m.starttime DESC
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)

    def extract_notes(
        self, patient_ids: Optional[List[str]] = None, hours_back: int = 48
    ) -> pd.DataFrame:
        params: dict = {"cutoff": datetime.utcnow() - timedelta(hours=hours_back)}
        where = "WHERE charttime >= :cutoff"
        if patient_ids:
            where += " AND patient_id = ANY(:pids)"
            params["pids"] = patient_ids
        query = f"""
            SELECT cn.*, p.mrn
            FROM clinical_notes cn
            JOIN patients p ON cn.patient_id = p.patient_id
            {where}
            ORDER BY charttime DESC
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)


# ─────────────────────────────────────────────
# CSV / File Extractor (Legacy systems)
# ─────────────────────────────────────────────
class CSVExtractor:
    """Load data from legacy CSV exports (e.g., SAS output)."""

    def load_vitals_csv(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["charttime"])
        logger.info("Loaded %d rows from %s", len(df), filepath)
        return df

    def load_labs_csv(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["charttime"])
        logger.info("Loaded %d rows from %s", len(df), filepath)
        return df

    def load_sofa_csv(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["charttime"])
        return df

    def load_ml_features_csv(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["charttime"])
        logger.info("ML feature file loaded: %d rows, %d columns", len(df), len(df.columns))
        return df


# ─────────────────────────────────────────────
# Synthetic Data Generator (for testing/demo)
# ─────────────────────────────────────────────
class SyntheticDataGenerator:
    """
    Generate medically plausible synthetic datasets when no
    live database is available (useful for CI/CD and testing).
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_vitals(
        self,
        n_patients: int = 10,
        hours: int = 72,
        deterioration_prob: float = 0.2,
    ) -> pd.DataFrame:
        """Generate synthetic vital sign time series."""
        records = []
        base_time = datetime.utcnow() - timedelta(hours=hours)

        for p in range(n_patients):
            patient_id = f"SYNTH-P{p:04d}"
            deteriorating = np.random.random() < deterioration_prob

            for h in range(hours):
                t = base_time + timedelta(hours=h)
                progress = h / hours  # 0→1 over time

                if deteriorating:
                    hr  = 75  + progress * 40 + np.random.randn() * 8
                    sbp = 120 - progress * 35 + np.random.randn() * 8
                    spo2 = 98 - progress * 12 + np.random.randn() * 2
                    rr  = 16  + progress * 10 + np.random.randn() * 2
                else:
                    hr   = 75  + np.random.randn() * 8
                    sbp  = 120 + np.random.randn() * 10
                    spo2 = 97  + np.random.randn() * 1.5
                    rr   = 16  + np.random.randn() * 2

                dbp  = sbp * 0.6 + np.random.randn() * 5
                map_ = (sbp + 2 * dbp) / 3
                temp = 36.8 + np.random.randn() * 0.4 + (0.8 if deteriorating else 0)

                records.append({
                    "patient_id":  patient_id,
                    "charttime":   t,
                    "heart_rate":  np.clip(hr,  40, 200),
                    "sbp":         np.clip(sbp, 60, 200),
                    "dbp":         np.clip(dbp, 40, 130),
                    "map":         np.clip(map_, 40, 150),
                    "temperature": np.clip(temp, 35, 41),
                    "spo2":        np.clip(spo2, 70, 100),
                    "resp_rate":   np.clip(rr,   8,  40),
                    "gcs_total":   max(3, 15 - int(progress * 5)) if deteriorating else 15,
                    "label_deteriorating": int(deteriorating),
                })

        df = pd.DataFrame(records)
        logger.info("Generated %d synthetic vital records (%d patients)",
                    len(df), n_patients)
        return df

    def generate_labs(self, n_patients: int = 10) -> pd.DataFrame:
        """Generate synthetic lab results."""
        labs = [
            ("Creatinine",  1.0, 0.5, 0.6,  1.2, "mg/dL"),
            ("WBC",         8.0, 3.0, 4.5, 11.0,  "K/uL"),
            ("Hemoglobin", 13.0, 2.0,12.0, 17.5,  "g/dL"),
            ("Platelets", 250.0,80.0,150.0,400.0, "K/uL"),
            ("Lactate",     1.5, 1.0, 0.5,  2.0, "mmol/L"),
            ("Glucose",    110.0,30.0,70.0,100.0, "mg/dL"),
            ("Troponin",    0.05,0.05,0.0,  0.04, "ng/mL"),
        ]
        records = []
        base_time = datetime.utcnow() - timedelta(hours=48)
        for p in range(n_patients):
            pid = f"SYNTH-P{p:04d}"
            for h in [0, 6, 12, 24, 36, 48]:
                for name, mean, std, lo, hi, uom in labs:
                    val = max(0, np.random.normal(mean, std))
                    flag = (
                        "critical_low"  if val < lo * 0.5 else
                        "critical_high" if val > hi * 2.0 else
                        "low"           if val < lo        else
                        "high"          if val > hi        else
                        "normal"
                    )
                    records.append({
                        "patient_id": pid,
                        "charttime":  base_time + timedelta(hours=h),
                        "lab_name":   name,
                        "value":      round(val, 3),
                        "value_uom":  uom,
                        "ref_range_lower": lo,
                        "ref_range_upper": hi,
                        "flag":       flag,
                    })
        return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────
class DataLoadPipeline:
    """
    Orchestrates extraction → validation → storage.
    """

    def __init__(self, use_synthetic: bool = False):
        self.use_synthetic = use_synthetic
        self.db_extractor  = DatabaseExtractor()
        self.csv_extractor = CSVExtractor()
        self.synth         = SyntheticDataGenerator()
        self.run_id        = hashlib.md5(
            str(datetime.utcnow()).encode()
        ).hexdigest()[:8]

    def run(self) -> Dict[str, pd.DataFrame]:
        """Full pipeline run. Returns dict of clean DataFrames."""
        logger.info("Pipeline run %s starting", self.run_id)
        results = {}

        if self.use_synthetic:
            logger.info("Using synthetic data generator")
            vitals_raw = self.synth.generate_vitals(n_patients=20, hours=72)
            labs_raw   = self.synth.generate_labs(n_patients=20)
        else:
            try:
                vitals_raw = self.db_extractor.extract_vitals(hours_back=72)
                labs_raw   = self.db_extractor.extract_labs(hours_back=72)
            except Exception as exc:
                logger.warning("DB extraction failed (%s), falling back to synthetic", exc)
                vitals_raw = self.synth.generate_vitals(n_patients=20, hours=72)
                labs_raw   = self.synth.generate_labs(n_patients=20)

        # Validate
        vitals_clean, vitals_rejected = validate_vitals(vitals_raw)
        labs_clean                    = validate_labs(labs_raw)

        # Save
        vitals_clean.to_csv(CLEAN_DIR / "vitals_clean.csv", index=False)
        labs_clean.to_csv(CLEAN_DIR / "labs_clean.csv", index=False)
        if not vitals_rejected.empty:
            vitals_rejected.to_csv(CLEAN_DIR / "vitals_rejected.csv", index=False)

        results["vitals"]    = vitals_clean
        results["labs"]      = labs_clean
        results["rejected"]  = vitals_rejected

        logger.info(
            "Pipeline complete – vitals: %d, labs: %d, rejected: %d",
            len(vitals_clean), len(labs_clean), len(vitals_rejected),
        )
        return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = DataLoadPipeline(use_synthetic=True)
    data = pipeline.run()
    for name, df in data.items():
        print(f"\n── {name}: {df.shape} ──")
        print(df.head(3).to_string())

-- ============================================================
-- AI HOSPITAL OPERATING SYSTEM - DATABASE SCHEMA
-- PostgreSQL 14+
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ------------------------------------------------------------
-- PATIENTS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patients (
    patient_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mrn                  VARCHAR(20) UNIQUE NOT NULL,
    first_name           VARCHAR(100) NOT NULL,
    last_name            VARCHAR(100) NOT NULL,
    date_of_birth        DATE NOT NULL,
    gender               VARCHAR(10) CHECK (gender IN ('M','F','Other')),
    race                 VARCHAR(50),
    ethnicity            VARCHAR(50),
    address              TEXT,
    phone                VARCHAR(20),
    emergency_contact    TEXT,
    insurance_id         VARCHAR(50),
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- ENCOUNTERS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS encounters (
    encounter_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_type       VARCHAR(50) CHECK (encounter_type IN
                           ('inpatient','outpatient','emergency','icu','observation')),
    admit_time           TIMESTAMPTZ NOT NULL,
    discharge_time       TIMESTAMPTZ,
    chief_complaint      TEXT,
    attending_md         VARCHAR(100),
    facility_id          VARCHAR(20) DEFAULT 'MAIN',
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- ADMISSIONS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS admissions (
    admission_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id         UUID REFERENCES encounters(encounter_id),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    admittime            TIMESTAMPTZ NOT NULL,
    dischtime            TIMESTAMPTZ,
    admission_type       VARCHAR(50),
    admission_location   VARCHAR(100),
    discharge_location   VARCHAR(100),
    insurance            VARCHAR(50),
    marital_status       VARCHAR(30),
    hospital_expire_flag BOOLEAN DEFAULT FALSE,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- ICU STAYS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS icustays (
    icustay_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admission_id         UUID REFERENCES admissions(admission_id),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    icu_unit             VARCHAR(50) CHECK (icu_unit IN
                           ('MICU','SICU','CSRU','CCU','TSICU','NICU')),
    intime               TIMESTAMPTZ NOT NULL,
    outtime              TIMESTAMPTZ,
    los                  FLOAT,
    first_careunit       VARCHAR(50),
    last_careunit        VARCHAR(50),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- VITAL SIGNS  (physiological ranges embedded as comments)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vital_signs (
    vital_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    icustay_id           UUID REFERENCES icustays(icustay_id),
    charttime            TIMESTAMPTZ NOT NULL,
    heart_rate           FLOAT,   -- Normal 60-100 bpm
    sbp                  FLOAT,   -- Systolic  90-140 mmHg
    dbp                  FLOAT,   -- Diastolic 60-90  mmHg
    map                  FLOAT,   -- Mean art. 70-100 mmHg
    temperature          FLOAT,   -- Celsius   36.5-37.5
    spo2                 FLOAT,   -- %         95-100
    resp_rate            FLOAT,   -- breaths/m 12-20
    gcs_total            INT,     -- GCS       3-15
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_vital_patient_time ON vital_signs(patient_id, charttime DESC);
CREATE INDEX idx_vital_icustay_time ON vital_signs(icustay_id,  charttime DESC);

-- ------------------------------------------------------------
-- LAB RESULTS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lab_results (
    lab_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    icustay_id           UUID REFERENCES icustays(icustay_id),
    charttime            TIMESTAMPTZ NOT NULL,
    lab_name             VARCHAR(100) NOT NULL,
    value                FLOAT,
    value_uom            VARCHAR(30),
    ref_range_lower      FLOAT,
    ref_range_upper      FLOAT,
    flag                 VARCHAR(20) CHECK (flag IN
                           ('normal','low','high','critical_low','critical_high')),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_lab_patient_time ON lab_results(patient_id, charttime DESC);

-- ------------------------------------------------------------
-- MEDICATIONS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS medications (
    med_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id         UUID REFERENCES encounters(encounter_id),
    drug_name            VARCHAR(200) NOT NULL,
    drug_class           VARCHAR(100),
    dose_val             FLOAT,
    dose_uom             VARCHAR(30),
    route                VARCHAR(50),
    starttime            TIMESTAMPTZ,
    endtime              TIMESTAMPTZ,
    prescriber           VARCHAR(100),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- PROCEDURES
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS procedures (
    proc_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id         UUID REFERENCES encounters(encounter_id),
    proc_time            TIMESTAMPTZ NOT NULL,
    proc_code            VARCHAR(20),
    proc_name            VARCHAR(200),
    proc_type            VARCHAR(50),
    icd_version          INT DEFAULT 10,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- DIAGNOSES
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnoses (
    diag_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id         UUID REFERENCES encounters(encounter_id),
    icd_code             VARCHAR(20) NOT NULL,
    icd_version          INT DEFAULT 10,
    description          TEXT,
    diag_priority        INT DEFAULT 1,
    diagnosis_time       TIMESTAMPTZ DEFAULT NOW(),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- CLINICAL NOTES
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id         UUID REFERENCES encounters(encounter_id),
    charttime            TIMESTAMPTZ NOT NULL,
    category             VARCHAR(100),
    description          TEXT,
    text                 TEXT NOT NULL,
    cgid                 VARCHAR(50),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_note_patient_time ON clinical_notes(patient_id, charttime DESC);
CREATE INDEX idx_note_text_gin     ON clinical_notes USING gin(to_tsvector('english', text));

-- ------------------------------------------------------------
-- AI ALERTS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_alerts (
    alert_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id),
    icustay_id           UUID REFERENCES icustays(icustay_id),
    alert_time           TIMESTAMPTZ DEFAULT NOW(),
    alert_type           VARCHAR(100),
    severity             VARCHAR(20) CHECK (severity IN ('info','warning','critical')),
    risk_score           FLOAT,
    model_name           VARCHAR(100),
    description          TEXT,
    acknowledged         BOOLEAN DEFAULT FALSE,
    acknowledged_by      VARCHAR(100),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- ML PREDICTIONS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_predictions (
    pred_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id           UUID REFERENCES patients(patient_id),
    icustay_id           UUID REFERENCES icustays(icustay_id),
    pred_time            TIMESTAMPTZ DEFAULT NOW(),
    model_name           VARCHAR(100),
    model_version        VARCHAR(50),
    prediction_type      VARCHAR(100),
    score                FLOAT,
    threshold            FLOAT,
    is_positive          BOOLEAN,
    feature_json         JSONB,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- EXPERIMENTS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    exp_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exp_name             VARCHAR(200),
    model_type           VARCHAR(100),
    dataset_version      VARCHAR(50),
    params_json          JSONB,
    metrics_json         JSONB,
    artifact_path        TEXT,
    status               VARCHAR(30) DEFAULT 'running',
    started_at           TIMESTAMPTZ DEFAULT NOW(),
    finished_at          TIMESTAMPTZ
);

-- ============================================================
-- VIEWS
-- ============================================================

CREATE OR REPLACE VIEW patient_summary AS
SELECT
    p.patient_id,
    p.mrn,
    p.first_name || ' ' || p.last_name               AS full_name,
    DATE_PART('year', AGE(p.date_of_birth))::INT      AS age,
    p.gender,
    COUNT(DISTINCT e.encounter_id)                    AS total_encounters,
    MAX(e.admit_time)                                 AS last_admit,
    COUNT(DISTINCT i.icustay_id)                      AS icu_stays,
    COUNT(DISTINCT d.diag_id)                         AS total_diagnoses
FROM patients p
LEFT JOIN encounters   e ON p.patient_id = e.patient_id
LEFT JOIN icustays     i ON p.patient_id = i.patient_id
LEFT JOIN diagnoses    d ON p.patient_id = d.patient_id
GROUP BY p.patient_id, p.mrn, p.first_name, p.last_name,
         p.date_of_birth, p.gender;

CREATE OR REPLACE VIEW icu_vitals_latest AS
SELECT DISTINCT ON (v.patient_id)
    v.patient_id, v.icustay_id, v.charttime,
    v.heart_rate, v.sbp, v.dbp, v.map,
    v.temperature, v.spo2, v.resp_rate, v.gcs_total
FROM vital_signs v
ORDER BY v.patient_id, v.charttime DESC;

CREATE OR REPLACE VIEW active_alerts AS
SELECT
    a.alert_id, a.patient_id, a.alert_time, a.alert_type,
    a.severity, a.risk_score, a.model_name, a.description,
    p.mrn, p.first_name || ' ' || p.last_name AS full_name
FROM ai_alerts a
JOIN patients  p ON a.patient_id = p.patient_id
WHERE a.acknowledged = FALSE
ORDER BY a.alert_time DESC;

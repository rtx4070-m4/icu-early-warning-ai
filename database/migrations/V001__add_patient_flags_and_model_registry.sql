-- Migration: V001 Add patient flags and model metadata tables
-- Created: 2024-01-01T00:00:00
-- Apply: New tables for production feature flags and model versioning

-- ── Patient flags (per-encounter clinical state flags) ────────────────────────
CREATE TABLE IF NOT EXISTS patient_flags (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id    UUID REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    flag_name       VARCHAR(64) NOT NULL,
    flag_value      BOOLEAN NOT NULL DEFAULT FALSE,
    set_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    set_by          VARCHAR(128),
    notes           TEXT,
    UNIQUE (patient_id, flag_name)
);

CREATE INDEX IF NOT EXISTS idx_patient_flags_patient_id ON patient_flags(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_flags_name ON patient_flags(flag_name);

-- Common flags: mechanically_ventilated, vasopressors_running, fluid_resuscitated,
--               dnr_status, isolation_required, dialysis_dependent

-- ── Model metadata registry ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(128) NOT NULL,
    task            VARCHAR(64) NOT NULL,
    version         VARCHAR(32) NOT NULL,
    stage           VARCHAR(32) NOT NULL DEFAULT 'staging',  -- staging | production | archived
    artifact_path   TEXT,
    metrics         JSONB,
    parameters      JSONB,
    tags            JSONB,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    promoted_at     TIMESTAMP WITH TIME ZONE,
    promoted_by     VARCHAR(128),
    retired_at      TIMESTAMP WITH TIME ZONE,
    UNIQUE (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_name_stage ON model_registry(model_name, stage);

-- ── Drift monitoring log ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(128) NOT NULL,
    task            VARCHAR(64) NOT NULL,
    check_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    current_window  DATE NOT NULL,
    n_reference     INTEGER,
    n_current       INTEGER,
    drift_severity  VARCHAR(32) NOT NULL DEFAULT 'none',
    overall_drift   BOOLEAN NOT NULL DEFAULT FALSE,
    retrain_needed  BOOLEAN NOT NULL DEFAULT FALSE,
    drift_results   JSONB,
    action_taken    TEXT
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_model ON drift_reports(model_name, task);
CREATE INDEX IF NOT EXISTS idx_drift_reports_ts ON drift_reports(check_timestamp DESC);

-- ── Add missing columns to existing tables ────────────────────────────────────
ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS language_preference VARCHAR(32) DEFAULT 'English',
    ADD COLUMN IF NOT EXISTS mrn                 VARCHAR(32);

ALTER TABLE clinical_notes
    ADD COLUMN IF NOT EXISTS ner_entities        JSONB,
    ADD COLUMN IF NOT EXISTS ner_processed_at    TIMESTAMP WITH TIME ZONE;

ALTER TABLE ai_alerts
    ADD COLUMN IF NOT EXISTS cdss_recommendations JSONB,
    ADD COLUMN IF NOT EXISTS notification_sent     BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS notification_sent_at  TIMESTAMP WITH TIME ZONE;

-- ── View: model performance summary ──────────────────────────────────────────
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    model_name,
    task,
    version,
    stage,
    (metrics->>'auc_roc')::FLOAT   AS auc_roc,
    (metrics->>'f1')::FLOAT        AS f1,
    (metrics->>'recall')::FLOAT    AS recall,
    (metrics->>'brier_score')::FLOAT AS brier_score,
    created_at,
    promoted_at
FROM model_registry
WHERE stage IN ('production', 'staging')
ORDER BY model_name, task, created_at DESC;

COMMENT ON TABLE patient_flags IS 'Per-patient clinical state flags (e.g., ventilated, vasopressors)';
COMMENT ON TABLE model_registry IS 'ML model versioning and promotion tracking';
COMMENT ON TABLE drift_reports IS 'Model drift monitoring log';

-- Rollback: V001 Add patient flags and model metadata tables
-- Undo all changes from the V001 up migration

DROP VIEW  IF EXISTS model_performance_summary;
DROP TABLE IF EXISTS drift_reports;
DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS patient_flags;

ALTER TABLE patients
    DROP COLUMN IF EXISTS language_preference,
    DROP COLUMN IF EXISTS mrn;

ALTER TABLE clinical_notes
    DROP COLUMN IF EXISTS ner_entities,
    DROP COLUMN IF EXISTS ner_processed_at;

ALTER TABLE ai_alerts
    DROP COLUMN IF EXISTS cdss_recommendations,
    DROP COLUMN IF EXISTS notification_sent,
    DROP COLUMN IF EXISTS notification_sent_at;

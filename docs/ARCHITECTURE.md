# AI Hospital OS — Architecture

## Overview

The AI Hospital Operating System is a layered, modular clinical decision support platform.
Each layer has clear responsibilities and communicates through well-defined interfaces.

```
┌──────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                         │
│   Dash Dashboard  │  REST API  │  CLI  │  WebSocket/SSE Server   │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    CLINICAL INTELLIGENCE                          │
│   CDSS Engine  │  Medication Safety  │  Lab Ordering  │  FHIR    │
│   Explainability  │  Patient Similarity  │  Workflow FSM          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                        ML LAYER                                   │
│   Risk Prediction  │  Anomaly Detection  │  LSTM  │  Autoencoder  │
│   Patient Similarity Engine  │  Model Evaluator  │  Explainer     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                       NLP LAYER                                   │
│         Medical NER  │  Clinical Text Processing                  │
│         Knowledge Graph (Disease↔Symptom↔Drug↔Procedure)         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                      DATA LAYER                                   │
│   EHR System  │  Feature Engineering  │  Preprocessing            │
│   Data Quality Monitor  │  SAS Pipeline  │  Simulation Engine     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                            │
│   PostgreSQL  │  MLflow  │  Redis  │  Airflow  │  Docker          │
│   Prometheus  │  Audit Log  │  Config Manager  │  Health Monitor  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Map

```
demo/system_demo.py
  ├── config/config_manager.py
  ├── security/audit_security.py
  ├── data_quality/dq_monitor.py
  ├── nlp_pipeline/medical_ner.py
  ├── nlp_pipeline/clinical_text_processing.py
  ├── knowledge_graph/graph_builder.py
  ├── clinical_decision_support/cdss_engine.py
  │     ├── knowledge_graph/graph_builder.py
  │     └── nlp_pipeline/medical_ner.py
  ├── clinical_decision_support/medication_safety.py
  ├── explainability/model_explainer.py
  ├── real_time_monitoring/vitals_stream_processor.py
  ├── lab_predictor/lab_ordering.py
  ├── patient_similarity/similarity_engine.py
  ├── workflow/clinical_workflow.py
  ├── simulation/patient_simulator.py
  ├── fhir_integration/fhir_client.py
  ├── reporting/report_generator.py
  ├── evaluation/model_evaluator.py
  └── system_health/health_monitor.py
```

---

## Data Flow

### Real-Time Monitoring Pipeline
```
Monitoring Device
      │
      ▼
VitalsStreamProcessor.ingest(reading)
      │
      ├── compute_news2(reading)
      ├── compute_shock_index(reading)
      ├── RuleBasedAlertEngine.evaluate(state, reading)
      │       ├── Critical threshold check
      │       ├── NEWS2 threshold check  
      │       ├── qSOFA sepsis screen
      │       └── Trend-based deterioration
      │
      └── [If alert] → EventBus.publish_alert()
                            │
                            ├── WebSocket server → Browser
                            ├── CDSS Engine evaluation
                            └── Workflow FSM transition
```

### ML Pipeline
```
Raw Data (DB / CSV / Synthetic)
      │
      ▼ DataLoadPipeline
Validated Records
      │
      ▼ PreprocessingPipeline (LOCF, normalisation, windows)
Clean Features
      │
      ▼ FeatureEngineeringPipeline (SOFA, NEWS2, delta, rolling)
Enriched Features
      │
      ├──▶ ClinicalRiskModel.fit()  [sepsis/mortality/cardiac]
      ├──▶ EnsembleAnomalyDetector.fit()
      ├──▶ LSTMTrainer.fit()
      └──▶ AutoencoderTrainer.fit()
              │
              ▼
         MLflow Experiment Tracking
              │
              ▼ EvaluationSuite (AUC/ECE/F1/backtest)
              │
              └──▶ [Pass] → MLflow Model Registry (Staging → Production)
```

### Clinical Decision Support Flow
```
Patient Data (vitals + labs + risk_scores + note_text)
      │
      ▼ ClinicalDecisionSupportEngine.evaluate()
      │
      ├── MedicalNER.full_extraction(note_text)      → entities
      ├── FeatureContext computation (MAP, qSOFA, SIRS)
      ├── Protocol evaluation (7 protocols)
      ├── Critical lab threshold checks
      ├── Vital sign alert rules
      ├── KnowledgeGraph.differential_diagnosis()
      └── MedicationSafetyChecker.check()
              │
              ▼
         CDSSOutput {
           risk_level, recommendations (STAT/URGENT/ROUTINE),
           triggered_protocols, differential_diagnosis,
           drug_safety, reasoning_chain, scores
         }
```

---

## Key Design Decisions

### 1. Graceful Degradation
Every module has a fallback path when optional dependencies are unavailable:
- PyTorch not installed → numpy/sklearn autoencoder & LSTM fallback
- PostgreSQL unavailable → synthetic data generation
- MLflow unavailable → local JSON file store
- FastAPI unavailable → stdlib `http.server`
- websockets unavailable → SSE over HTTP
- FHIR server unavailable → local JSON bundle

### 2. Stdlib-First
Core clinical logic (NER, KG, CDSS, alerts, scoring) runs with only:
`numpy`, `pandas`, `scikit-learn`, `networkx`

No heavy dependencies required for the critical path.

### 3. Thread Safety
- `VitalsStreamProcessor`: queue-based, single worker thread
- `EventBus`: `threading.Lock` on all publish/subscribe
- `AuditLogger`: `threading.Lock` on buffer
- `RateLimiter`: `threading.Lock` on token buckets
- `SessionManager`: `threading.Lock` on session map

### 4. Test Architecture
```
tests/test_models.py          ← unit tests (44, no external services)
integration_tests/            ← cross-module end-to-end (24)
benchmarks/system_benchmarks  ← latency/throughput regression
demo/system_demo.py           ← full system smoke test (18 modules)
```

### 5. Clinical Safety
All AI outputs are:
- Clearly labelled as decision support, not diagnosis
- Accompanied by confidence intervals and explanations
- Subject to audit logging (who accessed what, when)
- Protected by RBAC (role-based access control)
- De-identified before export (PHI protection)

---

## Security Architecture

```
Request
  │
  ▼ RateLimiter (120 req/min per IP)
  │
  ▼ JWT Verification (HS256, 8h expiry)
  │
  ▼ RBAC Check (5 roles: attending, resident, nurse, data_scientist, auditor)
  │
  ▼ Business Logic
  │
  ▼ AuditLogger (every access, modification, export)
  │
  └── PHIDeidentifier (on data export)
```

**Roles and key permissions:**

| Role | View Patients | Prescribe | Run Pipeline | View Audit Log |
|------|:---:|:---:|:---:|:---:|
| `attending_physician` | ✓ | ✓ | ✗ | ✗ |
| `resident` | ✓ | ✗ | ✗ | ✗ |
| `nurse` | ✓ | ✗ | ✗ | ✗ |
| `data_scientist` | ✓ | ✗ | ✓ | ✗ |
| `admin` | ✓ | ✓ | ✓ | ✓ |

---

## Database Schema Summary

```sql
patients          -- Demographics, admission info
encounters        -- ED/ward/ICU encounters
admissions        -- ICU admissions (bed, unit)
icustays          -- ICU stay periods
vital_signs       -- Hourly vital readings (linked to icustay)
lab_results       -- Lab values (linked to encounter)
medications       -- Medication administrations
procedures        -- Procedures (intubation, lines, etc.)
diagnoses         -- ICD-10 coded diagnoses
clinical_notes    -- Free-text notes
ai_alerts         -- System-generated alerts
ml_predictions    -- Stored ML predictions + SHAP values
experiments       -- ML experiment runs
_migrations       -- Applied DB migrations (version tracking)
```

Views: `patient_summary`, `icu_vitals_latest`, `active_alerts`

---

## Deployment Options

### Option 1: Local Development
```bash
pip install -r requirements.txt
python demo/system_demo.py
```

### Option 2: Docker Compose (Full Stack)
```bash
cd docker && DB_PASSWORD=secret docker compose up -d
# Dashboard: :8050 | MLflow: :5000 | Airflow: :8080 | API: :8000
```

### Option 3: Kubernetes
Deploy each service as a separate Deployment:
- `hospital-api` (2+ replicas, HPA)
- `hospital-dashboard` (1 replica)
- `hospital-worker` (Celery workers for pipeline)
- `postgres` (StatefulSet + PVC)
- `redis` (StatefulSet)
- `mlflow` (Deployment + PVC for artifacts)

### Option 4: pip install
```bash
pip install -e .             # Development
pip install -e ".[all]"      # All optional deps
hospital-cli health          # CLI entry point
```

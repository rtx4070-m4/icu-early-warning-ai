# AI Hospital Operating System

A production-grade, AI-powered clinical decision support platform for ICU monitoring,
early deterioration detection, risk stratification, NLP note analysis, and interoperability.

> ⚠️ **Clinical Disclaimer** — Research prototype only. Not validated for clinical use.

---

## Quick Start

```bash
pip install -r requirements.txt
python demo/system_demo.py          # 18-module end-to-end demo
./run_system.sh dashboard           # ICU dashboard → http://localhost:8050
./run_system.sh api                 # REST API    → http://localhost:8000/docs
cd docker && docker compose up -d   # Full stack
```

### CLI

```bash
python cli/hospital_cli.py icu
python cli/hospital_cli.py patient P001
python cli/hospital_cli.py ddx "fever,hypotension,tachycardia"
python cli/hospital_cli.py medsafe "vancomycin,amiodarone" --egfr=28
python cli/hospital_cli.py simulate septic_shock_treated
python cli/hospital_cli.py health
```

---

## 29 Modules

| Directory | Module | Description |
|-----------|--------|-------------|
| `database/` | Schema + seed | PostgreSQL: 13 tables, 3 views, 20 synthetic ICU patients |
| `ehr_system/` | Patient management | EHR service layer: CRUD, vitals, labs, notes |
| `sas_pipeline/` | SAS validation | Legacy SAS: physiological validation, SOFA calc |
| `data_engineering/` | ETL pipeline | Load → preprocess → feature engineering (SOFA, NEWS2, delta) |
| `ml_models/` | ML suite | Risk prediction, anomaly detection, LSTM, autoencoder |
| `nlp_pipeline/` | Clinical NLP | Medical NER (300+ lexicon), section segmentation, TF-IDF |
| `knowledge_graph/` | KG | Disease↔symptom↔drug graph, DDx scoring, drug interactions |
| `real_time_monitoring/` | Streaming | Thread-safe vitals stream, NEWS2/qSOFA/shock-index alerts |
| `websocket_server/` | WS/SSE server | Real-time push to browser; WebSocket + SSE fallback |
| `clinical_decision_support/` | CDSS + Med safety | 7 protocols, medication formulary, contraindications |
| `lab_predictor/` | Lab ordering | 21-test predictive lab ordering (LOINC-coded) |
| `patient_similarity/` | Cohort search | Vectorised similarity, outcome prediction from cohort |
| `workflow/` | State machine | Care pathway FSM, protocol adherence, audit log |
| `explainability/` | SHAP-style XAI | Kernel SHAP, counterfactuals, NL explanations |
| `evaluation/` | Model eval | AUC/ECE/F1, calibration, temporal backtest, HTML report |
| `dashboard/` | Dash dashboard | Plotly Dash ICU command centre (+ HTML fallback) |
| `reporting/` | Reports | HTML deterioration reports, note drafts, handover summary |
| `fhir_integration/` | FHIR R4 | Patient/Observation/Condition/RiskAssessment bundles |
| `simulation/` | Patient sim | 7 ICU deterioration trajectories with physiological noise |
| `benchmarks/` | Benchmarks | p50/p95/p99 latency + throughput for all subsystems |
| `security/` | HIPAA security | Audit trail, RBAC (5 roles), PHI de-identification |
| `config/` | Config mgmt | YAML + env-var config with validation |
| `system_health/` | Health monitor | 9-component checks, K8s readiness/liveness probes |
| `experiment_tracking/` | MLflow | Experiment logging, model registry (+ file fallback) |
| `orchestration/` | Airflow DAG | Daily pipeline + standalone runner |
| `api/` | REST API | FastAPI (+ stdlib fallback): 14 endpoints |
| `cli/` | CLI | 14-command terminal interface |
| `tests/` | Unit tests | 70+ pytest cases |
| `integration_tests/` | Integration | 24 end-to-end tests across all modules |
| `demo/` | System demo | Full 18-module ICU scenario demonstration |

---

## Make Targets

```bash
make install          # install runtime deps
make test             # unit tests
make test-integration # integration tests
make test-all         # all tests
make run-dashboard    # :8050
make run-api          # :8000
make eval             # model evaluation
make docker-up        # full stack
make clean
```

---

## ML Models

| Model | Task | Target AUC |
|-------|------|-----------|
| XGBoost/LightGBM | Sepsis risk | ≥0.80 |
| XGBoost/LightGBM | Mortality risk | ≥0.78 |
| Ensemble (IF+Statistical) | Vital anomalies | ≥0.72 |
| Bi-LSTM + Attention | Deterioration 6h ahead | ≥0.75 |

---

## CDSS Protocols

Sepsis Bundle · Vasopressor Protocol · Lung-Protective Ventilation (ARDS) ·
AKI Management · DVT Prophylaxis · Glycemic Management · Stress Ulcer Prophylaxis

---

## Simulation Scenarios

`septic_shock_treated` · `septic_shock_untreated` · `respiratory_failure` ·
`cardiogenic_shock` · `aki_progression` · `post_op_recovery` · `stable_icu`

---

## Environment Variables

Copy `.env.example` → `.env` and fill in values.

| Key | Default |
|-----|---------|
| `DB_HOST` | `localhost` |
| `DB_PASSWORD` | *(required)* |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns.db` |
| `API_PORT` | `8000` |
| `DASHBOARD_PORT` | `8050` |
| `LOG_LEVEL` | `INFO` |
| `HOSPITAL_OS__<SECTION>__<KEY>` | Override any config |

---

## Docker Compose

```bash
cd docker
DB_PASSWORD=secret docker compose up -d
# Dashboard: http://localhost:8050
# MLflow:    http://localhost:5000
# Airflow:   http://localhost:8080
```

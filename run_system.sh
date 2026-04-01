#!/usr/bin/env bash
# AI Hospital OS — System Startup Script
# Usage: ./run_system.sh [dashboard|pipeline|monitor|test|docker]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Data directories
mkdir -p /data/{raw,clean,features,models,reports,mlflow/artifacts,eval}
mkdir -p /data/models/{risk,anomaly,lstm,autoencoder}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[hospital_os]${NC} $1"; }
warn()  { echo -e "${YELLOW}[hospital_os]${NC} $1"; }
error() { echo -e "${RED}[hospital_os]${NC} $1" >&2; }

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_dashboard() {
  log "Starting ICU Dashboard on http://localhost:8050 ..."
  python "$PROJECT_ROOT/dashboard/hospital_dashboard.py"
}

cmd_pipeline() {
  log "Running full ML pipeline (standalone, no Airflow required) ..."
  python "$PROJECT_ROOT/orchestration/airflow_dag.py"
}

cmd_monitor() {
  log "Starting real-time vitals stream processor (demo mode) ..."
  python "$PROJECT_ROOT/real_time_monitoring/vitals_stream_processor.py"
}

cmd_ner_demo() {
  log "Running medical NER demo ..."
  python "$PROJECT_ROOT/nlp_pipeline/medical_ner.py"
}

cmd_kg_demo() {
  log "Running knowledge graph demo ..."
  python "$PROJECT_ROOT/knowledge_graph/graph_builder.py"
}

cmd_test() {
  log "Running test suite ..."
  python -m pytest "$PROJECT_ROOT/tests/test_models.py" -v --tb=short 2>&1 | tee /data/reports/test_results.txt
  log "Test results saved to /data/reports/test_results.txt"
}

cmd_docker() {
  log "Starting full Docker Compose stack ..."
  cd "$PROJECT_ROOT/docker"
  docker compose up --build -d
  log "Services starting. Dashboard: http://localhost:8050 | MLflow: http://localhost:5000 | Airflow: http://localhost:8080"
}

cmd_docker_down() {
  log "Stopping Docker Compose stack ..."
  cd "$PROJECT_ROOT/docker"
  docker compose down
}

cmd_db_init() {
  log "Initializing PostgreSQL schema and seed data ..."
  psql "$DATABASE_URL" -f "$PROJECT_ROOT/database/schema.sql"
  psql "$DATABASE_URL" -f "$PROJECT_ROOT/database/seed_data.sql"
  log "Database initialized."
}

cmd_report() {
  log "Generating static ICU HTML report ..."
  python -c "
from dashboard.hospital_dashboard import generate_static_report
path = generate_static_report('/data/reports/icu_latest.html')
print(f'Report: {path}')
"
}

cmd_api() {
  log "Starting REST API server on http://localhost:${API_PORT:-8000} ..."
  python "$PROJECT_ROOT/api/server.py"
}

cmd_cdss() {
  log "Running Clinical Decision Support Engine demo ..."
  python "$PROJECT_ROOT/clinical_decision_support/cdss_engine.py"
}

cmd_reports() {
  log "Generating all patient reports (HTML + Markdown) ..."
  python "$PROJECT_ROOT/reporting/report_generator.py"
}

cmd_security_demo() {
  log "Running security/audit logging demo ..."
  python "$PROJECT_ROOT/security/audit_security.py"
}

cmd_config() {
  log "Showing resolved configuration ..."
  python "$PROJECT_ROOT/config/config_manager.py"
}

cmd_help() {
  echo ""
  echo "AI Hospital OS — Command Reference"
  echo "==================================="
  echo ""
  echo "  ./run_system.sh dashboard    Start Dash ICU dashboard (port 8050)"
  echo "  ./run_system.sh api          Start REST API server (port 8000)"
  echo "  ./run_system.sh pipeline     Run full ML pipeline (no Airflow needed)"
  echo "  ./run_system.sh monitor      Real-time vitals stream demo"
  echo "  ./run_system.sh cdss         Clinical Decision Support Engine demo"
  echo "  ./run_system.sh reports      Generate all patient reports"
  echo "  ./run_system.sh ner          Medical NER demo"
  echo "  ./run_system.sh kg           Knowledge graph demo"
  echo "  ./run_system.sh security     Security & audit logging demo"
  echo "  ./run_system.sh config       Show resolved system configuration"
  echo "  ./run_system.sh test         Run pytest test suite"
  echo "  ./run_system.sh report       Generate static HTML ICU report"
  echo "  ./run_system.sh docker       Start full Docker Compose stack"
  echo "  ./run_system.sh docker-down  Stop Docker Compose stack"
  echo "  ./run_system.sh db-init      Initialize PostgreSQL schema"
  echo ""
  echo "Environment variables:"
  echo "  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
  echo "  MLFLOW_TRACKING_URI (default: sqlite:///mlruns.db)"
  echo "  API_PORT (default: 8000)"
  echo "  HOSPITAL_OS__SECTION__KEY (override any config value)"
  echo ""
}

# ── Router ────────────────────────────────────────────────────────────────────

COMMAND="${1:-help}"

case "$COMMAND" in
  dashboard)    cmd_dashboard ;;
  api)          cmd_api ;;
  pipeline)     cmd_pipeline ;;
  monitor)      cmd_monitor ;;
  cdss)         cmd_cdss ;;
  reports)      cmd_reports ;;
  ner)          cmd_ner_demo ;;
  kg)           cmd_kg_demo ;;
  security)     cmd_security_demo ;;
  config)       cmd_config ;;
  test)         cmd_test ;;
  report)       cmd_report ;;
  docker)       cmd_docker ;;
  docker-down)  cmd_docker_down ;;
  db-init)      cmd_db_init ;;
  help|--help)  cmd_help ;;
  *)
    error "Unknown command: $COMMAND"
    cmd_help
    exit 1
    ;;
esac

# AI Hospital OS — Makefile
# Usage: make <target>

PYTHON   ?= python3
PIP      ?= pip3
PROJECT  = ai_hospital_os
SRC_DIR  = .
TEST_DIR = tests
INT_TEST = integration_tests
REPORTS  = /data/reports
MODELS   = /data/models

.PHONY: help install install-dev test test-integration test-all \
        lint format typecheck \
        run-dashboard run-api run-pipeline run-monitor run-cdss \
        reports eval quality db-init docker-up docker-down \
        clean clean-cache clean-models clean-all

# ─────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AI Hospital OS — Makefile Targets"
	@echo "=================================="
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install runtime dependencies"
	@echo "    make install-dev      Install dev + test dependencies"
	@echo ""
	@echo "  Testing:"
	@echo "    make test             Run unit test suite"
	@echo "    make test-integration Run integration test suite"
	@echo "    make test-all         Run all tests"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint             Run flake8 linter"
	@echo "    make format           Run black formatter"
	@echo "    make typecheck        Run mypy type checker"
	@echo ""
	@echo "  Services:"
	@echo "    make run-dashboard    Start ICU Dash dashboard (port 8050)"
	@echo "    make run-api          Start FastAPI REST server (port 8000)"
	@echo "    make run-pipeline     Run full ML pipeline"
	@echo "    make run-monitor      Start real-time vitals stream demo"
	@echo "    make run-cdss         Run CDSS engine demo"
	@echo ""
	@echo "  Reports & Eval:"
	@echo "    make reports          Generate all patient reports"
	@echo "    make eval             Run model evaluation suite"
	@echo "    make quality          Run data quality checks"
	@echo ""
	@echo "  Infrastructure:"
	@echo "    make db-init          Initialise PostgreSQL schema"
	@echo "    make docker-up        Start Docker Compose stack"
	@echo "    make docker-down      Stop Docker Compose stack"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove temp files"
	@echo "    make clean-all        Remove all generated artifacts"
	@echo ""

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install flake8 black mypy pytest-cov pytest-xdist

# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────
test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_models.py -v --tb=short \
		--cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html:/data/reports/coverage \
		2>&1 | tee /tmp/unit_test_results.txt
	@echo "Unit test results: /tmp/unit_test_results.txt"

test-integration:
	@echo "Running integration tests..."
	$(PYTHON) -m pytest $(INT_TEST)/test_integration.py -v --tb=short \
		2>&1 | tee /tmp/integration_test_results.txt
	@echo "Integration test results: /tmp/integration_test_results.txt"

test-all: test test-integration
	@echo "All tests complete."

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) $(INT_TEST) -x -q --tb=line

# ─────────────────────────────────────────────────────────────
# Code quality
# ─────────────────────────────────────────────────────────────
lint:
	@echo "Running flake8..."
	$(PYTHON) -m flake8 $(SRC_DIR) --max-line-length=120 \
		--exclude=__pycache__,.git,venv,*.pyc \
		--ignore=E501,W503,E203

format:
	@echo "Running black formatter..."
	$(PYTHON) -m black $(SRC_DIR) --line-length=100 \
		--exclude="/__pycache__|/\.git|/venv/"

typecheck:
	@echo "Running mypy..."
	$(PYTHON) -m mypy $(SRC_DIR) --ignore-missing-imports \
		--exclude "test_|__pycache__"

# ─────────────────────────────────────────────────────────────
# Services
# ─────────────────────────────────────────────────────────────
run-dashboard:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) dashboard/hospital_dashboard.py

run-api:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) api/server.py

run-pipeline:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) orchestration/airflow_dag.py

run-monitor:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) real_time_monitoring/vitals_stream_processor.py

run-cdss:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) clinical_decision_support/cdss_engine.py

run-ner:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) nlp_pipeline/medical_ner.py

run-kg:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) knowledge_graph/graph_builder.py

run-security:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) security/audit_security.py

run-fhir:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) fhir_integration/fhir_client.py

run-config:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) config/config_manager.py

# ─────────────────────────────────────────────────────────────
# Reports & Evaluation
# ─────────────────────────────────────────────────────────────
reports:
	@echo "Generating clinical reports..."
	@mkdir -p $(REPORTS)
	PYTHONPATH=$(SRC_DIR) $(PYTHON) reporting/report_generator.py

eval:
	@echo "Running model evaluation suite..."
	@mkdir -p $(REPORTS)
	PYTHONPATH=$(SRC_DIR) $(PYTHON) evaluation/model_evaluator.py

quality:
	@echo "Running data quality checks..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) data_quality/dq_monitor.py

# ─────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────
db-init:
	@echo "Initialising database schema..."
	psql "$(DATABASE_URL)" -f database/schema.sql
	psql "$(DATABASE_URL)" -f database/seed_data.sql
	@echo "Database initialised."

docker-up:
	cd docker && docker compose up --build -d
	@echo "Stack started. Dashboard: http://localhost:8050 | MLflow: http://localhost:5000"

docker-down:
	cd docker && docker compose down

docker-logs:
	cd docker && docker compose logs -f

docker-ps:
	cd docker && docker compose ps

# ─────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned temp files."

clean-cache: clean

clean-models:
	@echo "Removing trained models (data preserved)..."
	rm -rf $(MODELS)/risk $(MODELS)/anomaly $(MODELS)/lstm $(MODELS)/autoencoder
	@echo "Models cleared."

clean-all: clean
	rm -rf /data/clean /data/features /data/raw
	@echo "All generated data cleared."

# ─────────────────────────────────────────────────────────────
# Shortcuts
# ─────────────────────────────────────────────────────────────
up: docker-up
down: docker-down
t: test
ti: test-integration
ta: test-all
d: run-dashboard
api: run-api

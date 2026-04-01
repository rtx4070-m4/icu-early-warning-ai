FROM python:3.11-slim

LABEL maintainer="hospital-os@example.com"
LABEL description="AI Hospital Operating System"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . /app/hospital_os

# ── Data directories ──────────────────────────────────────────────────────────
RUN mkdir -p /data/raw /data/clean /data/features /data/models \
             /data/mlflow/artifacts /data/reports

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DB_HOST=postgres \
    DB_PORT=5432 \
    DB_NAME=hospital_os \
    DB_USER=hospital \
    DB_PASSWORD=change_me \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_ARTIFACT_ROOT=/data/mlflow/artifacts

# ── Default command: run dashboard ────────────────────────────────────────────
EXPOSE 8050
CMD ["python", "/app/hospital_os/dashboard/hospital_dashboard.py"]

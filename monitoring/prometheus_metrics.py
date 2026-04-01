"""
Prometheus Metrics Exporter
Exposes /metrics endpoint with clinical and operational counters,
gauges, and histograms for the AI Hospital OS.

Metrics exported:
  - hospital_os_patients_total              (gauge)
  - hospital_os_alerts_total               (counter by severity)
  - hospital_os_news2_score                (histogram by patient)
  - hospital_os_sepsis_risk_score          (histogram)
  - hospital_os_api_requests_total         (counter by endpoint/status)
  - hospital_os_api_latency_seconds        (histogram by endpoint)
  - hospital_os_model_predictions_total    (counter by model/task)
  - hospital_os_vitals_ingested_total      (counter)
  - hospital_os_icu_occupancy             (gauge)
  - hospital_os_data_quality_score        (gauge)
  - hospital_os_pipeline_last_run_time    (gauge - unix timestamp)

Falls back to a simple text endpoint when prometheus_client not installed.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# In-process metric store (always available)
# ─────────────────────────────────────────

class SimpleMetricStore:
    """Thread-safe in-memory metric store (fallback for prometheus_client)."""

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, labels: str = "", amount: float = 1.0):
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def set_gauge(self, name: str, value: float, labels: str = ""):
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: str = ""):
        key = f"{name}{{{labels}}}" if labels else name
        with self._lock:
            self._histograms.setdefault(key, []).append(value)
            # Keep only last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def text_output(self) -> str:
        """Generate Prometheus text format."""
        lines = [
            f"# Hospital OS Metrics — {datetime.utcnow().isoformat()}",
            "",
        ]
        with self._lock:
            for key, val in sorted(self._counters.items()):
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{key} {val:.6f}")
            for key, val in sorted(self._gauges.items()):
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{key} {val:.6f}")
            for key, vals in sorted(self._histograms.items()):
                if not vals:
                    continue
                name = key.split("{")[0]
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(vals)}")
                lines.append(f"{name}_sum {sum(vals):.6f}")
                sv = sorted(vals)
                for q, label in [(0.5,"0.5"),(0.9,"0.9"),(0.95,"0.95"),(0.99,"0.99")]:
                    idx = min(int(q * len(sv)), len(sv) - 1)
                    lines.append(f"{name}_quantile{{quantile=\"{label}\"}} {sv[idx]:.6f}")
        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────
# Metrics facade
# ─────────────────────────────────────────

_store = SimpleMetricStore()

# Try to use prometheus_client if available
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )

    REGISTRY = CollectorRegistry()

    # Clinical metrics
    PATIENTS_TOTAL = Gauge(
        "hospital_os_patients_total",
        "Current ICU patient census",
        registry=REGISTRY,
    )
    ALERTS_TOTAL = Counter(
        "hospital_os_alerts_total",
        "Total alerts fired",
        ["severity", "alert_type"],
        registry=REGISTRY,
    )
    NEWS2_SCORE = Histogram(
        "hospital_os_news2_score",
        "Distribution of NEWS2 scores",
        buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        registry=REGISTRY,
    )
    SEPSIS_RISK = Histogram(
        "hospital_os_sepsis_risk_score",
        "Distribution of sepsis risk predictions",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        registry=REGISTRY,
    )
    ICU_OCCUPANCY = Gauge(
        "hospital_os_icu_occupancy",
        "Current ICU bed occupancy (0–1)",
        registry=REGISTRY,
    )

    # API metrics
    API_REQUESTS = Counter(
        "hospital_os_api_requests_total",
        "API request count",
        ["endpoint", "method", "status_code"],
        registry=REGISTRY,
    )
    API_LATENCY = Histogram(
        "hospital_os_api_latency_seconds",
        "API request latency",
        ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        registry=REGISTRY,
    )

    # ML metrics
    MODEL_PREDICTIONS = Counter(
        "hospital_os_model_predictions_total",
        "ML model prediction count",
        ["model", "task"],
        registry=REGISTRY,
    )

    # Data pipeline metrics
    VITALS_INGESTED = Counter(
        "hospital_os_vitals_ingested_total",
        "Vital sign readings ingested",
        registry=REGISTRY,
    )
    DATA_QUALITY_SCORE = Gauge(
        "hospital_os_data_quality_score",
        "Latest data quality score (0–1)",
        registry=REGISTRY,
    )
    PIPELINE_LAST_RUN = Gauge(
        "hospital_os_pipeline_last_run_timestamp",
        "Unix timestamp of last pipeline run",
        registry=REGISTRY,
    )

    _PROMETHEUS_AVAILABLE = True
    logger.info("prometheus_client available — full metrics enabled")

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed — using simple metric store")


# ─────────────────────────────────────────
# Public API (works with or without prometheus_client)
# ─────────────────────────────────────────

def record_alert(severity: str, alert_type: str = "DETERIORATION"):
    """Record a fired alert."""
    _store.inc("hospital_os_alerts_total", f'severity="{severity}",type="{alert_type}"')
    if _PROMETHEUS_AVAILABLE:
        ALERTS_TOTAL.labels(severity=severity, alert_type=alert_type).inc()


def record_news2(score: int):
    """Record a NEWS2 observation."""
    _store.observe("hospital_os_news2_score", float(score))
    if _PROMETHEUS_AVAILABLE:
        NEWS2_SCORE.observe(score)


def record_sepsis_risk(score: float):
    """Record a sepsis risk prediction."""
    _store.observe("hospital_os_sepsis_risk_score", score)
    if _PROMETHEUS_AVAILABLE:
        SEPSIS_RISK.observe(score)


def set_patient_census(n: int):
    """Set current ICU patient count."""
    _store.set_gauge("hospital_os_patients_total", float(n))
    if _PROMETHEUS_AVAILABLE:
        PATIENTS_TOTAL.set(n)


def set_icu_occupancy(occupancy: float):
    """Set ICU occupancy fraction (0–1)."""
    _store.set_gauge("hospital_os_icu_occupancy", occupancy)
    if _PROMETHEUS_AVAILABLE:
        ICU_OCCUPANCY.set(occupancy)


def record_api_request(endpoint: str, method: str, status_code: int, latency_s: float):
    """Record an API request."""
    _store.inc("hospital_os_api_requests_total",
                f'endpoint="{endpoint}",method="{method}",status="{status_code}"')
    _store.observe("hospital_os_api_latency_seconds", latency_s, f'endpoint="{endpoint}"')
    if _PROMETHEUS_AVAILABLE:
        API_REQUESTS.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
        API_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def record_prediction(model: str, task: str):
    """Record an ML model prediction."""
    _store.inc("hospital_os_model_predictions_total", f'model="{model}",task="{task}"')
    if _PROMETHEUS_AVAILABLE:
        MODEL_PREDICTIONS.labels(model=model, task=task).inc()


def record_vitals_ingested(count: int = 1):
    """Record vital sign readings ingested."""
    _store.inc("hospital_os_vitals_ingested_total", amount=count)
    if _PROMETHEUS_AVAILABLE:
        VITALS_INGESTED.inc(count)


def set_data_quality_score(score: float):
    """Set latest data quality score."""
    _store.set_gauge("hospital_os_data_quality_score", score)
    if _PROMETHEUS_AVAILABLE:
        DATA_QUALITY_SCORE.set(score)


def record_pipeline_run():
    """Record that the ML pipeline ran."""
    ts = time.time()
    _store.set_gauge("hospital_os_pipeline_last_run_timestamp", ts)
    if _PROMETHEUS_AVAILABLE:
        PIPELINE_LAST_RUN.set(ts)


def get_metrics_text() -> str:
    """Return metrics in Prometheus text exposition format."""
    if _PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY).decode("utf-8")
    return _store.text_output()


def get_metrics_summary() -> Dict:
    """Return a dict summary of current metrics for JSON API."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "prometheus_available": _PROMETHEUS_AVAILABLE,
        "counters": dict(_store._counters),
        "gauges": dict(_store._gauges),
        "histogram_counts": {k: len(v) for k, v in _store._histograms.items()},
    }


# ─────────────────────────────────────────
# HTTP metrics endpoint (stdlib)
# ─────────────────────────────────────────

def start_metrics_server(port: int = 9090):
    """Start a simple HTTP server exposing /metrics."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MetricsHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def do_GET(self):
            if self.path == "/metrics":
                body = get_metrics_text().encode()
                ct = "text/plain; version=0.0.4; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/metrics/json":
                body = __import__("json").dumps(get_metrics_summary(), indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Metrics server started on http://0.0.0.0:{port}/metrics")
    return server


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO)

    print("=== Hospital OS Metrics Demo ===")
    print(f"prometheus_client available: {_PROMETHEUS_AVAILABLE}")

    # Simulate some metrics
    set_patient_census(8)
    set_icu_occupancy(0.75)
    set_data_quality_score(0.92)
    record_pipeline_run()

    for _ in range(20):
        news2 = random.randint(0, 12)
        record_news2(news2)
        record_sepsis_risk(random.random())
        if news2 >= 7:
            record_alert("critical", "DETERIORATION")
        elif news2 >= 5:
            record_alert("high", "DETERIORATION")
        record_vitals_ingested()
        record_api_request("/patients", "GET", 200, random.uniform(0.01, 0.2))
        record_prediction("SepsisRiskModel", "sepsis_risk")

    print("\nSample metrics output:")
    print(get_metrics_text()[:1200])

    print("\nJSON summary:")
    summary = get_metrics_summary()
    print(f"  Counters: {len(summary['counters'])}")
    print(f"  Gauges: {len(summary['gauges'])}")
    print(f"  Histograms: {len(summary['histogram_counts'])}")

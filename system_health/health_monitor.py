"""
System Health Monitor
Checks all Hospital OS subsystems and reports status.
Provides a /health REST endpoint, continuous background polling,
and structured health reports for ops dashboards.
"""

import os
import sys
import time
import json
import logging
import platform
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Health check result
# ─────────────────────────────────────────

class HealthStatus:
    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"
    UNKNOWN  = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: str         # ok | degraded | down | unknown
    latency_ms: float
    message: str
    details: Dict = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at,
        }


@dataclass
class SystemHealthReport:
    timestamp: str
    overall_status: str
    version: str
    uptime_seconds: float
    components: List[ComponentHealth] = field(default_factory=list)
    system_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "components": [c.to_dict() for c in self.components],
            "system_info": self.system_info,
        }

    @property
    def healthy(self) -> bool:
        return self.overall_status == HealthStatus.OK

    def summary(self) -> str:
        down = [c.name for c in self.components if c.status == HealthStatus.DOWN]
        degraded = [c.name for c in self.components if c.status == HealthStatus.DEGRADED]
        return (f"Status={self.overall_status.upper()} | "
                f"Down={down or 'none'} | Degraded={degraded or 'none'} | "
                f"Uptime={self.uptime_seconds:.0f}s")


# ─────────────────────────────────────────
# Individual health checks
# ─────────────────────────────────────────

def _timed_check(fn: Callable[[], ComponentHealth]) -> ComponentHealth:
    """Execute a health check and record latency."""
    t0 = time.perf_counter()
    try:
        result = fn()
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result
    except Exception as e:
        return ComponentHealth(
            name="unknown", status=HealthStatus.DOWN,
            latency_ms=(time.perf_counter() - t0) * 1000,
            message=f"Check raised exception: {e}",
        )


def check_database() -> ComponentHealth:
    name = "database"
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = int(os.environ.get("DB_PORT", 5432))
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((db_host, db_port))
        sock.close()
        if result == 0:
            return ComponentHealth(name, HealthStatus.OK,
                                    0, f"PostgreSQL reachable at {db_host}:{db_port}",
                                    {"host": db_host, "port": db_port})
        else:
            return ComponentHealth(name, HealthStatus.DOWN,
                                    0, f"Cannot connect to PostgreSQL at {db_host}:{db_port}",
                                    {"host": db_host, "port": db_port})
    except Exception as e:
        return ComponentHealth(name, HealthStatus.UNKNOWN, 0, f"DB check error: {e}")


def check_mlflow() -> ComponentHealth:
    name = "mlflow"
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    if uri.startswith("sqlite"):
        db_path = uri.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path) or "."
        if os.path.exists(db_dir):
            return ComponentHealth(name, HealthStatus.OK, 0,
                                    f"MLflow local store: {uri}",
                                    {"uri": uri, "type": "sqlite"})
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                "MLflow SQLite directory not found", {"uri": uri})
    # HTTP check
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{uri}/health", timeout=3)
        if resp.status == 200:
            return ComponentHealth(name, HealthStatus.OK, 0,
                                    f"MLflow server OK at {uri}", {"uri": uri})
    except Exception as e:
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                f"MLflow server unreachable: {e}", {"uri": uri})
    return ComponentHealth(name, HealthStatus.UNKNOWN, 0, "MLflow status unknown")


def check_model_files() -> ComponentHealth:
    name = "model_files"
    model_dir = os.environ.get("MODEL_DIR", "/data/models")
    if not os.path.exists(model_dir):
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                f"Model directory not found: {model_dir}",
                                {"model_dir": model_dir})
    model_files = []
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith((".pkl", ".pt", ".joblib", ".json")):
                model_files.append(os.path.join(root, f))
    if model_files:
        return ComponentHealth(name, HealthStatus.OK, 0,
                                f"{len(model_files)} model file(s) found",
                                {"count": len(model_files), "dir": model_dir})
    return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                            "No trained model files found — models need training",
                            {"model_dir": model_dir})


def check_data_directories() -> ComponentHealth:
    name = "data_dirs"
    dirs = {
        "raw": os.environ.get("DATA_RAW_DIR", "/data/raw"),
        "clean": os.environ.get("DATA_CLEAN_DIR", "/data/clean"),
        "features": os.environ.get("DATA_FEATURES_DIR", "/data/features"),
        "reports": os.environ.get("DATA_REPORTS_DIR", "/data/reports"),
    }
    missing = [k for k, v in dirs.items() if not os.path.exists(v)]
    if not missing:
        return ComponentHealth(name, HealthStatus.OK, 0,
                                "All data directories present", dirs)
    return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                            f"Missing data directories: {missing}",
                            {"missing": missing, **dirs})


def check_python_dependencies() -> ComponentHealth:
    name = "python_deps"
    required = ["numpy", "pandas", "sklearn"]
    optional = ["torch", "xgboost", "lightgbm", "mlflow", "dash", "plotly",
                 "networkx", "fastapi", "uvicorn"]
    missing_required = []
    missing_optional = []
    available = []

    for pkg in required:
        try:
            __import__(pkg)
            available.append(pkg)
        except ImportError:
            missing_required.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg)
            available.append(pkg)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        return ComponentHealth(name, HealthStatus.DOWN, 0,
                                f"Required packages missing: {missing_required}",
                                {"missing_required": missing_required,
                                 "missing_optional": missing_optional})
    if missing_optional:
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                f"Optional packages missing: {missing_optional} "
                                f"(core functionality available)",
                                {"available": available, "missing_optional": missing_optional})
    return ComponentHealth(name, HealthStatus.OK, 0,
                            "All dependencies available",
                            {"available_count": len(available)})


def check_knowledge_graph() -> ComponentHealth:
    name = "knowledge_graph"
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from knowledge_graph.graph_builder import MedicalKnowledgeGraph
        kg = MedicalKnowledgeGraph()
        stats = kg.graph_stats()
        n_nodes = stats.get("total_nodes", 0)
        if n_nodes >= 50:
            return ComponentHealth(name, HealthStatus.OK, 0,
                                    f"Knowledge graph loaded ({n_nodes} nodes)",
                                    stats)
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                f"Knowledge graph has only {n_nodes} nodes (expected ≥50)",
                                stats)
    except Exception as e:
        return ComponentHealth(name, HealthStatus.DOWN, 0, f"KG load failed: {e}")


def check_nlp_pipeline() -> ComponentHealth:
    name = "nlp_pipeline"
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from nlp_pipeline.medical_ner import MedicalNER
        ner = MedicalNER()
        result = ner.process("Patient with septic shock on vancomycin 1g IV.")
        meds = result.get_by_label("MEDICATION")
        if meds:
            return ComponentHealth(name, HealthStatus.OK, 0,
                                    f"NER pipeline operational ({len(meds)} meds extracted from test)",
                                    {"test_meds_found": [m.text for m in meds]})
        return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                                "NER pipeline loaded but test extraction yielded no results")
    except Exception as e:
        return ComponentHealth(name, HealthStatus.DOWN, 0, f"NLP pipeline error: {e}")


def check_system_resources() -> ComponentHealth:
    name = "system_resources"
    details: Dict = {"python_version": platform.python_version(),
                      "platform": platform.system()}
    status = HealthStatus.OK
    messages = []

    # Disk space
    try:
        stat = os.statvfs("/")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        details["disk_free_gb"] = round(free_gb, 2)
        if free_gb < 1.0:
            status = HealthStatus.DEGRADED
            messages.append(f"Low disk space: {free_gb:.1f} GB free")
    except Exception:
        details["disk_free_gb"] = "unknown"

    # Memory (if psutil available)
    try:
        import psutil
        mem = psutil.virtual_memory()
        details["memory_used_pct"] = round(mem.percent, 1)
        details["memory_available_gb"] = round(mem.available / (1024 ** 3), 2)
        if mem.percent > 90:
            status = HealthStatus.DEGRADED
            messages.append(f"High memory usage: {mem.percent:.0f}%")
        cpu_pct = psutil.cpu_percent(interval=0.5)
        details["cpu_percent"] = round(cpu_pct, 1)
        if cpu_pct > 90:
            status = HealthStatus.DEGRADED
            messages.append(f"High CPU: {cpu_pct:.0f}%")
    except ImportError:
        details["memory"] = "psutil not installed"

    msg = "; ".join(messages) if messages else "System resources within normal bounds"
    return ComponentHealth(name, status, 0, msg, details)


def check_fhir_server() -> ComponentHealth:
    name = "fhir_server"
    fhir_url = os.environ.get("FHIR_SERVER_URL", "http://localhost:8080/fhir")
    try:
        import urllib.request
        req = urllib.request.Request(f"{fhir_url}/metadata")
        req.add_header("Accept", "application/fhir+json")
        resp = urllib.request.urlopen(req, timeout=3)
        if resp.status == 200:
            return ComponentHealth(name, HealthStatus.OK, 0,
                                    f"FHIR server OK at {fhir_url}", {"url": fhir_url})
    except Exception:
        pass
    return ComponentHealth(name, HealthStatus.DEGRADED, 0,
                            f"FHIR server not reachable at {fhir_url} (offline mode active)",
                            {"url": fhir_url, "mode": "offline"})


# ─────────────────────────────────────────
# Health monitor
# ─────────────────────────────────────────

class SystemHealthMonitor:
    """
    Runs all health checks and maintains a cache of the last report.
    Supports background polling and webhook-style alert callbacks.
    """

    VERSION = "1.0.0"
    CHECKS = [
        ("database",          check_database),
        ("mlflow",            check_mlflow),
        ("model_files",       check_model_files),
        ("data_directories",  check_data_directories),
        ("python_deps",       check_python_dependencies),
        ("knowledge_graph",   check_knowledge_graph),
        ("nlp_pipeline",      check_nlp_pipeline),
        ("system_resources",  check_system_resources),
        ("fhir_server",       check_fhir_server),
    ]

    def __init__(self):
        self._start_time = datetime.utcnow()
        self._last_report: Optional[SystemHealthReport] = None
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable[[SystemHealthReport], None]] = []
        self._lock = threading.Lock()

    def add_alert_callback(self, cb: Callable[[SystemHealthReport], None]):
        self._alert_callbacks.append(cb)

    def check_all(self, parallel: bool = False) -> SystemHealthReport:
        """Run all health checks and return a report."""
        components = []
        for check_name, check_fn in self.CHECKS:
            comp = _timed_check(check_fn)
            comp.name = check_name
            components.append(comp)

        # Overall status: worst of all components
        statuses = [c.status for c in components]
        if HealthStatus.DOWN in statuses:
            overall = HealthStatus.DOWN
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.OK

        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        report = SystemHealthReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_status=overall,
            version=self.VERSION,
            uptime_seconds=uptime,
            components=components,
            system_info={
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "hostname": platform.node(),
                "pid": os.getpid(),
            },
        )

        with self._lock:
            self._last_report = report

        # Fire alert callbacks if degraded/down
        if overall != HealthStatus.OK:
            for cb in self._alert_callbacks:
                try:
                    cb(report)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        return report

    def get_last_report(self) -> Optional[SystemHealthReport]:
        with self._lock:
            return self._last_report

    def start_polling(self, interval_seconds: int = 60):
        """Background polling thread."""
        self._polling = True
        def _poll():
            while self._polling:
                self.check_all()
                time.sleep(interval_seconds)
        self._poll_thread = threading.Thread(target=_poll, daemon=True)
        self._poll_thread.start()
        logger.info(f"Health monitor polling every {interval_seconds}s")

    def stop_polling(self):
        self._polling = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    def readiness_probe(self) -> bool:
        """K8s-style readiness probe — returns True only if core services are up."""
        report = self.check_all()
        core = {c.name: c.status for c in report.components
                if c.name in ("python_deps", "knowledge_graph", "nlp_pipeline")}
        return all(v == HealthStatus.OK for v in core.values())

    def liveness_probe(self) -> bool:
        """K8s-style liveness probe — process is alive."""
        return True

    def render_text(self, report: SystemHealthReport) -> str:
        """Render a health report as formatted text."""
        lines = [
            f"{'='*55}",
            f"  AI Hospital OS — Health Report",
            f"  {report.timestamp[:19]} UTC | Uptime: {report.uptime_seconds:.0f}s",
            f"  Overall: {report.overall_status.upper()}",
            f"{'='*55}",
        ]
        icon_map = {HealthStatus.OK: "✓", HealthStatus.DEGRADED: "⚠",
                     HealthStatus.DOWN: "✗", HealthStatus.UNKNOWN: "?"}
        for c in report.components:
            icon = icon_map.get(c.status, "?")
            lines.append(f"  {icon} {c.name:<22} {c.status:<10} "
                          f"{c.latency_ms:>6.0f}ms  {c.message[:40]}")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────
# WSGI health endpoint (stdlib only)
# ─────────────────────────────────────────

def make_health_wsgi_app(monitor: SystemHealthMonitor):
    """Create a minimal WSGI app that serves /health and /ready."""
    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path == "/health" or path == "/":
            report = monitor.check_all()
            body = json.dumps(report.to_dict(), default=str).encode()
            status_code = "200 OK" if report.healthy else "503 Service Unavailable"
            start_response(status_code,
                            [("Content-Type", "application/json"),
                             ("Content-Length", str(len(body)))])
            return [body]
        elif path == "/ready":
            ready = monitor.readiness_probe()
            status_code = "200 OK" if ready else "503 Not Ready"
            body = json.dumps({"ready": ready}).encode()
            start_response(status_code, [("Content-Type", "application/json")])
            return [body]
        elif path == "/live":
            body = json.dumps({"alive": True}).encode()
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body]
        else:
            body = b'{"error": "not found"}'
            start_response("404 Not Found", [("Content-Type", "application/json")])
            return [body]
    return app


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s — %(message)s")

    monitor = SystemHealthMonitor()

    def on_alert(report: SystemHealthReport):
        down = [c.name for c in report.components if c.status == HealthStatus.DOWN]
        print(f"  HEALTH ALERT: {report.overall_status.upper()} — down: {down}")

    monitor.add_alert_callback(on_alert)

    print("Running system health check...\n")
    report = monitor.check_all()
    print(monitor.render_text(report))

    print(f"\nReadiness probe: {'READY' if monitor.readiness_probe() else 'NOT READY'}")
    print(f"Liveness probe:  {'ALIVE' if monitor.liveness_probe() else 'DEAD'}")

    # Component breakdown
    print(f"\nDetailed component status:")
    for c in report.components:
        if c.status != HealthStatus.OK:
            print(f"  [{c.status.upper():8s}] {c.name}: {c.message}")
            if c.details:
                for k, v in list(c.details.items())[:3]:
                    print(f"           {k}: {v}")

    print(f"\n{report.summary()}")

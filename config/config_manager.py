"""
Configuration Management
Centralised, validated configuration for all Hospital OS subsystems.
Loads from YAML files + environment variable overrides.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "hospital_os"
    user: str = "hospital"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    connect_timeout: int = 10

    @property
    def url(self) -> str:
        return (f"postgresql+psycopg2://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.name}")

    @property
    def async_url(self) -> str:
        return (f"postgresql+asyncpg://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.name}")


@dataclass
class MLConfig:
    model_dir: str = "/data/models"
    retrain_schedule: str = "0 2 * * *"  # cron
    min_auc_for_promotion: float = 0.80
    sepsis_alert_threshold: float = 0.55
    mortality_alert_threshold: float = 0.40
    lstm_sequence_length: int = 24
    lstm_hidden_size: int = 64
    lstm_n_layers: int = 2
    anomaly_contamination: float = 0.08
    feature_window_hours: int = 6
    batch_size: int = 64
    max_train_epochs: int = 50
    early_stopping_patience: int = 7
    device: str = "cpu"  # cpu | cuda


@dataclass
class AlertConfig:
    news2_warning_threshold: int = 5
    news2_urgent_threshold: int = 7
    news2_critical_threshold: int = 9
    qsofa_alert_threshold: int = 2
    shock_index_threshold: float = 1.0
    lactate_alert_mmol: float = 2.0
    map_critical_mmhg: float = 65.0
    alert_cooldown_minutes: int = 10
    max_alerts_per_patient_per_hour: int = 5
    send_sms: bool = False
    send_email: bool = False
    sms_endpoint: str = ""
    email_recipients: list = field(default_factory=list)


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8050
    refresh_interval_ms: int = 5000
    max_patients_displayed: int = 20
    vital_history_hours: int = 24
    debug_mode: bool = False


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    rate_limit_per_minute: int = 120
    enable_cors: bool = True
    allowed_origins: list = field(default_factory=lambda: ["*"])
    jwt_secret: str = "change_in_production"
    token_expiry_minutes: int = 480


@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///mlruns.db"
    artifact_root: str = "/data/mlflow/artifacts"
    experiment_prefix: str = "Hospital_OS"
    auto_log: bool = True


@dataclass
class SecurityConfig:
    audit_log_dir: str = "/data/audit"
    session_timeout_minutes: int = 480
    require_2fa: bool = False
    password_min_length: int = 12
    max_login_attempts: int = 5
    lockout_minutes: int = 30
    phi_deidentify_exports: bool = True
    encrypt_at_rest: bool = False


@dataclass
class DataConfig:
    raw_dir: str = "/data/raw"
    clean_dir: str = "/data/clean"
    features_dir: str = "/data/features"
    reports_dir: str = "/data/reports"
    backup_dir: str = "/data/backup"
    max_vitals_history_days: int = 30
    synthetic_data_fallback: bool = True


@dataclass
class HospitalOSConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    environment: str = "development"   # development | staging | production
    log_level: str = "INFO"
    version: str = "1.0.0"


# ─────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────

class ConfigLoader:
    """
    Loads config from (in order of priority):
      1. Default dataclass values
      2. YAML config file (if present)
      3. Environment variable overrides (highest priority)

    Env-var naming: HOSPITAL_OS__SECTION__KEY
    e.g.  HOSPITAL_OS__DATABASE__HOST=mypostgres
          HOSPITAL_OS__ML__SEPSIS_ALERT_THRESHOLD=0.6
          HOSPITAL_OS__ALERTS__NEWS2_CRITICAL_THRESHOLD=9
    """

    ENV_PREFIX = "HOSPITAL_OS__"

    def load(self, config_path: Optional[str] = None) -> HospitalOSConfig:
        cfg = HospitalOSConfig()

        # Load from YAML if available
        if config_path and os.path.exists(config_path):
            cfg = self._load_yaml(cfg, config_path)
        else:
            # Try default locations
            for default_path in ["config/hospital_os.yaml", "/etc/hospital_os/config.yaml",
                                  os.path.expanduser("~/.hospital_os.yaml")]:
                if os.path.exists(default_path):
                    cfg = self._load_yaml(cfg, default_path)
                    break

        # Apply env var overrides
        cfg = self._apply_env_overrides(cfg)

        # Apply legacy single-var env overrides (backward compat)
        cfg = self._apply_legacy_env(cfg)

        self._validate(cfg)
        self._apply_to_environment(cfg)
        return cfg

    def _load_yaml(self, cfg: HospitalOSConfig, path: str) -> HospitalOSConfig:
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            cfg = self._merge(cfg, data)
            logger.info(f"Loaded config from {path}")
        except ImportError:
            # Try JSON fallback
            try:
                path_json = path.replace(".yaml", ".json")
                if os.path.exists(path_json):
                    with open(path_json) as f:
                        data = json.load(f)
                    cfg = self._merge(cfg, data)
            except Exception as e:
                logger.warning(f"Could not load config JSON: {e}")
        except Exception as e:
            logger.warning(f"Could not load config from {path}: {e}")
        return cfg

    def _merge(self, cfg: HospitalOSConfig, data: Dict) -> HospitalOSConfig:
        """Merge a dict into the config dataclass."""
        section_map = {
            "database": ("database", DatabaseConfig),
            "ml": ("ml", MLConfig),
            "alerts": ("alerts", AlertConfig),
            "dashboard": ("dashboard", DashboardConfig),
            "api": ("api", APIConfig),
            "mlflow": ("mlflow", MLflowConfig),
            "security": ("security", SecurityConfig),
            "data": ("data", DataConfig),
        }
        for section_key, (attr, cls) in section_map.items():
            if section_key in data:
                current = getattr(cfg, attr)
                current_dict = asdict(current)
                current_dict.update(data[section_key])
                try:
                    setattr(cfg, attr, cls(**{k: v for k, v in current_dict.items()
                                               if k in cls.__dataclass_fields__}))
                except Exception as e:
                    logger.warning(f"Config merge error for {section_key}: {e}")

        for top_key in ("environment", "log_level", "version"):
            if top_key in data:
                setattr(cfg, top_key, data[top_key])
        return cfg

    def _apply_env_overrides(self, cfg: HospitalOSConfig) -> HospitalOSConfig:
        """Apply HOSPITAL_OS__SECTION__KEY env vars."""
        section_map = {
            "DATABASE": "database", "ML": "ml", "ALERTS": "alerts",
            "DASHBOARD": "dashboard", "API": "api", "MLFLOW": "mlflow",
            "SECURITY": "security", "DATA": "data",
        }
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(self.ENV_PREFIX):
                continue
            parts = env_key[len(self.ENV_PREFIX):].split("__")
            if len(parts) < 2:
                continue
            section_upper, field_name = parts[0], "__".join(parts[1:]).lower()
            attr_name = section_map.get(section_upper)
            if not attr_name:
                continue
            section_obj = getattr(cfg, attr_name, None)
            if section_obj and hasattr(section_obj, field_name):
                old_val = getattr(section_obj, field_name)
                try:
                    new_val = _coerce(env_val, type(old_val))
                    setattr(section_obj, field_name, new_val)
                    logger.debug(f"Config override: {env_key} = {new_val}")
                except Exception as e:
                    logger.warning(f"Could not apply env override {env_key}: {e}")
        return cfg

    def _apply_legacy_env(self, cfg: HospitalOSConfig) -> HospitalOSConfig:
        """Apply simple DB_* and MLFLOW_* env vars for Docker Compose compatibility."""
        db = cfg.database
        db.host     = os.environ.get("DB_HOST", db.host)
        db.port     = int(os.environ.get("DB_PORT", db.port))
        db.name     = os.environ.get("DB_NAME", db.name)
        db.user     = os.environ.get("DB_USER", db.user)
        db.password = os.environ.get("DB_PASSWORD", db.password)

        mf = cfg.mlflow
        mf.tracking_uri  = os.environ.get("MLFLOW_TRACKING_URI", mf.tracking_uri)
        mf.artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", mf.artifact_root)

        cfg.log_level = os.environ.get("LOG_LEVEL", cfg.log_level)
        cfg.environment = os.environ.get("HOSPITAL_OS_ENV", cfg.environment)
        return cfg

    def _validate(self, cfg: HospitalOSConfig):
        errors = []
        if cfg.ml.min_auc_for_promotion < 0 or cfg.ml.min_auc_for_promotion > 1:
            errors.append("ml.min_auc_for_promotion must be 0–1")
        if cfg.alerts.news2_warning_threshold >= cfg.alerts.news2_critical_threshold:
            errors.append("alerts: warning threshold must be < critical threshold")
        if cfg.api.workers < 1:
            errors.append("api.workers must be ≥ 1")
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        logger.info(f"Config validated OK (env={cfg.environment})")

    def _apply_to_environment(self, cfg: HospitalOSConfig):
        """Set process-level environment and logging."""
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )
        os.makedirs(cfg.data.raw_dir, exist_ok=True)
        os.makedirs(cfg.data.clean_dir, exist_ok=True)
        os.makedirs(cfg.data.features_dir, exist_ok=True)
        os.makedirs(cfg.data.reports_dir, exist_ok=True)
        os.makedirs(cfg.ml.model_dir, exist_ok=True)
        os.makedirs(cfg.security.audit_log_dir, exist_ok=True)

    def save(self, cfg: HospitalOSConfig, path: str):
        """Save current config to JSON (YAML if PyYAML available)."""
        data = asdict(cfg)
        # Redact password
        data["database"]["password"] = "***REDACTED***"
        data["api"]["jwt_secret"] = "***REDACTED***"

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                logger.info(f"Config saved (YAML): {path}")
                return
            except ImportError:
                path = path.replace(".yaml", ".json").replace(".yml", ".json")

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Config saved (JSON): {path}")


def _coerce(value: str, target_type: type) -> Any:
    if target_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    if target_type == int:
        return int(value)
    if target_type == float:
        return float(value)
    if target_type == list:
        return [v.strip() for v in value.split(",")]
    return value


# ─────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────

_config: Optional[HospitalOSConfig] = None
_loader = ConfigLoader()


def get_config(config_path: Optional[str] = None) -> HospitalOSConfig:
    """Get or initialise the global config singleton."""
    global _config
    if _config is None:
        _config = _loader.load(config_path)
    return _config


def reload_config(config_path: Optional[str] = None) -> HospitalOSConfig:
    global _config
    _config = _loader.load(config_path)
    return _config


# ─────────────────────────────────────────
# Example YAML config template generator
# ─────────────────────────────────────────

def generate_config_template(output_path: str = "config/hospital_os.yaml"):
    """Write a commented YAML config template."""
    template = """# AI Hospital OS — Configuration
# Override any value with env vars: HOSPITAL_OS__SECTION__KEY=value
# e.g. HOSPITAL_OS__DATABASE__HOST=mypostgresql

environment: development   # development | staging | production
log_level: INFO

database:
  host: localhost
  port: 5432
  name: hospital_os
  user: hospital
  password: ""             # Use DB_PASSWORD env var in production
  pool_size: 10

ml:
  model_dir: /data/models
  retrain_schedule: "0 2 * * *"    # Daily at 02:00 UTC
  min_auc_for_promotion: 0.80
  sepsis_alert_threshold: 0.55
  mortality_alert_threshold: 0.40
  lstm_sequence_length: 24
  anomaly_contamination: 0.08
  device: cpu                      # cpu | cuda

alerts:
  news2_warning_threshold: 5
  news2_urgent_threshold: 7
  news2_critical_threshold: 9
  qsofa_alert_threshold: 2
  shock_index_threshold: 1.0
  lactate_alert_mmol: 2.0
  map_critical_mmhg: 65.0
  alert_cooldown_minutes: 10
  send_sms: false
  send_email: false

dashboard:
  host: "0.0.0.0"
  port: 8050
  refresh_interval_ms: 5000
  debug_mode: false

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  rate_limit_per_minute: 120
  token_expiry_minutes: 480

mlflow:
  tracking_uri: "sqlite:///mlruns.db"
  artifact_root: /data/mlflow/artifacts
  auto_log: true

security:
  audit_log_dir: /data/audit
  session_timeout_minutes: 480
  max_login_attempts: 5
  phi_deidentify_exports: true

data:
  raw_dir: /data/raw
  clean_dir: /data/clean
  features_dir: /data/features
  reports_dir: /data/reports
  synthetic_data_fallback: true
"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(template)
    logger.info(f"Config template written: {output_path}")
    return output_path


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpl = generate_config_template(os.path.join(tmpdir, "config/hospital_os.yaml"))
        print(f"Template written: {tmpl}")

        cfg = get_config()
        print(f"\nLoaded config:")
        print(f"  Environment  : {cfg.environment}")
        print(f"  Log level    : {cfg.log_level}")
        print(f"  DB host      : {cfg.database.host}:{cfg.database.port}/{cfg.database.name}")
        print(f"  ML device    : {cfg.ml.device}")
        print(f"  Sepsis thresh: {cfg.ml.sepsis_alert_threshold}")
        print(f"  NEWS2 crit   : {cfg.alerts.news2_critical_threshold}")
        print(f"  Dashboard    : {cfg.dashboard.host}:{cfg.dashboard.port}")
        print(f"  API          : {cfg.api.host}:{cfg.api.port} ({cfg.api.workers} workers)")
        print(f"  MLflow URI   : {cfg.mlflow.tracking_uri}")

        save_path = os.path.join(tmpdir, "saved_config.json")
        _loader.save(cfg, save_path)
        print(f"\nConfig saved to: {save_path}")

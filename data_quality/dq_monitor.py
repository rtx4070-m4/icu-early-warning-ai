"""
Data Quality Monitor
Continuously validates incoming data against clinical reference ranges,
detects schema drift, missing value spikes, and statistical distribution shifts.
Raises DataQualityAlerts with severity levels and remediation suggestions.
"""

import os
import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Clinical reference ranges
# ─────────────────────────────────────────

CLINICAL_RANGES = {
    # Vitals — (plausible_min, plausible_max, critical_low, critical_high)
    "heart_rate":         (20,   300,  30,   200),
    "sbp":                (40,   300,  60,   250),
    "dbp":                (20,   200,  30,   150),
    "map":                (20,   200,  40,   160),
    "respiratory_rate":   (4,    60,   6,    50),
    "temperature":        (30.0, 45.0, 33.0, 42.5),
    "spo2":               (50,   100,  70,   100),
    "gcs":                (3,    15,   3,    15),
    # Labs
    "heart_rate_lab":     (20,   300,  None, None),
    "wbc":                (0.1,  100,  0.5,  50),
    "hemoglobin":         (2.0,  25.0, 4.0,  20.0),
    "platelet":           (1,    2000, 10,   1500),
    "creatinine":         (0.1,  30.0, 0.2,  15.0),
    "bun":                (1,    300,  2,    200),
    "sodium":             (100,  175,  115,  165),
    "potassium":          (1.5,  9.0,  2.5,  7.5),
    "glucose":            (20,   1500, 40,   800),
    "lactate":            (0.1,  30.0, 0.2,  20.0),
    "inr":                (0.5,  15.0, 0.7,  10.0),
    "troponin":           (0.0,  100.0, None, None),
    "bnp":                (0,    10000, None, None),
    "ph":                 (6.5,  8.0,  6.8,  7.9),
    "pao2":               (20,   600,  30,   500),
    "procalcitonin":      (0.0,  1000, None, None),
}

REQUIRED_VITALS_FIELDS = [
    "patient_id", "timestamp", "heart_rate", "sbp", "dbp",
    "respiratory_rate", "temperature", "spo2"
]

REQUIRED_LABS_FIELDS = ["patient_id", "timestamp", "test_name", "value"]


# ─────────────────────────────────────────
# Alert structures
# ─────────────────────────────────────────

@dataclass
class DataQualityAlert:
    alert_id: str
    timestamp: str
    check_name: str
    severity: str           # info | warning | error | critical
    affected_field: str
    affected_records: int
    message: str
    remediation: str
    details: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class DataQualityReport:
    report_id: str
    timestamp: str
    source: str
    n_records: int
    n_fields_checked: int
    alerts: List[DataQualityAlert] = field(default_factory=list)
    completeness: Dict = field(default_factory=dict)      # field → % present
    range_violations: Dict = field(default_factory=dict)  # field → count out of range
    distribution_stats: Dict = field(default_factory=dict)
    overall_quality_score: float = 1.0                    # 0–1

    def to_dict(self):
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "n_records": self.n_records,
            "n_fields_checked": self.n_fields_checked,
            "overall_quality_score": round(self.overall_quality_score, 3),
            "alerts": [a.to_dict() for a in self.alerts],
            "completeness": self.completeness,
            "range_violations": self.range_violations,
            "distribution_stats": self.distribution_stats,
        }

    @property
    def critical_alerts(self) -> List[DataQualityAlert]:
        return [a for a in self.alerts if a.severity == "critical"]

    @property
    def error_alerts(self) -> List[DataQualityAlert]:
        return [a for a in self.alerts if a.severity in ("error", "critical")]

    def summary(self) -> str:
        return (f"DataQuality [{self.source}] score={self.overall_quality_score:.2f} "
                f"n={self.n_records} alerts={len(self.alerts)} "
                f"critical={len(self.critical_alerts)}")


# ─────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────

def _make_alert(check: str, severity: str, field: str,
                count: int, message: str, remediation: str,
                details: Dict = None) -> DataQualityAlert:
    import uuid
    return DataQualityAlert(
        alert_id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(),
        check_name=check,
        severity=severity,
        affected_field=field,
        affected_records=count,
        message=message,
        remediation=remediation,
        details=details or {},
    )


def check_schema(records: List[Dict], required_fields: List[str],
                  source: str) -> List[DataQualityAlert]:
    """Ensure all required fields are present in the dataset."""
    alerts = []
    if not records:
        alerts.append(_make_alert(
            "schema_empty", "critical", "all", 0,
            f"{source}: No records found in dataset",
            "Verify data pipeline is running and source system is connected"
        ))
        return alerts

    sample = records[0]
    missing = [f for f in required_fields if f not in sample]
    if missing:
        alerts.append(_make_alert(
            "schema_missing_fields", "error", ",".join(missing), len(records),
            f"Required fields missing: {missing}",
            f"Update ETL pipeline to include fields: {missing}",
            {"missing_fields": missing, "present_fields": list(sample.keys())},
        ))
    return alerts


def check_completeness(records: List[Dict],
                        fields: List[str]) -> Tuple[Dict[str, float], List[DataQualityAlert]]:
    """Compute % of records with non-null values per field."""
    n = len(records)
    if n == 0:
        return {}, []
    completeness = {}
    alerts = []
    for field in fields:
        present = sum(1 for r in records if r.get(field) is not None and r.get(field) != "")
        pct = present / n
        completeness[field] = round(pct, 4)
        if pct < 0.70:
            alerts.append(_make_alert(
                "completeness_critical", "critical", field,
                n - present,
                f"{field}: only {pct:.1%} complete ({n-present} nulls)",
                f"Investigate data source for {field}; consider LOCF imputation",
                {"completeness_pct": pct, "n_missing": n - present},
            ))
        elif pct < 0.90:
            alerts.append(_make_alert(
                "completeness_warning", "warning", field,
                n - present,
                f"{field}: {pct:.1%} complete ({n-present} missing values)",
                "Monitor for increasing missingness; apply imputation strategy",
                {"completeness_pct": pct, "n_missing": n - present},
            ))
    return completeness, alerts


def check_ranges(records: List[Dict]) -> Tuple[Dict[str, int], List[DataQualityAlert]]:
    """Check numeric fields against plausible clinical ranges."""
    violations: Dict[str, int] = {}
    alerts = []
    for field, (lo, hi, crit_lo, crit_hi) in CLINICAL_RANGES.items():
        field_key = field.replace("_lab", "")
        vals = [r.get(field_key) or r.get(field) for r in records
                if (r.get(field_key) is not None or r.get(field) is not None)]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue

        oor = [v for v in vals if v < lo or v > hi]
        critical = [v for v in vals
                    if (crit_lo is not None and v < crit_lo) or
                       (crit_hi is not None and v > crit_hi)]

        if oor:
            violations[field_key] = len(oor)
            pct = len(oor) / len(vals)
            severity = "critical" if pct > 0.10 or critical else "warning" if pct > 0.02 else "info"
            alerts.append(_make_alert(
                "range_violation", severity, field_key, len(oor),
                f"{field_key}: {len(oor)} values outside [{lo}, {hi}] ({pct:.1%})",
                f"Review sensor calibration; apply winsorisation or flag for manual review",
                {"n_oor": len(oor), "n_critical": len(critical),
                 "pct_oor": round(pct, 4), "examples": oor[:3]},
            ))
    return violations, alerts


def check_duplicates(records: List[Dict],
                      key_fields: List[str]) -> List[DataQualityAlert]:
    """Detect duplicate records by key field combination."""
    keys = [tuple(str(r.get(f, "")) for f in key_fields) for r in records]
    seen: Dict = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    dups = sum(v - 1 for v in seen.values() if v > 1)
    if dups > 0:
        pct = dups / len(records)
        severity = "error" if pct > 0.05 else "warning"
        return [_make_alert(
            "duplicates", severity, ",".join(key_fields), dups,
            f"{dups} duplicate records detected ({pct:.1%})",
            "Deduplicate at ingestion layer; check upsert logic in ETL",
            {"n_duplicates": dups, "pct_duplicates": round(pct, 4)},
        )]
    return []


def check_timestamps(records: List[Dict],
                      timestamp_field: str = "timestamp",
                      max_future_minutes: int = 5,
                      max_past_days: int = 365) -> List[DataQualityAlert]:
    """Check for future timestamps, very old records, or non-monotonic sequences."""
    alerts = []
    now = datetime.utcnow()
    future_count = old_count = parse_errors = 0

    for r in records:
        ts_raw = r.get(timestamp_field)
        if ts_raw is None:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", ""))
            if ts > now + timedelta(minutes=max_future_minutes):
                future_count += 1
            if ts < now - timedelta(days=max_past_days):
                old_count += 1
        except ValueError:
            parse_errors += 1

    if parse_errors > 0:
        alerts.append(_make_alert(
            "timestamp_parse_error", "error", timestamp_field, parse_errors,
            f"{parse_errors} records with unparseable timestamps",
            "Standardise timestamp format to ISO-8601 UTC in ETL",
        ))
    if future_count > 0:
        alerts.append(_make_alert(
            "timestamp_future", "warning", timestamp_field, future_count,
            f"{future_count} records with future timestamps",
            "Check system clock synchronisation on monitoring devices",
        ))
    if old_count > 0:
        alerts.append(_make_alert(
            "timestamp_stale", "info", timestamp_field, old_count,
            f"{old_count} records older than {max_past_days} days",
            "Verify historical data ingestion is expected; filter if backfill is complete",
        ))
    return alerts


def check_distribution_shift(current: List[float], baseline: List[float],
                               field: str, threshold: float = 0.15) -> Optional[DataQualityAlert]:
    """
    Detect distribution shift via Population Stability Index (PSI).
    PSI < 0.10: no change | 0.10–0.25: moderate | >0.25: significant
    """
    if not current or not baseline:
        return None
    n_bins = 10
    all_vals = current + baseline
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        return None
    bin_width = (hi - lo) / n_bins

    def bin_counts(vals):
        counts = [0] * n_bins
        for v in vals:
            idx = min(int((v - lo) / bin_width), n_bins - 1)
            counts[idx] += 1
        return counts

    curr_counts = bin_counts(current)
    base_counts = bin_counts(baseline)
    n_curr, n_base = len(current), len(baseline)

    psi = 0.0
    for c, b in zip(curr_counts, base_counts):
        p_curr = max(c / n_curr, 1e-6)
        p_base = max(b / n_base, 1e-6)
        psi += (p_curr - p_base) * math.log(p_curr / p_base)

    if psi > 0.25:
        severity = "error"
    elif psi > 0.10:
        severity = "warning"
    else:
        return None

    return _make_alert(
        "distribution_shift", severity, field, len(current),
        f"{field}: PSI={psi:.3f} — {'significant' if psi>0.25 else 'moderate'} distribution shift",
        "Investigate data source changes; consider model retraining if shift persists",
        {"psi": round(psi, 4), "current_mean": round(sum(current)/len(current), 3),
         "baseline_mean": round(sum(baseline)/len(baseline), 3)},
    )


def compute_distribution_stats(records: List[Dict],
                                 fields: List[str]) -> Dict[str, Dict]:
    """Compute basic distribution statistics per numeric field."""
    stats = {}
    for f in fields:
        vals = [r[f] for r in records if isinstance(r.get(f), (int, float))]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        std = math.sqrt(variance)
        q1 = vals_sorted[n // 4]
        q3 = vals_sorted[3 * n // 4]
        stats[f] = {
            "n": n, "mean": round(mean, 3), "std": round(std, 3),
            "min": round(vals_sorted[0], 3), "max": round(vals_sorted[-1], 3),
            "p25": round(q1, 3), "p50": round(vals_sorted[n // 2], 3), "p75": round(q3, 3),
            "iqr": round(q3 - q1, 3),
        }
    return stats


# ─────────────────────────────────────────
# Main monitor
# ─────────────────────────────────────────

class DataQualityMonitor:
    """
    Runs all data quality checks on an incoming batch of records
    and produces a DataQualityReport.
    """

    def __init__(self, baseline_stats: Optional[Dict] = None):
        """
        baseline_stats: previously computed distribution_stats to compare against.
        If None, distribution shift checks are skipped.
        """
        self.baseline_stats = baseline_stats or {}
        self._alert_history: List[DataQualityAlert] = []

    def check_vitals(self, records: List[Dict], source: str = "vitals") -> DataQualityReport:
        return self._check(records, source, REQUIRED_VITALS_FIELDS,
                            key_fields=["patient_id", "timestamp"])

    def check_labs(self, records: List[Dict], source: str = "labs") -> DataQualityReport:
        return self._check(records, source, REQUIRED_LABS_FIELDS,
                            key_fields=["patient_id", "timestamp", "test_name"])

    def _check(self, records: List[Dict], source: str,
                required_fields: List[str], key_fields: List[str]) -> DataQualityReport:
        import uuid
        report = DataQualityReport(
            report_id=str(uuid.uuid4())[:12],
            timestamp=datetime.utcnow().isoformat(),
            source=source,
            n_records=len(records),
            n_fields_checked=len(required_fields),
        )
        all_alerts: List[DataQualityAlert] = []

        # 1. Schema
        all_alerts += check_schema(records, required_fields, source)

        # 2. Completeness
        numeric_fields = [f for f in required_fields
                          if f not in ("patient_id", "timestamp", "test_name")]
        completeness, comp_alerts = check_completeness(records, numeric_fields)
        report.completeness = completeness
        all_alerts += comp_alerts

        # 3. Clinical ranges
        violations, range_alerts = check_ranges(records)
        report.range_violations = violations
        all_alerts += range_alerts

        # 4. Duplicates
        all_alerts += check_duplicates(records, key_fields)

        # 5. Timestamps
        all_alerts += check_timestamps(records)

        # 6. Distribution stats + shift
        dist_stats = compute_distribution_stats(records, numeric_fields)
        report.distribution_stats = dist_stats

        for field, stats in dist_stats.items():
            if field in self.baseline_stats:
                curr_vals = [r.get(field) for r in records
                              if isinstance(r.get(field), (int, float))]
                base_mean = self.baseline_stats[field].get("mean", stats["mean"])
                base_std  = self.baseline_stats[field].get("std", stats["std"])
                # Reconstruct approximate baseline distribution
                base_vals = [base_mean + random.gauss(0, base_std) for _ in range(200)]
                alert = check_distribution_shift(curr_vals, base_vals, field)
                if alert:
                    all_alerts.append(alert)

        report.alerts = sorted(all_alerts,
                                key=lambda a: {"critical": 0, "error": 1,
                                               "warning": 2, "info": 3}.get(a.severity, 4))
        self._alert_history += report.alerts

        # Quality score: penalise errors/criticals
        penalty = (len([a for a in all_alerts if a.severity == "critical"]) * 0.20 +
                   len([a for a in all_alerts if a.severity == "error"]) * 0.10 +
                   len([a for a in all_alerts if a.severity == "warning"]) * 0.02)
        report.overall_quality_score = max(0.0, 1.0 - penalty)
        return report

    def get_alert_history(self, n: int = 50) -> List[Dict]:
        return [a.to_dict() for a in self._alert_history[-n:]]


# ─────────────────────────────────────────
# Synthetic data generator for testing
# ─────────────────────────────────────────

def generate_test_vitals(n: int = 500, inject_errors: bool = True) -> List[Dict]:
    """Generate synthetic vital records, optionally with data quality issues."""
    records = []
    base = datetime.utcnow() - timedelta(hours=n // 12)
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        pid = f"P{(i % 20) + 1:03d}"
        record = {
            "patient_id": pid,
            "timestamp": ts.isoformat(),
            "heart_rate": round(80 + random.gauss(0, 10), 1),
            "sbp":        round(120 + random.gauss(0, 12), 1),
            "dbp":        round(75 + random.gauss(0, 7), 1),
            "respiratory_rate": round(16 + random.gauss(0, 2), 1),
            "temperature": round(37.0 + random.gauss(0, 0.3), 2),
            "spo2":       round(min(100, 97 + random.gauss(0, 1.2)), 1),
        }
        if inject_errors:
            r = random.random()
            if r < 0.02:       # 2%: out-of-range HR
                record["heart_rate"] = random.choice([5, 350])
            elif r < 0.04:     # 2%: missing SpO2
                record["spo2"] = None
            elif r < 0.045:    # 0.5%: future timestamp
                record["timestamp"] = (ts + timedelta(hours=24)).isoformat()
        records.append(record)
    return records


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    print("Generating synthetic vital records with injected errors...")
    records = generate_test_vitals(n=600, inject_errors=True)
    print(f"Generated {len(records)} records\n")

    monitor = DataQualityMonitor()
    report = monitor.check_vitals(records, source="synthetic_vitals")

    print(f"{'='*60}")
    print(f"Data Quality Report")
    print(f"{'='*60}")
    print(f"Source   : {report.source}")
    print(f"Records  : {report.n_records:,}")
    print(f"Quality  : {report.overall_quality_score:.2%}")
    print(f"Alerts   : {len(report.alerts)} "
          f"({len(report.critical_alerts)} critical, "
          f"{len(report.error_alerts) - len(report.critical_alerts)} error)")

    print(f"\nCompleteness:")
    for f, pct in report.completeness.items():
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        flag = " ⚠" if pct < 0.90 else ""
        print(f"  {f:25s} {bar} {pct:.1%}{flag}")

    print(f"\nRange Violations:")
    for f, count in report.range_violations.items():
        print(f"  {f:25s} {count} records out of range")

    print(f"\nAlerts:")
    for a in report.alerts:
        icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(a.severity, "⚪")
        print(f"  {icon} [{a.severity.upper():8s}] {a.check_name}: {a.message}")

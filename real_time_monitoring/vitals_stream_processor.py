"""
Real-Time Vitals Stream Processor
Simulates ICU streaming data, applies rule-based + ML deterioration detection,
fires alerts, and maintains a rolling patient state buffer
"""

import time
import logging
import threading
import queue
import random
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────

@dataclass
class VitalReading:
    patient_id: str
    timestamp: datetime
    heart_rate: float
    sbp: float
    dbp: float
    respiratory_rate: float
    temperature: float
    spo2: float
    map: float = 0.0
    gcs: int = 15
    source: str = "monitor"  # monitor | manual | calculated

    def __post_init__(self):
        if self.map == 0.0:
            self.map = (self.sbp + 2 * self.dbp) / 3

    def to_dict(self):
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


@dataclass
class Alert:
    patient_id: str
    timestamp: datetime
    alert_type: str          # CRITICAL_VITAL | DETERIORATION | SEPSIS_ALERT | ARRHYTHMIA
    severity: str            # low | moderate | high | critical
    message: str
    vitals_snapshot: Dict = field(default_factory=dict)
    score: float = 0.0
    acknowledged: bool = False

    def to_dict(self):
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


@dataclass
class PatientState:
    patient_id: str
    name: str = "Unknown"
    location: str = "ICU-1"
    readings: deque = field(default_factory=lambda: deque(maxlen=144))  # 12h @ 5-min
    alerts: List[Alert] = field(default_factory=list)
    baseline_hr: Optional[float] = None
    baseline_map: Optional[float] = None
    is_deteriorating: bool = False
    last_news2: int = 0
    last_sofa: float = 0.0


# ─────────────────────────────────────────
# Clinical scoring
# ─────────────────────────────────────────

def compute_news2(v: VitalReading) -> int:
    """National Early Warning Score 2 (0–20)."""
    score = 0

    # RR
    rr = v.respiratory_rate
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    # SpO2 (scale 1, not on supplemental O2)
    spo2 = v.spo2
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    # SBP
    sbp = v.sbp
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1

    # HR
    hr = v.heart_rate
    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif 41 <= hr <= 50 or 91 <= hr <= 110:
        score += 1

    # Temp
    t = v.temperature
    if t <= 35.0:
        score += 3
    elif 35.1 <= t <= 36.0 or t >= 39.1:
        score += 1
    elif 38.1 <= t <= 39.0:
        score += 1

    # Consciousness (GCS proxy)
    if v.gcs < 15:
        score += 3

    return score


def compute_shock_index(v: VitalReading) -> float:
    return v.heart_rate / v.sbp if v.sbp > 0 else 0.0


def compute_sofa_proxy(v: VitalReading) -> float:
    """Simplified SOFA using available vitals (no labs)."""
    score = 0.0
    # Cardiovascular (MAP)
    if v.map < 70:
        score += 2
    elif v.map < 80:
        score += 1
    # Respiratory (SpO2 proxy)
    if v.spo2 < 90:
        score += 3
    elif v.spo2 < 94:
        score += 2
    elif v.spo2 < 97:
        score += 1
    # CNS (GCS)
    if v.gcs < 6:
        score += 4
    elif v.gcs < 10:
        score += 3
    elif v.gcs < 13:
        score += 2
    elif v.gcs < 15:
        score += 1
    return score


# ─────────────────────────────────────────
# Rule-based alert engine
# ─────────────────────────────────────────

CRITICAL_THRESHOLDS = {
    "heart_rate":         {"low": 40,  "high": 150},
    "sbp":                {"low": 80,  "high": 200},
    "respiratory_rate":   {"low": 6,   "high": 30},
    "temperature":        {"low": 34.5,"high": 40.0},
    "spo2":               {"low": 88,  "high": None},
    "map":                {"low": 60,  "high": None},
}

WARNING_THRESHOLDS = {
    "heart_rate":         {"low": 50,  "high": 120},
    "sbp":                {"low": 90,  "high": 180},
    "respiratory_rate":   {"low": 8,   "high": 24},
    "temperature":        {"low": 35.5,"high": 39.0},
    "spo2":               {"low": 92,  "high": None},
    "map":                {"low": 65,  "high": None},
}


class RuleBasedAlertEngine:
    def __init__(self):
        self._alert_cooldown: Dict[str, datetime] = {}  # patient_id → last alert time
        self.cooldown_minutes = 10

    def _in_cooldown(self, patient_id: str, alert_key: str) -> bool:
        key = f"{patient_id}:{alert_key}"
        last = self._alert_cooldown.get(key)
        if last and (datetime.utcnow() - last).seconds < self.cooldown_minutes * 60:
            return True
        return False

    def _set_cooldown(self, patient_id: str, alert_key: str):
        key = f"{patient_id}:{alert_key}"
        self._alert_cooldown[key] = datetime.utcnow()

    def evaluate(self, state: PatientState, reading: VitalReading) -> List[Alert]:
        alerts = []
        vitals_dict = reading.to_dict()

        # Critical threshold checks
        for vital, thresholds in CRITICAL_THRESHOLDS.items():
            val = getattr(reading, vital, None)
            if val is None:
                continue
            lo = thresholds.get("low")
            hi = thresholds.get("high")
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                key = f"critical_{vital}"
                if not self._in_cooldown(reading.patient_id, key):
                    alerts.append(Alert(
                        patient_id=reading.patient_id,
                        timestamp=reading.timestamp,
                        alert_type="CRITICAL_VITAL",
                        severity="critical",
                        message=f"CRITICAL: {vital.replace('_', ' ').upper()} = {val:.1f} "
                                f"(limit: {lo}/{hi})",
                        vitals_snapshot=vitals_dict,
                        score=1.0,
                    ))
                    self._set_cooldown(reading.patient_id, key)

        # NEWS2
        news2 = compute_news2(reading)
        state.last_news2 = news2
        if news2 >= 7:
            key = "news2_critical"
            if not self._in_cooldown(reading.patient_id, key):
                alerts.append(Alert(
                    patient_id=reading.patient_id,
                    timestamp=reading.timestamp,
                    alert_type="DETERIORATION",
                    severity="critical" if news2 >= 9 else "high",
                    message=f"NEWS2 = {news2} — Immediate clinical review required",
                    vitals_snapshot=vitals_dict,
                    score=news2 / 20.0,
                ))
                self._set_cooldown(reading.patient_id, key)
        elif news2 >= 5:
            key = "news2_high"
            if not self._in_cooldown(reading.patient_id, key):
                alerts.append(Alert(
                    patient_id=reading.patient_id,
                    timestamp=reading.timestamp,
                    alert_type="DETERIORATION",
                    severity="moderate",
                    message=f"NEWS2 = {news2} — Urgent clinical review recommended",
                    vitals_snapshot=vitals_dict,
                    score=news2 / 20.0,
                ))
                self._set_cooldown(reading.patient_id, key)

        # Sepsis screen (qSOFA proxy)
        qsofa = 0
        if reading.sbp <= 100:
            qsofa += 1
        if reading.respiratory_rate >= 22:
            qsofa += 1
        if reading.gcs < 15:
            qsofa += 1
        if qsofa >= 2:
            key = "qsofa"
            if not self._in_cooldown(reading.patient_id, key):
                alerts.append(Alert(
                    patient_id=reading.patient_id,
                    timestamp=reading.timestamp,
                    alert_type="SEPSIS_ALERT",
                    severity="high",
                    message=f"qSOFA = {qsofa} — Sepsis screening positive",
                    vitals_snapshot=vitals_dict,
                    score=qsofa / 3.0,
                ))
                self._set_cooldown(reading.patient_id, key)

        # Shock index
        si = compute_shock_index(reading)
        if si > 1.0:
            key = "shock_index"
            if not self._in_cooldown(reading.patient_id, key):
                alerts.append(Alert(
                    patient_id=reading.patient_id,
                    timestamp=reading.timestamp,
                    alert_type="CRITICAL_VITAL",
                    severity="high",
                    message=f"Shock index = {si:.2f} (>1.0) — Hemodynamic compromise",
                    vitals_snapshot=vitals_dict,
                    score=min(si / 2.0, 1.0),
                ))
                self._set_cooldown(reading.patient_id, key)

        # Trend-based deterioration (need ≥6 readings)
        if len(state.readings) >= 6:
            recent = list(state.readings)[-6:]
            hr_trend = recent[-1].heart_rate - recent[0].heart_rate
            map_trend = recent[-1].map - recent[0].map
            spo2_trend = recent[-1].spo2 - recent[0].spo2

            if hr_trend > 30 and map_trend < -15 and spo2_trend < -5:
                key = "deterioration_trend"
                if not self._in_cooldown(reading.patient_id, key):
                    alerts.append(Alert(
                        patient_id=reading.patient_id,
                        timestamp=reading.timestamp,
                        alert_type="DETERIORATION",
                        severity="high",
                        message=f"Rapid deterioration trend: HR +{hr_trend:.0f}, "
                                f"MAP {map_trend:.0f}, SpO2 {spo2_trend:.0f}",
                        vitals_snapshot=vitals_dict,
                        score=0.85,
                    ))
                    self._set_cooldown(reading.patient_id, key)

        return alerts


# ─────────────────────────────────────────
# Synthetic stream generator
# ─────────────────────────────────────────

class PatientVitalsSimulator:
    """
    Generates realistic synthetic vital sign streams.
    Supports stable, mildly-ill, and deteriorating patient trajectories.
    """

    PROFILES = {
        "stable": dict(hr=80, sbp=120, dbp=75, rr=16, temp=37.0, spo2=98, gcs=15,
                       noise=dict(hr=5, sbp=8, dbp=5, rr=2, temp=0.2, spo2=0.5)),
        "mildly_ill": dict(hr=95, sbp=108, dbp=68, rr=19, temp=37.8, spo2=96, gcs=15,
                           noise=dict(hr=8, sbp=10, dbp=6, rr=3, temp=0.3, spo2=1.0)),
        "deteriorating": dict(hr=105, sbp=95, dbp=58, rr=22, temp=38.5, spo2=94, gcs=14,
                              noise=dict(hr=10, sbp=12, dbp=7, rr=4, temp=0.4, spo2=1.5),
                              drift=dict(hr=0.5, sbp=-0.4, rr=0.3, spo2=-0.2, map=-0.3)),
        "critical": dict(hr=130, sbp=82, dbp=52, rr=28, temp=39.2, spo2=88, gcs=12,
                         noise=dict(hr=12, sbp=15, dbp=8, rr=5, temp=0.5, spo2=2.0)),
    }

    def __init__(self, patient_id: str, profile: str = "stable"):
        self.patient_id = patient_id
        self.profile = profile
        p = self.PROFILES.get(profile, self.PROFILES["stable"])
        self.state = {k: v for k, v in p.items()
                      if k not in ("noise", "drift")}
        self.noise = p.get("noise", {})
        self.drift = p.get("drift", {})
        self.step = 0

    def _noisy(self, key: str, value: float) -> float:
        n = self.noise.get(key, 0)
        d = self.drift.get(key, 0) * self.step * 0.1
        jitter = random.gauss(0, n)
        return max(0.0, value + jitter + d)

    def next_reading(self) -> VitalReading:
        self.step += 1
        hr = round(self._noisy("hr", self.state["hr"]), 1)
        sbp = round(self._noisy("sbp", self.state["sbp"]), 1)
        dbp = round(min(sbp - 20, self._noisy("dbp", self.state["dbp"])), 1)
        rr = round(max(4, self._noisy("rr", self.state["rr"])), 1)
        temp = round(self._noisy("temp", self.state["temp"]), 2)
        spo2 = round(min(100, max(60, self._noisy("spo2", self.state["spo2"]))), 1)
        gcs = max(3, min(15, self.state["gcs"] + random.randint(-1, 0)
                         if self.profile == "deteriorating" else self.state["gcs"]))

        return VitalReading(
            patient_id=self.patient_id,
            timestamp=datetime.utcnow(),
            heart_rate=hr, sbp=sbp, dbp=dbp,
            respiratory_rate=rr, temperature=temp, spo2=spo2, gcs=gcs,
            source="monitor",
        )


# ─────────────────────────────────────────
# Stream Processor
# ─────────────────────────────────────────

class VitalsStreamProcessor:
    """
    Processes a real-time stream of vital readings.
    Maintains patient state, runs alert engine, calls registered callbacks.
    Thread-safe via queue.
    """

    def __init__(self):
        self.patient_states: Dict[str, PatientState] = {}
        self.alert_engine = RuleBasedAlertEngine()
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._all_alerts: List[Alert] = []
        self._lock = threading.Lock()

    def register_patient(self, patient_id: str, name: str = "", location: str = "ICU"):
        with self._lock:
            if patient_id not in self.patient_states:
                self.patient_states[patient_id] = PatientState(
                    patient_id=patient_id, name=name, location=location
                )

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        self._alert_handlers.append(handler)

    def ingest(self, reading: VitalReading):
        """Push a reading onto the processing queue (non-blocking)."""
        try:
            self._queue.put_nowait(reading)
        except queue.Full:
            logger.warning(f"Queue full — dropping reading for {reading.patient_id}")

    def _process_reading(self, reading: VitalReading):
        patient_id = reading.patient_id
        with self._lock:
            if patient_id not in self.patient_states:
                self.patient_states[patient_id] = PatientState(patient_id=patient_id)
            state = self.patient_states[patient_id]
            state.readings.append(reading)

            # Update baseline after 12 readings
            if len(state.readings) == 12:
                hrs = [r.heart_rate for r in state.readings]
                maps = [r.map for r in state.readings]
                state.baseline_hr = sum(hrs) / len(hrs)
                state.baseline_map = sum(maps) / len(maps)

        # Run alert engine (outside lock for speed)
        alerts = self.alert_engine.evaluate(state, reading)
        for alert in alerts:
            with self._lock:
                state.alerts.append(alert)
                self._all_alerts.append(alert)
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

    def _worker(self):
        while self._running:
            try:
                reading = self._queue.get(timeout=1.0)
                self._process_reading(reading)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("VitalsStreamProcessor started")

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("VitalsStreamProcessor stopped")

    def get_patient_summary(self, patient_id: str) -> Optional[Dict]:
        state = self.patient_states.get(patient_id)
        if not state or not state.readings:
            return None
        latest = state.readings[-1]
        news2 = compute_news2(latest)
        sofa = compute_sofa_proxy(latest)
        return {
            "patient_id": patient_id,
            "location": state.location,
            "latest_vitals": latest.to_dict(),
            "news2": news2,
            "sofa_proxy": sofa,
            "shock_index": compute_shock_index(latest),
            "active_alerts": len([a for a in state.alerts if not a.acknowledged]),
            "total_readings": len(state.readings),
            "is_deteriorating": news2 >= 5,
        }

    def get_icu_overview(self) -> List[Dict]:
        summaries = []
        for pid in self.patient_states:
            s = self.get_patient_summary(pid)
            if s:
                summaries.append(s)
        return sorted(summaries, key=lambda x: -x["news2"])

    def get_recent_alerts(self, n: int = 20) -> List[Dict]:
        with self._lock:
            return [a.to_dict() for a in self._all_alerts[-n:]]


# ─────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────

def run_demo(duration_seconds: int = 30, interval: float = 1.0):
    """Simulate 5 ICU patients for a short period, print alerts."""
    logging.basicConfig(level=logging.WARNING)

    processor = VitalsStreamProcessor()

    patients = [
        ("P001", "Alice Chen",    "ICU-1", "stable"),
        ("P002", "Bob Martinez",  "ICU-2", "mildly_ill"),
        ("P003", "Carol Singh",   "ICU-3", "deteriorating"),
        ("P004", "David Kim",     "ICU-4", "critical"),
        ("P005", "Eva Johnson",   "ICU-5", "stable"),
    ]

    simulators = {}
    for pid, name, loc, profile in patients:
        processor.register_patient(pid, name, loc)
        simulators[pid] = PatientVitalsSimulator(pid, profile)

    alert_log = []

    def on_alert(alert: Alert):
        alert_log.append(alert)
        print(f"🚨 [{alert.severity.upper()}] {alert.patient_id} — {alert.message}")

    processor.add_alert_handler(on_alert)
    processor.start()

    print(f"Streaming vitals for {len(patients)} ICU patients ({duration_seconds}s)...\n")
    start = time.time()
    step = 0

    while time.time() - start < duration_seconds:
        for pid, sim in simulators.items():
            reading = sim.next_reading()
            processor.ingest(reading)
        step += 1
        time.sleep(interval)

    time.sleep(2)  # flush queue
    processor.stop()

    print("\n=== ICU Overview ===")
    for s in processor.get_icu_overview():
        v = s["latest_vitals"]
        print(f"  {s['patient_id']} | NEWS2={s['news2']} | "
              f"HR={v['heart_rate']:.0f} BP={v['sbp']:.0f}/{v['dbp']:.0f} "
              f"SpO2={v['spo2']:.0f}% | Alerts={s['active_alerts']}")

    print(f"\nTotal alerts fired: {len(alert_log)}")
    return processor


if __name__ == "__main__":
    run_demo(duration_seconds=20, interval=0.5)

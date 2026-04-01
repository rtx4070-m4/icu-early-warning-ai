"""
Clinical Workflow State Machine
Models ICU care pathways as state machines with transitions triggered by
clinical events, AI alerts, and time-based rules.
Tracks protocol adherence and generates audit-ready workflow logs.

States: ADMISSION → MONITORING → ALERT → ASSESSMENT → INTERVENTION →
        STABILIZATION → RECOVERY | ESCALATION | DISCHARGE | DECEASED
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# States and events
# ─────────────────────────────────────────

class ClinicalState(str, Enum):
    ADMISSION       = "ADMISSION"
    MONITORING      = "MONITORING"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    ASSESSMENT      = "ASSESSMENT"
    INTERVENTION    = "INTERVENTION"
    STABILIZATION   = "STABILIZATION"
    RECOVERY        = "RECOVERY"
    ESCALATION      = "ESCALATION"
    DISCHARGE       = "DISCHARGE"
    DECEASED        = "DECEASED"


class ClinicalEvent(str, Enum):
    ADMIT               = "ADMIT"
    VITALS_STABLE       = "VITALS_STABLE"
    ALERT_FIRED         = "ALERT_FIRED"
    ALERT_ACKNOWLEDGED  = "ALERT_ACKNOWLEDGED"
    ASSESSMENT_COMPLETE = "ASSESSMENT_COMPLETE"
    INTERVENTION_GIVEN  = "INTERVENTION_GIVEN"
    IMPROVEMENT_NOTED   = "IMPROVEMENT_NOTED"
    DETERIORATION       = "DETERIORATION"
    CRITICAL_THRESHOLD  = "CRITICAL_THRESHOLD"
    STABILIZED          = "STABILIZED"
    CLEARED_FOR_STEP_DOWN = "CLEARED_FOR_STEP_DOWN"
    DISCHARGE_ORDER     = "DISCHARGE_ORDER"
    DEATH               = "DEATH"
    TIMEOUT             = "TIMEOUT"          # no action taken within window


# ─────────────────────────────────────────
# Transition table
# ─────────────────────────────────────────

# (current_state, event) → (next_state, max_response_minutes)
# max_response_minutes: how long the team has to trigger the next transition before TIMEOUT fires
TRANSITIONS: Dict[Tuple[ClinicalState, ClinicalEvent], Tuple[ClinicalState, Optional[int]]] = {
    # Admission flow
    (ClinicalState.ADMISSION,       ClinicalEvent.ADMIT):               (ClinicalState.MONITORING,      None),
    # Monitoring
    (ClinicalState.MONITORING,      ClinicalEvent.ALERT_FIRED):         (ClinicalState.ALERT_TRIGGERED, 10),
    (ClinicalState.MONITORING,      ClinicalEvent.VITALS_STABLE):       (ClinicalState.MONITORING,      None),
    (ClinicalState.MONITORING,      ClinicalEvent.DISCHARGE_ORDER):     (ClinicalState.DISCHARGE,       None),
    (ClinicalState.MONITORING,      ClinicalEvent.DEATH):               (ClinicalState.DECEASED,        None),
    # Alert triggered
    (ClinicalState.ALERT_TRIGGERED, ClinicalEvent.ALERT_ACKNOWLEDGED):  (ClinicalState.ASSESSMENT,      30),
    (ClinicalState.ALERT_TRIGGERED, ClinicalEvent.TIMEOUT):             (ClinicalState.ESCALATION,      None),
    (ClinicalState.ALERT_TRIGGERED, ClinicalEvent.CRITICAL_THRESHOLD):  (ClinicalState.ESCALATION,      None),
    (ClinicalState.ALERT_TRIGGERED, ClinicalEvent.DEATH):               (ClinicalState.DECEASED,        None),
    # Assessment
    (ClinicalState.ASSESSMENT,      ClinicalEvent.ASSESSMENT_COMPLETE): (ClinicalState.INTERVENTION,    60),
    (ClinicalState.ASSESSMENT,      ClinicalEvent.DETERIORATION):       (ClinicalState.ESCALATION,      None),
    (ClinicalState.ASSESSMENT,      ClinicalEvent.TIMEOUT):             (ClinicalState.ESCALATION,      None),
    (ClinicalState.ASSESSMENT,      ClinicalEvent.DEATH):               (ClinicalState.DECEASED,        None),
    # Intervention
    (ClinicalState.INTERVENTION,    ClinicalEvent.INTERVENTION_GIVEN):  (ClinicalState.STABILIZATION,   None),
    (ClinicalState.INTERVENTION,    ClinicalEvent.DETERIORATION):       (ClinicalState.ESCALATION,      None),
    (ClinicalState.INTERVENTION,    ClinicalEvent.TIMEOUT):             (ClinicalState.ESCALATION,      None),
    (ClinicalState.INTERVENTION,    ClinicalEvent.DEATH):               (ClinicalState.DECEASED,        None),
    # Stabilization
    (ClinicalState.STABILIZATION,   ClinicalEvent.STABILIZED):          (ClinicalState.MONITORING,      None),
    (ClinicalState.STABILIZATION,   ClinicalEvent.IMPROVEMENT_NOTED):   (ClinicalState.RECOVERY,        None),
    (ClinicalState.STABILIZATION,   ClinicalEvent.DETERIORATION):       (ClinicalState.ESCALATION,      None),
    (ClinicalState.STABILIZATION,   ClinicalEvent.ALERT_FIRED):         (ClinicalState.ALERT_TRIGGERED, 10),
    # Recovery
    (ClinicalState.RECOVERY,        ClinicalEvent.CLEARED_FOR_STEP_DOWN): (ClinicalState.DISCHARGE,     None),
    (ClinicalState.RECOVERY,        ClinicalEvent.DETERIORATION):       (ClinicalState.ALERT_TRIGGERED, 10),
    (ClinicalState.RECOVERY,        ClinicalEvent.DISCHARGE_ORDER):     (ClinicalState.DISCHARGE,       None),
    # Escalation
    (ClinicalState.ESCALATION,      ClinicalEvent.INTERVENTION_GIVEN):  (ClinicalState.STABILIZATION,   None),
    (ClinicalState.ESCALATION,      ClinicalEvent.STABILIZED):          (ClinicalState.MONITORING,      None),
    (ClinicalState.ESCALATION,      ClinicalEvent.DEATH):               (ClinicalState.DECEASED,        None),
}

# Protocol time standards (minutes)
RESPONSE_STANDARDS = {
    "alert_to_acknowledgement":    15,   # ALERT_TRIGGERED → ASSESSMENT
    "assessment_to_intervention":  60,   # ASSESSMENT → INTERVENTION
    "sepsis_bundle_completion":    60,   # Alert to first antibiotic
    "vasopressor_initiation":      30,   # Hypotension alert to norepinephrine
    "escalation_response":         10,   # ESCALATION → attending at bedside
}


# ─────────────────────────────────────────
# Workflow log entry
# ─────────────────────────────────────────

@dataclass
class WorkflowLogEntry:
    timestamp: str
    patient_id: str
    from_state: str
    event: str
    to_state: str
    triggered_by: str        # "clinician" | "ai_alert" | "timeout" | "system"
    clinician_id: Optional[str]
    duration_in_state_minutes: float
    protocol_compliant: bool
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "from_state": self.from_state,
            "event": self.event,
            "to_state": self.to_state,
            "triggered_by": self.triggered_by,
            "clinician_id": self.clinician_id,
            "duration_in_state_minutes": round(self.duration_in_state_minutes, 1),
            "protocol_compliant": self.protocol_compliant,
            "notes": self.notes,
        }


# ─────────────────────────────────────────
# Patient workflow instance
# ─────────────────────────────────────────

@dataclass
class PatientWorkflow:
    patient_id: str
    current_state: ClinicalState = ClinicalState.ADMISSION
    state_entered_at: datetime = field(default_factory=datetime.utcnow)
    max_response_deadline: Optional[datetime] = None
    log: List[WorkflowLogEntry] = field(default_factory=list)
    protocol_adherence_events: List[bool] = field(default_factory=list)
    active_alerts: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    escalations: int = 0
    timeouts: int = 0

    @property
    def protocol_adherence_rate(self) -> float:
        if not self.protocol_adherence_events:
            return 1.0
        return sum(self.protocol_adherence_events) / len(self.protocol_adherence_events)

    @property
    def time_in_current_state_minutes(self) -> float:
        return (datetime.utcnow() - self.state_entered_at).total_seconds() / 60

    @property
    def is_timeout_breached(self) -> bool:
        if self.max_response_deadline is None:
            return False
        return datetime.utcnow() > self.max_response_deadline

    def summary(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "current_state": self.current_state.value,
            "time_in_state_minutes": round(self.time_in_current_state_minutes, 1),
            "protocol_adherence": round(self.protocol_adherence_rate, 3),
            "escalations": self.escalations,
            "timeouts": self.timeouts,
            "active_alerts": len(self.active_alerts),
            "n_interventions": len(self.interventions),
            "log_entries": len(self.log),
            "timeout_breached": self.is_timeout_breached,
        }


# ─────────────────────────────────────────
# State machine engine
# ─────────────────────────────────────────

class ClinicalWorkflowEngine:
    """
    Manages clinical workflow state machines for all active patients.
    Thread-safe for concurrent updates from monitoring and alert systems.
    """

    def __init__(self):
        self._workflows: Dict[str, PatientWorkflow] = {}
        self._state_handlers: Dict[ClinicalState, List[Callable]] = {}
        self._transition_hooks: List[Callable] = []

    # ── Registration ──────────────────────

    def admit_patient(self, patient_id: str) -> PatientWorkflow:
        """Create a new workflow for an admitted patient."""
        wf = PatientWorkflow(patient_id=patient_id)
        self._workflows[patient_id] = wf
        self._fire_event(patient_id, ClinicalEvent.ADMIT, triggered_by="system")
        logger.info(f"Patient admitted: {patient_id}")
        return wf

    def register_state_handler(self, state: ClinicalState, handler: Callable):
        """Register a callback to fire whenever a patient enters a state."""
        self._state_handlers.setdefault(state, []).append(handler)

    def register_transition_hook(self, hook: Callable):
        """Register a callback fired on every state transition."""
        self._transition_hooks.append(hook)

    # ── Event firing ──────────────────────

    def _fire_event(self, patient_id: str, event: ClinicalEvent,
                     triggered_by: str = "clinician",
                     clinician_id: Optional[str] = None,
                     notes: str = "") -> Optional[ClinicalState]:
        wf = self._workflows.get(patient_id)
        if not wf:
            logger.warning(f"No workflow for patient {patient_id}")
            return None

        key = (wf.current_state, event)
        if key not in TRANSITIONS:
            logger.debug(f"No transition: {wf.current_state} + {event}")
            return wf.current_state

        next_state, response_window = TRANSITIONS[key]
        duration = wf.time_in_current_state_minutes

        # Protocol compliance check
        compliant = True
        if event == ClinicalEvent.ALERT_ACKNOWLEDGED:
            compliant = duration <= RESPONSE_STANDARDS["alert_to_acknowledgement"]
        elif event == ClinicalEvent.ASSESSMENT_COMPLETE:
            compliant = duration <= RESPONSE_STANDARDS["assessment_to_intervention"]
        elif event == ClinicalEvent.TIMEOUT:
            compliant = False
            wf.timeouts += 1
        if next_state == ClinicalState.ESCALATION:
            wf.escalations += 1

        wf.protocol_adherence_events.append(compliant)

        # Log the transition
        entry = WorkflowLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            patient_id=patient_id,
            from_state=wf.current_state.value,
            event=event.value,
            to_state=next_state.value,
            triggered_by=triggered_by,
            clinician_id=clinician_id,
            duration_in_state_minutes=duration,
            protocol_compliant=compliant,
            notes=notes,
        )
        wf.log.append(entry)

        # Advance state
        prev_state = wf.current_state
        wf.current_state = next_state
        wf.state_entered_at = datetime.utcnow()

        if response_window is not None:
            wf.max_response_deadline = datetime.utcnow() + timedelta(minutes=response_window)
        else:
            wf.max_response_deadline = None

        logger.info(f"[{patient_id}] {prev_state.value} → {next_state.value} "
                    f"via {event.value} ({'✓' if compliant else '✗'})")

        # Fire state handlers
        for handler in self._state_handlers.get(next_state, []):
            try:
                handler(patient_id, wf)
            except Exception as e:
                logger.error(f"State handler error: {e}")

        # Fire transition hooks
        for hook in self._transition_hooks:
            try:
                hook(patient_id, prev_state, event, next_state, wf)
            except Exception as e:
                logger.error(f"Transition hook error: {e}")

        return next_state

    # ── Public API ────────────────────────

    def alert_fired(self, patient_id: str, alert_id: str, severity: str,
                     notes: str = "") -> Optional[ClinicalState]:
        wf = self._workflows.get(patient_id)
        if wf:
            wf.active_alerts.append(alert_id)
        event = (ClinicalEvent.CRITICAL_THRESHOLD
                  if severity == "critical" else ClinicalEvent.ALERT_FIRED)
        return self._fire_event(patient_id, event, triggered_by="ai_alert", notes=notes)

    def acknowledge_alert(self, patient_id: str, clinician_id: str,
                           alert_id: str = "") -> Optional[ClinicalState]:
        wf = self._workflows.get(patient_id)
        if wf and alert_id in wf.active_alerts:
            wf.active_alerts.remove(alert_id)
        return self._fire_event(patient_id, ClinicalEvent.ALERT_ACKNOWLEDGED,
                                 triggered_by="clinician",
                                 clinician_id=clinician_id)

    def complete_assessment(self, patient_id: str, clinician_id: str,
                             notes: str = "") -> Optional[ClinicalState]:
        return self._fire_event(patient_id, ClinicalEvent.ASSESSMENT_COMPLETE,
                                 "clinician", clinician_id, notes)

    def record_intervention(self, patient_id: str, intervention: str,
                             clinician_id: str) -> Optional[ClinicalState]:
        wf = self._workflows.get(patient_id)
        if wf:
            wf.interventions.append(intervention)
        return self._fire_event(patient_id, ClinicalEvent.INTERVENTION_GIVEN,
                                 "clinician", clinician_id,
                                 notes=f"Intervention: {intervention}")

    def record_improvement(self, patient_id: str, clinician_id: str) -> Optional[ClinicalState]:
        return self._fire_event(patient_id, ClinicalEvent.IMPROVEMENT_NOTED,
                                 "clinician", clinician_id)

    def record_stabilization(self, patient_id: str, clinician_id: str) -> Optional[ClinicalState]:
        return self._fire_event(patient_id, ClinicalEvent.STABILIZED,
                                 "clinician", clinician_id)

    def discharge(self, patient_id: str, clinician_id: str) -> Optional[ClinicalState]:
        return self._fire_event(patient_id, ClinicalEvent.DISCHARGE_ORDER,
                                 "clinician", clinician_id)

    def record_death(self, patient_id: str, notes: str = "") -> Optional[ClinicalState]:
        return self._fire_event(patient_id, ClinicalEvent.DEATH,
                                 "system", notes=notes)

    def check_timeouts(self) -> List[str]:
        """Scan all workflows for timeout breaches and fire timeout events."""
        timed_out = []
        for pid, wf in self._workflows.items():
            if wf.is_timeout_breached:
                self._fire_event(pid, ClinicalEvent.TIMEOUT,
                                  triggered_by="system",
                                  notes="Automatic timeout — response window exceeded")
                timed_out.append(pid)
        return timed_out

    # ── Reporting ─────────────────────────

    def get_patient_summary(self, patient_id: str) -> Optional[Dict]:
        wf = self._workflows.get(patient_id)
        return wf.summary() if wf else None

    def get_ward_summary(self) -> Dict:
        summaries = [wf.summary() for wf in self._workflows.values()]
        state_counts: Dict[str, int] = {}
        for s in summaries:
            state_counts[s["current_state"]] = state_counts.get(s["current_state"], 0) + 1

        adherence_rates = [s["protocol_adherence"] for s in summaries]
        avg_adherence = sum(adherence_rates) / len(adherence_rates) if adherence_rates else 1.0

        return {
            "total_patients": len(self._workflows),
            "state_distribution": state_counts,
            "avg_protocol_adherence": round(avg_adherence, 3),
            "patients_with_alerts": sum(1 for s in summaries if s["active_alerts"] > 0),
            "timeout_breached": sum(1 for s in summaries if s["timeout_breached"]),
            "escalations_total": sum(s["escalations"] for s in summaries),
            "patients": summaries,
        }

    def get_patient_log(self, patient_id: str) -> List[Dict]:
        wf = self._workflows.get(patient_id)
        return [e.to_dict() for e in wf.log] if wf else []

    def protocol_adherence_report(self) -> Dict:
        """Compute protocol adherence metrics across all patients."""
        all_events: List[bool] = []
        per_transition: Dict[str, List[bool]] = {}

        for wf in self._workflows.values():
            all_events.extend(wf.protocol_adherence_events)
            for entry in wf.log:
                key = f"{entry.from_state}→{entry.to_state}"
                per_transition.setdefault(key, []).append(entry.protocol_compliant)

        overall = sum(all_events) / len(all_events) if all_events else 1.0
        by_transition = {k: round(sum(v) / len(v), 3)
                          for k, v in per_transition.items() if v}

        return {
            "overall_adherence": round(overall, 3),
            "n_events": len(all_events),
            "by_transition": by_transition,
            "grade": "A" if overall >= 0.90 else "B" if overall >= 0.75 else "C",
        }


# ─────────────────────────────────────────
# Demo simulation
# ─────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    engine = ClinicalWorkflowEngine()

    # Register a state handler
    def on_escalation(patient_id: str, wf: PatientWorkflow):
        print(f"  🚨 ESCALATION: {patient_id} — attending physician paged!")

    engine.register_state_handler(ClinicalState.ESCALATION, on_escalation)

    # Simulate two patients
    print("=== Clinical Workflow Simulation ===\n")

    # Patient 1: textbook sepsis bundle — compliant
    print("Patient P001 — Septic Shock (protocol compliant):")
    engine.admit_patient("P001")
    _time.sleep(0.05)
    engine.alert_fired("P001", "ALT-001", "high", "NEWS2=9")
    _time.sleep(0.05)
    engine.acknowledge_alert("P001", "dr_smith", "ALT-001")
    _time.sleep(0.05)
    engine.complete_assessment("P001", "dr_smith", "Septic shock — lactate 3.8")
    _time.sleep(0.05)
    engine.record_intervention("P001", "Vancomycin 25mg/kg IV + pip-tazo 4.5g IV", "dr_smith")
    _time.sleep(0.05)
    engine.record_stabilization("P001", "dr_smith")
    _time.sleep(0.05)
    engine.record_improvement("P001", "dr_smith")
    _time.sleep(0.05)
    engine.discharge("P001", "dr_smith")

    # Patient 2: no response — escalation
    print("\nPatient P002 — Unacknowledged Alert (timeout → escalation):")
    engine.admit_patient("P002")
    _time.sleep(0.05)
    engine.alert_fired("P002", "ALT-002", "critical", "MAP=58, lactate=5.2")
    _time.sleep(0.05)
    # Simulate timeout
    wf2 = engine._workflows["P002"]
    wf2.state_entered_at -= timedelta(minutes=20)  # fast-forward
    timed_out = engine.check_timeouts()
    print(f"  Timed out patients: {timed_out}")
    _time.sleep(0.05)
    engine.record_intervention("P002", "Norepinephrine 0.2 mcg/kg/min", "dr_jones")
    _time.sleep(0.05)
    engine.record_stabilization("P002", "dr_jones")

    # Ward summary
    print("\n=== Ward Summary ===")
    summary = engine.get_ward_summary()
    print(f"Total patients: {summary['total_patients']}")
    print(f"State distribution: {summary['state_distribution']}")
    print(f"Protocol adherence: {summary['avg_protocol_adherence']:.0%}")

    # Adherence report
    print("\n=== Protocol Adherence Report ===")
    report = engine.protocol_adherence_report()
    print(f"Overall: {report['overall_adherence']:.0%} (Grade: {report['grade']})")
    for transition, rate in sorted(report["by_transition"].items()):
        flag = "⚠" if rate < 0.80 else "✓"
        print(f"  {flag} {transition:<45} {rate:.0%}")

    # P001 audit log
    print(f"\n=== P001 Audit Log ({len(engine.get_patient_log('P001'))} entries) ===")
    for entry in engine.get_patient_log("P001"):
        flag = "✓" if entry["protocol_compliant"] else "✗"
        print(f"  {flag} {entry['from_state']:20s} → {entry['to_state']:20s} "
              f"via {entry['event']} ({entry['duration_in_state_minutes']:.0f}min)")

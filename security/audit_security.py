"""
Security & Audit Logging Module
HIPAA-aligned audit trail, role-based access control, PHI de-identification,
and session management for the AI Hospital OS.
"""

import os
import json
import uuid
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Roles and permissions
# ─────────────────────────────────────────

class Permission:
    VIEW_PATIENT         = "view_patient"
    EDIT_PATIENT         = "edit_patient"
    VIEW_VITALS          = "view_vitals"
    INGEST_VITALS        = "ingest_vitals"
    VIEW_LABS            = "view_labs"
    VIEW_NOTES           = "view_notes"
    WRITE_NOTES          = "write_notes"
    VIEW_MEDICATIONS     = "view_medications"
    PRESCRIBE            = "prescribe"
    VIEW_RISK_SCORES     = "view_risk_scores"
    ACKNOWLEDGE_ALERTS   = "acknowledge_alerts"
    VIEW_DASHBOARD       = "view_dashboard"
    RUN_PIPELINE         = "run_pipeline"
    MANAGE_MODELS        = "manage_models"
    VIEW_AUDIT_LOG       = "view_audit_log"
    EXPORT_DATA          = "export_data"
    ADMIN                = "admin"

ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "attending_physician": {
        Permission.VIEW_PATIENT, Permission.EDIT_PATIENT,
        Permission.VIEW_VITALS, Permission.VIEW_LABS,
        Permission.VIEW_NOTES, Permission.WRITE_NOTES,
        Permission.VIEW_MEDICATIONS, Permission.PRESCRIBE,
        Permission.VIEW_RISK_SCORES, Permission.ACKNOWLEDGE_ALERTS,
        Permission.VIEW_DASHBOARD, Permission.EXPORT_DATA,
    },
    "resident": {
        Permission.VIEW_PATIENT,
        Permission.VIEW_VITALS, Permission.VIEW_LABS,
        Permission.VIEW_NOTES, Permission.WRITE_NOTES,
        Permission.VIEW_MEDICATIONS, Permission.VIEW_RISK_SCORES,
        Permission.ACKNOWLEDGE_ALERTS, Permission.VIEW_DASHBOARD,
    },
    "nurse": {
        Permission.VIEW_PATIENT,
        Permission.VIEW_VITALS, Permission.INGEST_VITALS,
        Permission.VIEW_LABS, Permission.VIEW_MEDICATIONS,
        Permission.VIEW_NOTES, Permission.ACKNOWLEDGE_ALERTS,
        Permission.VIEW_DASHBOARD, Permission.VIEW_RISK_SCORES,
    },
    "data_scientist": {
        Permission.VIEW_PATIENT, Permission.VIEW_VITALS,
        Permission.VIEW_LABS, Permission.VIEW_RISK_SCORES,
        Permission.RUN_PIPELINE, Permission.MANAGE_MODELS,
        Permission.VIEW_DASHBOARD,
    },
    "admin": set(),   # filled below
    "auditor": {
        Permission.VIEW_AUDIT_LOG, Permission.VIEW_DASHBOARD,
    },
}

# Populate admin role with all permissions
ROLE_PERMISSIONS["admin"] = {
    v for k, v in vars(Permission).items()
    if not k.startswith("_") and isinstance(v, str)
}


# ─────────────────────────────────────────
# Audit event
# ─────────────────────────────────────────

@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    user_id: str
    user_role: str
    action: str          # ACCESS | MODIFY | DELETE | LOGIN | LOGOUT | EXPORT | ALERT_ACK
    resource_type: str   # PATIENT | VITALS | LAB | NOTE | ALERT | REPORT | MODEL
    resource_id: str
    patient_id: Optional[str]
    ip_address: str
    success: bool
    reason: Optional[str] = None
    details: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def to_log_line(self) -> str:
        status = "OK" if self.success else "DENIED"
        return (f"{self.timestamp} [{status}] user={self.user_id} role={self.user_role} "
                f"action={self.action} resource={self.resource_type}:{self.resource_id} "
                f"patient={self.patient_id or '-'} ip={self.ip_address}"
                + (f" reason={self.reason}" if self.reason else ""))


# ─────────────────────────────────────────
# User session
# ─────────────────────────────────────────

@dataclass
class UserSession:
    session_id: str
    user_id: str
    role: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    permissions: Set[str]
    is_active: bool = True

    def is_expired(self, timeout_minutes: int = 480) -> bool:
        return (datetime.utcnow() - self.last_activity).total_seconds() > timeout_minutes * 60

    def touch(self):
        self.last_activity = datetime.utcnow()

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


# ─────────────────────────────────────────
# Audit logger
# ─────────────────────────────────────────

class AuditLogger:
    """Thread-safe, file-backed audit trail with in-memory buffer."""

    def __init__(self, log_dir: str = "/data/audit", buffer_size: int = 500):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._buffer_size = buffer_size
        self._current_log_path = self._get_log_path()

    def _get_log_path(self) -> Path:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _flush(self):
        """Write buffer to disk."""
        path = self._get_log_path()
        with open(path, "a") as f:
            for event in self._buffer:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        self._buffer.clear()

    def log(self, event: AuditEvent):
        logger.debug(event.to_log_line())
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                self._flush()

    def flush(self):
        with self._lock:
            if self._buffer:
                self._flush()

    def query(self, patient_id: Optional[str] = None,
               user_id: Optional[str] = None,
               action: Optional[str] = None,
               start: Optional[datetime] = None,
               end: Optional[datetime] = None,
               limit: int = 100) -> List[Dict]:
        """Search audit logs."""
        results = []

        # Search recent buffer first
        with self._lock:
            buffer_copy = list(self._buffer)

        for event in reversed(buffer_copy):
            if patient_id and event.patient_id != patient_id:
                continue
            if user_id and event.user_id != user_id:
                continue
            if action and event.action != action:
                continue
            results.append(event.to_dict())
            if len(results) >= limit:
                return results

        # Search log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files[:7]:  # last 7 days
            try:
                with open(log_file) as f:
                    for line in reversed(f.readlines()):
                        try:
                            ev = json.loads(line)
                            if patient_id and ev.get("patient_id") != patient_id:
                                continue
                            if user_id and ev.get("user_id") != user_id:
                                continue
                            if action and ev.get("action") != action:
                                continue
                            results.append(ev)
                            if len(results) >= limit:
                                return results
                        except Exception:
                            continue
            except FileNotFoundError:
                continue

        return results


# ─────────────────────────────────────────
# Session manager
# ─────────────────────────────────────────

class SessionManager:
    def __init__(self, session_timeout_minutes: int = 480):
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()
        self.timeout_minutes = session_timeout_minutes

    def create_session(self, user_id: str, role: str, ip: str) -> UserSession:
        permissions = ROLE_PERMISSIONS.get(role, set())
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip,
            permissions=permissions,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            if not session.is_active or session.is_expired(self.timeout_minutes):
                session.is_active = False
                return None
            session.touch()
            return session

    def invalidate_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].is_active = False

    def active_sessions(self) -> List[Dict]:
        with self._lock:
            return [
                {"session_id": s.session_id[:8] + "...", "user_id": s.user_id,
                 "role": s.role, "ip": s.ip_address,
                 "last_activity": s.last_activity.isoformat()}
                for s in self._sessions.values()
                if s.is_active and not s.is_expired(self.timeout_minutes)
            ]


# ─────────────────────────────────────────
# Access control decorator / guard
# ─────────────────────────────────────────

class AccessControl:
    """Enforce RBAC on hospital OS operations."""

    def __init__(self, audit_logger: AuditLogger, session_manager: SessionManager):
        self.audit = audit_logger
        self.sessions = session_manager

    def _make_event(self, session: Optional[UserSession], action: str,
                    resource_type: str, resource_id: str,
                    patient_id: Optional[str], success: bool,
                    reason: Optional[str] = None, details: Dict = None) -> AuditEvent:
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            user_id=session.user_id if session else "anonymous",
            user_role=session.role if session else "none",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            ip_address=session.ip_address if session else "unknown",
            success=success,
            reason=reason,
            details=details or {},
        )

    def check_access(self, session_id: str, permission: str,
                     resource_type: str, resource_id: str,
                     patient_id: Optional[str] = None) -> bool:
        """Check and log access attempt."""
        session = self.sessions.get_session(session_id)

        if not session:
            self.audit.log(self._make_event(
                None, "ACCESS", resource_type, resource_id, patient_id,
                success=False, reason="Invalid or expired session"
            ))
            return False

        if not session.has_permission(permission):
            self.audit.log(self._make_event(
                session, "ACCESS", resource_type, resource_id, patient_id,
                success=False,
                reason=f"Permission denied: {permission} not in role {session.role}"
            ))
            return False

        self.audit.log(self._make_event(
            session, "ACCESS", resource_type, resource_id, patient_id, success=True
        ))
        return True

    def require(self, permission: str, resource_type: str = "SYSTEM",
                resource_id: str = "*") -> Callable:
        """Decorator for access-controlled functions."""
        def decorator(fn: Callable) -> Callable:
            def wrapper(session_id: str, *args, **kwargs):
                patient_id = kwargs.get("patient_id") or (args[0] if args else None)
                if not self.check_access(session_id, permission,
                                          resource_type, resource_id, patient_id):
                    raise PermissionError(
                        f"Access denied: {permission} required for {resource_type}:{resource_id}"
                    )
                return fn(session_id, *args, **kwargs)
            return wrapper
        return decorator


# ─────────────────────────────────────────
# PHI de-identification
# ─────────────────────────────────────────

class PHIDeidentifier:
    """
    De-identify protected health information from clinical text.
    Replaces names, MRNs, dates, phone numbers, addresses, and SSNs
    with synthetic placeholders.
    """

    PATTERNS = [
        # MRN
        (re.compile(r'\bMRN[-:\s]*\d{5,10}\b', re.IGNORECASE), "[MRN]"),
        # SSN
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
        # Phone
        (re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'), "[PHONE]"),
        # Date of birth
        (re.compile(r'\b(?:dob|date of birth|born)[-:\s]*\d{1,2}/\d{1,2}/\d{2,4}\b',
                    re.IGNORECASE), "[DOB]"),
        # Specific dates (preserve relative dates like "3 days ago")
        (re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'), "[DATE]"),
        (re.compile(r'\b(?:January|February|March|April|May|June|July|August|'
                    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                    re.IGNORECASE), "[DATE]"),
        # ZIP codes
        (re.compile(r'\b\d{5}(?:-\d{4})?\b'), "[ZIP]"),
        # IP addresses (patient portal)
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), "[IP]"),
        # Email
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL]"),
    ]

    # Common name patterns (first + last name after Dr/Mr/Ms/Mrs)
    NAME_PATTERNS = [
        re.compile(r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',
                   re.IGNORECASE),
    ]

    def deidentify(self, text: str) -> str:
        """Apply all de-identification patterns."""
        result = text
        for pattern, replacement in self.PATTERNS:
            result = pattern.sub(replacement, result)
        for pattern in self.NAME_PATTERNS:
            result = pattern.sub("[NAME]", result)
        return result

    def deidentify_patient_record(self, record: Dict) -> Dict:
        """De-identify a patient record dict (redacts key fields)."""
        safe = dict(record)
        # Redact direct identifiers
        for field in ("name", "date_of_birth", "ssn", "mrn", "address",
                      "phone", "email", "insurance_id"):
            if field in safe:
                safe[field] = f"[REDACTED-{field.upper()}]"
        # De-identify any text fields
        for field in ("notes", "clinical_note", "summary", "chief_complaint"):
            if field in safe and isinstance(safe[field], str):
                safe[field] = self.deidentify(safe[field])
        safe["_deidentified"] = True
        safe["_deidentified_at"] = datetime.utcnow().isoformat()
        return safe

    def hash_patient_id(self, patient_id: str, salt: str = "") -> str:
        """One-way hash a patient ID for research use."""
        return hashlib.sha256(f"{salt}{patient_id}".encode()).hexdigest()[:16]


# ─────────────────────────────────────────
# Security manager (facade)
# ─────────────────────────────────────────

class HospitalSecurityManager:
    """
    Unified security facade: sessions, RBAC, audit logging, de-identification.
    """

    def __init__(self, audit_dir: str = "/data/audit"):
        self.audit_logger = AuditLogger(log_dir=audit_dir)
        self.sessions = SessionManager()
        self.access_control = AccessControl(self.audit_logger, self.sessions)
        self.deidentifier = PHIDeidentifier()
        logger.info("HospitalSecurityManager initialised")

    def login(self, user_id: str, role: str, ip: str = "127.0.0.1") -> str:
        """Create a session and log the login event."""
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role: {role}")
        session = self.sessions.create_session(user_id, role, ip)
        self.audit_logger.log(AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id, user_role=role,
            action="LOGIN", resource_type="SYSTEM", resource_id="auth",
            patient_id=None, ip_address=ip, success=True,
        ))
        logger.info(f"Login: {user_id} ({role}) from {ip}")
        return session.session_id

    def logout(self, session_id: str):
        session = self.sessions.get_session(session_id)
        if session:
            self.sessions.invalidate_session(session_id)
            self.audit_logger.log(AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                user_id=session.user_id, user_role=session.role,
                action="LOGOUT", resource_type="SYSTEM", resource_id="auth",
                patient_id=None, ip_address=session.ip_address, success=True,
            ))

    def can_access_patient(self, session_id: str, patient_id: str) -> bool:
        return self.access_control.check_access(
            session_id, Permission.VIEW_PATIENT, "PATIENT", patient_id, patient_id
        )

    def log_data_export(self, session_id: str, export_type: str, record_count: int):
        session = self.sessions.get_session(session_id)
        if not session:
            return
        self.audit_logger.log(AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            user_id=session.user_id, user_role=session.role,
            action="EXPORT", resource_type="DATASET", resource_id=export_type,
            patient_id=None, ip_address=session.ip_address, success=True,
            details={"record_count": record_count, "export_type": export_type},
        ))

    def get_audit_trail(self, patient_id: Optional[str] = None,
                         limit: int = 50) -> List[Dict]:
        return self.audit_logger.query(patient_id=patient_id, limit=limit)

    def flush(self):
        self.audit_logger.flush()


# ─────────────────────────────────────────
# Demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        sec = HospitalSecurityManager(audit_dir=tmpdir)

        # Login different roles
        dr_session = sec.login("dr_smith", "attending_physician", "192.168.1.10")
        nurse_session = sec.login("nurse_jones", "nurse", "192.168.1.20")
        ds_session = sec.login("data_sci_1", "data_scientist", "10.0.0.5")

        print("\n=== Access Control Demo ===")

        # Physician can view patient
        ok = sec.can_access_patient(dr_session, "P001")
        print(f"Dr Smith → view P001: {'✓' if ok else '✗'}")

        # Data scientist cannot prescribe
        can_rx = sec.access_control.check_access(
            ds_session, Permission.PRESCRIBE, "MEDICATION", "P001", "P001"
        )
        print(f"Data scientist → prescribe P001: {'✓' if can_rx else '✗ (denied, correct)'}")

        # Nurse can acknowledge alerts
        can_ack = sec.access_control.check_access(
            nurse_session, Permission.ACKNOWLEDGE_ALERTS, "ALERT", "ALT-001", "P002"
        )
        print(f"Nurse Jones → acknowledge alert: {'✓' if can_ack else '✗'}")

        # PHI de-identification
        print("\n=== PHI De-identification ===")
        phi_text = ("Patient John Smith (DOB: 03/15/1952, MRN: 1234567, SSN: 123-45-6789) "
                    "called from 555-867-5309. Seen by Dr. Jane Doe on January 15, 2024.")
        deidentified = sec.deidentifier.deidentify(phi_text)
        print(f"Original:       {phi_text[:80]}...")
        print(f"De-identified:  {deidentified[:80]}...")

        # Audit trail
        sec.flush()
        trail = sec.get_audit_trail(limit=10)
        print(f"\n=== Audit Trail ({len(trail)} events) ===")
        for ev in trail[:5]:
            status = "✓" if ev["success"] else "✗"
            print(f"  {status} {ev['user_id']:12s} [{ev['user_role']:24s}] "
                  f"{ev['action']} {ev['resource_type']}")

        sec.logout(dr_session)
        print("\nAll demos complete.")

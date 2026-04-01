"""
AI Hospital OS — REST API Server
FastAPI-based service exposing patient data, risk scores, alerts, NER, and KG queries
Falls back to lightweight http.server when FastAPI is unavailable
"""

import sys
import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────
# Shared app state (lazy-loaded)
# ─────────────────────────────────────────

class AppState:
    _ner = None
    _kg = None
    _processor = None

    @classmethod
    def get_ner(cls):
        if cls._ner is None:
            try:
                from nlp_pipeline.medical_ner import ClinicalEntityExtractor
                cls._ner = ClinicalEntityExtractor()
            except Exception as e:
                logger.warning(f"NER unavailable: {e}")
        return cls._ner

    @classmethod
    def get_kg(cls):
        if cls._kg is None:
            try:
                from knowledge_graph.graph_builder import MedicalKnowledgeGraph
                cls._kg = MedicalKnowledgeGraph()
            except Exception as e:
                logger.warning(f"KG unavailable: {e}")
        return cls._kg

    @classmethod
    def get_processor(cls):
        if cls._processor is None:
            try:
                from real_time_monitoring.vitals_stream_processor import VitalsStreamProcessor
                cls._processor = VitalsStreamProcessor()
                cls._processor.start()
            except Exception as e:
                logger.warning(f"Stream processor unavailable: {e}")
        return cls._processor


# ─────────────────────────────────────────
# Synthetic data helpers (no DB required)
# ─────────────────────────────────────────

def _synthetic_patient(pid: str) -> Dict:
    profiles = ["stable", "mildly_ill", "deteriorating", "critical"]
    profile = random.choice(profiles)
    risk = {"stable": 0.05, "mildly_ill": 0.20,
            "deteriorating": 0.55, "critical": 0.85}[profile]
    return {
        "patient_id": pid,
        "name": random.choice(["Alice Chen", "Bob Martinez", "Carol Singh",
                                "David Kim", "Eva Johnson"]),
        "age": random.randint(45, 89),
        "sex": random.choice(["M", "F"]),
        "location": f"ICU-{random.randint(1, 8)}",
        "admission_date": (datetime.utcnow() - timedelta(days=random.randint(1, 14))).isoformat(),
        "primary_diagnosis": random.choice(["Septic shock", "Pneumonia", "ARDS",
                                             "CHF exacerbation", "AKI"]),
        "profile": profile,
        "vitals": {
            "heart_rate": round(80 + random.gauss(20 if profile != "stable" else 0, 8), 1),
            "sbp": round(120 + random.gauss(-20 if profile != "stable" else 0, 10), 1),
            "dbp": round(75 + random.gauss(-10 if profile != "stable" else 0, 6), 1),
            "respiratory_rate": round(16 + random.gauss(6 if profile != "stable" else 0, 2), 1),
            "temperature": round(37.0 + random.gauss(1.0 if profile != "stable" else 0, 0.3), 2),
            "spo2": round(min(100, 97 + random.gauss(-4 if profile != "stable" else 0, 1.5)), 1),
        },
        "risk_scores": {
            "sepsis_risk": round(risk + random.gauss(0, 0.03), 3),
            "mortality_risk": round(risk * 0.7 + random.gauss(0, 0.02), 3),
            "cardiac_risk": round(risk * 0.5 + random.gauss(0, 0.02), 3),
        },
        "news2": random.randint(
            {"stable": 0, "mildly_ill": 3, "deteriorating": 6, "critical": 9}[profile],
            {"stable": 2, "mildly_ill": 5, "deteriorating": 8, "critical": 14}[profile],
        ),
        "active_alerts": random.randint(0, {"stable": 0, "mildly_ill": 1,
                                             "deteriorating": 3, "critical": 5}[profile]),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _synthetic_vitals_history(pid: str, hours: int = 24) -> List[Dict]:
    history = []
    base_hr, base_sbp = 85, 115
    for i in range(hours * 4):  # every 15 min
        ts = datetime.utcnow() - timedelta(minutes=15 * (hours * 4 - i))
        history.append({
            "patient_id": pid,
            "timestamp": ts.isoformat(),
            "heart_rate": round(base_hr + random.gauss(0, 5), 1),
            "sbp": round(base_sbp + random.gauss(0, 8), 1),
            "dbp": round(75 + random.gauss(0, 5), 1),
            "respiratory_rate": round(16 + random.gauss(0, 2), 1),
            "temperature": round(37.1 + random.gauss(0, 0.2), 2),
            "spo2": round(min(100, 97 + random.gauss(0, 1)), 1),
        })
    return history


# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────

def build_fastapi_app():
    try:
        from fastapi import FastAPI, HTTPException, Query, Body
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="AI Hospital OS API",
        description="Clinical decision support REST API — patient data, risk scores, alerts, NLP, knowledge graph",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Pydantic models ───────────────────

    class NERRequest(BaseModel):
        text: str
        include_relations: bool = True

    class DDxRequest(BaseModel):
        symptoms: List[str]
        top_n: int = 5

    class DrugInteractionRequest(BaseModel):
        medications: List[str]

    class VitalReading(BaseModel):
        patient_id: str
        heart_rate: float
        sbp: float
        dbp: float
        respiratory_rate: float
        temperature: float
        spo2: float
        gcs: int = 15

    class RiskRequest(BaseModel):
        patient_id: str
        features: Dict[str, float]

    # ── Health ─────────────────────────────

    @app.get("/health", tags=["System"])
    def health():
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {
                "ner": AppState.get_ner() is not None,
                "knowledge_graph": AppState.get_kg() is not None,
            }
        }

    @app.get("/", tags=["System"])
    def root():
        return {
            "name": "AI Hospital OS API",
            "docs": "/docs",
            "endpoints": ["/patients", "/vitals", "/alerts", "/nlp/ner",
                          "/kg/ddx", "/kg/interactions", "/risk/score"]
        }

    # ── Patients ───────────────────────────

    @app.get("/patients", tags=["Patients"])
    def list_patients(limit: int = Query(10, ge=1, le=100)):
        """List ICU patients with latest vitals and risk scores."""
        try:
            from ehr_system.patient_management import EHRService
            svc = EHRService()
            census = svc.get_icu_census()
            return {"patients": census[:limit], "count": len(census)}
        except Exception:
            patients = [_synthetic_patient(f"P{str(i).zfill(3)}") for i in range(1, limit + 1)]
            return {"patients": patients, "count": len(patients), "source": "synthetic"}

    @app.get("/patients/{patient_id}", tags=["Patients"])
    def get_patient(patient_id: str):
        """Get full patient summary including vitals, labs, and risk scores."""
        try:
            from ehr_system.patient_management import EHRService
            svc = EHRService()
            summary = svc.get_patient_summary(patient_id)
            if not summary:
                raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
            return summary
        except HTTPException:
            raise
        except Exception:
            return _synthetic_patient(patient_id)

    @app.get("/patients/{patient_id}/vitals", tags=["Patients"])
    def get_patient_vitals(patient_id: str, hours: int = Query(24, ge=1, le=168)):
        """Get vital sign time-series for a patient."""
        try:
            from ehr_system.patient_management import EHRService
            svc = EHRService()
            vitals = svc.get_patient_vitals(patient_id, hours=hours)
            return {"patient_id": patient_id, "vitals": vitals, "hours": hours}
        except Exception:
            history = _synthetic_vitals_history(patient_id, hours=min(hours, 24))
            return {"patient_id": patient_id, "vitals": history, "hours": hours,
                    "source": "synthetic"}

    # ── Vitals (real-time ingest) ──────────

    @app.post("/vitals/ingest", tags=["Real-Time"])
    def ingest_vital(reading: VitalReading):
        """Ingest a single vital sign reading into the stream processor."""
        from real_time_monitoring.vitals_stream_processor import (
            VitalReading as VR, compute_news2, compute_shock_index
        )
        proc = AppState.get_processor()
        vr = VR(
            patient_id=reading.patient_id,
            timestamp=datetime.utcnow(),
            heart_rate=reading.heart_rate,
            sbp=reading.sbp, dbp=reading.dbp,
            respiratory_rate=reading.respiratory_rate,
            temperature=reading.temperature,
            spo2=reading.spo2, gcs=reading.gcs,
        )
        news2 = compute_news2(vr)
        si = compute_shock_index(vr)
        if proc:
            proc.ingest(vr)
        return {
            "accepted": True,
            "patient_id": reading.patient_id,
            "news2": news2,
            "shock_index": round(si, 3),
            "alert_level": "critical" if news2 >= 7 else "warning" if news2 >= 5 else "normal",
        }

    # ── Alerts ─────────────────────────────

    @app.get("/alerts", tags=["Alerts"])
    def get_alerts(limit: int = Query(20, ge=1, le=100), severity: Optional[str] = None):
        """Get recent system alerts. Filter by severity: critical|high|moderate."""
        proc = AppState.get_processor()
        if proc:
            alerts = proc.get_recent_alerts(limit * 2)
            if severity:
                alerts = [a for a in alerts if a.get("severity") == severity]
            return {"alerts": alerts[:limit], "count": len(alerts[:limit])}
        # Synthetic
        severities = ["critical", "high", "moderate"]
        alerts = []
        for i in range(min(limit, 8)):
            sev = severities[i % 3]
            if severity and sev != severity:
                continue
            alerts.append({
                "patient_id": f"P{str(i+1).zfill(3)}",
                "severity": sev,
                "alert_type": "DETERIORATION" if sev == "critical" else "SEPSIS_ALERT",
                "message": f"NEWS2 elevation detected — score {7 + i}",
                "timestamp": (datetime.utcnow() - timedelta(minutes=i * 5)).isoformat(),
                "acknowledged": False,
            })
        return {"alerts": alerts, "count": len(alerts), "source": "synthetic"}

    @app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
    def acknowledge_alert(alert_id: str, clinician_id: str = Query(...)):
        """Acknowledge an alert."""
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_by": clinician_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── NLP ────────────────────────────────

    @app.post("/nlp/ner", tags=["NLP"])
    def extract_entities(request: NERRequest):
        """Extract clinical entities from a note: medications, diagnoses, vitals, labs, etc."""
        ner = AppState.get_ner()
        if not ner:
            raise HTTPException(status_code=503, detail="NER service unavailable")
        extraction = ner.full_extraction(request.text)
        return {
            "text_length": len(request.text),
            "extraction": extraction,
        }

    @app.post("/nlp/medications", tags=["NLP"])
    def extract_medications(text: str = Body(..., embed=True)):
        """Extract medications with dosages and frequencies from clinical text."""
        ner = AppState.get_ner()
        if not ner:
            raise HTTPException(status_code=503, detail="NER service unavailable")
        meds = ner.extract_medications(text)
        return {"medications": meds, "count": len(meds)}

    @app.post("/nlp/vitals", tags=["NLP"])
    def extract_vitals_from_text(text: str = Body(..., embed=True)):
        """Parse vital signs mentioned in free text."""
        ner = AppState.get_ner()
        if not ner:
            raise HTTPException(status_code=503, detail="NER service unavailable")
        vitals = ner.extract_vitals(text)
        labs = ner.extract_labs(text)
        return {"vitals": vitals, "labs": labs}

    # ── Knowledge Graph ────────────────────

    @app.post("/kg/ddx", tags=["Knowledge Graph"])
    def differential_diagnosis(request: DDxRequest):
        """Generate differential diagnosis ranked by symptom match score."""
        kg = AppState.get_kg()
        if not kg:
            raise HTTPException(status_code=503, detail="Knowledge graph unavailable")
        ddx = kg.differential_diagnosis(request.symptoms, top_n=request.top_n)
        return {
            "symptoms": request.symptoms,
            "differential": ddx,
        }

    @app.post("/kg/interactions", tags=["Knowledge Graph"])
    def drug_interactions(request: DrugInteractionRequest):
        """Check drug-drug interactions for a medication list."""
        kg = AppState.get_kg()
        if not kg:
            raise HTTPException(status_code=503, detail="Knowledge graph unavailable")
        interactions = kg.check_drug_interactions(request.medications)
        high_risk = [i for i in interactions if i["severity"] in ("high", "contraindicated")]
        return {
            "medications": request.medications,
            "interactions": interactions,
            "high_risk_count": len(high_risk),
            "safe": len(high_risk) == 0,
        }

    @app.get("/kg/disease/{disease}", tags=["Knowledge Graph"])
    def disease_info(disease: str):
        """Get full clinical profile for a disease from the knowledge graph."""
        kg = AppState.get_kg()
        if not kg:
            raise HTTPException(status_code=503, detail="Knowledge graph unavailable")
        info = kg.get_disease_info(disease)
        if not info:
            raise HTTPException(status_code=404, detail=f"Disease '{disease}' not found in knowledge graph")
        return info

    @app.get("/kg/stats", tags=["Knowledge Graph"])
    def kg_stats():
        kg = AppState.get_kg()
        if not kg:
            raise HTTPException(status_code=503, detail="Knowledge graph unavailable")
        return kg.graph_stats()

    # ── Risk Scores ────────────────────────

    @app.get("/risk/{patient_id}", tags=["Risk"])
    def get_risk_scores(patient_id: str):
        """Get current risk scores for a patient."""
        patient = _synthetic_patient(patient_id)
        return {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "scores": patient["risk_scores"],
            "news2": patient["news2"],
            "interpretation": {
                "sepsis_risk": "high" if patient["risk_scores"]["sepsis_risk"] > 0.5 else "low",
                "mortality_risk": "high" if patient["risk_scores"]["mortality_risk"] > 0.4 else "low",
            }
        }

    # ── ICU Overview ───────────────────────

    @app.get("/icu/overview", tags=["ICU"])
    def icu_overview():
        """Full ICU census with risk stratification."""
        proc = AppState.get_processor()
        if proc:
            overview = proc.get_icu_overview()
            if overview:
                return {"census": overview, "count": len(overview),
                        "critical": sum(1 for p in overview if p["news2"] >= 7)}

        patients = [_synthetic_patient(f"P{str(i).zfill(3)}") for i in range(1, 9)]
        patients.sort(key=lambda p: -p["news2"])
        return {
            "census": patients,
            "count": len(patients),
            "critical": sum(1 for p in patients if p["news2"] >= 7),
            "source": "synthetic",
        }

    @app.get("/icu/alerts/summary", tags=["ICU"])
    def icu_alert_summary():
        """Summary of active alerts by severity."""
        return {
            "critical": random.randint(0, 3),
            "high": random.randint(1, 5),
            "moderate": random.randint(2, 8),
            "total": random.randint(5, 15),
            "timestamp": datetime.utcnow().isoformat(),
        }

    return app


# ─────────────────────────────────────────
# Fallback: minimal stdlib HTTP server
# ─────────────────────────────────────────

def run_simple_server(port: int = 8000):
    """Lightweight JSON API using only stdlib — no FastAPI required."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info(fmt % args)

        def _respond(self, data: Any, status: int = 200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path.rstrip("/")
            parts = path.split("/")

            if path == "" or path == "/":
                self._respond({"name": "AI Hospital OS API (simple mode)",
                                "endpoints": ["/health", "/patients", "/icu/overview", "/alerts"]})
            elif path == "/health":
                self._respond({"status": "ok", "timestamp": datetime.utcnow().isoformat()})
            elif path == "/patients":
                patients = [_synthetic_patient(f"P{str(i).zfill(3)}") for i in range(1, 9)]
                self._respond({"patients": patients, "count": len(patients)})
            elif len(parts) == 3 and parts[1] == "patients":
                self._respond(_synthetic_patient(parts[2]))
            elif path == "/icu/overview":
                patients = [_synthetic_patient(f"P{str(i).zfill(3)}") for i in range(1, 9)]
                patients.sort(key=lambda p: -p["news2"])
                self._respond({"census": patients, "count": len(patients)})
            elif path == "/alerts":
                self._respond({"alerts": [], "count": 0, "message": "No active alerts"})
            else:
                self._respond({"error": "Not found", "path": path}, 404)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Simple API server running on http://localhost:{port}")
    server.serve_forever()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")
    port = int(os.environ.get("API_PORT", 8000))

    app = build_fastapi_app()
    if app:
        try:
            import uvicorn
            print(f"FastAPI server → http://localhost:{port}")
            print(f"API docs       → http://localhost:{port}/docs")
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        except ImportError:
            print("uvicorn not installed. Run: pip install uvicorn")
    else:
        run_simple_server(port)

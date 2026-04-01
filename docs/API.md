# AI Hospital OS — REST API Reference

Base URL: `http://localhost:8000`  
Auth: `Authorization: Bearer <jwt_token>` (development mode: optional)  
Content-Type: `application/json`

---

## Authentication

### POST /auth/token
Obtain a JWT access token.

**Request**
```json
{ "user_id": "dr_smith", "password": "password123" }
```
**Response**
```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 28800, "role": "attending_physician" }
```

### GET /auth/me
Get current user info from token.

---

## System

### GET /health
```json
{ "status": "ok", "timestamp": "2024-01-15T10:30:00", "version": "1.0.0" }
```

### GET /
API index with available endpoints.

---

## Patients

### GET /patients
List ICU patients with latest vitals and risk scores.

**Query params**: `limit` (1–100, default 10)

**Response**
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "name": "Alice Chen",
      "location": "ICU-1",
      "diagnosis": "Septic shock",
      "news2": 9,
      "sepsis_risk": 0.78,
      "active_alerts": 2,
      "vitals": { "heart_rate": 118, "sbp": 88, "spo2": 91 }
    }
  ],
  "count": 8
}
```

### GET /patients/{patient_id}
Full patient summary including vitals, labs, and risk scores.

### GET /patients/{patient_id}/vitals
Vital sign time-series.

**Query params**: `hours` (1–168, default 24)

---

## Real-Time

### POST /vitals/ingest
Ingest a single vital sign reading.

**Request**
```json
{
  "patient_id": "P001",
  "heart_rate": 118.0,
  "sbp": 88.0,
  "dbp": 55.0,
  "respiratory_rate": 26.0,
  "temperature": 38.8,
  "spo2": 91.0,
  "gcs": 14
}
```
**Response**
```json
{
  "accepted": true,
  "patient_id": "P001",
  "news2": 9,
  "shock_index": 1.341,
  "alert_level": "critical"
}
```

---

## Alerts

### GET /alerts
List recent alerts.

**Query params**: `limit` (1–100), `severity` (critical|high|moderate|low)

**Response**
```json
{
  "alerts": [
    {
      "patient_id": "P001",
      "severity": "critical",
      "alert_type": "DETERIORATION",
      "message": "NEWS2 = 9 — Immediate clinical review required",
      "timestamp": "2024-01-15T10:30:00",
      "acknowledged": false
    }
  ],
  "count": 3
}
```

### POST /alerts/{alert_id}/acknowledge
Acknowledge an alert.

**Query params**: `clinician_id` (required)

---

## NLP

### POST /nlp/ner
Extract clinical entities from a note.

**Request**
```json
{ "text": "Patient with septic shock on vancomycin 25mg/kg q8h. HR 118, BP 82/54.", "include_relations": true }
```
**Response**
```json
{
  "text_length": 74,
  "extraction": {
    "medications": [{"medication": "vancomycin", "dosages": ["25mg/kg"], "frequencies": ["q8h"]}],
    "diagnoses": ["septic shock"],
    "vitals": {"HR": "118", "BP": "82/54"},
    "entity_count": 8,
    "relation_count": 1
  }
}
```

### POST /nlp/medications
Extract medications only.

**Request body**: `"text"` (string)

### POST /nlp/vitals
Parse vital signs from free text.

---

## Knowledge Graph

### POST /kg/ddx
Differential diagnosis from symptoms.

**Request**
```json
{ "symptoms": ["fever", "hypotension", "tachycardia", "tachypnea"], "top_n": 5 }
```
**Response**
```json
{
  "symptoms": ["fever", "hypotension", "tachycardia", "tachypnea"],
  "differential": [
    { "disease": "septic_shock", "score": 0.84, "severity": "critical", "icd10": "A41.9" },
    { "disease": "sepsis",       "score": 0.76, "severity": "critical", "icd10": "A41.9" }
  ]
}
```

### POST /kg/interactions
Check drug-drug interactions.

**Request**
```json
{ "medications": ["vancomycin", "piperacillin-tazobactam", "heparin", "furosemide"] }
```
**Response**
```json
{
  "interactions": [
    { "drug1": "vancomycin", "drug2": "furosemide", "severity": "moderate", "effect": "↑ ototoxicity" }
  ],
  "high_risk_count": 0,
  "safe": true
}
```

### GET /kg/disease/{disease}
Full clinical profile for a disease.

### GET /kg/stats
Knowledge graph statistics.

---

## Risk Scores

### GET /risk/{patient_id}
Current AI risk scores for a patient.

**Response**
```json
{
  "patient_id": "P001",
  "timestamp": "2024-01-15T10:30:00",
  "scores": {
    "sepsis_risk": 0.783,
    "mortality_risk": 0.312,
    "cardiac_risk": 0.189
  },
  "interpretation": {
    "sepsis_risk": "high",
    "mortality_risk": "moderate"
  }
}
```

---

## ICU

### GET /icu/overview
Full ICU census with risk stratification.

**Response**
```json
{
  "census": [ ... ],
  "count": 8,
  "critical": 3
}
```

### GET /icu/alerts/summary
Summary of active alerts by severity.

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Bad request — invalid parameters |
| 401 | Authentication required or token expired |
| 403 | Permission denied for this role |
| 404 | Resource not found |
| 422 | Validation error — request body malformed |
| 429 | Rate limit exceeded (120 req/min) |
| 503 | Service unavailable (component down) |

```json
{ "error": "Rate limit exceeded", "retry_after_seconds": 60 }
```

---

## Rate Limits

- 120 requests/minute per IP address
- Rate limit headers returned on every response:
  - `X-RateLimit-Limit: 120`
  - `X-RateLimit-Remaining: 119`
  - `X-Response-Time: 12.3ms`

---

## Development Quick Start

```bash
# Start server (no auth required in development)
python api/server.py

# Get all patients
curl http://localhost:8000/patients

# Run NER
curl -X POST http://localhost:8000/nlp/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Septic shock. Vancomycin 25mg/kg. HR 118."}'

# Differential diagnosis
curl -X POST http://localhost:8000/kg/ddx \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "hypotension", "tachycardia"]}'

# Get auth token
curl -X POST http://localhost:8000/auth/token \
  -d '{"user_id":"dr_smith","password":"password123"}'
```

Interactive docs: `http://localhost:8000/docs`

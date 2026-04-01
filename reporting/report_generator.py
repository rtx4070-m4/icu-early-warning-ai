"""
Clinical Report Generator
Produces structured HTML/Markdown deterioration reports, shift handover summaries,
and AI-generated clinical note drafts for ICU patients.
"""

import os
import json
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Report data structures
# ─────────────────────────────────────────

@dataclass
class PatientSnapshot:
    patient_id: str
    name: str
    age: int
    sex: str
    location: str
    primary_diagnosis: str
    admission_date: str
    vitals: Dict
    labs: Dict
    medications: List[str]
    news2: int
    risk_scores: Dict
    active_alerts: int
    hours_in_icu: int
    events_24h: List[str] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    differential: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────
# Synthetic patient snapshot
# ─────────────────────────────────────────

def _make_snapshot(patient_id: str = "P003") -> PatientSnapshot:
    profiles = {
        "P001": ("Alice Chen", 68, "F", "ICU-1", "Septic shock", 7, 0.72, 0.38),
        "P002": ("Bob Martinez", 74, "M", "ICU-2", "ARDS", 9, 0.61, 0.44),
        "P003": ("Carol Singh", 55, "F", "ICU-3", "Pneumonia", 5, 0.34, 0.18),
        "P004": ("David Kim", 82, "M", "ICU-4", "CHF exacerbation", 4, 0.22, 0.12),
    }
    pid = patient_id
    name, age, sex, loc, dx, news2, sep_r, mort_r = profiles.get(
        pid, ("Unknown Patient", 65, "M", "ICU-5", "Sepsis", 6, 0.50, 0.25)
    )

    def jitter(val, std=0.05): return round(val + random.gauss(0, std), 3)

    return PatientSnapshot(
        patient_id=pid, name=name, age=age, sex=sex, location=loc,
        primary_diagnosis=dx,
        admission_date=(datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
        vitals={
            "heart_rate": round(85 + news2 * 3 + random.gauss(0, 4), 1),
            "sbp": round(130 - news2 * 4 + random.gauss(0, 5), 1),
            "dbp": round(80 - news2 * 2 + random.gauss(0, 3), 1),
            "respiratory_rate": round(14 + news2 * 1.5 + random.gauss(0, 1), 1),
            "temperature": round(36.8 + news2 * 0.2 + random.gauss(0, 0.2), 2),
            "spo2": round(min(100, 99 - news2 * 1.0 + random.gauss(0, 0.8)), 1),
            "map": round(95 - news2 * 3.5 + random.gauss(0, 4), 1),
        },
        labs={
            "wbc": round(8 + news2 * 1.2 + random.gauss(0, 1), 1),
            "hemoglobin": round(12 - news2 * 0.3, 1),
            "lactate": round(1.0 + news2 * 0.4 + random.gauss(0, 0.2), 1),
            "creatinine": round(0.9 + news2 * 0.18 + random.gauss(0, 0.1), 2),
            "potassium": round(4.0 + news2 * 0.05, 1),
            "sodium": 138,
            "glucose": round(120 + news2 * 8, 0),
            "procalcitonin": round(0.1 + news2 * 0.8, 2),
            "bnp": round(50 + news2 * 60, 0),
        },
        medications=[
            "Vancomycin 1.5g IV q12h",
            "Piperacillin-tazobactam 4.5g IV q6h",
            "Norepinephrine 0.12 mcg/kg/min",
            "Pantoprazole 40mg IV daily",
            "Enoxaparin 40mg SC daily",
            "Propofol 20 mcg/kg/min",
        ][:3 + news2 // 3],
        news2=news2,
        risk_scores={"sepsis_risk": jitter(sep_r), "mortality_risk": jitter(mort_r)},
        active_alerts=max(0, news2 - 3),
        hours_in_icu=72 + random.randint(-12, 12),
        events_24h=[
            "02:14 — BP drop to 82/50; fluid bolus 500 mL given, norepinephrine uptitrated",
            "08:30 — Blood cultures drawn × 2 peripheral",
            "11:00 — Chest X-ray: bilateral infiltrates, worsening from prior",
            "14:45 — Nephrology consulted for rising creatinine",
            "18:00 — MAP stabilised at 68 mmHg",
        ][:max(1, news2 - 2)],
        recommendations=[
            {"urgency": "STAT",    "action": "Obtain repeat lactate in 2h to assess perfusion"},
            {"urgency": "URGENT",  "action": "Obtain 12-lead ECG — persistent tachycardia"},
            {"urgency": "ROUTINE", "action": "DVT prophylaxis — sequential compression devices"},
        ][:max(1, news2 // 2)],
        differential=[
            {"disease": "septic_shock", "score": 0.84},
            {"disease": "pneumonia", "score": 0.71},
            {"disease": "ards", "score": 0.52},
        ][:2 + news2 // 4],
    )


# ─────────────────────────────────────────
# Report generators
# ─────────────────────────────────────────

class ReportGenerator:

    COLORS = {
        "critical": "#f85149", "high": "#e3b341",
        "moderate": "#3fb950", "low": "#58a6ff",
        "bg": "#0d1117", "surface": "#161b22",
        "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e",
    }

    # ── Deterioration Report (HTML) ────────

    def generate_deterioration_report(self, snapshot: PatientSnapshot,
                                       output_path: Optional[str] = None) -> str:
        news2_color = (self.COLORS["critical"] if snapshot.news2 >= 7 else
                       self.COLORS["high"] if snapshot.news2 >= 5 else
                       self.COLORS["moderate"] if snapshot.news2 >= 3 else self.COLORS["low"])
        sep_color = (self.COLORS["critical"] if snapshot.risk_scores.get("sepsis_risk", 0) > 0.6
                     else self.COLORS["high"] if snapshot.risk_scores.get("sepsis_risk", 0) > 0.35
                     else self.COLORS["moderate"])

        vitals_rows = ""
        v = snapshot.vitals
        labs = snapshot.labs
        for label, val, unit, lo, hi in [
            ("Heart Rate",    v.get("heart_rate"),     "bpm",    40,  150),
            ("SBP",           v.get("sbp"),             "mmHg",   90,  180),
            ("DBP",           v.get("dbp"),             "mmHg",   60,  110),
            ("MAP",           v.get("map"),             "mmHg",   65,  None),
            ("Resp. Rate",    v.get("respiratory_rate"),"br/min", 8,   25),
            ("Temperature",   v.get("temperature"),    "°C",     35,  39),
            ("SpO2",          v.get("spo2"),            "%",      92,  None),
        ]:
            if val is None: continue
            abnormal = (lo is not None and val < lo) or (hi is not None and val > hi)
            color = self.COLORS["critical"] if abnormal else self.COLORS["text"]
            flag = " ⚠" if abnormal else ""
            vitals_rows += f"<tr><td>{label}</td><td style='color:{color};font-weight:{'bold' if abnormal else 'normal'}'>{val}{flag}</td><td style='color:{self.COLORS['muted']}'>{unit}</td></tr>"

        lab_rows = ""
        for label, val, unit, lo, hi in [
            ("WBC",          labs.get("wbc"),          "k/μL",   4.0, 12.0),
            ("Lactate",      labs.get("lactate"),      "mmol/L", None, 2.0),
            ("Creatinine",   labs.get("creatinine"),   "mg/dL",  None, 1.5),
            ("Procalcitonin",labs.get("procalcitonin"),"ng/mL",  None, 0.5),
            ("Glucose",      labs.get("glucose"),      "mg/dL",  70,  180),
            ("BNP",          labs.get("bnp"),          "pg/mL",  None, 100),
            ("Potassium",    labs.get("potassium"),    "mEq/L",  3.5, 5.0),
        ]:
            if val is None: continue
            abnormal = (lo is not None and val < lo) or (hi is not None and val > hi)
            color = self.COLORS["critical"] if abnormal else self.COLORS["text"]
            flag = " ⚠" if abnormal else ""
            lab_rows += f"<tr><td>{label}</td><td style='color:{color};font-weight:{'bold' if abnormal else 'normal'}'>{val}{flag}</td><td style='color:{self.COLORS['muted']}'>{unit}</td></tr>"

        events_html = "".join(
            f"<li style='margin-bottom:6px'><span style='color:{self.COLORS['muted']}'>{e.split('—')[0].strip()}</span> — {e.split('—')[1].strip() if '—' in e else e}</li>"
            for e in snapshot.events_24h
        )

        recs_html = "".join(
            f"<li style='margin-bottom:8px'><span style='background:{'#f8514922' if r['urgency']=='STAT' else '#e3b34122' if r['urgency']=='URGENT' else '#3fb95022'};color:{'#f85149' if r['urgency']=='STAT' else '#e3b341' if r['urgency']=='URGENT' else '#3fb950'};padding:2px 6px;border-radius:4px;font-size:11px'>{r['urgency']}</span> {r['action']}</li>"
            for r in snapshot.recommendations
        )

        meds_html = "".join(
            f"<li style='margin-bottom:4px;font-size:13px'>{m}</li>"
            for m in snapshot.medications
        )

        ddx_html = "".join(
            f"<li style='margin-bottom:4px'>{d['disease'].replace('_',' ').title()} <span style='color:{self.COLORS['muted']}'>(match score: {d['score']:.2f})</span></li>"
            for d in snapshot.differential
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ICU Deterioration Report — {snapshot.patient_id}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {self.COLORS['bg']}; color: {self.COLORS['text']}; font-family: 'JetBrains Mono', monospace; padding: 32px; font-size: 13px; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; color: {self.COLORS['text']}; }}
    h2 {{ font-size: 14px; color: {self.COLORS['muted']}; text-transform: uppercase; margin: 0 0 12px; letter-spacing: 0.05em; }}
    .header {{ background: {self.COLORS['surface']}; border: 1px solid {self.COLORS['border']}; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }}
    .header-left .subtitle {{ color: {self.COLORS['muted']}; font-size: 12px; margin-top: 4px; }}
    .badges {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .card {{ background: {self.COLORS['surface']}; border: 1px solid {self.COLORS['border']}; border-radius: 8px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ color: {self.COLORS['muted']}; text-align: left; padding: 6px 4px; border-bottom: 1px solid {self.COLORS['border']}; font-size: 11px; text-transform: uppercase; }}
    td {{ padding: 6px 4px; border-bottom: 1px solid {self.COLORS['border']}22; }}
    ul {{ list-style: none; padding: 0; }}
    .footer {{ margin-top: 24px; color: {self.COLORS['muted']}; font-size: 11px; border-top: 1px solid {self.COLORS['border']}; padding-top: 12px; }}
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <h1>🏥 ICU Deterioration Report</h1>
      <div class="subtitle">
        {snapshot.name} · {snapshot.age}{snapshot.sex} · {snapshot.location} ·
        {snapshot.primary_diagnosis} · Admitted {snapshot.admission_date} ·
        {snapshot.hours_in_icu}h in ICU
      </div>
      <div class="subtitle" style="margin-top:6px;color:{self.COLORS['muted']}">
        Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
      </div>
    </div>
    <div class="badges">
      <div class="badge" style="background:{news2_color}22;color:{news2_color}">
        NEWS2: {snapshot.news2}
      </div>
      <div class="badge" style="background:{sep_color}22;color:{sep_color}">
        Sepsis: {snapshot.risk_scores.get('sepsis_risk',0):.0%}
      </div>
      <div class="badge" style="background:{self.COLORS['muted']}22;color:{self.COLORS['muted']}">
        Alerts: {snapshot.active_alerts}
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Vital Signs</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
        {vitals_rows}
      </table>
    </div>
    <div class="card">
      <h2>Laboratory Results</h2>
      <table>
        <tr><th>Test</th><th>Value</th><th>Unit</th></tr>
        {lab_rows}
      </table>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>24-Hour Events</h2>
      <ul>{events_html}</ul>
    </div>
    <div class="card">
      <h2>Active Medications</h2>
      <ul>{meds_html}</ul>
      <div style="margin-top:16px">
        <h2>Differential Diagnosis</h2>
        <ul>{ddx_html}</ul>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>AI Recommendations</h2>
    <ul>{recs_html}</ul>
  </div>

  <div class="footer">
    ⚠ This report is AI-generated for decision support only. Not a substitute for clinical judgement.
    Patient: {snapshot.patient_id} · Report ID: RPT-{snapshot.patient_id}-{datetime.utcnow().strftime('%Y%m%d%H%M')}
  </div>
</body>
</html>"""

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"Deterioration report written: {output_path}")

        return html

    # ── Shift Handover Summary (Markdown) ─

    def generate_handover_summary(self, snapshots: List[PatientSnapshot]) -> str:
        lines = [
            f"# ICU Shift Handover Summary",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Census:** {len(snapshots)} patients  ",
            f"**Critical (NEWS2≥7):** {sum(1 for s in snapshots if s.news2 >= 7)}",
            "",
            "---",
            "",
            "## Patient Summary",
            "",
            "| ID | Patient | Location | Dx | NEWS2 | Sepsis% | Alerts | Key Issue |",
            "|----|---------|----------|----|-------|---------|--------|-----------|",
        ]

        for s in sorted(snapshots, key=lambda x: -x.news2):
            status = "🔴" if s.news2 >= 7 else "🟡" if s.news2 >= 5 else "🟢"
            key_issue = s.events_24h[-1].split("—")[-1].strip()[:40] if s.events_24h else "Stable"
            lines.append(
                f"| {s.patient_id} | {s.name} | {s.location} | {s.primary_diagnosis} | "
                f"{status} {s.news2} | {s.risk_scores.get('sepsis_risk',0):.0%} | "
                f"{s.active_alerts} | {key_issue} |"
            )

        lines += ["", "---", "", "## Pending Actions"]
        for s in sorted(snapshots, key=lambda x: -x.news2):
            if s.recommendations:
                lines.append(f"\n### {s.patient_id} — {s.name}")
                for r in s.recommendations:
                    urgency_marker = "🚨" if r["urgency"] == "STAT" else "⚡" if r["urgency"] == "URGENT" else "📋"
                    lines.append(f"- {urgency_marker} `{r['urgency']}` {r['action']}")

        lines += [
            "",
            "---",
            "",
            "> *AI-generated handover. All clinical decisions require physician review.*",
        ]
        return "\n".join(lines)

    # ── Clinical Note Draft ────────────────

    def generate_note_draft(self, snapshot: PatientSnapshot) -> str:
        v = snapshot.vitals
        l = snapshot.labs

        sbp, dbp = v.get("sbp", "?"), v.get("dbp", "?")
        hr = v.get("heart_rate", "?")
        rr = v.get("respiratory_rate", "?")
        temp = v.get("temperature", "?")
        spo2 = v.get("spo2", "?")

        sep_risk = snapshot.risk_scores.get("sepsis_risk", 0)
        assessment_risk = ("high" if sep_risk > 0.6 else
                           "moderate" if sep_risk > 0.3 else "low")

        note = f"""## ICU Progress Note — {snapshot.patient_id}
**Date/Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC  
**Patient:** {snapshot.name}, {snapshot.age}{snapshot.sex}  
**Location:** {snapshot.location}  
**Physician:** [Attending Physician]

---

### Subjective
Patient is a {snapshot.age}-year-old {snapshot.sex} with a history of {snapshot.primary_diagnosis},
admitted {snapshot.hours_in_icu} hours ago. [Insert interval history and patient-reported symptoms here.]

### Objective

**Vitals:**
- Temp {temp}°C | HR {hr} bpm | BP {sbp}/{dbp} mmHg | RR {rr} br/min | SpO2 {spo2}%
- MAP {v.get('map', '?')} mmHg | NEWS2 Score: **{snapshot.news2}**

**Labs:**
- WBC {l.get('wbc', '?')} k/μL | Hgb {l.get('hemoglobin', '?')} g/dL
- Lactate {l.get('lactate', '?')} mmol/L | Creatinine {l.get('creatinine', '?')} mg/dL
- Procalcitonin {l.get('procalcitonin', '?')} ng/mL | Glucose {l.get('glucose', '?')} mg/dL

**Active Medications:**
{chr(10).join('- ' + m for m in snapshot.medications)}

**24h Events:**
{chr(10).join('- ' + e for e in snapshot.events_24h)}

### Assessment

{snapshot.age}{snapshot.sex} with {snapshot.primary_diagnosis}, currently in the ICU for
{snapshot.hours_in_icu} hours.

**AI Risk Assessment:**
- Sepsis risk: {sep_risk:.0%} ({assessment_risk})
- Mortality risk: {snapshot.risk_scores.get('mortality_risk', 0):.0%}

**Clinical Impression:** [Attending physician impression to be entered here]

**Differential Diagnosis:**
{chr(10).join(f'- {d["disease"].replace("_"," ").title()} (AI score: {d["score"]:.2f})' for d in snapshot.differential)}

### Plan

{chr(10).join(f'{i+1}. [{r["urgency"]}] {r["action"]}' for i, r in enumerate(snapshot.recommendations))}

---
*Note drafted with AI decision support assistance. All content requires physician review and co-signature.*  
*Report ID: NOTE-{snapshot.patient_id}-{datetime.utcnow().strftime('%Y%m%d%H%M')}*
"""
        return note


# ─────────────────────────────────────────
# Batch report runner
# ─────────────────────────────────────────

def generate_all_reports(patient_ids: List[str] = None,
                          output_dir: str = "/data/reports") -> Dict:
    """Generate all report types for a list of patients."""
    os.makedirs(output_dir, exist_ok=True)
    patient_ids = patient_ids or ["P001", "P002", "P003", "P004"]
    generator = ReportGenerator()
    snapshots = [_make_snapshot(pid) for pid in patient_ids]
    generated = []

    # Individual deterioration reports (HTML)
    for snap in snapshots:
        path = os.path.join(output_dir, f"deterioration_{snap.patient_id}.html")
        generator.generate_deterioration_report(snap, output_path=path)
        generated.append(path)

    # Shift handover (Markdown)
    handover = generator.generate_handover_summary(snapshots)
    handover_path = os.path.join(output_dir, "shift_handover.md")
    with open(handover_path, "w") as f:
        f.write(handover)
    generated.append(handover_path)

    # Clinical note drafts
    for snap in snapshots:
        note = generator.generate_note_draft(snap)
        note_path = os.path.join(output_dir, f"note_draft_{snap.patient_id}.md")
        with open(note_path, "w") as f:
            f.write(note)
        generated.append(note_path)

    logger.info(f"Generated {len(generated)} reports in {output_dir}")
    return {"generated": generated, "count": len(generated), "output_dir": output_dir}


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_all_reports(output_dir=tmpdir)
        print(f"Generated {result['count']} reports:")
        for path in result["generated"]:
            size = os.path.getsize(path)
            print(f"  {os.path.basename(path):40s}  {size:>7,} bytes")

    # Also demo single patient report to stdout
    gen = ReportGenerator()
    snap = _make_snapshot("P003")
    print("\n--- Clinical Note Draft ---")
    print(gen.generate_note_draft(snap)[:800] + "\n  [...]")

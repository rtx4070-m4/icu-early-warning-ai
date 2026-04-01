"""
AI Hospital OS — Interactive ICU Dashboard
Built with Plotly Dash; falls back to static HTML report when Dash unavailable
Displays: patient census, vital trends, alert feed, risk scores, NEWS2 heatmap
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Synthetic demo data (always available)
# ─────────────────────────────────────────

def _generate_demo_patients(n: int = 8) -> List[Dict]:
    profiles = ["stable", "stable", "mildly_ill", "mildly_ill",
                "deteriorating", "deteriorating", "critical", "stable"]
    names = ["Alice Chen", "Bob Martinez", "Carol Singh", "David Kim",
             "Eva Johnson", "Frank Patel", "Grace Lee", "Henry Wilson"]
    locations = [f"ICU-{i+1}" for i in range(n)]
    diagnoses = ["Septic shock", "ARDS", "CHF exacerbation", "COPD exacerbation",
                 "Pneumonia", "AKI", "Pulmonary embolism", "Post-op monitoring"]

    data = []
    for i in range(min(n, len(names))):
        profile = profiles[i % len(profiles)]
        base = {
            "stable":       dict(hr=80,  sbp=122, dbp=76,  rr=16, temp=37.0, spo2=98, news2=1),
            "mildly_ill":   dict(hr=98,  sbp=108, dbp=66,  rr=20, temp=37.9, spo2=95, news2=4),
            "deteriorating":dict(hr=118, sbp=90,  dbp=58,  rr=25, temp=38.7, spo2=91, news2=7),
            "critical":     dict(hr=138, sbp=78,  dbp=48,  rr=30, temp=39.3, spo2=86, news2=11),
        }[profile]

        # Add some noise
        hr   = base["hr"]  + random.gauss(0, 5)
        sbp  = base["sbp"] + random.gauss(0, 6)
        dbp  = base["dbp"] + random.gauss(0, 4)
        rr   = base["rr"]  + random.gauss(0, 2)
        temp = base["temp"]+ random.gauss(0, 0.2)
        spo2 = min(100, base["spo2"] + random.gauss(0, 1))
        map_ = (sbp + 2 * dbp) / 3

        data.append({
            "patient_id": f"P{str(i+1).zfill(3)}",
            "name": names[i],
            "location": locations[i],
            "diagnosis": diagnoses[i % len(diagnoses)],
            "profile": profile,
            "hr": round(hr, 1),
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "rr": round(rr, 1),
            "temp": round(temp, 2),
            "spo2": round(spo2, 1),
            "map": round(map_, 1),
            "news2": base["news2"] + random.randint(-1, 1),
            "lactate": round(random.uniform(0.8, 5.5) if profile != "stable" else random.uniform(0.8, 1.5), 1),
            "creatinine": round(random.uniform(0.8, 4.5) if profile in ["deteriorating","critical"] else random.uniform(0.7, 1.3), 1),
            "sepsis_risk": round({"stable": 0.05, "mildly_ill": 0.25,
                                   "deteriorating": 0.65, "critical": 0.88}[profile] + random.gauss(0, 0.03), 2),
            "mortality_risk": round({"stable": 0.02, "mildly_ill": 0.08,
                                      "deteriorating": 0.28, "critical": 0.52}[profile] + random.gauss(0, 0.02), 2),
            "active_alerts": {"stable": 0, "mildly_ill": 1, "deteriorating": 2, "critical": 3}[profile],
            "hours_in_icu": random.randint(2, 168),
        })
    return data


def _generate_vital_history(patient: Dict, hours: int = 24) -> Dict[str, List]:
    """Generate synthetic vital sign history for a patient."""
    n = hours * 12  # every 5 minutes
    ts = [datetime.utcnow() - timedelta(minutes=5 * (n - i)) for i in range(n)]

    trend = {"stable": 0, "mildly_ill": 0.02,
             "deteriorating": 0.08, "critical": 0.15}[patient["profile"]]

    hr_vals, sbp_vals, spo2_vals, rr_vals = [], [], [], []
    for i, t in enumerate(ts):
        frac = i / n
        drift_factor = 1 + trend * frac
        hr_vals.append(patient["hr"] * drift_factor + random.gauss(0, 4))
        sbp_vals.append(patient["sbp"] / drift_factor + random.gauss(0, 5))
        spo2_vals.append(min(100, patient["spo2"] - trend * frac * 8 + random.gauss(0, 0.8)))
        rr_vals.append(patient["rr"] * drift_factor + random.gauss(0, 1.5))

    return {
        "timestamps": [t.strftime("%H:%M") for t in ts],
        "hr": hr_vals,
        "sbp": sbp_vals,
        "spo2": spo2_vals,
        "rr": rr_vals,
    }


def _generate_alerts(patients: List[Dict]) -> List[Dict]:
    alert_templates = {
        "critical": ["CRITICAL: HR {val:.0f} bpm — severe tachycardia",
                     "CRITICAL: SpO2 {val:.0f}% — hypoxemia",
                     "CRITICAL: MAP {val:.0f} mmHg — hemodynamic compromise"],
        "high":     ["NEWS2 = {val:.0f} — urgent clinical review",
                     "Sepsis screen positive (qSOFA ≥ 2)",
                     "Rapid deterioration trend detected"],
        "moderate": ["NEWS2 = {val:.0f} — close monitoring required",
                     "Elevated lactate trend",
                     "Temperature spike — fever workup recommended"],
    }
    alerts = []
    for p in patients:
        for _ in range(p["active_alerts"]):
            if p["news2"] >= 9:
                sev = "critical"
                val = p["news2"]
            elif p["news2"] >= 5:
                sev = "high"
                val = p["news2"]
            else:
                sev = "moderate"
                val = p["hr"]
            template = random.choice(alert_templates[sev])
            alerts.append({
                "patient_id": p["patient_id"],
                "patient_name": p["name"],
                "location": p["location"],
                "severity": sev,
                "message": template.format(val=val),
                "time": (datetime.utcnow() - timedelta(minutes=random.randint(1, 30))).strftime("%H:%M"),
            })
    return sorted(alerts, key=lambda x: {"critical": 0, "high": 1, "moderate": 2}.get(x["severity"], 3))


# ─────────────────────────────────────────
# Dash dashboard (requires: pip install dash plotly)
# ─────────────────────────────────────────

def build_dash_app():
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.warning("Dash/Plotly not installed. Run: pip install dash plotly pandas")
        return None

    app = dash.Dash(__name__, title="AI Hospital OS — ICU Dashboard")

    COLORS = {
        "bg":        "#0d1117",
        "surface":   "#161b22",
        "border":    "#30363d",
        "text":      "#e6edf3",
        "muted":     "#8b949e",
        "critical":  "#f85149",
        "high":      "#e3b341",
        "moderate":  "#3fb950",
        "stable":    "#58a6ff",
        "accent":    "#1f6feb",
    }

    def severity_color(s: str) -> str:
        return {"critical": COLORS["critical"], "high": COLORS["high"],
                "moderate": COLORS["moderate"], "stable": COLORS["stable"],
                "mildly_ill": COLORS["high"],
                "deteriorating": COLORS["critical"]}.get(s, COLORS["muted"])

    def news2_color(score: int) -> str:
        if score >= 7: return COLORS["critical"]
        if score >= 5: return COLORS["high"]
        if score >= 3: return COLORS["moderate"]
        return COLORS["stable"]

    # ── Layout ────────────────────────────

    app.layout = html.Div(style={
        "backgroundColor": COLORS["bg"],
        "color": COLORS["text"],
        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
        "minHeight": "100vh",
        "padding": "0",
    }, children=[

        # Header
        html.Div(style={
            "backgroundColor": COLORS["surface"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "16px 24px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
        }, children=[
            html.Div([
                html.H1("🏥 AI Hospital OS", style={"margin": 0, "fontSize": "20px",
                                                      "color": COLORS["text"]}),
                html.Span("ICU Command Center", style={"color": COLORS["muted"], "fontSize": "13px"}),
            ]),
            html.Div(id="live-clock", style={"color": COLORS["muted"], "fontSize": "13px"}),
        ]),

        # Body
        html.Div(style={"padding": "20px", "display": "grid",
                        "gridTemplateColumns": "1fr 320px", "gap": "16px"}, children=[

            # Left column
            html.Div([

                # KPI row
                html.Div(id="kpi-row", style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "12px", "marginBottom": "16px",
                }),

                # Patient table
                html.Div(style={
                    "backgroundColor": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "marginBottom": "16px",
                }, children=[
                    html.H3("ICU Census", style={"margin": "0 0 12px", "fontSize": "14px",
                                                   "color": COLORS["muted"]}),
                    html.Div(id="patient-table"),
                ]),

                # Vital trend chart
                html.Div(style={
                    "backgroundColor": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "8px",
                    "padding": "16px",
                }, children=[
                    html.Div(style={"display": "flex", "justifyContent": "space-between",
                                     "alignItems": "center", "marginBottom": "12px"}, children=[
                        html.H3("Vital Trends (24h)", style={"margin": 0, "fontSize": "14px",
                                                               "color": COLORS["muted"]}),
                        dcc.Dropdown(
                            id="patient-select",
                            style={"width": "200px", "fontSize": "13px"},
                        ),
                    ]),
                    dcc.Graph(id="vital-trend-chart", config={"displayModeBar": False}),
                ]),
            ]),

            # Right column — alerts
            html.Div([
                html.Div(style={
                    "backgroundColor": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "marginBottom": "16px",
                }, children=[
                    html.H3("🚨 Active Alerts", style={"margin": "0 0 12px", "fontSize": "14px",
                                                        "color": COLORS["muted"]}),
                    html.Div(id="alert-feed"),
                ]),

                # NEWS2 heatmap
                html.Div(style={
                    "backgroundColor": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "8px",
                    "padding": "16px",
                }, children=[
                    html.H3("NEWS2 Overview", style={"margin": "0 0 12px", "fontSize": "14px",
                                                      "color": COLORS["muted"]}),
                    html.Div(id="news2-grid"),
                ]),
            ]),
        ]),

        # Refresh interval
        dcc.Interval(id="refresh", interval=5000, n_intervals=0),
        dcc.Store(id="demo-data"),
    ])

    # ── Callbacks ────────────────────────

    @app.callback(
        Output("demo-data", "data"),
        Input("refresh", "n_intervals"),
    )
    def refresh_data(n):
        patients = _generate_demo_patients(8)
        alerts = _generate_alerts(patients)
        return {"patients": patients, "alerts": alerts}

    @app.callback(
        Output("live-clock", "children"),
        Input("refresh", "n_intervals"),
    )
    def update_clock(n):
        return datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S")

    @app.callback(
        Output("kpi-row", "children"),
        Input("demo-data", "data"),
    )
    def update_kpis(data):
        if not data:
            return []
        patients = data["patients"]
        alerts = data["alerts"]
        critical_count = sum(1 for p in patients if p["news2"] >= 7)
        alert_count = len([a for a in alerts if a["severity"] == "critical"])

        def kpi_card(title, value, color, unit=""):
            return html.Div(style={
                "backgroundColor": COLORS["surface"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "8px",
                "padding": "16px",
            }, children=[
                html.Div(title, style={"color": COLORS["muted"], "fontSize": "11px",
                                        "marginBottom": "8px", "textTransform": "uppercase"}),
                html.Div([
                    html.Span(str(value), style={"fontSize": "28px", "fontWeight": "bold",
                                                   "color": color}),
                    html.Span(unit, style={"color": COLORS["muted"], "fontSize": "13px",
                                            "marginLeft": "4px"}),
                ]),
            ])

        return [
            kpi_card("ICU Census", len(patients), COLORS["stable"]),
            kpi_card("Critical (NEWS2≥7)", critical_count, COLORS["critical"]),
            kpi_card("Active Alerts", alert_count, COLORS["high"]),
            kpi_card("Avg Sepsis Risk",
                     f"{sum(p['sepsis_risk'] for p in patients)/len(patients):.0%}",
                     COLORS["moderate"]),
        ]

    @app.callback(
        Output("patient-table", "children"),
        Output("patient-select", "options"),
        Output("patient-select", "value"),
        Input("demo-data", "data"),
    )
    def update_patient_table(data):
        if not data:
            return [], [], None
        patients = data["patients"]
        header = html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "80px 140px 100px 60px 60px 60px 60px 60px 70px 80px",
            "gap": "4px",
            "padding": "6px 8px",
            "fontSize": "11px",
            "color": COLORS["muted"],
            "borderBottom": f"1px solid {COLORS['border']}",
        }, children=[html.Span(h) for h in
                     ["ID", "Patient", "Diagnosis", "HR", "BP", "SpO2", "Temp",
                      "NEWS2", "Sepsis%", "Status"]])

        rows = [header]
        for p in sorted(patients, key=lambda x: -x["news2"]):
            status_color = severity_color(p["profile"])
            news2_c = news2_color(p["news2"])
            rows.append(html.Div(style={
                "display": "grid",
                "gridTemplateColumns": "80px 140px 100px 60px 60px 60px 60px 60px 70px 80px",
                "gap": "4px",
                "padding": "6px 8px",
                "fontSize": "12px",
                "borderBottom": f"1px solid {COLORS['border']}",
                "alignItems": "center",
            }, children=[
                html.Span(p["patient_id"]),
                html.Span(p["name"], style={"overflow": "hidden", "textOverflow": "ellipsis"}),
                html.Span(p["diagnosis"][:14] + "…" if len(p["diagnosis"]) > 14 else p["diagnosis"],
                          style={"color": COLORS["muted"], "fontSize": "11px"}),
                html.Span(f"{p['hr']:.0f}"),
                html.Span(f"{p['sbp']:.0f}/{p['dbp']:.0f}"),
                html.Span(f"{p['spo2']:.0f}%",
                          style={"color": COLORS["critical"] if p["spo2"] < 92 else COLORS["text"]}),
                html.Span(f"{p['temp']:.1f}°"),
                html.Span(f"{p['news2']}",
                          style={"color": news2_c, "fontWeight": "bold"}),
                html.Span(f"{p['sepsis_risk']:.0%}",
                          style={"color": COLORS["critical"] if p["sepsis_risk"] > 0.5 else COLORS["text"]}),
                html.Span("●", style={"color": status_color}),
            ]))

        options = [{"label": f"{p['patient_id']} — {p['name']}", "value": p["patient_id"]}
                   for p in patients]
        return rows, options, patients[0]["patient_id"] if patients else None

    @app.callback(
        Output("vital-trend-chart", "figure"),
        Input("patient-select", "value"),
        Input("demo-data", "data"),
    )
    def update_vital_chart(pid, data):
        import plotly.graph_objects as go
        fig = go.Figure()
        if not pid or not data:
            fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"])
            return fig

        patient = next((p for p in data["patients"] if p["patient_id"] == pid), None)
        if not patient:
            return fig

        history = _generate_vital_history(patient, hours=24)
        ts = history["timestamps"]

        # Downsample for display
        step = max(1, len(ts) // 200)
        ts_d = ts[::step]

        for metric, vals, color, yaxis in [
            ("Heart Rate", history["hr"], "#58a6ff", "y"),
            ("SpO2",       history["spo2"], "#3fb950", "y2"),
            ("SBP",        history["sbp"], "#e3b341", "y"),
            ("RR",         history["rr"],  "#bc8cff", "y2"),
        ]:
            vals_d = vals[::step]
            fig.add_trace(go.Scatter(
                x=ts_d, y=vals_d, name=metric, mode="lines",
                line=dict(color=color, width=1.5),
                yaxis=yaxis,
            ))

        fig.update_layout(
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"], size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
            margin=dict(l=40, r=40, t=10, b=40),
            height=280,
            xaxis=dict(gridcolor=COLORS["border"], showgrid=True,
                       tickmode="array",
                       tickvals=ts_d[::len(ts_d)//8] if len(ts_d) > 8 else ts_d),
            yaxis=dict(gridcolor=COLORS["border"], showgrid=True, title="HR / SBP"),
            yaxis2=dict(overlaying="y", side="right", title="SpO2 / RR",
                        gridcolor="rgba(0,0,0,0)"),
        )
        return fig

    @app.callback(
        Output("alert-feed", "children"),
        Input("demo-data", "data"),
    )
    def update_alerts(data):
        if not data:
            return []
        alerts = data["alerts"][:12]
        items = []
        for a in alerts:
            color = severity_color(a["severity"])
            items.append(html.Div(style={
                "borderLeft": f"3px solid {color}",
                "padding": "8px 12px",
                "marginBottom": "8px",
                "backgroundColor": f"{color}11",
                "borderRadius": "0 4px 4px 0",
            }, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
                    html.Span(f"{a['patient_id']} — {a['location']}",
                              style={"fontSize": "11px", "color": COLORS["muted"]}),
                    html.Span(a["time"], style={"fontSize": "11px", "color": COLORS["muted"]}),
                ]),
                html.Div(a["message"],
                         style={"fontSize": "12px", "marginTop": "4px", "color": color}),
            ]))
        return items if items else [html.Div("No active alerts", style={"color": COLORS["muted"]})]

    @app.callback(
        Output("news2-grid", "children"),
        Input("demo-data", "data"),
    )
    def update_news2_grid(data):
        if not data:
            return []
        patients = sorted(data["patients"], key=lambda x: -x["news2"])
        cells = []
        for p in patients:
            color = news2_color(p["news2"])
            cells.append(html.Div(style={
                "textAlign": "center",
                "padding": "10px 6px",
                "borderRadius": "6px",
                "backgroundColor": f"{color}22",
                "border": f"1px solid {color}44",
            }, children=[
                html.Div(str(p["news2"]),
                         style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
                html.Div(p["patient_id"],
                         style={"fontSize": "10px", "color": COLORS["muted"], "marginTop": "2px"}),
                html.Div(p["location"],
                         style={"fontSize": "10px", "color": COLORS["muted"]}),
            ]))
        return html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "6px"},
                        children=cells)

    return app


# ─────────────────────────────────────────
# Fallback: static HTML report
# ─────────────────────────────────────────

def generate_static_report(output_path: str = "/tmp/icu_report.html"):
    """Generate a static HTML ICU report when Dash is unavailable."""
    patients = _generate_demo_patients(8)
    alerts = _generate_alerts(patients)

    rows = ""
    for p in sorted(patients, key=lambda x: -x["news2"]):
        status_colors = {"stable": "#3fb950", "mildly_ill": "#e3b341",
                         "deteriorating": "#f85149", "critical": "#f85149"}
        color = status_colors.get(p["profile"], "#8b949e")
        news2_c = "#f85149" if p["news2"] >= 7 else "#e3b341" if p["news2"] >= 5 else "#3fb950"
        rows += f"""
        <tr>
            <td>{p['patient_id']}</td>
            <td>{p['name']}</td>
            <td style="color:#8b949e">{p['location']}</td>
            <td>{p['diagnosis']}</td>
            <td>{p['hr']:.0f}</td>
            <td>{p['sbp']:.0f}/{p['dbp']:.0f}</td>
            <td style="color:{'#f85149' if p['spo2']<92 else 'inherit'}">{p['spo2']:.0f}%</td>
            <td>{p['temp']:.1f}°C</td>
            <td style="color:{news2_c};font-weight:bold">{p['news2']}</td>
            <td style="color:{color}">{'●' * p['active_alerts'] if p['active_alerts'] else '—'}</td>
            <td style="color:{'#f85149' if p['sepsis_risk']>0.5 else 'inherit'}">{p['sepsis_risk']:.0%}</td>
        </tr>"""

    alert_rows = ""
    for a in alerts[:10]:
        ac = {"critical": "#f85149", "high": "#e3b341", "moderate": "#3fb950"}.get(a["severity"], "#8b949e")
        alert_rows += f"""
        <tr>
            <td style="color:{ac}">{a['severity'].upper()}</td>
            <td>{a['patient_id']} — {a['location']}</td>
            <td>{a['message']}</td>
            <td style="color:#8b949e">{a['time']}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI Hospital OS — ICU Report</title>
  <style>
    body {{ background:#0d1117; color:#e6edf3; font-family:'Fira Code',monospace; margin:0; padding:24px; }}
    h1 {{ color:#58a6ff; margin-bottom:4px; }}
    .subtitle {{ color:#8b949e; font-size:13px; margin-bottom:24px; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:24px; font-size:12px; }}
    th {{ color:#8b949e; text-align:left; padding:8px; border-bottom:1px solid #30363d; font-size:11px; text-transform:uppercase; }}
    td {{ padding:8px; border-bottom:1px solid #21262d; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }}
    .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
    .kpi-val {{ font-size:28px; font-weight:bold; }}
    .kpi-lbl {{ color:#8b949e; font-size:11px; text-transform:uppercase; margin-bottom:8px; }}
    .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:16px; }}
    .section h2 {{ margin:0 0 12px; font-size:14px; color:#8b949e; }}
  </style>
</head>
<body>
  <h1>🏥 AI Hospital OS</h1>
  <div class="subtitle">ICU Report — Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>

  <div class="kpis">
    <div class="kpi"><div class="kpi-lbl">ICU Census</div>
      <div class="kpi-val" style="color:#58a6ff">{len(patients)}</div></div>
    <div class="kpi"><div class="kpi-lbl">Critical (NEWS2≥7)</div>
      <div class="kpi-val" style="color:#f85149">{sum(1 for p in patients if p['news2']>=7)}</div></div>
    <div class="kpi"><div class="kpi-lbl">Critical Alerts</div>
      <div class="kpi-val" style="color:#e3b341">{len([a for a in alerts if a['severity']=='critical'])}</div></div>
    <div class="kpi"><div class="kpi-lbl">Avg Sepsis Risk</div>
      <div class="kpi-val" style="color:#3fb950">{sum(p['sepsis_risk'] for p in patients)/len(patients):.0%}</div></div>
  </div>

  <div class="section">
    <h2>ICU Census</h2>
    <table>
      <tr><th>ID</th><th>Name</th><th>Location</th><th>Diagnosis</th>
          <th>HR</th><th>BP</th><th>SpO2</th><th>Temp</th><th>NEWS2</th><th>Alerts</th><th>Sepsis</th></tr>
      {rows}
    </table>
  </div>

  <div class="section">
    <h2>🚨 Active Alerts</h2>
    <table>
      <tr><th>Severity</th><th>Patient</th><th>Message</th><th>Time</th></tr>
      {alert_rows}
    </table>
  </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    logger.info(f"Static ICU report written to {output_path}")
    return output_path


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_dash_app()
    if app:
        print("Starting Dash ICU Dashboard on http://localhost:8050")
        app.run(debug=False, port=8050)
    else:
        path = generate_static_report()
        print(f"Dash unavailable. Static report written to: {path}")

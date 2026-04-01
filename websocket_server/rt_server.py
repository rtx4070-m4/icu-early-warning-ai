"""
WebSocket Real-Time Push Server
Streams live vital signs, alerts, and risk score updates to connected clients.
Uses asyncio + stdlib only; optional websockets library for production.
Falls back to Server-Sent Events (SSE) over HTTP when WebSockets unavailable.
"""

import asyncio
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Message types
# ─────────────────────────────────────────

class MsgType:
    VITALS        = "vitals"
    ALERT         = "alert"
    RISK_UPDATE   = "risk_update"
    ALERT_ACK     = "alert_ack"
    PATIENT_LIST  = "patient_list"
    SYSTEM_STATUS = "system_status"
    PING          = "ping"
    PONG          = "pong"
    ERROR         = "error"


def make_message(msg_type: str, payload: Dict) -> str:
    return json.dumps({
        "type": msg_type,
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload,
    })


# ─────────────────────────────────────────
# In-process event bus (thread-safe)
# ─────────────────────────────────────────

class EventBus:
    """Simple pub/sub event bus for decoupling producers and consumers."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        with self._lock:
            subs = self._subscribers.get(topic, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, topic: str, data: Dict):
        with self._lock:
            subscribers = list(self._subscribers.get(topic, []) +
                               self._subscribers.get("*", []))
        for cb in subscribers:
            try:
                cb(topic, data)
            except Exception as e:
                logger.error(f"EventBus subscriber error: {e}")

    def publish_vital(self, patient_id: str, vitals: Dict):
        self.publish("vitals", {"patient_id": patient_id, **vitals})

    def publish_alert(self, patient_id: str, alert: Dict):
        self.publish("alert", {"patient_id": patient_id, **alert})

    def publish_risk(self, patient_id: str, scores: Dict):
        self.publish("risk_update", {"patient_id": patient_id, "scores": scores})


# ─────────────────────────────────────────
# Client connection abstraction
# ─────────────────────────────────────────

@dataclass
class ClientConnection:
    client_id: str
    user_id: str
    role: str
    connected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    subscribed_patients: Set[str] = field(default_factory=set)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_connected: bool = True

    def to_dict(self) -> Dict:
        return {
            "client_id": self.client_id,
            "user_id": self.user_id,
            "role": self.role,
            "connected_at": self.connected_at,
            "subscribed_patients": list(self.subscribed_patients),
        }


# ─────────────────────────────────────────
# WebSocket server (asyncio-based)
# ─────────────────────────────────────────

class HospitalWSServer:
    """
    WebSocket server that pushes real-time ICU updates.
    Requires: pip install websockets
    Falls back to SSE server if websockets unavailable.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                  event_bus: Optional[EventBus] = None):
        self.host = host
        self.port = port
        self.event_bus = event_bus or EventBus()
        self._clients: Dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()
        self._running = False

        # Subscribe to all events from the bus
        self.event_bus.subscribe("*", self._on_event)

    def _on_event(self, topic: str, data: Dict):
        """Bridge sync event bus → async client queues."""
        msg = make_message({
            "vitals": MsgType.VITALS,
            "alert": MsgType.ALERT,
            "risk_update": MsgType.RISK_UPDATE,
        }.get(topic, topic), data)

        patient_id = data.get("patient_id")
        for client in self._clients.values():
            if not client.is_connected:
                continue
            # Send to clients subscribed to this patient or to all
            if not patient_id or patient_id in client.subscribed_patients or not client.subscribed_patients:
                try:
                    client.message_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for client {client.client_id}")

    async def _handle_client(self, websocket, path: str = "/"):
        """Handle a single WebSocket connection."""
        import uuid
        client_id = str(uuid.uuid4())[:8]
        client = ClientConnection(
            client_id=client_id,
            user_id="anonymous",
            role="viewer",
        )

        async with self._lock:
            self._clients[client_id] = client

        logger.info(f"Client connected: {client_id} from {websocket.remote_address}")

        # Send welcome
        await websocket.send(make_message(MsgType.SYSTEM_STATUS, {
            "message": "Connected to AI Hospital OS real-time feed",
            "client_id": client_id,
            "server_version": "1.0.0",
        }))

        try:
            # Concurrent: receive commands + drain send queue
            async def _sender():
                while client.is_connected:
                    try:
                        msg = await asyncio.wait_for(
                            client.message_queue.get(), timeout=30.0
                        )
                        await websocket.send(msg)
                    except asyncio.TimeoutError:
                        # Keepalive ping
                        await websocket.send(make_message(MsgType.PING, {}))
                    except Exception:
                        break

            async def _receiver():
                async for raw in websocket:
                    try:
                        msg = json.loads(raw)
                        await self._handle_client_message(client, msg, websocket)
                    except json.JSONDecodeError:
                        await websocket.send(make_message(MsgType.ERROR,
                                                           {"message": "Invalid JSON"}))

            await asyncio.gather(_sender(), _receiver())

        except Exception as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            client.is_connected = False
            async with self._lock:
                self._clients.pop(client_id, None)

    async def _handle_client_message(self, client: ClientConnection,
                                      msg: Dict, websocket):
        """Process incoming client commands."""
        msg_type = msg.get("type")
        payload = msg.get("payload", {})

        if msg_type == MsgType.PONG:
            return

        elif msg_type == "subscribe":
            patients = payload.get("patients", [])
            client.subscribed_patients.update(patients)
            await websocket.send(make_message("subscribed", {
                "patients": list(client.subscribed_patients)
            }))

        elif msg_type == "unsubscribe":
            patients = payload.get("patients", [])
            client.subscribed_patients -= set(patients)

        elif msg_type == MsgType.ALERT_ACK:
            alert_id = payload.get("alert_id")
            patient_id = payload.get("patient_id")
            logger.info(f"Alert {alert_id} acknowledged by {client.user_id}")
            self.event_bus.publish("alert_ack", {
                "alert_id": alert_id,
                "patient_id": patient_id,
                "acknowledged_by": client.user_id,
            })

        elif msg_type == "auth":
            client.user_id = payload.get("user_id", "anonymous")
            client.role = payload.get("role", "viewer")
            await websocket.send(make_message("auth_ok", {
                "user_id": client.user_id,
                "role": client.role,
            }))

        elif msg_type == "get_patients":
            # Return current patient list
            patients = self._get_patient_snapshot()
            await websocket.send(make_message(MsgType.PATIENT_LIST,
                                               {"patients": patients}))
        else:
            await websocket.send(make_message(MsgType.ERROR,
                                               {"message": f"Unknown message type: {msg_type}"}))

    def _get_patient_snapshot(self) -> List[Dict]:
        """Return a snapshot of current ICU patients (synthetic when no DB)."""
        patients = []
        for i in range(1, 9):
            news2 = random.randint(0, 12)
            patients.append({
                "patient_id": f"P{i:03d}",
                "name": f"Patient {i}",
                "location": f"ICU-{i}",
                "news2": news2,
                "risk_level": "critical" if news2 >= 7 else "high" if news2 >= 5 else "normal",
            })
        return patients

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
            self._running = True
            logger.info(f"WebSocket server starting on ws://{self.host}:{self.port}")
            async with websockets.serve(self._handle_client, self.host, self.port):
                await asyncio.Future()  # Run until cancelled
        except ImportError:
            logger.warning("websockets not installed — starting SSE fallback server")
            await self._sse_server()

    async def _sse_server(self):
        """Server-Sent Events fallback over plain HTTP."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        event_bus = self.event_bus
        clients_ref = self._clients

        class SSEHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args): pass

            def do_GET(self):
                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    try:
                        while True:
                            # Push synthetic vitals every 2 seconds
                            pid = f"P{random.randint(1, 8):03d}"
                            data = {
                                "patient_id": pid,
                                "heart_rate": round(80 + random.gauss(0, 8), 1),
                                "sbp": round(120 + random.gauss(0, 10), 1),
                                "spo2": round(min(100, 97 + random.gauss(0, 1.2)), 1),
                                "news2": random.randint(0, 10),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            msg = f"data: {json.dumps(data)}\n\n"
                            self.wfile.write(msg.encode())
                            self.wfile.flush()
                            time.sleep(2)
                    except Exception:
                        pass
                elif self.path == "/health":
                    body = b'{"status": "ok", "mode": "sse"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer((self.host, self.port), SSEHandler)
        logger.info(f"SSE server running on http://{self.host}:{self.port}/events")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.serve_forever)

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    def broadcast(self, msg_type: str, payload: Dict):
        """Broadcast a message to all connected clients (thread-safe)."""
        msg = make_message(msg_type, payload)
        for client in list(self._clients.values()):
            if client.is_connected:
                try:
                    client.message_queue.put_nowait(msg)
                except Exception:
                    pass


# ─────────────────────────────────────────
# Vitals feed producer (simulates monitoring)
# ─────────────────────────────────────────

class VitalsFeedProducer:
    """
    Continuously generates synthetic vital signs and publishes to EventBus.
    In production: replace with real monitoring device data.
    """

    def __init__(self, event_bus: EventBus, n_patients: int = 8,
                  interval_seconds: float = 5.0):
        self.event_bus = event_bus
        self.n_patients = n_patients
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._patient_states = self._init_states()

    def _init_states(self) -> Dict[str, Dict]:
        states = {}
        profiles = ["stable"] * 3 + ["mildly_ill"] * 3 + ["deteriorating"] * 2
        for i in range(self.n_patients):
            pid = f"P{i+1:03d}"
            profile = profiles[i % len(profiles)]
            states[pid] = {
                "profile": profile,
                "hr": {"stable": 78, "mildly_ill": 96, "deteriorating": 115}[profile],
                "sbp": {"stable": 122, "mildly_ill": 108, "deteriorating": 90}[profile],
                "spo2": {"stable": 98, "mildly_ill": 95, "deteriorating": 91}[profile],
                "rr": {"stable": 15, "mildly_ill": 19, "deteriorating": 24}[profile],
                "temp": {"stable": 37.0, "mildly_ill": 37.8, "deteriorating": 38.6}[profile],
                "sepsis_risk": {"stable": 0.05, "mildly_ill": 0.22, "deteriorating": 0.68}[profile],
            }
        return states

    def _generate_reading(self, patient_id: str) -> Dict:
        state = self._patient_states[patient_id]
        drift = {"stable": 0, "mildly_ill": 0.01, "deteriorating": 0.03}[state["profile"]]

        hr = state["hr"] * (1 + drift) + random.gauss(0, 4)
        sbp = state["sbp"] * (1 - drift * 0.5) + random.gauss(0, 6)
        dbp = sbp * 0.62 + random.gauss(0, 3)
        spo2 = min(100, state["spo2"] - drift * 0.5 + random.gauss(0, 0.8))
        rr = state["rr"] * (1 + drift * 0.3) + random.gauss(0, 1.5)
        temp = state["temp"] + drift * 0.1 + random.gauss(0, 0.15)

        # Compute NEWS2 (simplified)
        news2 = 0
        if rr >= 25 or rr <= 8: news2 += 3
        elif rr >= 21: news2 += 2
        elif rr <= 11: news2 += 1
        if spo2 <= 91: news2 += 3
        elif spo2 <= 93: news2 += 2
        elif spo2 <= 95: news2 += 1
        if sbp <= 90 or sbp >= 220: news2 += 3
        elif sbp <= 100: news2 += 2
        elif sbp <= 110: news2 += 1
        if hr >= 131 or hr <= 40: news2 += 3
        elif hr >= 111: news2 += 2
        elif hr <= 50 or hr >= 91: news2 += 1
        if temp >= 39.1 or temp <= 35.0: news2 += 1

        return {
            "heart_rate": round(hr, 1),
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "map": round((sbp + 2 * dbp) / 3, 1),
            "respiratory_rate": round(rr, 1),
            "temperature": round(temp, 2),
            "spo2": round(spo2, 1),
            "news2": news2,
            "sepsis_risk": round(min(0.99, state["sepsis_risk"] + drift * 0.01 + random.gauss(0, 0.02)), 3),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _run(self):
        while self._running:
            for pid in list(self._patient_states.keys()):
                reading = self._generate_reading(pid)
                self.event_bus.publish_vital(pid, reading)

                # Fire alert if NEWS2 critical
                news2 = reading.get("news2", 0)
                if news2 >= 7:
                    self.event_bus.publish_alert(pid, {
                        "severity": "critical" if news2 >= 9 else "high",
                        "message": f"NEWS2 = {news2} — urgent review required",
                        "alert_type": "DETERIORATION",
                    })

                # Publish risk scores
                self.event_bus.publish_risk(pid, {
                    "sepsis_risk": reading["sepsis_risk"],
                    "news2": news2,
                })

            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"VitalsFeedProducer started ({self.n_patients} patients, {self.interval}s interval)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


# ─────────────────────────────────────────
# JavaScript client snippet generator
# ─────────────────────────────────────────

def generate_js_client(ws_url: str = "ws://localhost:8765") -> str:
    """Generate a browser JavaScript snippet for connecting to this server."""
    return f"""
// AI Hospital OS — WebSocket Client
// Connect to: {ws_url}

const ws = new WebSocket('{ws_url}');

ws.onopen = () => {{
    console.log('Connected to Hospital OS real-time feed');
    // Authenticate
    ws.send(JSON.stringify({{
        type: 'auth',
        payload: {{ user_id: 'dr_smith', role: 'attending_physician' }}
    }}));
    // Subscribe to specific patients (empty = all)
    ws.send(JSON.stringify({{
        type: 'subscribe',
        payload: {{ patients: ['P001', 'P002', 'P003'] }}
    }}));
    // Request patient list
    ws.send(JSON.stringify({{ type: 'get_patients', payload: {{}} }}));
}};

ws.onmessage = (event) => {{
    const msg = JSON.parse(event.data);
    switch(msg.type) {{
        case 'vitals':
            updateVitalsDisplay(msg.payload);
            break;
        case 'alert':
            showAlert(msg.payload);
            break;
        case 'risk_update':
            updateRiskScores(msg.payload);
            break;
        case 'patient_list':
            renderPatientList(msg.payload.patients);
            break;
        case 'ping':
            ws.send(JSON.stringify({{ type: 'pong', payload: {{}} }}));
            break;
    }}
}};

ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('Disconnected from Hospital OS feed');

// Acknowledge an alert
function acknowledgeAlert(alertId, patientId) {{
    ws.send(JSON.stringify({{
        type: 'alert_ack',
        payload: {{ alert_id: alertId, patient_id: patientId }}
    }}));
}}
"""


# ─────────────────────────────────────────
# Threaded demo runner
# ─────────────────────────────────────────

class RTDemoRunner:
    """Run the WebSocket server + vitals producer in a background thread."""

    def __init__(self, ws_port: int = 8765, sse_fallback: bool = True):
        self.ws_port = ws_port
        self.event_bus = EventBus()
        self.producer = VitalsFeedProducer(self.event_bus, n_patients=8, interval_seconds=2.0)
        self.server = HospitalWSServer("0.0.0.0", ws_port, self.event_bus)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._received: List[Dict] = []

        # Local message capture for testing
        self.event_bus.subscribe("*", lambda topic, data: self._received.append(
            {"topic": topic, "data": data}
        ))

    def start_producer_only(self):
        """Start just the producer (for testing without a server)."""
        self.producer.start()

    def stop(self):
        self.producer.stop()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def get_received_messages(self, n: int = 10) -> List[Dict]:
        return self._received[-n:]


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s")

    print("AI Hospital OS — WebSocket Real-Time Server")
    print("=" * 48)

    runner = RTDemoRunner(ws_port=8765)
    runner.start_producer_only()

    print(f"Vitals producer running — collecting 10 seconds of data...\n")
    time.sleep(3)

    messages = runner.get_received_messages(20)
    vitals_msgs = [m for m in messages if m["topic"] == "vitals"]
    alert_msgs = [m for m in messages if m["topic"] == "alert"]
    risk_msgs = [m for m in messages if m["topic"] == "risk_update"]

    print(f"Messages received:")
    print(f"  Vitals updates : {len(vitals_msgs)}")
    print(f"  Alert events   : {len(alert_msgs)}")
    print(f"  Risk updates   : {len(risk_msgs)}")

    if vitals_msgs:
        sample = vitals_msgs[0]["data"]
        print(f"\nSample vital reading ({sample.get('patient_id')}):")
        print(f"  HR={sample.get('heart_rate'):.0f} "
              f"BP={sample.get('sbp'):.0f}/{sample.get('dbp'):.0f} "
              f"SpO2={sample.get('spo2'):.0f}% NEWS2={sample.get('news2')}")

    if alert_msgs:
        alert = alert_msgs[0]["data"]
        print(f"\nSample alert:")
        print(f"  [{alert.get('severity','?').upper()}] {alert.get('patient_id')}: {alert.get('message')}")

    runner.stop()

    print(f"\nJavaScript client snippet (save as hospital_client.js):")
    print(generate_js_client())

    try:
        import websockets
        print("websockets installed — full WebSocket server available.")
        print("Run: python websocket_server/rt_server.py")
        print("     (starts on ws://localhost:8765)")
    except ImportError:
        print("websockets not installed — SSE fallback available.")
        print("Install with: pip install websockets")

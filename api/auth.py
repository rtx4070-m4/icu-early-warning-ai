"""
API Authentication & Middleware
JWT token generation/validation, rate limiting, request logging,
and CORS for the FastAPI server.
Falls back gracefully when FastAPI/JWT libraries are unavailable.
"""

import os
import time
import json
import hmac
import hashlib
import base64
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# JWT implementation (stdlib only)
# ─────────────────────────────────────────

JWT_SECRET = os.environ.get("API_JWT_SECRET", "change_in_production_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = int(os.environ.get("JWT_EXPIRY_MINUTES", "480"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    return base64.urlsafe_b64decode(data + "=" * padding)


def create_token(user_id: str, role: str,
                  expiry_minutes: int = JWT_EXPIRY_MINUTES) -> str:
    """Create a signed JWT token."""
    now = datetime.utcnow()
    header = _b64url_encode(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}).encode())
    payload = _b64url_encode(json.dumps({
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expiry_minutes)).timestamp()),
    }).encode())
    signing_input = f"{header}.{payload}"
    signature = _b64url_encode(
        hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    return f"{signing_input}.{signature}"


def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode a JWT token. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = _b64url_encode(
            hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            logger.warning("JWT signature mismatch")
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("exp", 0) < int(datetime.utcnow().timestamp()):
            logger.info("JWT expired")
            return None
        return payload
    except Exception as e:
        logger.debug(f"JWT verify error: {e}")
        return None


def extract_token(auth_header: Optional[str]) -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    if not auth_header:
        return None
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


# ─────────────────────────────────────────
# Rate limiter (in-memory, per IP)
# ─────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter. Thread-safe."""

    def __init__(self, requests_per_minute: int = 120):
        self.rpm = requests_per_minute
        self._buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": requests_per_minute,
            "last_refill": time.monotonic(),
        })
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        with self._lock:
            bucket = self._buckets[client_ip]
            now = time.monotonic()
            elapsed = now - bucket["last_refill"]
            # Refill tokens based on elapsed time
            refill = elapsed * (self.rpm / 60.0)
            bucket["tokens"] = min(self.rpm, bucket["tokens"] + refill)
            bucket["last_refill"] = now
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    def remaining(self, client_ip: str) -> int:
        with self._lock:
            return int(self._buckets[client_ip]["tokens"])


# ─────────────────────────────────────────
# Request logger
# ─────────────────────────────────────────

class RequestLogger:
    """Structured request/response logging."""

    def log_request(self, method: str, path: str, client_ip: str,
                     user_id: Optional[str] = None, status_code: int = 200,
                     duration_ms: float = 0.0):
        logger.info(json.dumps({
            "type": "api_request",
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "path": path,
            "client_ip": client_ip,
            "user_id": user_id or "anonymous",
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }))


# ─────────────────────────────────────────
# FastAPI middleware (optional)
# ─────────────────────────────────────────

_rate_limiter = RateLimiter(requests_per_minute=120)
_req_logger = RequestLogger()


def add_middleware(app):
    """Add auth, rate-limiting, and logging middleware to a FastAPI app."""
    try:
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware

        class AuthRateLimitMiddleware(BaseHTTPMiddleware):
            OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
            AUTH_REQUIRED_PATHS = {"/patients", "/vitals", "/alerts", "/nlp",
                                     "/kg", "/risk", "/icu"}

            async def dispatch(self, request: Request, call_next: Callable) -> Response:
                t0 = time.perf_counter()
                client_ip = request.client.host if request.client else "unknown"
                path = request.url.path

                # Rate limiting
                if not _rate_limiter.is_allowed(client_ip):
                    return JSONResponse(
                        {"error": "Rate limit exceeded", "retry_after_seconds": 60},
                        status_code=429,
                        headers={"Retry-After": "60",
                                  "X-RateLimit-Limit": str(_rate_limiter.rpm),
                                  "X-RateLimit-Remaining": "0"},
                    )

                # Token auth for protected paths
                user_id = None
                role = "anonymous"
                if any(path.startswith(p) for p in self.AUTH_REQUIRED_PATHS):
                    auth_header = request.headers.get("Authorization")
                    token_str = extract_token(auth_header)
                    if token_str:
                        payload = verify_token(token_str)
                        if payload:
                            user_id = payload.get("sub")
                            role = payload.get("role", "viewer")
                            request.state.user_id = user_id
                            request.state.role = role
                        else:
                            return JSONResponse(
                                {"error": "Invalid or expired token"},
                                status_code=401,
                            )
                    # In development mode, skip auth
                    elif os.environ.get("HOSPITAL_OS_ENV") == "production":
                        return JSONResponse(
                            {"error": "Authentication required"},
                            status_code=401,
                        )

                response = await call_next(request)
                duration_ms = (time.perf_counter() - t0) * 1000
                _req_logger.log_request(
                    request.method, path, client_ip,
                    user_id, response.status_code, duration_ms
                )

                # Add standard headers
                response.headers["X-RateLimit-Limit"] = str(_rate_limiter.rpm)
                response.headers["X-RateLimit-Remaining"] = str(
                    _rate_limiter.remaining(client_ip))
                response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
                response.headers["X-Hospital-OS-Version"] = "1.0.0"
                return response

        app.add_middleware(AuthRateLimitMiddleware)
        logger.info("Auth/rate-limit middleware registered")

    except ImportError:
        logger.info("FastAPI not available — middleware skipped")

    return app


# ─────────────────────────────────────────
# Auth endpoints builder
# ─────────────────────────────────────────

def add_auth_routes(app):
    """Add /auth/token and /auth/refresh endpoints to FastAPI app."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        class LoginRequest(BaseModel):
            user_id: str
            password: str
            role: str = "viewer"

        class TokenResponse(BaseModel):
            access_token: str
            token_type: str = "bearer"
            expires_in: int = JWT_EXPIRY_MINUTES * 60
            role: str

        # Stub credential store (replace with real DB lookup in production)
        DEMO_USERS = {
            "dr_smith":   ("password123", "attending_physician"),
            "nurse_jones": ("password123", "nurse"),
            "admin":      ("adminpass",   "admin"),
            "data_sci":   ("password123", "data_scientist"),
        }

        @app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
        def login(req: LoginRequest):
            """Obtain a JWT access token."""
            stored = DEMO_USERS.get(req.user_id)
            if not stored or stored[0] != req.password:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            actual_role = stored[1]
            token = create_token(req.user_id, actual_role)
            return TokenResponse(access_token=token, role=actual_role)

        @app.post("/auth/refresh", tags=["Auth"])
        def refresh_token(authorization: str):
            """Refresh a valid JWT token."""
            token_str = extract_token(authorization)
            if not token_str:
                raise HTTPException(status_code=401, detail="No token provided")
            payload = verify_token(token_str)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            new_token = create_token(payload["sub"], payload["role"])
            return {"access_token": new_token, "token_type": "bearer"}

        @app.get("/auth/me", tags=["Auth"])
        def whoami(authorization: str = ""):
            """Get current user info from token."""
            token_str = extract_token(authorization)
            payload = verify_token(token_str) if token_str else None
            if not payload:
                return {"user_id": "anonymous", "role": "none", "authenticated": False}
            return {
                "user_id": payload["sub"],
                "role": payload["role"],
                "authenticated": True,
                "expires_at": datetime.fromtimestamp(payload["exp"]).isoformat(),
            }

        logger.info("Auth routes registered: /auth/token, /auth/refresh, /auth/me")

    except ImportError:
        logger.info("FastAPI not available — auth routes skipped")

    return app


# ─────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== JWT Token Demo ===")
    token = create_token("dr_smith", "attending_physician")
    print(f"Token: {token[:60]}...")
    payload = verify_token(token)
    print(f"Verified: user={payload['sub']} role={payload['role']}")
    print(f"Expires: {datetime.fromtimestamp(payload['exp']).isoformat()}")

    # Test expired token
    import time as _time
    expired_payload = {
        "sub": "old_user", "role": "nurse",
        "iat": int(datetime.utcnow().timestamp()) - 3600,
        "exp": int(datetime.utcnow().timestamp()) - 1,  # 1 second ago
    }
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_b64 = _b64url_encode(json.dumps(expired_payload).encode())
    signing = f"{header}.{payload_b64}"
    sig = _b64url_encode(hmac.new(JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest())
    expired_token = f"{signing}.{sig}"
    result = verify_token(expired_token)
    print(f"Expired token result: {result}")  # Should be None

    print("\n=== Rate Limiter Demo ===")
    rl = RateLimiter(requests_per_minute=5)
    for i in range(7):
        allowed = rl.is_allowed("192.168.1.1")
        print(f"  Request {i+1}: {'✓ allowed' if allowed else '✗ blocked'} "
              f"(remaining={rl.remaining('192.168.1.1')})")

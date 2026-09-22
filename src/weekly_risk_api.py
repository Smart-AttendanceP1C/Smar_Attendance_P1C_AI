r"""
ATT-ML: Weekly Attendance Risk API
 
Wraps the logistic regression trained in train_weekly_risk_model.py.
The Node.js backend calls POST /predict-weekly-risk with a student's
current weekly features and gets back a next-week risk probability.
 
GRACEFUL DEGRADATION: if the model file is missing or prediction fails for
any reason, this service falls back to the SAME rule-based baseline used
during training/evaluation (flag_any logic) and marks the response
"baseline_fallback". It never raises a 500 that would block anything
downstream.
 
Run locally (from project root, e.g. ATT-AIML\):
    pip install fastapi uvicorn joblib scikit-learn python-dotenv slowapi
    uvicorn src.weekly_risk_api:app --reload --port 8000
 
Test:
    curl -X POST http://localhost:8000/predict-weekly-risk \
         -H "Content-Type: application/json" \
         -d '{"attendance_rate_to_date": 0.65, "trend_last_3": -0.1, \
              "rejected_last_2": 1, "consecutive_absences": 2, "course_load": 5}'
"""
 
import hashlib
import logging
import os
import secrets
from pathlib import Path
from typing import Optional
 
import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Project layout -------------------------------------------------------
# This file lives at <project_root>/src/weekly_risk_api.py. Everything below
# is resolved relative to that, NOT relative to the current working
# directory — so the API works the same whether you launch uvicorn from
# the project root, from src/, or from anywhere else.
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

# Reads <project_root>/.env and loads its values into os.environ, so
# os.environ.get("AI_SERVICE_KEY") etc. below pick them up. Does nothing if
# .env doesn't exist — real deployments should set real environment
# variables instead of using a file.
load_dotenv(PROJECT_ROOT / ".env")
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weekly_risk_api")

# Security Finding 4 fix: exception messages can leak details about input
# data or internal model structure. The main "weekly_risk_api" logger only
# ever gets a fixed, generic message. Full exception detail goes to this
# separate debug-only logger instead, so it can be routed to an
# access-controlled sink (or disabled) independently of normal API logs.
debug_logger = logging.getLogger("weekly_risk_api.debug")
debug_logger.setLevel(logging.DEBUG)
 
MODEL_PATH = PROJECT_ROOT / "models" / "weekly_risk_model.joblib"
MODEL_HASH_PATH = PROJECT_ROOT / "models" / "weekly_risk_model.sha256"  # written by train_weekly_risk_model.py
 
 
def verify_model_integrity(model_path: Path, hash_path: Path) -> bool:
    """
    Security Finding 2 mitigation: refuse to load the model file if its
    SHA256 hash doesn't match the hash recorded at training time. This
    doesn't stop pickle's arbitrary-code-execution risk on its own, but it
    stops a SILENTLY swapped/tampered file from ever being loaded.
    """
    try:
        expected_hash = hash_path.read_text().strip()
    except FileNotFoundError:
        logger.warning("No hash manifest found at %s — cannot verify model integrity.", hash_path)
        return False
 
    try:
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        logger.warning("No model file found at %s — cannot verify model integrity.", model_path)
        return False
 
    if not secrets.compare_digest(actual_hash, expected_hash):
        logger.error(
            "MODEL INTEGRITY CHECK FAILED: %s does not match the hash recorded at training "
            "time. Refusing to load — the file may have been tampered with.",
            model_path,
        )
        return False
 
    return True
 
# Shared secret between the Backend and this AI service, per the squad's own
# convention ("every request carries X-Service-Key"). Set it as an
# environment variable in real deployment.
#
# Security Finding 1 fix: a hardcoded fallback key is visible to anyone who
# reads the source (i.e. anyone with repo access), so it must never be used
# outside local development. The fallback is now gated behind an explicit
# ENVIRONMENT=development flag; everywhere else (staging, production, or if
# ENVIRONMENT isn't set at all) the service refuses to start rather than run
# with a known key.
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production").lower()
_DEV_ONLY_DEFAULT_KEY = "local-dev-only-change-me"

SERVICE_KEY = os.environ.get("AI_SERVICE_KEY")
if not SERVICE_KEY:
    if ENVIRONMENT == "development":
        SERVICE_KEY = _DEV_ONLY_DEFAULT_KEY
        logger.warning(
            "AI_SERVICE_KEY not set — using the known local-dev default because "
            "ENVIRONMENT=development. This must never happen outside local dev."
        )
    else:
        raise RuntimeError(
            "AI_SERVICE_KEY environment variable is not set. Refusing to start "
            "with a known/hardcoded key. Set AI_SERVICE_KEY, or set "
            "ENVIRONMENT=development explicitly if this really is a local dev run."
        )
 
 
def require_service_key(x_service_key: str = Header(default=None)):
    """FastAPI dependency: rejects the request before any model code runs
    if the caller didn't send the correct X-Service-Key header."""
    if x_service_key is None or not secrets.compare_digest(x_service_key, SERVICE_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Service-Key")
 
# Same thresholds as baseline_features.py — keep in sync if those change
LOW_ATTENDANCE_THRESHOLD = 0.70
REPEATED_FAILED_THRESHOLD = 2
STREAK_THRESHOLD = 2
 
FEATURE_ORDER = [
    "attendance_rate_to_date",
    "trend_last_3",
    "rejected_last_2",
    "consecutive_absences",
    "course_load",
]
 
app = FastAPI(title="ATT Weekly Attendance Risk API", version="1.0")

# Security Finding 5 fix: no max request body size was enforced anywhere,
# so a client could send an arbitrarily large body and make the service
# spend CPU/memory parsing it before Pydantic ever gets a chance to reject
# the fields. This service's real payload is five small numeric fields, so
# the cap is generous but still bounded.
MAX_REQUEST_BODY_BYTES = 10 * 1024  # 10 KB


@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    # Note: HTTPException raised from a raw ASGI/Starlette middleware (as
    # opposed to inside a route handler) is NOT caught by FastAPI's usual
    # exception handling, since ExceptionMiddleware sits *inside* this
    # middleware layer, not outside it. So we return a JSONResponse
    # directly here instead of raising.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            # Malformed Content-Length header — let normal request handling
            # deal with it rather than guessing.
            pass
    else:
        # No Content-Length (e.g. chunked transfer): read and enforce the
        # cap manually so an unbounded stream can't be smuggled through.
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        # Re-inject the already-consumed body so downstream handlers can
        # still read it normally.
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request._receive = receive

    return await call_next(request)
 
# Security Finding 2 fix: allow_origins=["*"] let any website in a user's
# browser call this API cross-origin. Origins now come from an env var
# (comma-separated), so the real backend's origin(s) can be configured per
# environment without touching code. In local dev with nothing set, we fall
# back to typical localhost dev ports only — never "*".
_default_dev_origins = "http://localhost:5173,http://127.0.0.1:5173"
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    _default_dev_origins if ENVIRONMENT == "development" else "",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

if not ALLOWED_ORIGINS:
    logger.warning(
        "ALLOWED_ORIGINS is empty — no browser origin will be able to call this "
        "API cross-origin. Set ALLOWED_ORIGINS to the real frontend/backend "
        "origin(s) (comma-separated) once known."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Service-Key"],
)
 
# Security Finding 3 fix: cap requests per client IP. Combined with
# Finding 1's auth, this stops both simple flooding and systematic
# input-scanning to reverse-engineer the model's scoring behavior.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
# Security Finding 3 (accepted risk, partially mitigated):
# joblib.load() deserializes via pickle, which can execute arbitrary code
# embedded in the file. verify_model_integrity() only protects against a
# SILENT swap/tamper *after* the hash was recorded — it does not eliminate
# pickle's deserialization risk if the model file and its hash manifest are
# compromised together (e.g. inside the training pipeline itself).
# Mitigation in place: write access to MODEL_PATH and MODEL_HASH_PATH is
# restricted to the training pipeline; both should be treated as trusted
# inputs, not attacker-controlled.
# Recommended future work: move to a non-executable serialization format
# (e.g. ONNX) so loading the model can never run arbitrary code, regardless
# of file provenance.
model = None
if verify_model_integrity(MODEL_PATH, MODEL_HASH_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s (integrity verified)", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load model (%s). Running in fallback-only mode.", exc)
else:
    logger.warning("Model integrity check failed or hash missing. Running in fallback-only mode.")
 
 
class RiskRequest(BaseModel):
    attendance_rate_to_date: float = Field(..., ge=0, le=1)
    trend_last_3: float = Field(..., description="Recent 3-week attendance trend, can be negative")
    rejected_last_2: int = Field(..., ge=0, description="Rejected/expired scans in the last 2 weeks")
    consecutive_absences: int = Field(..., ge=0)
    course_load: int = Field(..., ge=0, description="Number of sections the student is enrolled in")
 
 
class RiskResponse(BaseModel):
    risk_probability: Optional[float]
    risk_band: str
    is_flagged: bool
    source: str  # "model" or "baseline_fallback"
 
 
def band_for(prob: float) -> str:
    if prob < 0.25:
        return "low"
    if prob < 0.50:
        return "medium"
    if prob < 0.75:
        return "high"
    return "very_high"
 
 
def baseline_fallback(req: "RiskRequest") -> RiskResponse:
    """
    Same deterministic rules as baseline_features.py, but the reported
    probability is now a graded score built from how far each rule is
    exceeded, not one of two fixed constants (Security Finding 5 fix).
    This isn't a real learned probability — it's still just the baseline —
    but it's no longer trivially guessable from source alone.
    """
    attendance_gap = max(0.0, LOW_ATTENDANCE_THRESHOLD - req.attendance_rate_to_date)
    attendance_score = min(1.0, attendance_gap / LOW_ATTENDANCE_THRESHOLD) * 0.6
 
    rejected_score = min(1.0, req.rejected_last_2 / (REPEATED_FAILED_THRESHOLD * 2)) * 0.2
    streak_score = min(1.0, req.consecutive_absences / (STREAK_THRESHOLD * 2)) * 0.2
 
    prob = round(min(0.97, max(0.03, attendance_score + rejected_score + streak_score)), 3)
    is_flagged = (
        req.attendance_rate_to_date < LOW_ATTENDANCE_THRESHOLD
        or req.rejected_last_2 >= REPEATED_FAILED_THRESHOLD
        or req.consecutive_absences >= STREAK_THRESHOLD
    )
    return RiskResponse(
        risk_probability=prob,
        risk_band=band_for(prob),
        is_flagged=is_flagged,
        source="baseline_fallback",
    )
 
 
@app.get("/health")
def health():
    # Security Finding 4 fix: public health check reveals nothing about
    # internal state anymore — just "is the process up".
    return {"status": "ok"}
 
 
@app.get("/admin/model-status", dependencies=[Depends(require_service_key)])
def model_status():
    # Detailed status now requires the same X-Service-Key as predictions,
    # so it's no longer free reconnaissance for an unauthenticated caller.
    return {"model_loaded": model is not None}
 
 
@app.post("/predict-weekly-risk", response_model=RiskResponse, dependencies=[Depends(require_service_key)])
@limiter.limit("20/minute")  # Security Finding 3 fix
def predict_weekly_risk(request: Request, req: RiskRequest):
    if model is not None:
        try:
            features = [[getattr(req, col) for col in FEATURE_ORDER]]
            prob = float(model.predict_proba(features)[0, 1])
            return RiskResponse(
                risk_probability=prob,
                risk_band=band_for(prob),
                is_flagged=prob >= 0.5,
                source="model",
            )
        except (ValueError, TypeError) as exc:
            # Security Finding 6 fix: expected failure category — malformed
            # or out-of-range input reaching the model. Routine, not suspicious.
            # Security Finding 4 fix: only a fixed, generic message goes to the
            # main log; the raw exception text (which may echo back input
            # values or internal details) goes to the debug-only logger.
            logger.warning("Prediction input rejected by model; using fallback.")
            debug_logger.debug("Prediction input rejected by model: %s", exc)
        except Exception as exc:  # noqa: BLE001
            # Anything else is unexpected and logged at a higher severity so
            # it stands out from routine input errors in the logs/alerts.
            logger.error("UNEXPECTED model prediction failure; using fallback.")
            debug_logger.debug("UNEXPECTED model prediction failure: %s", exc, exc_info=True)
 
    return baseline_fallback(req)
from contextlib import asynccontextmanager
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

from app.db.pgvector import init_db
from app.routers import session, stream, enroll, verify, live_ws, training, auth_router, face_match_admin
from app.security.auth import API_KEY, AUTH_REQUIRED
from app.security.rate_limit import RateLimitMiddleware
from app.services.bg_trainer import training_loop


MOBILE_WEB_DIR = Path(__file__).resolve().parents[2] / "mobile-web"
MOBILE_WEB_DIST = MOBILE_WEB_DIR / "dist"

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    ok = await init_db()
    print(
        f"[BOOT] database_ready={ok} LOG_LEVEL={_LOG_LEVEL} "
        f"AUTH_REQUIRED={AUTH_REQUIRED} API_KEY_SET={bool(API_KEY)}",
        flush=True,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(training_loop(stop), name="bg-trainer")
    print("[BOOT] background training loop started", flush=True)
    try:
        yield
    finally:
        stop.set()
        try:
            await asyncio.wait_for(task, timeout=5)
        except Exception:
            task.cancel()


app = FastAPI(title="Liveness Verification - Decision API", version="1.1.0", lifespan=lifespan)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(verify.router, tags=["verify"])
app.include_router(session.router, prefix="/session", tags=["session"])
app.include_router(stream.router, prefix="/session", tags=["stream"])
app.include_router(enroll.router, prefix="/account", tags=["enrollment"])
app.include_router(live_ws.router, tags=["live"])
app.include_router(training.router, prefix="/training", tags=["training"])
app.include_router(face_match_admin.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "auth_required": AUTH_REQUIRED,
        "version": "1.1.0",
    }


# Prefer Vite production build; fall back to source folder for legacy static files
_mobile_static = MOBILE_WEB_DIST if MOBILE_WEB_DIST.is_dir() else MOBILE_WEB_DIR
if _mobile_static.is_dir():
    app.mount("/app", StaticFiles(directory=str(_mobile_static), html=True), name="mobile-web")

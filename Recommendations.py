"""
StyleSphere – FastAPI Application Entry Point
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import init_db
from routes.garments import router as garments_router
from routes.recommendations import router as recommendations_router
from services.ai_pipeline import pose_estimator, body_segmentor

app = FastAPI(
    title="StyleSphere API",
    description="Virtual Wardrobe & AI Try-On backend with DIP pipeline",
    version="1.0.0",
)

# ── CORS (allow React dev server) ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files for processed images ────────────────────────────────────────
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(garments_router)
app.include_router(recommendations_router)


@app.on_event("startup")
def on_startup():
    """Initialise DB tables and AI models on server start."""
    init_db()
    # Lazy-load AI models (graceful if mediapipe not installed)
    try:
        pose_estimator.initialise()
    except Exception as e:
        print(f"[Startup] Pose estimator init skipped: {e}")
    try:
        body_segmentor.initialise()
    except Exception as e:
        print(f"[Startup] Body segmentor init skipped: {e}")


@app.get("/")
def root():
    return {
        "app": "StyleSphere",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}
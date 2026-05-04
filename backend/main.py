"""
StyleSphere – FastAPI Application Entry Point
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import init_db, get_db
from sqlalchemy.orm import Session
from models import Garment
from routes.garments import router as garments_router
from services.ai_pipeline import pose_estimator, body_segmentor, virtual_try_on
import cv2
import numpy as np
import base64
from fastapi import File, Form, UploadFile, HTTPException, Depends

# Load Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    print(f"[Startup] Hugging Face token loaded (authenticated mode)")
else:
    print(f"[Startup] No HF_TOKEN found in .env — using anonymous mode (limited quota)")

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


@app.post("/api/tryon")
async def try_on(
    garment_id: str = Form(...),
    user_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Virtual Try-On Endpoint — powered by OOTDiffusion + IDM-VTON fallback.
    
    OOTDiffusion supports Upper-body, Lower-body, and Dress categories,
    making it ideal for full-length garments like shalwar kameez.
    """
    import tempfile
    import os

    # Fetch garment from DB
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")

    # Read user image bytes
    image_bytes = await user_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    user_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if user_img is None:
        raise HTTPException(status_code=400, detail="Invalid user image")

    # Use the ORIGINAL unprocessed image for AI try-on (not the GrabCut-processed one)
    garment_path = garment.original_image_path or garment.image_path
    if not os.path.exists(garment_path):
        raise HTTPException(status_code=500, detail="Garment image missing from disk")

    # --- Upscale small images for better AI results ---
    MIN_DIM = 768  # minimum pixels on shortest side for good AI output

    def upscale_if_needed(img, label="image"):
        """Upscale image to at least MIN_DIM on shortest side using high-quality LANCZOS4."""
        h, w = img.shape[:2]
        shortest = min(h, w)
        if shortest < MIN_DIM:
            scale = MIN_DIM / shortest
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"[TryOn] Upscaling {label}: {w}x{h} → {new_w}x{new_h} (LANCZOS4)")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # Sharpen after upscale to counteract blurriness from resize
            gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
            print(f"[TryOn] Applied unsharp mask sharpening to {label}")
        return img

    user_img = upscale_if_needed(user_img, "user photo")

    garment_img = cv2.imread(garment_path, cv2.IMREAD_COLOR)
    if garment_img is not None:
        garment_img = upscale_if_needed(garment_img, "garment")

    # Save processed images to temp files for the API call
    tmp_user = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="uploads")
    cv2.imwrite(tmp_user.name, user_img)
    tmp_user.close()

    tmp_garment = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="uploads")
    if garment_img is not None:
        cv2.imwrite(tmp_garment.name, garment_img)
    else:
        # Fallback: copy original file
        import shutil
        shutil.copy2(garment_path, tmp_garment.name)
    tmp_garment.close()

    try:
        from gradio_client import Client, handle_file

        # Determine the correct garment category for OOTDiffusion
        category = (garment.category or "other").lower()
        name = (garment.name or "").lower()

        # South Asian / full-length garment keywords → force "Dress" mode
        sa_keywords = ["shalwar", "kameez", "suit", "kurta", "lehnga", "lehenga",
                       "anarkali", "sharara", "gharara", "frock", "maxi", "abaya",
                       "dress", "gown", "jumpsuit", "romper", "saree", "sari"]

        is_full_length = any(kw in name for kw in sa_keywords) or category in ("dress", "other")

        # Map to OOTDiffusion categories: 'Upper-body', 'Lower-body', 'Dress'
        if is_full_length:
            ootd_category = "Dress"
        elif category == "bottom":
            ootd_category = "Lower-body"
        else:
            ootd_category = "Upper-body"

        print(f"[TryOn] Using OOTDiffusion | Category: {ootd_category} | Garment: {garment.name}")

        result_image_path = None
        ai_model_used = ""

        # === PRIMARY MODEL: OOTDiffusion (supports full-body / Dress) ===
        try:
            client = Client("levihsu/OOTDiffusion", token=HF_TOKEN)

            result = client.predict(
                vton_img=handle_file(tmp_user.name),
                garm_img=handle_file(tmp_garment.name),
                category=ootd_category,
                n_samples=1,
                n_steps=20,
                image_scale=2.0,
                seed=-1,
                api_name="/process_dc"
            )

            # Result is a list of dicts with 'image' key
            if result and len(result) > 0:
                result_image_path = result[0].get("image") if isinstance(result[0], dict) else result[0]
                ai_model_used = "OOTDiffusion"

        except Exception as ootd_err:
            print(f"[TryOn] OOTDiffusion failed: {ootd_err}")

        # === FALLBACK: IDM-VTON ===
        if not result_image_path:
            try:
                print("[TryOn] Falling back to IDM-VTON...")
                client = Client("yisol/IDM-VTON", token=HF_TOKEN)

                color = garment.dominant_color_name or ""
                garment_des = f"{color} {garment.name or 'clothing'}, full length outfit"

                result_paths = client.predict(
                    dict={
                        "background": handle_file(tmp_user.name),
                        "layers": [],
                        "composite": None,
                    },
                    garm_img=handle_file(tmp_garment.name),
                    garment_des=garment_des,
                    is_checked=True,
                    is_checked_crop=False,
                    denoise_steps=30,
                    seed=42,
                    api_name="/tryon"
                )
                result_image_path = result_paths[0]
                ai_model_used = "IDM-VTON"
            except Exception as idm_err:
                print(f"[TryOn] IDM-VTON also failed: {idm_err}")
                raise HTTPException(
                    status_code=503,
                    detail="Both AI models are currently unavailable. Please try again in a few minutes."
                )

        # Read the result image and convert to base64
        result_img = cv2.imread(result_image_path)
        if result_img is None:
            raise HTTPException(status_code=500, detail="AI model returned an invalid image")

        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64_str = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "compatibility_score": 0.95,
            "style_feedback": "",
            "tryon_image_base64": f"data:image/jpeg;base64,{b64_str}"
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"[TryOn AI Error] {error_msg}")
        if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            raise HTTPException(status_code=503, detail="AI GPU quota exceeded. Please try again later or sign up for a free Hugging Face account.")
        raise HTTPException(status_code=500, detail=f"AI Try-On failed: {error_msg}")

    finally:
        try:
            os.unlink(tmp_user.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_garment.name)
        except OSError:
            pass

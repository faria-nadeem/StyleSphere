# main.py - entry point for the StyleSphere backend
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
from routes.recommendations import router as recommendations_router
from services.ai_pipeline import pose_estimator, body_segmentor, virtual_try_on
import cv2
import numpy as np
import base64
from fastapi import File, Form, UploadFile, HTTPException, Depends

# loading the HF token so we can call the AI models
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    print(f"[Startup] Hugging Face token loaded")
else:
    print(f"[Startup] No HF_TOKEN found - anonymous mode")

app = FastAPI(
    title="StyleSphere API",
    description="Virtual Wardrobe & AI Try-On backend",
    version="1.0.0",
)

# allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve uploaded images as static files
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# register route files
app.include_router(garments_router)
app.include_router(recommendations_router)


@app.on_event("startup")
def on_startup():
    init_db()
    # try to load AI models, skip if mediapipe isnt installed
    try:
        pose_estimator.initialise()
    except Exception as e:
        print(f"[Startup] Pose estimator skipped: {e}")
    try:
        body_segmentor.initialise()
    except Exception as e:
        print(f"[Startup] Body segmentor skipped: {e}")


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
    """handles the virtual try-on - uses OOTDiffusion for bottoms, IDM-VTON for tops"""
    import tempfile
    import os

    # get the garment from db
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")

    # read and decode the user's photo
    image_bytes = await user_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    user_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if user_img is None:
        raise HTTPException(status_code=400, detail="Invalid user image")

    # use the processed image (white background) for try-on
    garment_path = garment.image_path
    if not os.path.exists(garment_path):
        raise HTTPException(status_code=500, detail="Garment image missing from disk")

    # upscale small images so the AI models give better results
    MIN_DIM = 768

    def upscale_if_needed(img, label="image"):
        h, w = img.shape[:2]
        shortest = min(h, w)
        if shortest < MIN_DIM:
            scale = MIN_DIM / shortest
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"[TryOn] Upscaling {label}: {w}x{h} -> {new_w}x{new_h}")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # sharpen after upscale so it doesnt look blurry
            gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        return img

    user_img = upscale_if_needed(user_img, "user photo")
    
    # --- Untucked Hack: Stretch the shirt downwards to trick IDM-VTON's auto-parser ---
    category = (garment.category or "other").lower()
    name = (garment.name or "").lower()
    sa_keywords = ["shalwar", "kameez", "suit", "kurta", "lehnga", "lehenga",
                   "anarkali", "sharara", "gharara", "frock", "maxi", "abaya",
                   "dress", "gown", "jumpsuit", "romper", "saree", "sari"]
    is_full_length = any(kw in name for kw in sa_keywords) or category in ("dress", "other")

    if category != "bottom" and not is_full_length:
        try:
            pose = pose_estimator.estimate(user_img)
            seg = body_segmentor.segment(user_img)
            if pose.get("status") == "ok" and seg.get("status") == "ok":
                lm = pose["landmarks"]
                h, w = user_img.shape[:2]
                
                # Check if we have hip landmarks
                if len(lm) > 24:
                    l_hip = (int(lm[23]["x"] * w), int(lm[23]["y"] * h))
                    r_hip = (int(lm[24]["x"] * w), int(lm[24]["y"] * h))
                    
                    hip_y = min(l_hip[1], r_hip[1])
                    hip_width = abs(r_hip[0] - l_hip[0])
                    waist_y = max(0, hip_y - int(hip_width * 0.3))
                    
                    band_height = int(hip_width * 0.15)
                    if waist_y - band_height > 0:
                        band = user_img[waist_y - band_height:waist_y, :]
                        
                        # Target stretch goes down past the hips
                        stretch_target_y = min(h, hip_y + int(hip_width * 0.5))
                        stretch_height = stretch_target_y - waist_y
                        
                        if stretch_height > 0:
                            stretched = cv2.resize(band, (w, stretch_height))
                            mask = np.array(seg["mask"], dtype=np.uint8)
                            
                            for y in range(waist_y, stretch_target_y):
                                row_idx = y - waist_y
                                body_pixels = mask[y, :] > 128
                                user_img[y, body_pixels] = stretched[row_idx, body_pixels]
                                
                            print(f"[TryOn] Stretched shirt downwards by {stretch_height}px to force untucked mask")
        except Exception as e:
            print(f"[TryOn] Failed to stretch shirt: {e}")
    # --- End Untucked Hack ---

    garment_img = cv2.imread(garment_path, cv2.IMREAD_COLOR)
    if garment_img is not None:
        garment_img = upscale_if_needed(garment_img, "garment")

    # save to temp files so we can pass them to the API
    tmp_user = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="uploads")
    cv2.imwrite(tmp_user.name, user_img)
    tmp_user.close()

    tmp_garment = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="uploads")
    if garment_img is not None:
        cv2.imwrite(tmp_garment.name, garment_img)
    else:
        import shutil
        shutil.copy2(garment_path, tmp_garment.name)
    tmp_garment.close()

    try:
        category = (garment.category or "other").lower()
        name = (garment.name or "").lower()
        color = garment.dominant_color_name or ""

        result_img = None

        from gradio_client import Client, handle_file

        # bottoms -> use OOTDiffusion (it supports lower-body)
        if category == "bottom":
            print(f"[TryOn] Bottom category - using OOTDiffusion")
            try:
                client = Client("levihsu/OOTDiffusion", token=HF_TOKEN)
                result = client.predict(
                    vton_img=handle_file(tmp_user.name),
                    garm_img=handle_file(tmp_garment.name),
                    category="Lower-body",
                    n_samples=1,
                    n_steps=30,
                    image_scale=2.0,
                    seed=-1,
                    api_name="/process_dc"
                )
                if result and len(result) > 0:
                    path = result[0].get("image") if isinstance(result[0], dict) else result[0]
                    result_img = cv2.imread(path)
            except Exception as err:
                print(f"[TryOn] OOTDiffusion failed: {err}")
                raise HTTPException(status_code=503, detail="AI model unavailable. Please try again in a few seconds.")

        # tops and dresses -> use IDM-VTON
        else:
            if is_full_length:
                garment_des = f"{color} {name or 'clothing'}, full length flowing outfit, long dress, photorealistic"
            else:
                garment_des = f"{color} {name or 'clothing'}, untucked upper body top shirt blouse, hanging loose outside pants, not tucked in, photorealistic"

            print(f"[TryOn] Using IDM-VTON | Prompt: {garment_des}")

            try:
                client = Client("yisol/IDM-VTON", token=HF_TOKEN)
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
                    denoise_steps=40,
                    seed=-1,
                    api_name="/tryon"
                )
                result_img = cv2.imread(result_paths[0])
            except Exception as idm_err:
                print(f"[TryOn] IDM-VTON failed: {idm_err}")
                if "11001" in str(idm_err) or "getaddrinfo" in str(idm_err):
                    raise HTTPException(status_code=503, detail="Network error: Could not connect to AI server.")
                raise HTTPException(status_code=503, detail="AI model unavailable. Please try again in a few seconds.")

        if result_img is None:
            raise HTTPException(status_code=500, detail="Try-on failed to produce an image.")

        # encode result as base64 jpeg to send back to frontend
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
        print(f"[TryOn Error] {error_msg}")
        if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            raise HTTPException(status_code=503, detail="AI GPU quota exceeded. Please try again later or sign up for a free Hugging Face account.")
        raise HTTPException(status_code=500, detail=f"AI Try-On failed: {error_msg}")

    finally:
        # cleanup temp files
        try:
            os.unlink(tmp_user.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_garment.name)
        except OSError:
            pass

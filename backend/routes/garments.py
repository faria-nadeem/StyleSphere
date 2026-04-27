"""
Garment upload & processing routes.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Garment, User
from services.image_processing import run_full_pipeline
from services.ai_pipeline import pose_estimator, body_segmentor

router = APIRouter(prefix="/api/garments", tags=["garments"])

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _validate_image(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


@router.post("/upload")
async def upload_garment(
    file: UploadFile = File(...),
    name: str = Form("Untitled Garment"),
    category: str = Form("other"),
    user_id: str = Form("default-user"),
    db: Session = Depends(get_db),
):
    """
    Upload an image → run the full DIP pipeline → store features in DB.
    Returns garment metadata + extracted features.
    """
    _validate_image(file.filename)

    # Ensure user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, username=user_id)
        db.add(user)
        db.commit()

    image_bytes = await file.read()
    garment_id = str(uuid.uuid4())
    save_dir = str(UPLOAD_DIR / garment_id)

    # Run DIP pipeline
    try:
        result = run_full_pipeline(image_bytes, save_dir, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Optional AI processing
    pose_data = None
    seg_class = None
    try:
        import cv2, numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            pose_result = pose_estimator.estimate(img)
            pose_data = pose_result if pose_result["status"] == "ok" else None
            seg_result = body_segmentor.segment(img)
            seg_class = "person" if seg_result["status"] == "ok" else None
    except Exception:
        pass  # AI features are optional

    garment = Garment(
        id=garment_id,
        user_id=user_id,
        name=name,
        category=category,
        original_filename=file.filename,
        image_path=result["processed_image_path"],
        mask_path=result["mask_path"],
        dominant_color_hex=result["dominant_color_hex"],
        dominant_color_name=result["dominant_color_name"],
        color_histogram=result["histogram"],
        confidence_score=0.85,
        pose_data=pose_data,
        segmentation_class=seg_class,
    )
    db.add(garment)
    db.commit()
    db.refresh(garment)

    return {
        "id": garment.id,
        "name": garment.name,
        "category": garment.category,
        "dominant_color": {
            "hex": garment.dominant_color_hex,
            "name": garment.dominant_color_name,
        },
        "image_path": garment.image_path,
        "mask_path": garment.mask_path,
        "confidence_score": garment.confidence_score,
        "created_at": str(garment.created_at),
    }


@router.get("/")
def list_garments(user_id: str = "default-user", db: Session = Depends(get_db)):
    """Return all garments for a user."""
    garments = db.query(Garment).filter(Garment.user_id == user_id).order_by(Garment.created_at.desc()).all()
    return [
        {
            "id": g.id,
            "name": g.name,
            "category": g.category,
            "dominant_color": {"hex": g.dominant_color_hex, "name": g.dominant_color_name},
            "image_path": g.image_path,
            "confidence_score": g.confidence_score,
            "created_at": str(g.created_at),
        }
        for g in garments
    ]


@router.get("/{garment_id}")
def get_garment(garment_id: str, db: Session = Depends(get_db)):
    """Fetch a single garment by ID."""
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    return {
        "id": garment.id,
        "name": garment.name,
        "category": garment.category,
        "dominant_color": {"hex": garment.dominant_color_hex, "name": garment.dominant_color_name},
        "image_path": garment.image_path,
        "mask_path": garment.mask_path,
        "color_histogram": garment.color_histogram,
        "pose_data": garment.pose_data,
        "segmentation_class": garment.segmentation_class,
        "confidence_score": garment.confidence_score,
        "created_at": str(garment.created_at),
    }


@router.delete("/{garment_id}")
def delete_garment(garment_id: str, db: Session = Depends(get_db)):
    """Delete a garment."""
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    db.delete(garment)
    db.commit()
    return {"detail": "Garment deleted", "id": garment_id}

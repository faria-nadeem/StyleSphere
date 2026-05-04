# models.py - database tables for users and garments
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    wardrobe = relationship("Garment", back_populates="owner", cascade="all, delete-orphan")


class Garment(Base):
    __tablename__ = "garments"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    category = Column(String(50), default="other")
    original_filename = Column(String(255))
    original_image_path = Column(Text)       # original unprocessed image
    image_path = Column(Text)                # processed image (bg removed)
    mask_path = Column(Text)                 # segmentation mask

    # color features extracted by our pipeline
    dominant_color_hex = Column(String(7))
    dominant_color_name = Column(String(50))
    color_histogram = Column(JSON)
    confidence_score = Column(Float, default=0.0)

    # pose and segmentation data
    pose_data = Column(JSON)
    segmentation_class = Column(String(50))

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="wardrobe")

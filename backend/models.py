"""
SQLAlchemy models for StyleSphere.
Stores garment metadata, extracted features, and user wardrobe data.
"""
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
    category = Column(String(50), default="other")  # top, bottom, dress, shoes, accessory, other
    original_filename = Column(String(255))
    original_image_path = Column(Text)      # path to original unprocessed image (for AI try-on)
    image_path = Column(Text)               # path to processed image (for wardrobe display)
    mask_path = Column(Text)                # path to segmentation mask

    # Extracted features
    dominant_color_hex = Column(String(7))  # e.g. #FF5733
    dominant_color_name = Column(String(50))
    color_histogram = Column(JSON)          # HSV histogram data
    confidence_score = Column(Float, default=0.0)

    # AI pipeline metadata
    pose_data = Column(JSON)                # placeholder for pose keypoints
    segmentation_class = Column(String(50)) # placeholder for body-seg class

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="wardrobe")

# recommendations.py - skin tone detection and garment recommendations
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
import cv2
import numpy as np

from database import get_db
from models import Garment

router = APIRouter()

# skin tone profiles - maps each tone to colors that look good with it
SKIN_TONE_DATA = {
    "Fair": {
        "emoji": "🌸",
        "description": "Your cool, fair complexion pairs beautifully with rich jewel tones and deep contrasts.",
        "style_tip": "Go bold with jewel tones — navy, emerald, and deep burgundy are your power colors.",
        "best_colors": ["navy", "blue", "emerald", "green", "burgundy", "purple",
                        "magenta", "charcoal", "black", "pink", "red", "cobalt",
                        "indigo", "forest", "teal", "dark"],
        "avoid": ["beige", "cream", "ivory", "pale", "light tan", "nude"],
    },
    "Light": {
        "emoji": "🌼",
        "description": "Your warm, light skin tone glows beautifully with earthy and warm hues.",
        "style_tip": "Coral, terracotta, and warm olive tones bring out your natural warmth.",
        "best_colors": ["coral", "teal", "rose", "terracotta", "olive", "rust",
                        "warm", "red", "cobalt", "brown", "peach", "sage",
                        "orange", "burnt", "camel"],
        "avoid": ["neon", "very pale"],
    },
    "Medium": {
        "emoji": "🌻",
        "description": "Your medium skin tone is incredibly versatile — most colors work brilliantly for you!",
        "style_tip": "Earth tones and warm brights like orange, warm red, and mustard yellow complement your skin.",
        "best_colors": ["orange", "red", "yellow", "white", "turquoise", "olive",
                        "green", "earth", "brown", "warm", "tan", "gold",
                        "mustard", "rust", "amber", "caramel"],
        "avoid": [],
    },
    "Tan": {
        "emoji": "🌞",
        "description": "Your gorgeous tan complexion shines with vibrant, bright, and saturated colors.",
        "style_tip": "Bright whites, vivid yellows, and turquoises make your skin absolutely radiate.",
        "best_colors": ["white", "yellow", "orange", "turquoise", "pink", "bright",
                        "red", "blue", "cyan", "lime", "gold", "coral",
                        "fuchsia", "magenta", "aqua", "vivid"],
        "avoid": ["dark brown", "khaki", "mustard", "olive"],
    },
    "Deep": {
        "emoji": "✨",
        "description": "Your rich, deep complexion is stunning — bold and vivid colors make you unforgettable.",
        "style_tip": "Bright whites, yellows, and electric hues are your superpower — go vivid and fearless.",
        "best_colors": ["white", "yellow", "orange", "red", "bright", "hot pink",
                        "lime", "cobalt", "gold", "silver", "electric", "neon",
                        "fuchsia", "turquoise", "aqua", "ivory", "cream"],
        "avoid": ["dark brown", "black", "navy", "deep purple", "charcoal"],
    },
}


def _extract_skin_pixels(img_bgr):
    """uses HSV color ranges to find skin-colored pixels in the image"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # two ranges to cover lighter and deeper skin tones
    lo1 = np.array([0,  15,  80], dtype=np.uint8)
    hi1 = np.array([25, 200, 255], dtype=np.uint8)
    lo2 = np.array([0,  10,  30], dtype=np.uint8)
    hi2 = np.array([20, 180, 200], dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                          cv2.inRange(hsv, lo2, hi2))

    # clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return img_bgr[mask > 0]


def detect_skin_tone(img_bgr):
    """
    figures out the skin tone from a photo
    uses HSV to find skin pixels, then LAB lightness to classify the tone
    returns one of: Fair, Light, Medium, Tan, Deep
    """
    skin_px = _extract_skin_pixels(img_bgr)

    # if we didnt find enough skin pixels, just sample the face area
    if len(skin_px) < 150:
        h, w = img_bgr.shape[:2]
        cx, cy = w // 2, h // 4
        pad = 60
        region = img_bgr[max(0, cy - pad): cy + pad,
                         max(0, cx - pad): cx + pad]
        skin_px = region.reshape(-1, 3)

    if len(skin_px) == 0:
        return "Medium"  # default if nothing works

    # convert to LAB and use the L channel (lightness) to classify
    sample = skin_px.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)
    L = float(lab.reshape(-1, 3)[:, 0].mean())

    if   L >= 185: return "Fair"
    elif L >= 158: return "Light"
    elif L >= 125: return "Medium"
    elif L >=  92: return "Tan"
    else:          return "Deep"


def score_garment(garment, skin_tone):
    """scores how well a garment color matches the skin tone (0 to 1)"""
    tone = SKIN_TONE_DATA.get(skin_tone, SKIN_TONE_DATA["Medium"])
    color_name = (garment.dominant_color_name or "").lower()

    # check if its a color to avoid
    for bad in tone["avoid"]:
        if bad in color_name:
            return 0.15

    # check if its a recommended color
    for good in tone["best_colors"]:
        if good in color_name:
            return 1.0

    return 0.45  # neutral


@router.post("/api/recommendations")
async def get_recommendations(
    user_image: UploadFile = File(...),
    user_id:    str        = Form(default="default-user"),
    db:         Session    = Depends(get_db),
):
    """takes a selfie, detects skin tone, and scores wardrobe garments"""

    # decode the uploaded image
    raw = await user_image.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image — please upload a valid JPEG/PNG.")

    # detect skin tone
    skin_tone = detect_skin_tone(img)
    tone_info = SKIN_TONE_DATA[skin_tone]

    # get all garments for this user
    garments = db.query(Garment).filter(Garment.user_id == user_id).all()

    if not garments:
        return {
            "skin_tone":               skin_tone,
            "skin_tone_emoji":         tone_info["emoji"],
            "description":             tone_info["description"],
            "style_tip":               tone_info["style_tip"],
            "best_colors":             tone_info["best_colors"][:5],
            "recommended_garment_ids": [],
            "all_scores":              {},
        }

    # score each garment and sort by best match
    scored = sorted(
        [(g, score_garment(g, skin_tone)) for g in garments],
        key=lambda x: x[1],
        reverse=True,
    )

    recommended_ids = [g.id for g, s in scored if s >= 0.5]

    return {
        "skin_tone":               skin_tone,
        "skin_tone_emoji":         tone_info["emoji"],
        "description":             tone_info["description"],
        "style_tip":               tone_info["style_tip"],
        "best_colors":             tone_info["best_colors"][:5],
        "recommended_garment_ids": recommended_ids,
        "all_scores":              {g.id: round(s, 2) for g, s in scored},
    }

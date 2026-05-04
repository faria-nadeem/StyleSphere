"""
Digital Image Processing pipeline for StyleSphere.
Handles: AI background removal (rembg/U2-Net), HSV color analysis.
"""
import cv2
import numpy as np
from pathlib import Path
from rembg import remove
from PIL import Image
import io

# ─── Colour name lookup table ───────────────────────────────────────────────
COLOR_RANGES = [
    ("Red",        (0,   50,  50),  (10,  255, 255)),
    ("Orange",     (11,  50,  50),  (25,  255, 255)),
    ("Yellow",     (26,  50,  50),  (34,  255, 255)),
    ("Green",      (35,  50,  50),  (85,  255, 255)),
    ("Cyan",       (86,  50,  50),  (95,  255, 255)),
    ("Blue",       (96,  50,  50),  (130, 255, 255)),
    ("Purple",     (131, 50,  50),  (160, 255, 255)),
    ("Pink",       (161, 50,  50),  (175, 255, 255)),
    ("Red",        (176, 50,  50),  (180, 255, 255)),  # red wraps around
]


def ai_background_removal(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background using rembg (U2-Net deep learning model).
    This is the same class of model used by Zara, remove.bg, etc.
    Returns (foreground_rgba, binary_mask).
    """
    # rembg works on raw bytes and returns RGBA PNG
    result_bytes = remove(image_bytes)

    # Convert result to numpy arrays
    result_pil = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    result_np = np.array(result_pil)

    # Extract the alpha channel as the mask
    alpha = result_np[:, :, 3]
    binary_mask = np.where(alpha > 128, 255, 0).astype(np.uint8)

    # Convert RGBA to BGR for OpenCV (drop alpha, apply mask)
    bgr = cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR)
    foreground = cv2.bitwise_and(bgr, bgr, mask=binary_mask)

    return foreground, binary_mask


def extract_color_features(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Convert to HSV, compute histogram, find dominant colour.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Filter out black/grey background pixels
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Require Saturation > 30 and Value > 30 for it to be considered a 'color'
    color_mask = cv2.bitwise_and(mask, cv2.inRange(s_channel, 30, 255))
    color_mask = cv2.bitwise_and(color_mask, cv2.inRange(v_channel, 30, 255))

    # If the garment is actually just black or white, fallback to original mask
    if cv2.countNonZero(color_mask) < 50:
        color_mask = mask

    # Compute histogram on Hue channel within the valid color mask
    hist_h = cv2.calcHist([hsv], [0], color_mask, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], color_mask, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], color_mask, [256], [0, 256])

    dominant_hue = int(np.argmax(hist_h))
    dominant_sat = int(np.argmax(hist_s))
    dominant_val = int(np.argmax(hist_v))

    # Map hue to a human-readable colour name
    color_name = _hue_to_name(dominant_hue, dominant_sat, dominant_val)

    # Convert dominant HSV → BGR → HEX
    sample = np.uint8([[[dominant_hue, dominant_sat, dominant_val]]])
    bgr = cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)[0][0]
    hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

    return {
        "dominant_color_hex": hex_color,
        "dominant_color_name": color_name,
        "histogram": {
            "hue": hist_h.flatten().tolist(),
            "saturation": hist_s.flatten().tolist(),
            "value": hist_v.flatten().tolist(),
        },
    }


def _hue_to_name(h: int, s: int, v: int) -> str:
    """Map an HSV triplet to a human-readable colour name."""
    if s < 40 and v > 200:
        return "White"
    if v < 40:
        return "Black"
    if s < 40:
        return "Gray"

    for name, lower, upper in COLOR_RANGES:
        if lower[0] <= h <= upper[0]:
            return name
    return "Unknown"


def run_full_pipeline(image_bytes: bytes, save_dir: str, filename: str) -> dict:
    """
    Execute the complete DIP pipeline on raw image bytes.
    Returns paths + extracted features.
    """
    # Decode original image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    # Step 1 — AI Background Removal (U2-Net via rembg)
    foreground, mask = ai_background_removal(image_bytes)

    # Step 2 — Colour Feature Extraction
    features = extract_color_features(foreground, mask)

    # Save artefacts
    original_path = str(save_path / f"{stem}_original.png")
    processed_path = str(save_path / f"{stem}_processed.png")
    mask_path = str(save_path / f"{stem}_mask.png")
    cv2.imwrite(original_path, image)       # Original for AI try-on
    cv2.imwrite(processed_path, foreground)  # Processed for wardrobe display
    cv2.imwrite(mask_path, mask)

    return {
        "original_image_path": original_path,
        "processed_image_path": processed_path,
        "mask_path": mask_path,
        **features,
    }

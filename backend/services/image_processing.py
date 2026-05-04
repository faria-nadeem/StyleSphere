"""
Digital Image Processing pipeline for StyleSphere.
Handles: Gaussian blur, GrabCut segmentation, HSV color analysis.
"""
import cv2
import numpy as np
from pathlib import Path

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


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Step 1 – Noise reduction via Gaussian Blur.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def grabcut_segmentation(image: np.ndarray, iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 2 – Background removal using GrabCut.
    Returns (foreground_image, binary_mask).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Foreground rectangle – keep a 2 pixel margin on every side to capture trousers/long garments
    margin_x, margin_y = 2, 2
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Convert mask: 0/2 → background, 1/3 → foreground
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    foreground = cv2.bitwise_and(image, image, mask=binary_mask)

    return foreground, binary_mask


def extract_color_features(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Step 3 – Convert to HSV, compute histogram, find dominant colour.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Filter out black/grey background pixels that were set to 0 by GrabCut mask
    # This prevents 'H=0' (Black) from being detected as Red.
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
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    # Step 1 — Gaussian Blur
    blurred = apply_gaussian_blur(image)

    # Step 2 — GrabCut Segmentation
    foreground, mask = grabcut_segmentation(blurred)

    # Step 3 — Colour Feature Extraction
    features = extract_color_features(foreground, mask)

    # Save artefacts
    processed_path = str(save_path / f"{stem}_processed.png")
    mask_path = str(save_path / f"{stem}_mask.png")
    cv2.imwrite(processed_path, foreground)
    cv2.imwrite(mask_path, mask)

    return {
        "processed_image_path": processed_path,
        "mask_path": mask_path,
        **features,
    }

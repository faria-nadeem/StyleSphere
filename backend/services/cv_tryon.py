"""
Custom CV-based Virtual Try-On for pants/bottoms.

Instead of pasting a flat garment on top (looks like a sticker),
this uses a structure-preserving color/texture transfer approach:
1. Precise body segmentation (rembg U2-Net) restricted to lower body via pose
2. Warp the pants texture to fill the leg region
3. Extract the 3D structure (shadows, folds, creases) from the original image
4. Multiply-blend: new pants texture × original structure = realistic result
5. Feathered edge blending for seamless compositing
"""
import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from rembg import remove
from PIL import Image
import io

_POSE_MODEL = "pose_landmarker_lite.task"
_POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _ensure_model():
    if not os.path.exists(_POSE_MODEL):
        print("[CV-TryOn] Downloading pose model...")
        urllib.request.urlretrieve(_POSE_URL, _POSE_MODEL)


def _get_body_mask(user_img: np.ndarray) -> np.ndarray:
    """Get precise body segmentation using rembg (U2-Net)."""
    _, img_bytes = cv2.imencode('.png', user_img)
    result_bytes = remove(img_bytes.tobytes())
    result_pil = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    alpha = np.array(result_pil)[:, :, 3]
    return (alpha > 128).astype(np.uint8) * 255


def _get_pose_landmarks(user_img: np.ndarray):
    """Detect pose landmarks using MediaPipe Tasks API."""
    _ensure_model()
    h, w = user_img.shape[:2]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        raise ValueError("Could not detect body pose in user image")

    lm = results.pose_landmarks[0]

    def px(i):
        return (int(lm[i].x * w), int(lm[i].y * h))

    return {
        'l_hip': px(23), 'r_hip': px(24),
        'l_knee': px(25), 'r_knee': px(26),
        'l_ankle': px(27), 'r_ankle': px(28),
        'l_shoulder': px(11), 'r_shoulder': px(12),
    }


def try_on_bottom(user_img: np.ndarray, pants_img: np.ndarray) -> np.ndarray:
    """
    Structure-preserving pants try-on.
    Keeps the person's natural body shape, shadows, and folds.
    Changes only the color/texture to match the target pants.
    """
    h, w = user_img.shape[:2]

    # ── Step 1: Get precise body mask ──
    print("[CV-TryOn] Running body segmentation...")
    body_mask = _get_body_mask(user_img)

    # ── Step 2: Get pose landmarks ──
    print("[CV-TryOn] Detecting pose landmarks...")
    lm = _get_pose_landmarks(user_img)

    l_hip, r_hip = lm['l_hip'], lm['r_hip']
    l_ankle, r_ankle = lm['l_ankle'], lm['r_ankle']

    # Define waist line (above hips — where real waistband sits)
    hip_y = min(l_hip[1], r_hip[1])
    hip_width = abs(r_hip[0] - l_hip[0])
    waist_y = max(0, hip_y - int(hip_width * 0.35))

    # Define ankle cutoff (stop above shoes)
    ankle_y = max(l_ankle[1], r_ankle[1]) - int(hip_width * 0.1)

    # Restrict body mask to lower body only (waist to ankles, no shoes)
    leg_mask = body_mask.copy()
    leg_mask[:waist_y, :] = 0
    leg_mask[min(h, ankle_y):, :] = 0

    # Clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    leg_mask = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    leg_mask = cv2.morphologyEx(leg_mask, cv2.MORPH_OPEN, kernel)

    if cv2.countNonZero(leg_mask) < 100:
        raise ValueError("Could not segment the lower body region")

    print(f"[CV-TryOn] Leg mask: {cv2.countNonZero(leg_mask)} pixels, waist={waist_y}, ankle_cutoff={ankle_y}")

    # ── Step 3: Extract pants color/texture ──
    pants_gray = cv2.cvtColor(pants_img, cv2.COLOR_BGR2GRAY)
    pants_pixel_mask = (pants_gray < 240).astype(np.uint8) * 255

    if cv2.countNonZero(pants_pixel_mask) < 50:
        raise ValueError("Could not detect garment in pants image")

    pants_lab = cv2.cvtColor(pants_img, cv2.COLOR_BGR2LAB).astype(np.float64)
    target_l = np.mean(pants_lab[:, :, 0][pants_pixel_mask > 0])
    target_a = np.mean(pants_lab[:, :, 1][pants_pixel_mask > 0])
    target_b = np.mean(pants_lab[:, :, 2][pants_pixel_mask > 0])

    target_l_std = np.std(pants_lab[:, :, 0][pants_pixel_mask > 0])
    target_a_std = np.std(pants_lab[:, :, 1][pants_pixel_mask > 0])
    target_b_std = np.std(pants_lab[:, :, 2][pants_pixel_mask > 0])

    # ── Step 4: Structure-preserving color transfer ──
    user_lab = cv2.cvtColor(user_img, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Get current stats in the leg region
    src_l = user_lab[:, :, 0][leg_mask > 0]
    src_a = user_lab[:, :, 1][leg_mask > 0]
    src_b = user_lab[:, :, 2][leg_mask > 0]

    src_l_mean, src_l_std = np.mean(src_l), max(np.std(src_l), 1.0)
    src_a_mean, src_a_std = np.mean(src_a), max(np.std(src_a), 1.0)
    src_b_mean, src_b_std = np.mean(src_b), max(np.std(src_b), 1.0)

    # Reinhard color transfer
    recolored_lab = user_lab.copy()

    recolored_lab[:, :, 0] = ((user_lab[:, :, 0] - src_l_mean) *
                               (max(target_l_std, 10.0) / src_l_std) + target_l)
    recolored_lab[:, :, 1] = ((user_lab[:, :, 1] - src_a_mean) *
                               (max(target_a_std, 1.0) / src_a_std) + target_a)
    recolored_lab[:, :, 2] = ((user_lab[:, :, 2] - src_b_mean) *
                               (max(target_b_std, 1.0) / src_b_std) + target_b)

    recolored_lab[:, :, 0] = np.clip(recolored_lab[:, :, 0], 0, 255)
    recolored_lab[:, :, 1] = np.clip(recolored_lab[:, :, 1], 0, 255)
    recolored_lab[:, :, 2] = np.clip(recolored_lab[:, :, 2], 0, 255)

    recolored = cv2.cvtColor(recolored_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Step 5: Feathered edge blending ──
    feathered_mask = cv2.GaussianBlur(leg_mask, (31, 31), 12)

    # Smooth transition at waist (larger zone for natural blend)
    transition_height = int(hip_width * 0.4)
    for y in range(max(0, waist_y), min(h, waist_y + transition_height)):
        t = (y - waist_y) / max(transition_height, 1)
        feathered_mask[y, :] = (feathered_mask[y, :].astype(np.float32) * t).astype(np.uint8)

    # Smooth transition at ankle hem
    hem_height = int(hip_width * 0.2)
    for y in range(max(0, ankle_y - hem_height), min(h, ankle_y)):
        t = 1.0 - ((y - (ankle_y - hem_height)) / max(hem_height, 1))
        feathered_mask[y, :] = (feathered_mask[y, :].astype(np.float32) * t).astype(np.uint8)

    alpha = feathered_mask.astype(np.float32)[:, :, np.newaxis] / 255.0

    result = (recolored.astype(np.float32) * alpha +
              user_img.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    print("[CV-TryOn] Structure-preserving color transfer complete")
    return result


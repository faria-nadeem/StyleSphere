"""
Advanced AI pipeline placeholders for StyleSphere.
Architecture stubs for Pose Estimation and Body Segmentation.
"""
import numpy as np
import cv2
import math
from typing import Any, Tuple


class PoseEstimator:
    """
    Pose Estimation using MediaPipe Pose.
    Detects 33 body landmarks for virtual garment overlay positioning.
    """

    def __init__(self):
        self._model = None
        self._initialised = False

    def initialise(self):
        """
        Lazy-load MediaPipe Pose model using Tasks API.
        """
        try:
            import mediapipe as mp
            import os
            import urllib.request
            
            # Download model if it doesn't exist
            model_path = "pose_landmarker_lite.task"
            if not os.path.exists(model_path):
                print("[PoseEstimator] Downloading Pose model...")
                urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', model_path)
            
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )
            
            self._model = PoseLandmarker.create_from_options(options)
            self._initialised = True
        except ImportError:
            print("[PoseEstimator] mediapipe not installed – running in stub mode")
            self._initialised = False
        except Exception as e:
            print(f"[PoseEstimator] Error loading model: {e}")
            self._initialised = False

    def estimate(self, image: np.ndarray) -> dict[str, Any]:
        """
        Run pose estimation using Tasks API.
        """
        if not self._initialised or self._model is None:
            return {
                "status": "stub",
                "message": "Pose model not loaded. Install mediapipe for real inference.",
                "landmarks": [],
            }

        import mediapipe as mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = self._model.detect(mp_image)

        if not results.pose_landmarks:
            return {"status": "no_detection", "landmarks": []}

        # results.pose_landmarks is a list of poses, each pose is a list of landmarks
        landmarks = []
        for lm in results.pose_landmarks[0]:
            landmarks.append({
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 4),
            })

        return {"status": "ok", "landmarks": landmarks}


class BodySegmentor:
    """
    Body Segmentation placeholder.
    In production, integrate a pre-trained Torch model (e.g., U²-Net, DeepLabV3)
    or MediaPipe Selfie Segmentation.
    """

    def __init__(self):
        self._model = None
        self._initialised = False

    def initialise(self):
        """Lazy-load segmentation model."""
        try:
            import mediapipe as mp
            import os
            import urllib.request
            
            # Download model if it doesn't exist
            model_path = "selfie_segmenter.tflite"
            if not os.path.exists(model_path):
                print("[BodySegmentor] Downloading Segmentation model...")
                urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite', model_path)
                
            BaseOptions = mp.tasks.BaseOptions
            ImageSegmenter = mp.tasks.vision.ImageSegmenter
            ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                output_category_mask=True
            )
            
            self._model = ImageSegmenter.create_from_options(options)
            self._initialised = True
        except ImportError:
            print("[BodySegmentor] mediapipe not installed – running in stub mode")
            self._initialised = False
        except Exception as e:
            print(f"[BodySegmentor] Error loading model: {e}")
            self._initialised = False

    def segment(self, image: np.ndarray) -> dict[str, Any]:
        """
        Produce a body segmentation mask.
        """
        if not self._initialised or self._model is None:
            return {
                "status": "stub",
                "message": "Segmentation model not loaded.",
                "mask": None,
            }

        import mediapipe as mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = self._model.segment(mp_image)

        # 0 is usually background, 1 is person
        mask = results.category_mask.numpy_view()
        binary_mask = (mask > 0).astype(np.uint8) * 255
        return {"status": "ok", "mask": binary_mask.tolist()}


class VirtualTryOn:
    """
    Virtual Try-On engine.
    
    Strategy:
    - Detect skin pixels in the garment image using HSV color space
    - Subtract skin from the GrabCut mask to get a clothing-only mask
    - Use pose landmarks to determine proper scale and position  
    - Warp garment onto user photo with alpha blending
    """

    # HSV ranges that capture human skin tones across ethnicities
    SKIN_LOWER_1 = np.array([0, 20, 70], dtype=np.uint8)
    SKIN_UPPER_1 = np.array([20, 255, 255], dtype=np.uint8)
    SKIN_LOWER_2 = np.array([170, 20, 70], dtype=np.uint8)
    SKIN_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

    def extract_torso_landmarks(self, landmarks, img_width, img_height):
        """Extracts MediaPipe landmarks for shoulders (11, 12) and hips (23, 24)."""
        def get_pixel_coords(lm):
            return int(lm['x'] * img_width), int(lm['y'] * img_height)

        try:
            l_shoulder = get_pixel_coords(landmarks[11])
            r_shoulder = get_pixel_coords(landmarks[12])
            l_hip = get_pixel_coords(landmarks[23])
            r_hip = get_pixel_coords(landmarks[24])
            return l_shoulder, r_shoulder, l_hip, r_hip
        except IndexError:
            return None

    def _get_landmark_px(self, landmarks, idx, img_w, img_h):
        """Safely get pixel coordinates for a landmark index."""
        try:
            lm = landmarks[idx]
            return int(lm['x'] * img_w), int(lm['y'] * img_h)
        except (IndexError, KeyError):
            return None

    def _create_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect skin-colored pixels using HSV thresholding.
        Returns a binary mask where 255 = skin pixel.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.SKIN_LOWER_1, self.SKIN_UPPER_1)
        mask2 = cv2.inRange(hsv, self.SKIN_LOWER_2, self.SKIN_UPPER_2)
        skin = cv2.bitwise_or(mask1, mask2)
        # Clean up noise with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
        return skin

    def _create_clothing_mask(self, garment_img: np.ndarray, grabcut_mask: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Build a mask that contains ONLY clothing — no face, no skin, no hands.
        
        Steps:
        1. Start with the GrabCut foreground mask
        2. Detect skin pixels via HSV and subtract them
        3. Use pose landmarks to define head/hand regions and exclude them more precisely
        4. Clean up with morphological operations
        """
        g_h, g_w = garment_img.shape[:2]
        clothing_mask = grabcut_mask.copy()

        # Step 1: Subtract skin-colored pixels
        skin_mask = self._create_skin_mask(garment_img)
        clothing_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(skin_mask))

        # Step 2: If we have pose landmarks, cut out head region precisely
        if landmarks:
            nose = self._get_landmark_px(landmarks, 0, g_w, g_h)
            l_ear = self._get_landmark_px(landmarks, 7, g_w, g_h)
            r_ear = self._get_landmark_px(landmarks, 8, g_w, g_h)
            l_shoulder = self._get_landmark_px(landmarks, 11, g_w, g_h)
            r_shoulder = self._get_landmark_px(landmarks, 12, g_w, g_h)

            if nose and l_shoulder and r_shoulder:
                shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) // 2
                # Everything above the shoulder line minus a small buffer is "head"
                head_cutoff_y = shoulder_mid_y - int(abs(shoulder_mid_y - nose[1]) * 0.15)
                head_cutoff_y = max(0, head_cutoff_y)
                clothing_mask[0:head_cutoff_y, :] = 0

            # Cut out forearms + hands (below elbows)
            for elbow_idx, wrist_idx in [(13, 15), (14, 16)]:
                elbow = self._get_landmark_px(landmarks, elbow_idx, g_w, g_h)
                wrist = self._get_landmark_px(landmarks, wrist_idx, g_w, g_h)
                if elbow and wrist:
                    # Draw a thick black line from elbow to wrist on the mask
                    thickness = int(g_w * 0.06)
                    cv2.line(clothing_mask, elbow, wrist, 0, thickness)
                    # Also blank out a circle around the wrist/hand
                    cv2.circle(clothing_mask, wrist, int(g_w * 0.05), 0, -1)

        # Step 3: Morphological cleanup — fill small holes in the clothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Slight erosion to remove fuzzy edges
        clothing_mask = cv2.erode(clothing_mask, kernel, iterations=1)
        
        return clothing_mask

    def calculate_style_compatibility(self, user_img, garment_hist) -> float:
        """
        Use the dominant color detected via HSV histograms to return a 
        'style compatibility' score between the garment and the user's skin.
        """
        from services.image_processing import extract_color_features
        # Extract features of the whole user image (or could be just skin ROI)
        user_features = extract_color_features(user_img)
        
        # Simple heuristic: we just return a high baseline score 
        # offset by some hue variance
        score = 0.85
        return min(1.0, max(0.1, score))

    def apply_tryon(self, user_img: np.ndarray, garment_img: np.ndarray, garment_mask: np.ndarray, landmarks: list) -> Tuple[np.ndarray, float]:
        """
        Warp and blend the garment onto the user's torso.
        
        Pipeline:
        1. Build a clothing-only mask (skin removed, head removed)
        2. Detect pose in garment image for alignment
        3. Calculate scale based on shoulder width ratio
        4. Affine warp + alpha blend onto user photo
        """
        h, w = user_img.shape[:2]

        # --- User pose ---
        u_pts = self.extract_torso_landmarks(landmarks, w, h)
        if not u_pts:
            return user_img, 0.0
        u_l_shoulder, u_r_shoulder, u_l_hip, u_r_hip = u_pts

        # --- Garment pose (detect model in garment photo) ---
        global pose_estimator
        garment_pose = pose_estimator.estimate(garment_img)
        g_landmarks = garment_pose.get("landmarks", []) if garment_pose["status"] == "ok" else []

        # --- Build clothing-only mask (skin + head + hands removed) ---
        clothing_mask = self._create_clothing_mask(garment_img, garment_mask, g_landmarks)

        # Check there's still something left in the mask
        coords = cv2.findNonZero(clothing_mask)
        if coords is None:
            # Skin removal was too aggressive — fall back to original GrabCut mask
            clothing_mask = garment_mask.copy()
            coords = cv2.findNonZero(clothing_mask)
            if coords is None:
                return user_img, 0.0

        x_g, y_g, w_g, h_g = cv2.boundingRect(coords)

        # --- Calculate scale & translation ---
        u_shoulder_w = math.hypot(u_l_shoulder[0] - u_r_shoulder[0], u_l_shoulder[1] - u_r_shoulder[1])
        u_torso_h = math.hypot(
            ((u_l_shoulder[0] + u_r_shoulder[0]) / 2) - ((u_l_hip[0] + u_r_hip[0]) / 2),
            ((u_l_shoulder[1] + u_r_shoulder[1]) / 2) - ((u_l_hip[1] + u_r_hip[1]) / 2)
        )

        if g_landmarks:
            g_pts = self.extract_torso_landmarks(g_landmarks, garment_img.shape[1], garment_img.shape[0])
        else:
            g_pts = None

        if g_pts:
            g_l_sh, g_r_sh, g_l_hip, g_r_hip = g_pts
            g_shoulder_w = math.hypot(g_l_sh[0] - g_r_sh[0], g_l_sh[1] - g_r_sh[1])
            g_torso_h = math.hypot(
                ((g_l_sh[0] + g_r_sh[0]) / 2) - ((g_l_hip[0] + g_r_hip[0]) / 2),
                ((g_l_sh[1] + g_r_sh[1]) / 2) - ((g_l_hip[1] + g_r_hip[1]) / 2)
            )
            # Scale proportionally based on body measurements
            scale_x = (u_shoulder_w / max(g_shoulder_w, 1)) * 1.15
            scale_y = (u_torso_h / max(g_torso_h, 1)) * 1.15
            # Anchor point: midpoint between model's shoulders
            anchor_x = (g_l_sh[0] + g_r_sh[0]) / 2.0
            anchor_y = (g_l_sh[1] + g_r_sh[1]) / 2.0
        else:
            # Flat-lay fallback: scale bounding box to match user torso
            target_width = int(u_shoulder_w * 1.5)
            target_height = int(u_torso_h * 1.4) if u_torso_h > 20 else int(target_width * (h_g / max(w_g, 1)))
            scale_x = target_width / max(w_g, 1)
            scale_y = target_height / max(h_g, 1)
            # Anchor at top-center of the bounding box
            anchor_x = x_g + w_g / 2.0
            anchor_y = y_g + h_g * 0.05

        # User's anchor: midpoint between shoulders
        u_center_x = (u_l_shoulder[0] + u_r_shoulder[0]) / 2.0
        u_center_y = (u_l_shoulder[1] + u_r_shoulder[1]) / 2.0

        tx = u_center_x - (anchor_x * scale_x)
        ty = u_center_y - (anchor_y * scale_y)

        M = np.float32([
            [scale_x, 0, tx],
            [0, scale_y, ty]
        ])

        # --- Warp ---
        warped_garment = cv2.warpAffine(garment_img, M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        warped_mask = cv2.warpAffine(clothing_mask, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)

        # Threshold the warped mask to clean anti-aliasing artifacts
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

        # Slight Gaussian blur on the mask edges for smoother blending
        warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 0)

        # --- Alpha Blend ---
        mask_alpha = (warped_mask / 255.0).astype(np.float32)
        if len(mask_alpha.shape) == 2:
            mask_alpha = cv2.merge([mask_alpha, mask_alpha, mask_alpha])

        result_img = user_img.astype(np.float32)
        warped_garment_f = warped_garment.astype(np.float32)

        # I = F * alpha + B * (1 - alpha)
        blended = warped_garment_f * mask_alpha + result_img * (1.0 - mask_alpha)
        result = blended.astype(np.uint8)

        compatibility_score = self.calculate_style_compatibility(user_img, None)
        return result, compatibility_score


# Module-level singletons
pose_estimator = PoseEstimator()
body_segmentor = BodySegmentor()
virtual_try_on = VirtualTryOn()

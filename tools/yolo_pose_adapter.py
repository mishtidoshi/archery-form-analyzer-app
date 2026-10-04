"""
YOLO Pose Adapter — MediaPipe-compatible wrapper
=================================================
Wraps Ultralytics YOLO pose output (v11/v26+) in an interface identical to
MediaPipe Pose, so that archery_analyzer_v2.py requires only minimal changes.

MediaPipe returns 33 named landmarks; COCO-17 (YOLO output) has 17.
All 33 MediaPipe indices are supported — finger-tip landmarks (INDEX, PINKY,
THUMB) not present in COCO-17 are returned with visibility=0.0 and (x,y)=(0,0).
Callers should check .visibility before using those landmarks.

COCO-17 → MediaPipe index mapping:
  COCO 0  nose              → MP 0
  COCO 1  left_eye          → MP 2
  COCO 2  right_eye         → MP 5
  COCO 3  left_ear          → MP 7
  COCO 4  right_ear         → MP 8
  COCO 5  left_shoulder     → MP 11
  COCO 6  right_shoulder    → MP 12
  COCO 7  left_elbow        → MP 13
  COCO 8  right_elbow       → MP 14
  COCO 9  left_wrist        → MP 15
  COCO 10 right_wrist       → MP 16
  COCO 11 left_hip          → MP 23
  COCO 12 right_hip         → MP 24
  COCO 13 left_knee         → MP 25
  COCO 14 right_knee        → MP 26
  COCO 15 left_ankle        → MP 27
  COCO 16 right_ankle       → MP 28

  MP 17/18 (L/R PINKY), 19/20 (L/R INDEX), 21/22 (L/R THUMB) → not in COCO
  MP 29-32 (heel/foot) → not in COCO
"""

import os

import numpy as np

# ── Model path resolution ───────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_ROOT, "models")
STOCK_MODEL_PATH = os.path.join(_MODELS_DIR, "yolo26m-pose.pt")
FINETUNED_MODEL_PATH = os.path.join(_MODELS_DIR, "archery_pose_finetuned.pt")
_LEGACY_ROOT_MODEL_PATH = os.path.join(_ROOT, "yolo26m-pose.pt")


def resolve_pose_model_path() -> str:
    """Single source of truth for "which pose model to load" (previously computed
    independently in archery_analyzer_v2.py and each analyze_*_yolo.py script).

    Prefers a locally fine-tuned model over the stock pretrained one, mirroring the
    load-or-fallback pattern in ml_shot_detector.MLShotDetector: check the file
    exists, use it if so, otherwise fall back with an explanatory print. See
    tools/export_pose_dataset.py + tools/finetune_pose_model.py to produce a
    fine-tuned model from your own footage."""
    if os.path.exists(FINETUNED_MODEL_PATH):
        print(f"[pose] Using fine-tuned model: {FINETUNED_MODEL_PATH}")
        return FINETUNED_MODEL_PATH
    if os.path.exists(STOCK_MODEL_PATH):
        return STOCK_MODEL_PATH
    if os.path.exists(_LEGACY_ROOT_MODEL_PATH):   # legacy location, pre-models/ dir
        return _LEGACY_ROOT_MODEL_PATH
    print(f"[pose] No fine-tuned model found at '{FINETUNED_MODEL_PATH}'. "
          f"Using stock model (auto-downloads on first use): {STOCK_MODEL_PATH}. "
          f"Run tools/export_pose_dataset.py + tools/finetune_pose_model.py to train "
          f"your own on your own footage.")
    return STOCK_MODEL_PATH


# COCO keypoint index → MediaPipe PoseLandmark index
_COCO_TO_MP = {
    0:  0,   # nose
    1:  2,   # left_eye
    2:  5,   # right_eye
    3:  7,   # left_ear
    4:  8,   # right_ear
    5:  11,  # left_shoulder
    6:  12,  # right_shoulder
    7:  13,  # left_elbow
    8:  14,  # right_elbow
    9:  15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

# YOLO11 skeleton connections (pairs of COCO indices) for overlay drawing
YOLO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]


class _Landmark:
    """Mimics a single MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class _PoseLandmarks:
    """Mimics MediaPipe NormalizedLandmarkList."""
    def __init__(self, landmark_list):
        self.landmark = landmark_list   # list of 33 _Landmark objects


class _YoloResult:
    """Mimics MediaPipe Pose process() result."""
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks   # None or _PoseLandmarks


class YoloPoseAdapter:
    """
    Drop-in replacement for mp.solutions.pose.Pose().

    Usage (identical to MediaPipe):
        with YoloPoseAdapter(model_path) as pose:
            result = pose.process(bgr_frame)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                x = lm[12].x   # right shoulder, normalized 0-1
    """

    # Minimum mean keypoint confidence to accept a detection as valid
    MIN_CONFIDENCE = 0.30

    def __init__(self, model_path: str, conf: float = 0.25):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf = conf
        self._empty_landmarks = [_Landmark() for _ in range(33)]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def process(self, bgr_frame: np.ndarray) -> _YoloResult:
        h, w = bgr_frame.shape[:2]
        results = self._model(bgr_frame, verbose=False, conf=self._conf)

        boxes = results[0].boxes
        kpts  = results[0].keypoints

        if kpts is None or len(kpts.xy) == 0:
            return _YoloResult(None)

        # Pick the person with the largest bounding box (primary subject is
        # always closest to camera, so largest in frame).
        if boxes is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best = int(areas.argmax())
        else:
            best = 0

        xy   = kpts.xy[best].cpu().numpy()    # (17, 2) pixel coords
        conf = (kpts.conf[best].cpu().numpy()
                if kpts.conf is not None
                else np.ones(17, dtype=np.float32))

        # Reject if overall confidence is too low
        if float(conf.mean()) < self.MIN_CONFIDENCE:
            return _YoloResult(None)

        # Build 33-element MediaPipe-compatible landmark list
        lm_list = [_Landmark() for _ in range(33)]
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            px, py = xy[coco_idx]
            c = float(conf[coco_idx])
            lm_list[mp_idx] = _Landmark(
                x=float(px) / w,
                y=float(py) / h,
                z=0.0,
                visibility=c,
            )

        return _YoloResult(_PoseLandmarks(lm_list))

    def draw_skeleton(self, frame: np.ndarray, pose_landmarks) -> np.ndarray:
        """Draw COCO-17 skeleton on frame. Returns frame (in-place)."""
        import cv2
        lm = pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Draw connections
        for (a, b) in YOLO_CONNECTIONS:
            mp_a = _COCO_TO_MP.get(a)
            mp_b = _COCO_TO_MP.get(b)
            if mp_a is None or mp_b is None:
                continue
            la, lb = lm[mp_a], lm[mp_b]
            if la.visibility < 0.2 or lb.visibility < 0.2:
                continue
            pa = (int(la.x * w), int(la.y * h))
            pb = (int(lb.x * w), int(lb.y * h))
            cv2.line(frame, pa, pb, (0, 255, 0), 2)

        # Draw keypoints
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            l = lm[mp_idx]
            if l.visibility < 0.2:
                continue
            px = (int(l.x * w), int(l.y * h))
            color = (0, 255, 255) if l.visibility > 0.6 else (0, 165, 255)
            cv2.circle(frame, px, 5, color, -1)
            cv2.circle(frame, px, 5, (255, 255, 255), 1)

        return frame

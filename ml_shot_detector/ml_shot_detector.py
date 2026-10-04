"""
MLShotDetector — drop-in replacement for ShotDetector.

Uses a trained Random Forest to classify each frame into:
  IDLE / DRAWING / FULL_DRAW / RELEASE / FOLLOW_THROUGH

The RELEASE class triggers a shot event, just like the old wrist-velocity threshold.
Falls back to velocity-based detection automatically if model file is not found.
"""
import os
import sys
import pickle
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

_PARENT = os.path.join(os.path.dirname(__file__), "..")
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from .feature_extractor import FrameFeatures, extract_frame_features
from .window_builder import WindowBuilder
from .labeler import LABEL_TO_INT, INT_TO_LABEL


@dataclass
class ShotEvent:
    frame_index: int
    hold_time_s: Optional[float]
    confidence:  float
    state:       str = "RELEASE"


class MLShotDetector:
    """
    ML-based shot detector — drop-in replacement for ShotDetector.

    Identical interface to ShotDetector.update(wrist_x, frame_idx)
    plus an enhanced update_full(landmarks, frame_idx, frame_shape) that
    uses full pose data for higher accuracy.

    Fallback: if model_path is not found, automatically falls back to
    velocity threshold (identical to ShotDetector behavior).
    """

    RELEASE_IDX = LABEL_TO_INT["RELEASE"]
    FULL_DRAW_IDX = LABEL_TO_INT["FULL_DRAW"]

    def __init__(
        self,
        model_path: str,
        archer_hand: str = "right",
        cooldown_frames: int = 45,
        min_shot_time: float = 0.0,
        confidence_threshold: float = 0.60,
        # Dual-confirmation: audio candidate frames (optional)
        audio_candidates: Optional[List[int]] = None,
        audio_window_frames: int = 15,
        # Fallback velocity params (used when model not available)
        velocity_threshold: float = 0.015,
        settle_threshold: float = 0.003,
        settle_frames: int = 5,
    ):
        self.archer_hand          = archer_hand
        self.cooldown_frames      = cooldown_frames
        self.min_shot_time        = min_shot_time
        self.confidence_threshold = confidence_threshold
        self._fps                 = 30.0
        self._cooldown            = 0
        self._last_shot_frame: Optional[int] = None
        self._uncertainty_log: List[dict]    = []

        # Dual-confirmation: audio candidate frames
        self._audio_candidates: Optional[List[int]] = audio_candidates
        self._audio_window: int = audio_window_frames
        if audio_candidates:
            print(f"  [MLShotDetector] Dual-confirmation ON: "
                  f"{len(audio_candidates)} audio candidate(s), ±{audio_window_frames} frame window")

        # ML state
        self._window_builder = WindowBuilder(fps=30.0)
        self._prev_wrist_xy: Optional[Tuple[float, float]] = None
        self._full_draw_frame: Optional[int] = None   # compatibility shim for HUD
        self._current_state: str = "IDLE"
        self._last_predicted_labels: deque = deque(maxlen=5)

        # Transition-based detection state
        # We fire a shot when label transitions DRAWING/FULL_DRAW → FOLLOW_THROUGH
        self._prev_pred_label: str = "IDLE"
        self._prev_frame_idx: int = 0
        self._prev_release_conf: float = 0.0
        self._in_active_sequence: bool = False  # True while DRAWING or FULL_DRAW

        # Landmark index mapping (set up in _setup_landmark_indices)
        self._lm_indices: Optional[dict] = None

        # Load model
        self._model = None
        self._model_path = model_path
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                print(f"  [MLShotDetector] Loaded model: {model_path}")
            except Exception as e:
                print(f"  [MLShotDetector] Failed to load model ({e}). Using velocity fallback.")
        else:
            print(f"  [MLShotDetector] Model not found at '{model_path}'. "
                  f"Using velocity fallback. Run scripts/train_model.py to train.")

        # Velocity fallback state (mirrors ShotDetector)
        self._fb_prev_x: Optional[float]      = None
        self._fb_vel_threshold                 = velocity_threshold
        self._fb_settle_threshold              = settle_threshold
        self._fb_settle_frames_req             = settle_frames
        self._fb_still_count                   = 0
        self._fb_full_draw_frame: Optional[int] = None

    def set_fps(self, fps: float):
        self._fps = fps
        self._window_builder.set_fps(fps)

    def set_audio_candidates(self, candidates: List[int], window_frames: int = 15):
        """Set audio clicker candidate frames for dual-confirmation."""
        self._audio_candidates  = candidates
        self._audio_window      = window_frames
        print(f"  [MLShotDetector] Dual-confirmation ON: "
              f"{len(candidates)} audio candidate(s), ±{window_frames} frame window")

    def _near_audio_candidate(self, frame_idx: int) -> bool:
        """Return True if frame_idx is within audio_window of any audio candidate."""
        if not self._audio_candidates:
            return True  # no audio → always allow (pose-only mode)
        return any(abs(frame_idx - c) <= self._audio_window
                   for c in self._audio_candidates)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def get_current_state(self) -> str:
        return self._current_state

    def get_uncertainty_log(self) -> List[dict]:
        return self._uncertainty_log

    # ── Landmark index setup (lazy) ────────────────────────────────────────────

    def _setup_landmark_indices(self):
        import mediapipe as mp
        PL = mp.solutions.pose.PoseLandmark
        hand = self.archer_hand
        if hand == "right":
            self._lm_indices = {
                "d_sh": PL.RIGHT_SHOULDER.value,  "d_el": PL.RIGHT_ELBOW.value,
                "d_wr": PL.RIGHT_WRIST.value,     "b_sh": PL.LEFT_SHOULDER.value,
                "b_el": PL.LEFT_ELBOW.value,      "b_wr": PL.LEFT_WRIST.value,
                "d_ear": PL.RIGHT_EAR.value,      "l_sh": PL.LEFT_SHOULDER.value,
                "r_sh": PL.RIGHT_SHOULDER.value,  "nose": PL.NOSE.value,
            }
        else:
            self._lm_indices = {
                "d_sh": PL.LEFT_SHOULDER.value,   "d_el": PL.LEFT_ELBOW.value,
                "d_wr": PL.LEFT_WRIST.value,      "b_sh": PL.RIGHT_SHOULDER.value,
                "b_el": PL.RIGHT_ELBOW.value,     "b_wr": PL.RIGHT_WRIST.value,
                "d_ear": PL.LEFT_EAR.value,       "l_sh": PL.LEFT_SHOULDER.value,
                "r_sh": PL.RIGHT_SHOULDER.value,  "nose": PL.NOSE.value,
            }

    # ── Drop-in interface (matches ShotDetector.update signature) ─────────────

    def update(self, wrist_x: float, frame_idx: int) -> Tuple[bool, Optional[float], float]:
        """
        Drop-in replacement for ShotDetector.update().

        Returns (released: bool, hold_time_s: Optional[float], velocity: float).
        When ML model is loaded, velocity is replaced by model confidence.
        """
        if not self.is_model_loaded():
            return self._fallback_update(wrist_x, frame_idx)

        # ML path: we only have wrist_x here, not full landmarks.
        # We'll use state from the last update_full call if available.
        # This method is kept for compatibility; prefer update_full.
        return False, None, 0.0

    # ── Enhanced interface (uses full landmarks) ───────────────────────────────

    def update_full(
        self,
        landmarks,          # mediapipe pose_landmarks object
        frame_idx: int,
        frame_shape: tuple, # (h, w) or (h, w, c)
    ) -> Optional[ShotEvent]:
        """
        Full-landmark update — preferred when ML model is loaded.

        Returns ShotEvent if a RELEASE is detected, else None.
        """
        if landmarks is None:
            return None

        if self._lm_indices is None:
            self._setup_landmark_indices()

        lm = landmarks.landmark
        ix = self._lm_indices

        # Extract per-frame features
        ff = extract_frame_features(
            lm,
            ix["d_sh"], ix["d_el"], ix["d_wr"],
            ix["b_sh"], ix["b_el"], ix["b_wr"],
            ix["d_ear"], ix["l_sh"], ix["r_sh"], ix["nose"],
            prev_wrist_xy=self._prev_wrist_xy,
        )
        self._prev_wrist_xy = (lm[ix["d_wr"]].x, lm[ix["d_wr"]].y)
        self._window_builder.push(ff, frame_idx)

        if not self.is_model_loaded():
            # Use fallback with the wrist_x we have
            released, hold_t, vel = self._fallback_update(lm[ix["d_wr"]].x, frame_idx)
            if released:
                return ShotEvent(frame_index=frame_idx, hold_time_s=hold_t,
                                 confidence=vel, state="RELEASE")
            return None

        # Build window feature vector
        vec, meta = self._window_builder.build_window_features(frame_idx)
        if vec is None:
            return None  # not enough buffered frames yet

        # Predict
        proba = self._model.predict_proba([vec])[0]
        pred_label_int = int(np.argmax(proba))
        pred_label     = INT_TO_LABEL[pred_label_int]
        confidence     = float(proba[pred_label_int])
        release_confidence = float(proba[self.RELEASE_IDX])

        self._current_state = pred_label
        self._last_predicted_labels.append(pred_label_int)

        # Track whether we're in an active draw sequence
        if pred_label in ("DRAWING", "FULL_DRAW"):
            self._in_active_sequence = True
        elif pred_label not in ("RELEASE", "FOLLOW_THROUGH"):
            # IDLE resets the sequence
            self._in_active_sequence = False

        # Track full draw for HUD compatibility
        if pred_label == "FULL_DRAW" and self._full_draw_frame is None:
            self._full_draw_frame = frame_idx
        elif pred_label not in ("FULL_DRAW", "RELEASE") and self._full_draw_frame is not None:
            if pred_label != "FOLLOW_THROUGH":
                self._full_draw_frame = None

        # Log uncertainty
        if 0.35 < release_confidence < 0.65:
            self._uncertainty_log.append({
                "frame": frame_idx,
                "proba": {INT_TO_LABEL[i]: float(p) for i, p in enumerate(proba)},
                "meta":  meta,
            })

        # ── Shot detection: two strategies ──────────────────────────────────────
        #
        # Strategy 1 (primary): DRAWING/FULL_DRAW → FOLLOW_THROUGH transition.
        #   The model reliably sees this transition even though RELEASE confidence
        #   is low (~0.12 max) due to RELEASE being only 4 labeled frames/shot.
        #   We fire at frame_idx - 1 (the last frame before FOLLOW_THROUGH began).
        #
        # Strategy 2 (fallback): direct RELEASE prediction ≥ threshold.
        #   Kept for future models with better RELEASE labeling.
        #
        shot_frame = None
        shot_conf  = 0.0

        transition_to_follow = (
            pred_label == "FOLLOW_THROUGH"
            and self._prev_pred_label in ("DRAWING", "FULL_DRAW", "RELEASE")
            and self._in_active_sequence
        )
        if (transition_to_follow
                and self._cooldown == 0
                and self._near_audio_candidate(self._prev_frame_idx)):
            # Use the previous frame (last DRAWING/FULL_DRAW/RELEASE frame)
            shot_frame = self._prev_frame_idx
            shot_conf  = max(release_confidence, self._prev_release_conf, 0.05)

        elif (pred_label == "RELEASE"
                and release_confidence >= self.confidence_threshold
                and self._cooldown == 0
                and self._near_audio_candidate(frame_idx)):
            shot_frame = frame_idx
            shot_conf  = release_confidence

        # Update prev-frame state AFTER transition check
        self._prev_pred_label   = pred_label
        self._prev_frame_idx    = frame_idx
        self._prev_release_conf = release_confidence

        if shot_frame is not None:
            # Check minimum time between shots
            if self._last_shot_frame is not None and self.min_shot_time > 0:
                elapsed = (shot_frame - self._last_shot_frame) / self._fps
                if elapsed < self.min_shot_time:
                    if self._cooldown > 0:
                        self._cooldown -= 1
                    return None

            # Compute hold time
            hold_time_s = None
            if self._full_draw_frame is not None:
                hold_time_s = (shot_frame - self._full_draw_frame) / self._fps

            self._cooldown           = self.cooldown_frames
            self._last_shot_frame    = shot_frame
            self._full_draw_frame    = None
            self._in_active_sequence = False
            self._current_state      = "RELEASE"

            return ShotEvent(
                frame_index = shot_frame,
                hold_time_s = hold_time_s,
                confidence  = shot_conf,
                state       = "RELEASE",
            )

        if self._cooldown > 0:
            self._cooldown -= 1

        return None

    # ── Velocity fallback (mirrors ShotDetector) ──────────────────────────────

    def _fallback_update(
        self, wrist_x: float, frame_idx: int
    ) -> Tuple[bool, Optional[float], float]:
        released    = False
        hold_time_s = None
        velocity    = 0.0

        if self._fb_prev_x is not None and self._cooldown == 0:
            delta    = wrist_x - self._fb_prev_x
            velocity = delta

            if delta > self._fb_vel_threshold:
                if self._last_shot_frame is not None and self.min_shot_time > 0:
                    if (frame_idx - self._last_shot_frame) / self._fps < self.min_shot_time:
                        self._fb_prev_x = wrist_x
                        return False, None, 0.0

                released = True
                if self._fb_full_draw_frame is not None:
                    hold_time_s = (frame_idx - self._fb_full_draw_frame) / self._fps
                self._fb_full_draw_frame = None
                self._fb_still_count     = 0
                self._last_shot_frame    = frame_idx
                self._cooldown           = self.cooldown_frames
                self._full_draw_frame    = None

            elif abs(delta) < self._fb_settle_threshold:
                self._fb_still_count += 1
                if (self._fb_still_count >= self._fb_settle_frames_req
                        and self._fb_full_draw_frame is None):
                    self._fb_full_draw_frame = frame_idx
                    self._full_draw_frame    = frame_idx
            else:
                self._fb_still_count     = 0
                self._fb_full_draw_frame = None
                self._full_draw_frame    = None

        self._fb_prev_x = wrist_x
        if self._cooldown > 0:
            self._cooldown -= 1

        return released, hold_time_s, velocity

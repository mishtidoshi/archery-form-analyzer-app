# -*- coding: utf-8 -*-
"""
Archery Form Analyzer - v3.0 (YOLO11 Pose Edition)
===================================================
Analyzes Olympic recurve archery form from up to 3 simultaneous camera angles:
  - Face view   : draw/bow elbow angles, anchor point, follow-through
                  (perpendicular to shooting line, facing archer's FACE side)
  - Back view   : shoulder levelness, bow arm alignment, lateral head tilt
                  (perpendicular to shooting line, facing archer's BACK/SPINE — T-shape)
  - Target view : draw elbow height, back tension, string alignment
                  (directly BEHIND archer along arrow line, facing the target)
  - ALL views   : hold time (time from full draw to release, in seconds)

Each angle is processed independently, shots are synchronized by number,
and a unified multi-panel report is generated combining all available angles.

Requirements:
  pip install ultralytics opencv-python numpy matplotlib
  # Download model weights (once): yolo26m-pose.pt from ultralytics/assets
  # (stored in models/ directory)

  # Legacy MediaPipe fallback (--pose-backend mediapipe):
  pip install mediapipe==0.10.9

Usage (provide any combination of angles):
  python archery_analyzer_v2.py --face face.MOV --back back.MOV --target target.MOV

  # Just two angles:
  python archery_analyzer_v2.py --face face.MOV --back back.MOV

  # Single angle (same as v1):
  python archery_analyzer_v2.py --face face.MOV

Optional flags:
  --hand        right|left   archer's dominant hand (default: right)
  --out         ./output     output folder
  --threshold   0.015        shot detection sensitivity (higher = less sensitive, default: 0.015)
  --min-time    0            minimum seconds between shots (default: 0; AV detector enforces 10s floor)
  --max-shots   0            keep only the N strongest detections (0 = keep all, useful for target angle)
"""

import cv2
# mediapipe is imported lazily, only when --pose-backend mediapipe is used
# (its legacy solutions API is gone from Python-3.13 wheels). See tools/pose_landmarks.py.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import json
import os
import sys
import collections
from dataclasses import dataclass
from typing import Optional, List, Dict

# Backend-independent pose landmark constants (no mediapipe dependency).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from pose_landmarks import PoseLandmark
import hold_time as _hold_time   # shared hold-time definition (BB11)
from yolo_pose_adapter import resolve_pose_model_path

# Default YOLO model path — override with --yolo-model. Prefers a fine-tuned model
# over the stock one if present; see yolo_pose_adapter.resolve_pose_model_path().
_DEFAULT_YOLO_MODEL = resolve_pose_model_path()


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def angle_at_joint(a, b, c) -> float:
    """Angle in degrees at joint B, given points A, B, C as (x,y) tuples."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def pt(lm, idx, w, h):
    """Pixel (x,y) of landmark index idx."""
    l = lm[idx]
    return (int(l.x * w), int(l.y * h))


def norm(lm, idx):
    """Normalized (x,y) of landmark index idx."""
    return (lm[idx].x, lm[idx].y)


def horizontal_angle(p1, p2) -> float:
    """Angle of the line p1->p2 relative to horizontal, in degrees."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-shot data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SideShot:
    shot_number: int
    frame_index: int
    draw_elbow_angle:    Optional[float] = None   # deg - angle at draw elbow (at release)
    bow_elbow_angle:     Optional[float] = None   # deg - angle at bow elbow
    anchor_x:            Optional[float] = None   # normalized (at release)
    anchor_y:            Optional[float] = None   # normalized (at release)
    shoulder_tilt:       Optional[float] = None   # deg - shoulder line vs horizontal
    head_tilt:           Optional[float] = None   # deg - nose vs midpoint of shoulders
    hold_time_s:         Optional[float] = None   # seconds from full draw to release
    detection_velocity:  Optional[float] = None   # velocity of release detection (strength)
    # Full draw metrics — captured at anchor settle, before the release snap
    draw_elbow_angle_fd: Optional[float] = None   # deg - draw elbow angle when settled at anchor
    anchor_x_fd:         Optional[float] = None   # normalized anchor X at full draw
    anchor_y_fd:         Optional[float] = None   # normalized anchor Y at full draw
    # Bow grip metrics — captured at release frame
    bow_wrist_angle:         Optional[float] = None   # deg - angle at bow wrist (b_el→b_wr→b_index); high-wrist push grip ≈150-165°
    bow_grip_spread:         Optional[float] = None   # index-to-pinky dist / wrist-to-index dist; low = relaxed fingers (good)
    # Bow shoulder and draw anchor metrics
    bow_shoulder_elevation:  Optional[float] = None   # normalized ear-to-shoulder dist; larger = shoulder depressed (good)
    draw_pinky_neck_dist:    Optional[float] = None   # normalized draw-pinky to draw-ear dist; smaller = pinky at neck (good)
    # Nose / head metrics — captured at full draw and release
    nose_string_gap:         Optional[float] = None   # anchor_x_fd - nose_x_fd; near 0 = light contact; negative = squish
    nose_y_fd:               Optional[float] = None   # normalized nose Y at full draw; increasing = chin dropping
    nose_movement:           Optional[float] = None   # normalized distance nose traveled FD→release; >0.025 = head moved
    nose_preanchor_drift:    Optional[float] = None   # nose Y drift (pixels) from ~25fr before shot to release; positive = chin dropped approaching anchor
    # Follow-through metrics — captured ~20 frames after release
    draw_elbow_followthrough: Optional[float] = None  # draw elbow angle post-release; should be > release angle (expansion)
    bow_elbow_followthrough:  Optional[float] = None  # bow elbow angle post-release; drop = bow arm collapsed


@dataclass
class FrontShot:
    shot_number: int
    frame_index: int
    shoulder_level:     Optional[float] = None   # deg - shoulder line vs horizontal (should ~0)
    bow_wrist_x:        Optional[float] = None   # normalized - lateral bow wrist position
    bow_wrist_y:        Optional[float] = None   # normalized
    head_lateral_tilt:  Optional[float] = None   # deg - ear-to-ear line vs horizontal
    hip_alignment:      Optional[float] = None   # deg - hip line vs horizontal
    bow_arm_vertical:   Optional[float] = None   # deg - bow shoulder to bow wrist vertical alignment
    t_draw_angle:       Optional[float] = None   # deg - horizontal angle between bow elbow and draw elbow; 0 = perfect T
    hold_time_s:        Optional[float] = None   # seconds from full draw to release
    detection_velocity: Optional[float] = None   # velocity of release detection (strength)
    # Full draw metrics — captured at anchor settle
    shoulder_level_fd:      Optional[float] = None   # deg - shoulder level at full draw onset
    hip_alignment_fd:       Optional[float] = None   # deg - hip alignment at full draw onset
    bow_shoulder_elevation: Optional[float] = None   # normalized ear-to-shoulder dist; larger = shoulder depressed (good)


@dataclass
class BehindShot:
    shot_number: int
    frame_index: int
    draw_elbow_height:  Optional[float] = None   # normalized Y - elbow height relative to shoulder (neg = above)
    draw_elbow_lateral: Optional[float] = None   # normalized X - elbow flare behind body line
    shoulder_rotation:  Optional[float] = None   # deg - shoulder line from behind (rotation / cant indicator)
    back_tension_proxy: Optional[float] = None   # normalized shoulder width (proxy for scapula engagement)
    draw_wrist_x:       Optional[float] = None   # for shot detection
    hold_time_s:        Optional[float] = None   # seconds from full draw to release
    detection_velocity: Optional[float] = None   # velocity of release detection (strength)
    # Full draw metrics — captured at anchor settle
    draw_elbow_height_fd: Optional[float] = None   # draw elbow height at full draw onset
    back_tension_fd:      Optional[float] = None   # back tension at full draw onset


# ─────────────────────────────────────────────────────────────────────────────
#  Shot detector
# ─────────────────────────────────────────────────────────────────────────────

class ShotDetector:
    """
    Detects full draw and release by tracking the draw wrist X position.

    Full draw is detected when the wrist stops moving (settles into anchor).
    Release is detected when the wrist snaps forward rapidly.
    Hold time = elapsed frames between full-draw onset and release / fps.
    """
    def __init__(self, velocity_threshold=0.015, settle_threshold=0.003,
                 settle_frames=5, cooldown_frames=45, min_shot_time=0):
        self.prev_x             = None
        self.release_threshold  = velocity_threshold   # forward snap speed
        self.settle_threshold   = settle_threshold     # wrist considered "settled"
        self.settle_frames      = settle_frames        # consecutive still frames = full draw
        self.cooldown_frames    = cooldown_frames
        self.cooldown           = 0
        self.min_shot_time      = min_shot_time        # minimum seconds between shots

        # Hold time tracking state
        self._still_count            = 0       # consecutive frames wrist has been still
        self._moving_count           = 0       # consecutive frames wrist has been moving (BB11)
        self._track                  = {}      # frame -> (x, y) draw-wrist buffer (BB11)
        self._track_frames           = 240     # ~8s at 30fps; covers the longest holds seen
        self._full_draw_frame        = None    # frame index when full draw was detected
        self._fps                    = 30.0    # updated by processor before use
        self._last_shot_frame        = None    # frame of last detected shot
        self.full_draw_just_detected = False   # True for exactly one update() call at FD onset

    def set_fps(self, fps: float):
        self._fps = fps

    def update(self, wrist_x: float, frame_idx: int, wrist_y: float = None):
        """
        Returns (released: bool, hold_time_s: Optional[float], velocity: float).
        hold_time_s is only non-None on the frame where release is detected.
        velocity is the forward snap speed (used to measure detection strength).
        Sets self.full_draw_just_detected=True for exactly one frame at full-draw onset.

        BB11 (2026-07-28): when `wrist_y` is supplied, hold time is measured by
        `tools/hold_time.measure_hold_from_release` over the buffered 2D track, so
        this path and the per-view analysis share one definition. Without it, the
        legacy X-only settle logic below is used, which systematically overshoots
        because vertical draw motion is invisible to it.
        """
        self.full_draw_just_detected = False   # reset at start of every call
        released    = False
        hold_time_s = None
        velocity    = 0.0

        # Buffer the 2D track so hold time can use the shared velocity definition.
        if wrist_y is not None:
            self._track[frame_idx] = (wrist_x, wrist_y)
            cutoff = frame_idx - self._track_frames
            if cutoff in self._track:
                for f in [g for g in self._track if g < cutoff]:
                    del self._track[f]

        if self.prev_x is not None and self.cooldown == 0:
            delta = wrist_x - self.prev_x
            velocity = delta

            # Detect release: wrist snaps forward
            if delta > self.release_threshold:
                # Check minimum time between shots
                if self._last_shot_frame is not None and self.min_shot_time > 0:
                    time_since_last = (frame_idx - self._last_shot_frame) / self._fps
                    if time_since_last < self.min_shot_time:
                        # Too soon after last shot, ignore this detection
                        self.prev_x = wrist_x
                        return False, None, 0.0

                released = True
                if wrist_y is not None and len(self._track) > 10:
                    # Shared definition (BB11) — same onset rule as the per-view path.
                    hold_time_s = _hold_time.measure_hold_from_release(
                        self._track, frame_idx, self._fps)["hold_time_s"]
                elif self._full_draw_frame is not None:
                    # Legacy X-only fallback. Overshoots; kept only so callers that
                    # do not pass wrist_y keep working.
                    hold_frames = frame_idx - self._full_draw_frame
                    hold_time_s = hold_frames / self._fps
                self._full_draw_frame = None
                self._still_count     = 0
                self._moving_count    = 0
                self._last_shot_frame = frame_idx
                self.cooldown         = self.cooldown_frames

            # Detect full draw: wrist has been still for several frames
            elif abs(delta) < self.settle_threshold:
                self._still_count += 1
                self._moving_count = 0
                if self._still_count >= self.settle_frames and self._full_draw_frame is None:
                    self._full_draw_frame        = frame_idx
                    self.full_draw_just_detected = True
            else:
                # Wrist is moving. Require SUSTAINED motion before concluding the
                # archer is drawing again — a single frame over the gate is keypoint
                # jitter, not movement.
                #
                # BB11 (2026-07-28): this reset used to fire on ONE frame, which
                # truncated the hold to whatever uninterrupted quiet run happened to
                # precede the release. On noisier sessions that produced 0.37-0.67s
                # holds where the same shots measure 2.5-5.0s under the shared
                # velocity definition in tools/hold_time.py — the source of the
                # long-standing "is it half a second or three seconds?" conflict.
                # `settle_frames` consecutive moving frames mirrors the sustain rule
                # used there.
                self._moving_count += 1
                if self._moving_count >= self.settle_frames:
                    self._still_count     = 0
                    self._full_draw_frame = None

        self.prev_x = wrist_x
        if self.cooldown > 0:
            self.cooldown -= 1

        return released, hold_time_s, velocity


# ─────────────────────────────────────────────────────────────────────────────
#  Single-angle processor base
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
C_GREEN   = "#00ff88"
C_ORANGE  = "#ff6b6b"
C_YELLOW  = "#ffaa00"
C_BLUE    = "#60a5fa"
C_PURPLE  = "#a78bfa"
C_CYAN    = "#22d3ee"
WHITE     = "#ffffff"


class AngleProcessor:
    """Processes a single video for a given camera angle."""

    def __init__(self, video_path: str, angle: str, archer_hand: str, output_dir: str,
                 velocity_threshold: float = 0.015, min_shot_time: float = 0,
                 use_ml: bool = False, ml_model_path: str = "models/rf_current.pkl",
                 write_video: bool = True, use_av: bool = True,
                 n_shots: Optional[int] = None,
                 pose_backend: str = "yolo", yolo_model: str = _DEFAULT_YOLO_MODEL):
        self.video_path  = video_path
        self.angle       = angle.lower()          # "face" | "back" | "target"
        self.hand        = archer_hand.lower()    # "right" | "left"
        self.output_dir  = output_dir
        self.use_ml      = use_ml
        self.use_av      = use_av and not use_ml  # audio-visual is default unless ML requested
        self.write_video = write_video
        self.n_shots     = n_shots
        os.makedirs(output_dir, exist_ok=True)

        self.pose_backend = pose_backend    # "yolo" or "mediapipe"
        self.yolo_model   = yolo_model

        # Landmark constants are backend-independent (local IntEnum, no mediapipe).
        self.PL = PoseLandmark

        # mediapipe objects only exist when the mediapipe backend is selected.
        self.mp_pose = self.mp_drawing = self.mp_styles = None
        if self.pose_backend == "mediapipe":
            import mediapipe as mp
            self.mp_pose    = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_styles  = mp.solutions.drawing_styles

        if use_ml:
            from ml_shot_detector import MLShotDetector
            from ml_shot_detector.clicker_detector import detect_all_clicker_frames
            # Pre-scan audio for clicker candidates (dual-confirmation)
            audio_candidates = []
            try:
                frames, info = detect_all_clicker_frames(video_path)
                if frames:
                    # Audio lags ~2 frames behind actual release (sound travel time)
                    audio_candidates = [max(0, f - 2) for f in frames]
                    print(f"  [audio] Clicker candidate(s): {info}")
                else:
                    print(f"  [audio] No clicker detected — pose-only mode")
            except Exception as e:
                print(f"  [audio] Audio scan failed ({e}) — pose-only mode")

            self.detector = MLShotDetector(
                model_path        = ml_model_path,
                archer_hand       = archer_hand,
                cooldown_frames   = 45,
                min_shot_time     = min_shot_time,
                audio_candidates  = audio_candidates or None,
                audio_window_frames = 15,
            )
        elif self.use_av:
            # Default: audio onset + visual anchor zone check
            import sys, os as _os
            sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "tools"))
            from audio_visual_shot_detector import AudioVisualShotDetector
            # prefer_strongest_audio default in the detector is tuned for face
            # view, where the bow-mounted clicker is loud relative to noise.
            # In back/target views the mic sits behind the archer's body and
            # the clicker is barely louder than wind/equipment noise; the
            # strongest-pick filter then favors non-shot loud noise events
            # over real clicker fires. Disable per-view to avoid that regression.
            prefer_strongest = (angle.lower() == "face")
            self.detector = AudioVisualShotDetector(
                angle             = angle,
                archer_hand       = archer_hand,
                fps               = 30.0,          # updated by set_fps() in process()
                min_shot_interval = max(min_shot_time, 12.0),
                n_shots           = n_shots,
                prefer_strongest_audio = prefer_strongest,
            )
        else:
            self.detector = ShotDetector(velocity_threshold=velocity_threshold,
                                         min_shot_time=min_shot_time)
        self.shots: list  = []
        self._fd_snapshot: dict = {}        # buffered pose metrics at full-draw onset
        self._pending_followthrough: dict = {}  # {shot, target_frame} — sample elbow angles ~20f post release

        # Landmark indices per hand
        PL = self.PL
        if self.hand == "right":
            self.D_SH, self.D_EL, self.D_WR = PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST
            self.B_SH, self.B_EL, self.B_WR = PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,  PL.LEFT_WRIST
            self.D_EAR, self.B_EAR           = PL.RIGHT_EAR,      PL.LEFT_EAR
            self.D_HIP, self.B_HIP           = PL.RIGHT_HIP,      PL.LEFT_HIP
        else:
            self.D_SH, self.D_EL, self.D_WR = PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,  PL.LEFT_WRIST
            self.B_SH, self.B_EL, self.B_WR = PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST
            self.D_EAR, self.B_EAR           = PL.LEFT_EAR,       PL.RIGHT_EAR
            self.D_HIP, self.B_HIP           = PL.LEFT_HIP,       PL.RIGHT_HIP
        self.NOSE = PL.NOSE

    def _update_detector(self, wrist_x: float, frame_idx: int, wrist_y: float):
        """Wraps self.detector.update() — only the local ShotDetector class (BB11)
        understands the wrist_y kwarg for velocity-based hold time; AudioVisualShotDetector
        and MLShotDetector each compute hold time their own way from just (wrist_x,
        frame_idx) and don't accept it. Calling every backend the same way from the three
        _extract_* methods without this check crashes as soon as AudioVisualShotDetector
        (the default) confirms a shot — see BB11 follow-up, 2026-08-14."""
        if isinstance(self.detector, ShotDetector):
            return self.detector.update(wrist_x, frame_idx, wrist_y=wrist_y)
        return self.detector.update(wrist_x, frame_idx)

    # ── Metric extraction per angle ───────────────────────────────────────

    def _extract_side(self, lm, w, h, frame_idx, shot_count,
                       landmarks_obj=None) -> Optional[SideShot]:
        d_sh = pt(lm, self.D_SH, w, h);  d_el = pt(lm, self.D_EL, w, h);  d_wr = pt(lm, self.D_WR, w, h)
        b_sh = pt(lm, self.B_SH, w, h);  b_el = pt(lm, self.B_EL, w, h);  b_wr = pt(lm, self.B_WR, w, h)
        l_sh = pt(lm, self.PL.LEFT_SHOULDER,  w, h)
        r_sh = pt(lm, self.PL.RIGHT_SHOULDER, w, h)
        nose = pt(lm, self.NOSE, w, h)
        mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)

        # Bow hand finger landmarks for grip analysis
        # YOLO11 (COCO-17) does not include finger keypoints — degrade gracefully
        PL = self.PL
        if self.hand == "right":
            b_idx_lm = lm[PL.LEFT_INDEX]
            b_pnk_lm = lm[PL.LEFT_PINKY]
            d_pnk_lm = lm[PL.RIGHT_PINKY]
        else:
            b_idx_lm = lm[PL.RIGHT_INDEX]
            b_pnk_lm = lm[PL.RIGHT_PINKY]
            d_pnk_lm = lm[PL.LEFT_PINKY]

        _has_fingers = (b_idx_lm.visibility > 0.1 and b_pnk_lm.visibility > 0.1)

        if _has_fingers:
            b_idx = (int(b_idx_lm.x * w), int(b_idx_lm.y * h))
            b_pnk = (int(b_pnk_lm.x * w), int(b_pnk_lm.y * h))
            bow_wrist_angle = angle_at_joint(b_el, b_wr, b_idx)
            finger_dist     = np.linalg.norm(np.array(b_idx) - np.array(b_pnk))
            wrist_to_idx    = np.linalg.norm(np.array(b_wr)  - np.array(b_idx)) + 1e-9
            bow_grip_spread = float(finger_dist / wrist_to_idx)
        else:
            bow_wrist_angle = None   # not available from YOLO COCO-17
            bow_grip_spread = None

        # Bow shoulder elevation: normalized distance from bow ear to bow shoulder.
        # When the shoulder rides up toward the ear, this value shrinks — coach cue "keep shoulder low".
        b_ear_px = pt(lm, self.B_EAR, w, h)
        bow_shoulder_elevation = np.linalg.norm(np.array(b_ear_px) - np.array(b_sh)) / h

        # Draw pinky to neck: only available when finger landmarks present (MediaPipe)
        d_ear_px = pt(lm, self.D_EAR, w, h)
        if _has_fingers and d_pnk_lm.visibility > 0.1:
            d_pnk = (int(d_pnk_lm.x * w), int(d_pnk_lm.y * h))
            draw_pinky_neck_dist = np.linalg.norm(np.array(d_pnk) - np.array(d_ear_px)) / h
        else:
            draw_pinky_neck_dist = None

        if self.use_ml and landmarks_obj is not None:
            from ml_shot_detector import MLShotDetector
            event = self.detector.update_full(landmarks_obj, frame_idx, (h, w))
            fired    = event is not None
            hold_time_s = event.hold_time_s if event else None
            velocity    = event.confidence  if event else 0.0
        else:
            fired, hold_time_s, velocity = self._update_detector(
                lm[self.D_WR].x, frame_idx, lm[self.D_WR].y)
        if fired:
            s = SideShot(
                shot_number        = shot_count,
                frame_index        = frame_idx,
                draw_elbow_angle   = angle_at_joint(d_sh, d_el, d_wr),
                bow_elbow_angle    = angle_at_joint(b_sh, b_el, b_wr),
                anchor_x           = lm[self.D_WR].x,
                anchor_y           = lm[self.D_WR].y,
                shoulder_tilt      = horizontal_angle(l_sh, r_sh),
                head_tilt          = horizontal_angle(mid_sh, nose),
                hold_time_s        = hold_time_s,
                detection_velocity = velocity,
                bow_wrist_angle    = bow_wrist_angle,
                bow_grip_spread    = bow_grip_spread,
                bow_shoulder_elevation = bow_shoulder_elevation,
                draw_pinky_neck_dist   = draw_pinky_neck_dist,
            )
            return s
        return None

    def _extract_front(self, lm, w, h, frame_idx, shot_count,
                        landmarks_obj=None) -> Optional[FrontShot]:
        PL = self.PL
        l_sh  = pt(lm, PL.LEFT_SHOULDER,  w, h)
        r_sh  = pt(lm, PL.RIGHT_SHOULDER, w, h)
        l_ear = pt(lm, PL.LEFT_EAR,       w, h)
        r_ear = pt(lm, PL.RIGHT_EAR,      w, h)
        l_hip = pt(lm, PL.LEFT_HIP,       w, h)
        r_hip = pt(lm, PL.RIGHT_HIP,      w, h)
        b_sh  = pt(lm, self.B_SH, w, h)
        b_wr  = pt(lm, self.B_WR, w, h)
        d_el  = pt(lm, self.D_EL, w, h)
        b_el  = pt(lm, self.B_EL, w, h)

        # T-draw alignment: horizontal angle of the line from bow elbow to draw elbow.
        # At a perfect "T" both elbows are at the same height → angle = 0°.
        # Positive = draw elbow higher; negative = bow elbow higher.
        t_draw_angle = horizontal_angle(b_el, d_el)

        # Bow shoulder elevation (back view): ear-to-shoulder gap normalized by frame height.
        # Smaller = bow shoulder riding up toward ear = coach cue violation.
        b_ear_bk = pt(lm, self.B_EAR, w, h)
        bow_shoulder_elevation = np.linalg.norm(np.array(b_ear_bk) - np.array(b_sh)) / h

        if self.use_ml and landmarks_obj is not None:
            from ml_shot_detector import MLShotDetector
            event = self.detector.update_full(landmarks_obj, frame_idx, (h, w))
            fired    = event is not None
            hold_time_s = event.hold_time_s if event else None
            velocity    = event.confidence  if event else 0.0
        else:
            fired, hold_time_s, velocity = self._update_detector(
                lm[self.D_WR].x, frame_idx, lm[self.D_WR].y)
        if fired:
            # Bow arm vertical alignment: how vertical is the line from bow shoulder to bow wrist?
            bow_vert = horizontal_angle(b_sh, b_wr) + 90  # offset so 0 = perfectly vertical
            s = FrontShot(
                shot_number        = shot_count,
                frame_index        = frame_idx,
                shoulder_level     = horizontal_angle(l_sh, r_sh),
                bow_wrist_x        = lm[self.B_WR].x,
                bow_wrist_y        = lm[self.B_WR].y,
                head_lateral_tilt  = horizontal_angle(l_ear, r_ear),
                hip_alignment      = horizontal_angle(l_hip, r_hip),
                bow_arm_vertical   = bow_vert,
                t_draw_angle       = t_draw_angle,
                bow_shoulder_elevation = bow_shoulder_elevation,
                hold_time_s        = hold_time_s,
                detection_velocity = velocity,
            )
            return s
        return None

    def _extract_behind(self, lm, w, h, frame_idx, shot_count,
                         landmarks_obj=None) -> Optional[BehindShot]:
        PL = self.PL
        d_sh  = pt(lm, self.D_SH, w, h)
        d_el  = pt(lm, self.D_EL, w, h)
        l_sh  = pt(lm, PL.LEFT_SHOULDER,  w, h)
        r_sh  = pt(lm, PL.RIGHT_SHOULDER, w, h)

        # Back tension proxy: normalized distance between shoulder blade landmarks
        # Using shoulder width as a proxy (back tension pulls them together/apart)
        sh_dist = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
        frame_diag = np.sqrt(w**2 + h**2)
        back_tension = sh_dist / frame_diag

        # Draw elbow height relative to draw shoulder (negative = elbow below shoulder)
        elbow_relative_y = (d_el[1] - d_sh[1]) / h   # normalized; negative = higher

        # Draw elbow lateral flare (how far behind the body line)
        elbow_lateral = lm[self.D_EL].x - lm[self.D_SH].x

        # Bow cant proxy: from behind, bow cant is best reflected in the shoulder line tilt.
        # shoulder_rotation captures this — no separate bow_tilt field needed.

        if self.use_ml and landmarks_obj is not None:
            from ml_shot_detector import MLShotDetector
            event = self.detector.update_full(landmarks_obj, frame_idx, (h, w))
            fired    = event is not None
            hold_time_s = event.hold_time_s if event else None
            velocity    = event.confidence  if event else 0.0
        else:
            fired, hold_time_s, velocity = self._update_detector(
                lm[self.D_WR].x, frame_idx, lm[self.D_WR].y)
        if fired:
            s = BehindShot(
                shot_number         = shot_count,
                frame_index         = frame_idx,
                draw_elbow_height   = elbow_relative_y,
                draw_elbow_lateral  = elbow_lateral,
                shoulder_rotation   = horizontal_angle(l_sh, r_sh),
                back_tension_proxy  = back_tension,
                draw_wrist_x        = lm[self.D_WR].x,
                hold_time_s         = hold_time_s,
                detection_velocity  = velocity,
            )
            return s
        return None

    # ── Overlay drawing ───────────────────────────────────────────────────

    def _draw_overlay(self, frame, landmarks, shot_count, pose_estimator=None):
        if self.pose_backend == "yolo" and pose_estimator is not None:
            pose_estimator.draw_skeleton(frame, landmarks)
        elif self.mp_drawing is not None:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
            )
        h, w = frame.shape[:2]
        lm = landmarks.landmark

        if self.angle == "face":
            d_el = pt(lm, self.D_EL, w, h)
            b_el = pt(lm, self.B_EL, w, h)
            d_sh = pt(lm, self.D_SH, w, h)
            d_wr = pt(lm, self.D_WR, w, h)
            b_sh = pt(lm, self.B_SH, w, h)
            b_wr = pt(lm, self.B_WR, w, h)
            a1 = angle_at_joint(d_sh, d_el, d_wr)
            a2 = angle_at_joint(b_sh, b_el, b_wr)
            cv2.putText(frame, f"Draw: {a1:.1f}deg", (d_el[0]+10, d_el[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,136), 2)
            cv2.putText(frame, f"Bow:  {a2:.1f}deg", (b_el[0]+10, b_el[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
            anchor_px = (int(lm[self.D_WR].x * w), int(lm[self.D_WR].y * h))
            cv2.drawMarker(frame, anchor_px, (0,200,255), cv2.MARKER_CROSS, 20, 2)

        elif self.angle == "back":
            PL = self.PL
            l_sh = pt(lm, PL.LEFT_SHOULDER,  w, h)
            r_sh = pt(lm, PL.RIGHT_SHOULDER, w, h)
            sh_angle = horizontal_angle(l_sh, r_sh)
            mid = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2 - 20)
            cv2.putText(frame, f"Shoulders: {sh_angle:.1f}deg", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,100), 2)
            # Draw shoulder line
            cv2.line(frame, l_sh, r_sh, (255,100,100), 2)

        elif self.angle == "target":
            d_el = pt(lm, self.D_EL, w, h)
            d_sh = pt(lm, self.D_SH, w, h)
            PL_b = self.PL
            b_sh = pt(lm, self.B_SH, w, h)
            rel_h = (d_el[1] - d_sh[1]) / h
            cv2.putText(frame, f"Elbow rel: {rel_h:.3f}", (d_el[0]+10, d_el[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,200,255), 2)
            # Shoulder line — proxy for bow cant from behind
            sh_tilt = horizontal_angle(b_sh, d_sh)   # bow→draw: +ve means draw side is lower
            tilt_color = (0, 255, 0) if abs(sh_tilt) <= 8 else (0, 80, 255)
            mid_x = (d_sh[0] + b_sh[0]) // 2
            mid_y = min(d_sh[1], b_sh[1]) - 18
            cv2.putText(frame, f"Sh tilt: {sh_tilt:+.1f}deg", (mid_x - 60, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tilt_color, 2)
            cv2.line(frame, d_sh, b_sh, tilt_color, 2)

        # HUD — compatible with both ShotDetector and MLShotDetector
        if hasattr(self.detector, 'get_current_state'):
            ml_state   = self.detector.get_current_state()
            hold_state = "AT FULL DRAW" if ml_state == "FULL_DRAW" else ml_state
        else:
            hold_state = "AT FULL DRAW" if self.detector._full_draw_frame is not None else "drawing..."
        last_hold  = self.shots[-1].hold_time_s if self.shots and self.shots[-1].hold_time_s else None
        hud = [
            f"Angle : {self.angle.upper()}",
            f"Shots : {shot_count}",
            f"State : {hold_state}",
            f"Last hold: {last_hold:.2f}s" if last_hold else "Last hold: --",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (5,5), (240,100), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        state_color = (0, 255, 136) if hold_state == "AT FULL DRAW" else (200, 200, 200)
        for i, line in enumerate(hud):
            color = state_color if i == 2 else (255, 255, 255)
            cv2.putText(frame, line, (12, 26 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return frame

    # ── Main process loop ─────────────────────────────────────────────────

    def process(self) -> str:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.video_path}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_path = os.path.join(self.output_dir, f"analyzed_{self.angle}.mp4")
        if self.write_video:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        else:
            writer = None

        shot_count = 0
        frame_idx  = 0
        self.detector.set_fps(fps)
        # Rolling nose-Y buffer for pre-anchor drift detection (face view only)
        _nose_y_buf: collections.deque = collections.deque(maxlen=25)

        # Audio-visual detector: extract candidates before the frame loop
        if self.use_av and hasattr(self.detector, 'extract_candidates'):
            self.detector.extract_candidates(self.video_path)

        extract_fn = {"face": self._extract_side,
                      "back": self._extract_front,
                      "target": self._extract_behind}[self.angle]

        print(f"\n  Processing [{self.angle.upper()}] view: {self.video_path}")

        # Build the pose estimator per backend. YOLO adapter is its own context
        # manager; mediapipe Pose is only instantiated for the mediapipe backend.
        if self.pose_backend == "yolo":
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
            from yolo_pose_adapter import YoloPoseAdapter
            pose_ctx = YoloPoseAdapter(self.yolo_model)
            print(f"  [pose] Using YOLO backend: {self.yolo_model}")
        else:
            pose_ctx = self.mp_pose.Pose(min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5,
                                         model_complexity=1)
            print(f"  [pose] Using MediaPipe backend")

        with pose_ctx as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                if self.pose_backend == "yolo":
                    results = pose.process(frame)   # YOLO accepts BGR directly
                else:
                    results = pose.process(rgb)
                rgb.flags.writeable = True

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    shot_count_before = shot_count

                    # Track nose Y for pre-anchor drift (face view only)
                    if self.angle == "face":
                        _nose_y_buf.append(lm[self.NOSE].y)

                    # Feed landmarks to AV detector before shot check
                    if self.use_av and hasattr(self.detector, 'feed_pose'):
                        self.detector.feed_pose(frame_idx, lm, w, h)

                    shot = extract_fn(lm, w, h, frame_idx, shot_count + 1,
                                      landmarks_obj=results.pose_landmarks)

                    # ── FD snapshot: two paths ───────────────────────────────────────
                    # Path A (AV detector): FD lm is stored on the detector at shot time
                    av_fd_lm = getattr(self.detector, 'fd_lm', None) if shot is not None else None
                    if av_fd_lm is not None:
                        fd_lm = av_fd_lm
                        if self.angle == "face":
                            d_sh = pt(fd_lm, self.D_SH, w, h)
                            d_el = pt(fd_lm, self.D_EL, w, h)
                            self._fd_snapshot = {
                                'draw_elbow_angle_fd': angle_at_joint(d_sh, d_el, pt(fd_lm, self.D_WR, w, h)),
                                'anchor_x_fd': fd_lm[self.D_WR].x,
                                'anchor_y_fd': fd_lm[self.D_WR].y,
                                'nose_x_fd':   fd_lm[self.NOSE].x,
                                'nose_y_fd':   fd_lm[self.NOSE].y,
                            }
                        elif self.angle == "back":
                            PL_fd = self.PL
                            l_sh_fd = pt(fd_lm, PL_fd.LEFT_SHOULDER,  w, h)
                            r_sh_fd = pt(fd_lm, PL_fd.RIGHT_SHOULDER, w, h)
                            l_hip_fd = pt(fd_lm, PL_fd.LEFT_HIP,  w, h)
                            r_hip_fd = pt(fd_lm, PL_fd.RIGHT_HIP, w, h)
                            self._fd_snapshot = {
                                'shoulder_level_fd': horizontal_angle(l_sh_fd, r_sh_fd),
                                'hip_alignment_fd':  horizontal_angle(l_hip_fd, r_hip_fd),
                            }
                        elif self.angle == "target":
                            d_sh_fd = pt(fd_lm, self.D_SH, w, h)
                            d_el_fd = pt(fd_lm, self.D_EL, w, h)
                            PL_fd = self.PL
                            l_sh_fd = pt(fd_lm, PL_fd.LEFT_SHOULDER,  w, h)
                            r_sh_fd = pt(fd_lm, PL_fd.RIGHT_SHOULDER, w, h)
                            sh_dist_fd = np.linalg.norm(np.array(l_sh_fd) - np.array(r_sh_fd))
                            self._fd_snapshot = {
                                'draw_elbow_height_fd': (d_el_fd[1] - d_sh_fd[1]) / h,
                                'back_tension_fd':      sh_dist_fd / np.sqrt(w**2 + h**2),
                            }

                    # Path B (velocity / ML detector): FD detected at the settle frame
                    elif getattr(self.detector, 'full_draw_just_detected', False):
                        if self.angle == "face":
                            d_sh = pt(lm, self.D_SH, w, h)
                            d_el = pt(lm, self.D_EL, w, h)
                            self._fd_snapshot = {
                                'draw_elbow_angle_fd': angle_at_joint(d_sh, d_el, pt(lm, self.D_WR, w, h)),
                                'anchor_x_fd': lm[self.D_WR].x,
                                'anchor_y_fd': lm[self.D_WR].y,
                                'nose_x_fd':   lm[self.NOSE].x,
                                'nose_y_fd':   lm[self.NOSE].y,
                            }
                        elif self.angle == "back":
                            PL_fd = self.PL
                            l_sh_fd = pt(lm, PL_fd.LEFT_SHOULDER,  w, h)
                            r_sh_fd = pt(lm, PL_fd.RIGHT_SHOULDER, w, h)
                            l_hip_fd = pt(lm, PL_fd.LEFT_HIP,  w, h)
                            r_hip_fd = pt(lm, PL_fd.RIGHT_HIP, w, h)
                            self._fd_snapshot = {
                                'shoulder_level_fd': horizontal_angle(l_sh_fd, r_sh_fd),
                                'hip_alignment_fd':  horizontal_angle(l_hip_fd, r_hip_fd),
                            }
                        elif self.angle == "target":
                            d_sh_fd = pt(lm, self.D_SH, w, h)
                            d_el_fd = pt(lm, self.D_EL, w, h)
                            PL_fd = self.PL
                            l_sh_fd = pt(lm, PL_fd.LEFT_SHOULDER,  w, h)
                            r_sh_fd = pt(lm, PL_fd.RIGHT_SHOULDER, w, h)
                            sh_dist_fd = np.linalg.norm(np.array(l_sh_fd) - np.array(r_sh_fd))
                            self._fd_snapshot = {
                                'draw_elbow_height_fd': (d_el_fd[1] - d_sh_fd[1]) / h,
                                'back_tension_fd':      sh_dist_fd / np.sqrt(w**2 + h**2),
                            }

                    if shot is not None:
                        # Attach buffered full-draw metrics
                        if self._fd_snapshot:
                            if self.angle == "face":
                                shot.draw_elbow_angle_fd = self._fd_snapshot.get('draw_elbow_angle_fd')
                                shot.anchor_x_fd         = self._fd_snapshot.get('anchor_x_fd')
                                shot.anchor_y_fd         = self._fd_snapshot.get('anchor_y_fd')
                                shot.nose_y_fd           = self._fd_snapshot.get('nose_y_fd')
                                # Nose-string gap: how far draw wrist (string) is ahead of nose at full draw.
                                # Near 0 = light contact (correct); negative = nose squished into string.
                                ax_fd  = self._fd_snapshot.get('anchor_x_fd')
                                nx_fd  = self._fd_snapshot.get('nose_x_fd')
                                if ax_fd is not None and nx_fd is not None:
                                    shot.nose_string_gap = ax_fd - nx_fd
                                # Head movement: distance nose traveled from full draw to release.
                                nx_rel = lm[self.NOSE].x
                                ny_rel = lm[self.NOSE].y
                                if nx_fd is not None:
                                    shot.nose_movement = float(np.sqrt(
                                        (nx_rel - nx_fd)**2 + (ny_rel - self._fd_snapshot.get('nose_y_fd', ny_rel))**2
                                    ))
                                # Pre-anchor drift: nose Y change (px) from ~25 frames before shot to release.
                                # Positive = chin dropped during approach to anchor.
                                if len(_nose_y_buf) >= 20:
                                    shot.nose_preanchor_drift = float((ny_rel - _nose_y_buf[0]) * h)
                            elif self.angle == "back":
                                shot.shoulder_level_fd = self._fd_snapshot.get('shoulder_level_fd')
                                shot.hip_alignment_fd  = self._fd_snapshot.get('hip_alignment_fd')
                            elif self.angle == "target":
                                shot.draw_elbow_height_fd = self._fd_snapshot.get('draw_elbow_height_fd')
                                shot.back_tension_fd      = self._fd_snapshot.get('back_tension_fd')
                            self._fd_snapshot = {}
                        # Schedule follow-through measurement ~20 frames after release (face view only)
                        if self.angle == "face":
                            self._pending_followthrough = {
                                'shot': shot,
                                'target_frame': frame_idx + 20,
                            }
                        shot_count += 1
                        shot.shot_number = shot_count
                        self.shots.append(shot)
                        hold_str = f"{shot.hold_time_s:.2f}s" if shot.hold_time_s else "n/a"
                        print(f"    Shot {shot_count:02d} @ frame {frame_idx}  |  Hold: {hold_str}")

                    # Follow-through sampling: once the target frame is reached, record elbow angles
                    if (self.angle == "face" and self._pending_followthrough
                            and frame_idx >= self._pending_followthrough['target_frame']):
                        ft_shot = self._pending_followthrough['shot']
                        d_sh_ft = pt(lm, self.D_SH, w, h)
                        d_el_ft = pt(lm, self.D_EL, w, h)
                        d_wr_ft = pt(lm, self.D_WR, w, h)
                        b_sh_ft = pt(lm, self.B_SH, w, h)
                        b_el_ft = pt(lm, self.B_EL, w, h)
                        b_wr_ft = pt(lm, self.B_WR, w, h)
                        ft_shot.draw_elbow_followthrough = angle_at_joint(d_sh_ft, d_el_ft, d_wr_ft)
                        ft_shot.bow_elbow_followthrough  = angle_at_joint(b_sh_ft, b_el_ft, b_wr_ft)
                        self._pending_followthrough = {}

                    frame = self._draw_overlay(frame, results.pose_landmarks, shot_count, pose)

                if writer is not None:
                    writer.write(frame)
                frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"    Saved annotated video -> {out_path}  ({shot_count} shots detected)")
        else:
            print(f"    Skipped video output (--no-video).  ({shot_count} shots detected)")
        return out_path

    def filter_strongest_shots(self, max_shots: int):
        """Keep only the N strongest detections based on velocity."""
        if max_shots <= 0 or len(self.shots) <= max_shots:
            return  # No filtering needed

        # Sort by detection velocity (strongest first) and keep top N
        self.shots.sort(key=lambda s: s.detection_velocity if s.detection_velocity else 0, reverse=True)
        removed_shots = self.shots[max_shots:]
        self.shots = self.shots[:max_shots]

        # Re-sort by frame index (chronological order) and renumber
        self.shots.sort(key=lambda s: s.frame_index)
        for i, shot in enumerate(self.shots, 1):
            shot.shot_number = i

        print(f"    Filtered to {max_shots} strongest detections (removed {len(removed_shots)} weak detections)")


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-angle report generator
# ─────────────────────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def plot_metric_line(ax, shot_nums, values, color, title, ylabel, ideal=None):
    """Reusable line chart with mean band."""
    style_ax(ax)
    if not values:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=WHITE)
        ax.set_title(title)
        return
    arr = np.array(values)
    ax.plot(shot_nums, values, "o-", color=color, linewidth=2, markersize=7)
    ax.axhline(arr.mean(), color=C_YELLOW, linestyle="--", linewidth=1.5,
               label=f"Mean: {arr.mean():.2f}")
    ax.fill_between(shot_nums, arr.mean()-arr.std(), arr.mean()+arr.std(),
                    alpha=0.15, color=color, label=f"SD: {arr.std():.2f}")
    if ideal is not None:
        ax.axhline(ideal, color="#888", linestyle=":", linewidth=1.2, label=f"Ideal: {ideal}")
    ax.set_title(title)
    ax.set_xlabel("Shot #")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)


def generate_unified_report(
    face_shots:   List[SideShot],
    back_shots:   List[FrontShot],
    target_shots: List[BehindShot],
    output_dir:   str
):
    has_face   = len(face_shots)   > 0
    has_back   = len(back_shots)   > 0
    has_target = len(target_shots) > 0

    available = sum([has_face, has_back, has_target])
    print(f"\n  Building unified report ({available} angle(s) available)...")

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MULTI-ANGLE ARCHERY FORM REPORT")
    print("=" * 65)

    # Display shot counts for each angle
    shot_counts = []
    if has_face:
        shot_counts.append(f"Face: {len(face_shots)}")
    if has_back:
        shot_counts.append(f"Back: {len(back_shots)}")
    if has_target:
        shot_counts.append(f"Target: {len(target_shots)}")

    if shot_counts:
        print(f"  Arrows Detected: {', '.join(shot_counts)}")
        print("=" * 65)

    def stats_row(label, values):
        if not values:
            return
        arr = np.array(values)
        print(f"  {label:<30} {arr.mean():>7.2f}  {arr.std():>7.2f}  "
              f"{arr.min():>7.2f}  {arr.max():>7.2f}")

    print(f"  {'Metric':<30} {'Mean':>7}  {'StdDev':>7}  {'Min':>7}  {'Max':>7}")
    print("  " + "-" * 63)

    if has_face:
        print("  -- FACE VIEW --")
        stats_row("Draw Elbow Angle (deg)",     [s.draw_elbow_angle    for s in face_shots])
        stats_row("Draw Angle at Full Draw",    [s.draw_elbow_angle_fd for s in face_shots if s.draw_elbow_angle_fd])
        stats_row("Bow Elbow Angle (deg)",      [s.bow_elbow_angle     for s in face_shots])
        stats_row("Anchor X (norm)",            [s.anchor_x            for s in face_shots])
        stats_row("Anchor Y (norm)",            [s.anchor_y            for s in face_shots])
        stats_row("Anchor X at Full Draw",      [s.anchor_x_fd         for s in face_shots if s.anchor_x_fd])
        stats_row("Anchor Y at Full Draw",      [s.anchor_y_fd         for s in face_shots if s.anchor_y_fd])
        stats_row("Bow Wrist Angle (deg)",      [s.bow_wrist_angle     for s in face_shots if s.bow_wrist_angle])
        stats_row("Bow Grip Spread",            [s.bow_grip_spread     for s in face_shots if s.bow_grip_spread])
        stats_row("Bow Shoulder Elevation",     [s.bow_shoulder_elevation  for s in face_shots if s.bow_shoulder_elevation])
        stats_row("Draw Pinky-Neck Dist",       [s.draw_pinky_neck_dist    for s in face_shots if s.draw_pinky_neck_dist    is not None])
        stats_row("Nose-String Gap",            [s.nose_string_gap         for s in face_shots if s.nose_string_gap         is not None])
        stats_row("Nose Y at Full Draw",        [s.nose_y_fd               for s in face_shots if s.nose_y_fd               is not None])
        stats_row("Nose Movement (FD→rel)",     [s.nose_movement           for s in face_shots if s.nose_movement           is not None])
        stats_row("Nose Pre-Anchor Drift (px)",  [s.nose_preanchor_drift    for s in face_shots if s.nose_preanchor_drift    is not None])
        stats_row("Draw Elbow Follow-Through",  [s.draw_elbow_followthrough for s in face_shots if s.draw_elbow_followthrough is not None])
        stats_row("Bow Elbow Follow-Through",   [s.bow_elbow_followthrough  for s in face_shots if s.bow_elbow_followthrough  is not None])
        stats_row("Shoulder Tilt (deg)",        [s.shoulder_tilt       for s in face_shots])
        stats_row("Hold Time (s)",              [s.hold_time_s for s in face_shots if s.hold_time_s])

    if has_back:
        print("  -- BACK VIEW --")
        stats_row("Shoulder Level (deg)",      [s.shoulder_level    for s in back_shots])
        stats_row("Shoulder Level at FD (deg)",[s.shoulder_level_fd for s in back_shots if s.shoulder_level_fd is not None])
        stats_row("Head Lateral Tilt (deg)",   [s.head_lateral_tilt for s in back_shots])
        stats_row("Hip Alignment (deg)",       [s.hip_alignment     for s in back_shots])
        stats_row("Hip Alignment at FD (deg)", [s.hip_alignment_fd  for s in back_shots if s.hip_alignment_fd  is not None])
        stats_row("T-Draw Angle (deg)",        [s.t_draw_angle      for s in back_shots if s.t_draw_angle      is not None])
        stats_row("Bow Shoulder Elevation",    [s.bow_shoulder_elevation for s in back_shots if s.bow_shoulder_elevation is not None])
        stats_row("Bow Arm Vertical (deg)",    [s.bow_arm_vertical  for s in back_shots])
        stats_row("Hold Time (s)",             [s.hold_time_s for s in back_shots if s.hold_time_s])

    if has_target:
        print("  -- TARGET VIEW --")
        stats_row("Draw Elbow Height",         [s.draw_elbow_height    for s in target_shots])
        stats_row("Draw Elbow Height at FD",   [s.draw_elbow_height_fd for s in target_shots if s.draw_elbow_height_fd is not None])
        stats_row("Draw Elbow Lateral",        [s.draw_elbow_lateral   for s in target_shots])
        stats_row("Back Tension Proxy",        [s.back_tension_proxy   for s in target_shots])
        stats_row("Back Tension at FD",        [s.back_tension_fd      for s in target_shots if s.back_tension_fd      is not None])
        stats_row("Shoulder Rotation (deg)",   [s.shoulder_rotation    for s in target_shots])
        stats_row("Hold Time (s)",             [s.hold_time_s for s in target_shots if s.hold_time_s])

    print("=" * 65)

    # ── Coaching insights ────────────────────────────────────────────────
    print("\n  COACHING INSIGHTS:")
    insights = []

    if has_face:
        de = np.array([s.draw_elbow_angle for s in face_shots])
        if de.std() > 3.0:
            insights.append(f"  [FACE]   Draw elbow angle varies by {de.std():.1f} deg (SD) -- "
                            "work on consistent draw length.")
        ax_x = np.array([s.anchor_x for s in face_shots])
        ax_y = np.array([s.anchor_y for s in face_shots])
        spread = np.sqrt(ax_x.var() + ax_y.var()) * 1000
        if spread > 12:
            insights.append(f"  [FACE]   Anchor point spread = {spread:.1f} units -- "
                            "anchor position needs tightening.")

        # Draw length at full draw
        de_fd = np.array([s.draw_elbow_angle_fd for s in face_shots if s.draw_elbow_angle_fd is not None])
        if len(de_fd) >= 3:
            if de_fd.std() > 3.0:
                insights.append(f"  [FACE]   Draw length at full draw varies by {de_fd.std():.1f} deg (SD) -- "
                                "inconsistent settling at anchor; focus on reaching the same draw length each shot.")
            de_rel = np.array([s.draw_elbow_angle for s in face_shots if s.draw_elbow_angle_fd is not None])
            if len(de_rel) == len(de_fd):
                drift = float(np.mean(de_rel - de_fd))
                if abs(drift) > 5.0:
                    direction = "opening" if drift > 0 else "closing"
                    insights.append(f"  [FACE]   Draw arm moves {abs(drift):.1f} deg ({direction}) between full draw and release -- "
                                    "large movement during shot cycle; check for anticipation or heeling.")

        # Bow grip consistency
        bwa = np.array([s.bow_wrist_angle for s in face_shots if s.bow_wrist_angle is not None])
        if len(bwa) >= 3:
            if bwa.std() > 5.0:
                insights.append(f"  [FACE]   Bow wrist angle varies by {bwa.std():.1f} deg (SD) -- "
                                "grip contact point shifting between shots; inconsistent pressure on the riser "
                                "can cause arrows to fly high (bow rotates if palm is not fully touching).")
            if bwa.mean() > 168.0:
                insights.append(f"  [FACE]   Bow wrist angle avg {bwa.mean():.1f} deg (near straight) -- "
                                "possible wrapped/death grip; Olympic recurve technique uses a high-wrist push grip (~150-165 deg); "
                                "ensure full palm contact on the riser to prevent torque at release.")
        bgs = np.array([s.bow_grip_spread for s in face_shots if s.bow_grip_spread is not None])
        if len(bgs) >= 3 and bgs.std() > 0.15:
            insights.append(f"  [FACE]   Bow finger spread inconsistent (SD={bgs.std():.2f}) -- "
                            "finger position on the riser changing between shots; relax all fingers and maintain "
                            "consistent full palm contact to avoid torque that sends arrows high.")

        # Bow shoulder elevation (face view)
        bse = np.array([s.bow_shoulder_elevation for s in face_shots if s.bow_shoulder_elevation is not None])
        if len(bse) >= 3:
            if bse.mean() < 0.18:
                insights.append(f"  [FACE]   Bow shoulder may be elevated (ear-to-shoulder gap avg {bse.mean():.3f}) -- "
                                "keep the lower scapula low; press the bow shoulder down from setup and hold it "
                                "throughout the shot cycle.")
            if bse.std() > 0.03:
                insights.append(f"  [FACE]   Bow shoulder elevation varies by {bse.std():.3f} (SD) -- "
                                "shoulder height inconsistent between shots; focus on a locked-down bow shoulder "
                                "from initial setup through follow-through.")

        # Draw pinky to neck (anchor strength)
        dpn = np.array([s.draw_pinky_neck_dist for s in face_shots if s.draw_pinky_neck_dist is not None])
        if len(dpn) >= 3:
            if dpn.mean() > 0.10:
                insights.append(f"  [FACE]   Draw pinky is far from neck (avg {dpn.mean():.3f} normalized) -- "
                                "establish a strong anchor by touching the draw-hand pinky to the neck; "
                                "this prevents collapse and locks in the back-tension connection.")
            if dpn.std() > 0.03:
                insights.append(f"  [FACE]   Draw pinky-to-neck distance varies by {dpn.std():.3f} (SD) -- "
                                "anchor contact inconsistent between shots; make pinky-to-neck contact a "
                                "deliberate part of every shot to eliminate collapse.")

        # Nose squish / string-nose contact
        nsg = np.array([s.nose_string_gap for s in face_shots if s.nose_string_gap is not None])
        if len(nsg) >= 3:
            if nsg.mean() < 0:
                insights.append(f"  [FACE]   Nose squish detected (avg gap {nsg.mean():.3f}) -- "
                                "nose is being pushed into the string; keep chin up and close mouth at anchor. "
                                "Squish shifts the bow and sends arrows right or up.")
            elif nsg.mean() < 0.01:
                insights.append(f"  [FACE]   Nose very close to string (avg gap {nsg.mean():.3f}) -- "
                                "watch for squish; light contact is correct but ensure mouth is closed "
                                "and head is not pressing forward.")
            if nsg.std() > 0.02:
                insights.append(f"  [FACE]   Nose-to-string distance varies by {nsg.std():.3f} (SD) -- "
                                "head position at anchor is inconsistent shot to shot; "
                                "aim for the same 'pretty nose' contact every arrow.")

        # Head movement FD → release
        nm = np.array([s.nose_movement for s in face_shots if s.nose_movement is not None])
        if len(nm) >= 3:
            if nm.mean() > 0.025:
                insights.append(f"  [FACE]   Head moving during shot (avg {nm.mean():.3f} normalized) -- "
                                "head should stay perfectly still from full draw through release; "
                                "head movement shifts the string and causes squish or pulls the anchor.")
            if nm.std() > 0.015:
                insights.append(f"  [FACE]   Head movement inconsistent between shots (SD={nm.std():.3f}) -- "
                                "some shots show much more head movement than others; "
                                "focus on locking the head position at setup and holding it.")

        # Pre-anchor head drift (chin drop during draw approach).
        # A common high-arrow tell: at anchor the head pushes DOWN into the
        # string just before release, which forces the release hand down, breaks
        # the up/down line, and sends arrows HIGH. Fix: pull the string to the
        # head, leave the head in the same place — don't push the head to the string.
        pad = np.array([s.nose_preanchor_drift for s in face_shots if s.nose_preanchor_drift is not None])
        if len(pad) >= 2:
            if pad.mean() > 2.0:
                insights.append(
                    f"  [FACE]   Head pushing DOWN into the string at anchor (avg +{pad.mean():.1f}px) -- "
                    "a common high-arrow tell: the head drops into the string before release, "
                    "forcing the draw hand down so the arrow flies high. "
                    "Pull the string to the head and leave the head still — don't push the head to the string.")
            elif pad.mean() < -2.0:
                insights.append(
                    f"  [FACE]   Chin drifting UP during draw approach (avg {pad.mean():.1f}px) -- "
                    "head is rising toward anchor; check that you're not lifting the chin "
                    "to find the string contact.")

        # Follow-through
        de_ft = np.array([s.draw_elbow_followthrough for s in face_shots if s.draw_elbow_followthrough is not None])
        de_rel = np.array([s.draw_elbow_angle for s in face_shots if s.draw_elbow_followthrough is not None])
        if len(de_ft) >= 3 and len(de_rel) == len(de_ft):
            expansion = float(np.mean(de_ft - de_rel))
            if expansion < 3.0:
                insights.append(f"  [FACE]   Draw elbow not continuing after release (avg +{expansion:.1f} deg) -- "
                                "expansion is stopping at the clicker; keep moving the elbow behind "
                                "using back tension — compass analogy: rotate elbow around toward spine.")
        be_ft  = np.array([s.bow_elbow_followthrough for s in face_shots if s.bow_elbow_followthrough is not None])
        be_rel = np.array([s.bow_elbow_angle         for s in face_shots if s.bow_elbow_followthrough is not None])
        if len(be_ft) >= 3 and len(be_rel) == len(be_ft):
            bow_drop = float(np.mean(be_rel - be_ft))
            if bow_drop > 10.0:
                insights.append(f"  [FACE]   Bow arm collapsing after release (avg -{bow_drop:.1f} deg drop) -- "
                                "bow arm is falling sideways or downward; keep strong triceps and maintain "
                                "bow arm direction toward the target through the full follow-through.")
            if be_ft.std() > 6.0:
                insights.append(f"  [FACE]   Bow arm follow-through inconsistent (SD={be_ft.std():.1f} deg) -- "
                                "bow arm is landing in a different position each shot; "
                                "aim for the bow to drop consistently to thigh level after release.")

    if has_back:
        sl = np.array([s.shoulder_level for s in back_shots])
        if abs(sl.mean()) > 4.0:
            insights.append(f"  [BACK]   Shoulders tilted avg {sl.mean():.1f} deg -- "
                            "check stance and hip rotation.")
        if sl.std() > 3.0:
            insights.append(f"  [BACK]   Shoulder level varies by {sl.std():.1f} deg (SD) -- "
                            "inconsistent shoulder height between shots; check for muscular fatigue or stance shifting.")
        ht = np.array([s.head_lateral_tilt for s in back_shots])
        if abs(ht.mean()) > 5.0:
            insights.append(f"  [BACK]   Head tilting {ht.mean():.1f} deg laterally -- "
                            "focus on keeping head upright at full draw.")
        if ht.std() > 3.0:
            insights.append(f"  [BACK]   Head lateral tilt varies by {ht.std():.1f} deg (SD) -- "
                            "inconsistent head position between shots; aim for a stable, upright head at full draw.")
        # Bow wrist position consistency
        bwx = np.array([s.bow_wrist_x for s in back_shots if s.bow_wrist_x is not None])
        if len(bwx) >= 3 and bwx.std() > 0.02:
            insights.append(f"  [BACK]   Bow wrist lateral position varies by {bwx.std():.3f} norm units (SD) -- "
                            "bow arm extension is inconsistent; aim for the same bow hand position each shot.")
        # T-draw alignment
        td = np.array([s.t_draw_angle for s in back_shots if s.t_draw_angle is not None])
        if len(td) >= 3:
            if abs(td.mean()) > 5.0:
                direction = "draw elbow is high" if td.mean() > 0 else "bow elbow is high"
                insights.append(f"  [BACK]   T-draw alignment avg {td.mean():+.1f} deg -- {direction}; "
                                "both elbows should be level forming a horizontal T at full draw.")
            if td.std() > 4.0:
                insights.append(f"  [BACK]   T-draw alignment varies by {td.std():.1f} deg (SD) -- "
                                "arm height inconsistent at full draw; focus on a repeatable horizontal T position.")
        # Bow shoulder elevation (back view — both shoulders visible in T-shape)
        bse_bk = np.array([s.bow_shoulder_elevation for s in back_shots if s.bow_shoulder_elevation is not None])
        if len(bse_bk) >= 3:
            if bse_bk.mean() < 0.18:
                insights.append(f"  [BACK]   Bow shoulder appears elevated (ear-to-shoulder gap avg {bse_bk.mean():.3f}) -- "
                                "keep the lower scapula actively depressed; bow shoulder should stay low from "
                                "setup through the entire shot cycle.")
            if bse_bk.std() > 0.03:
                insights.append(f"  [BACK]   Bow shoulder elevation inconsistent (SD={bse_bk.std():.3f}) -- "
                                "shoulder height is varying between shots; lock the bow shoulder down at setup "
                                "and maintain that position.")

    if has_target:
        eh = np.array([s.draw_elbow_height for s in target_shots])
        if eh.mean() > 0.05:
            insights.append(f"  [TARGET] Draw elbow is below shoulder line (avg {eh.mean():.3f}) -- "
                            "raise elbow to improve back tension.")
        if eh.std() > 0.025:
            insights.append(f"  [TARGET] Draw elbow height varies by {eh.std():.3f} normalized (SD) -- "
                            "inconsistent elbow elevation between shots; work on reaching the same height each draw.")
        bt = np.array([s.back_tension_proxy for s in target_shots])
        if bt.std() > 0.02:
            insights.append(f"  [TARGET] Back tension inconsistent (SD={bt.std():.3f}) -- "
                            "focus on scapula engagement through the shot.")
        # Draw elbow lateral flare
        el = np.array([s.draw_elbow_lateral for s in target_shots if s.draw_elbow_lateral is not None])
        if len(el) >= 3:
            if abs(el.mean()) > 0.08:
                insights.append(f"  [TARGET] Draw elbow flaring {abs(el.mean()):.2f} normalized units -- "
                                "elbow is swinging out rather than rotating back; focus on rotating the elbow tip "
                                "around toward the target to engage the back muscles.")
            if el.std() > 0.04:
                insights.append(f"  [TARGET] Draw elbow flare inconsistent (SD={el.std():.3f}) -- "
                                "elbow path varies between shots; aim for a consistent rotation through the draw.")
        # Bow cant via shoulder rotation (primary indicator from target view)
        sr = np.array([s.shoulder_rotation for s in target_shots])
        if abs(sr.mean()) > 8.0:
            side = "bow" if sr.mean() > 0 else "draw"
            insights.append(f"  [TARGET] Shoulder line tilted {sr.mean():+.1f} deg avg -- "
                            f"{side} shoulder is higher; an asymmetric shoulder line often accompanies bow cant. "
                            f"Ensure the bow is held vertical at full draw.")

    # Hold time insight — use whichever view has the most shots with hold data
    all_hold_times = []
    for shots_list in [face_shots, back_shots, target_shots]:
        ht = [s.hold_time_s for s in shots_list if s.hold_time_s]
        if len(ht) > len(all_hold_times):
            all_hold_times = ht

    if all_hold_times:
        ht_arr = np.array(all_hold_times)
        if ht_arr.std() > 0.4:
            insights.append(f"  [HOLD]   Hold time varies by {ht_arr.std():.2f}s (SD) -- "
                            "inconsistent timing; aim for a consistent 1-3s window.")
        if ht_arr.mean() < 1.0:
            insights.append(f"  [HOLD]   Average hold of {ht_arr.mean():.2f}s is very short -- "
                            "consider holding longer to ensure full settling at anchor.")
        if ht_arr.mean() > 4.0:
            insights.append(f"  [HOLD]   Average hold of {ht_arr.mean():.2f}s is long -- "
                            "extended holds can cause aiming breakdown and target panic.")

    if insights:
        for ins in insights:
            print(ins)
    else:
        print("  Form looks consistent across all measured metrics. Great shooting!")

    # ── Tournament quick reference ────────────────────────────────────────
    print()
    print("  TOURNAMENT QUICK REFERENCE")
    print("  " + "─" * 58)
    print("  Vertical corrections:")
    print("    Arrow HIGH  →  Check shoulders/anchor.")
    print("                   Fix: 'Scapulas DOWN. Pinky to neck.'")
    print("    Arrow LOW   →  Check expansion through clicker.")
    print("                   Fix: 'EXPAND. Don't wait for the clicker.'")
    print()
    print("  Horizontal corrections:")
    print("    Arrow LEFT  →  Check the release/follow-through.")
    print("                   Fix: 'Finish the hand behind the ear.'")
    print("    Arrow RIGHT →  Check grip contact and hold timing.")
    print("                   Fix: 'Touch ALL grip. Even pressure.'")
    print("                   Note: both too fast AND too slow hold times")
    print("                         send arrows right in tournament data.")
    print()
    print("  Mental game patterns (observed across 2024-2026 tournaments):")
    print("    3rd arrow after two 9s/10s:")
    print("      Take a breath and reset. Treat it like arrow #1.")
    print("      Don't change anything — the process was already working.")
    print("    First 1-2 ends:")
    print("      Allow the warm-up; focus on form process, not the score.")
    print("      Nervousness in first ends is a known pattern — expected.")
    print("    After a 10:")
    print("      Stay with identical process. Resist the urge to grip")
    print("      harder or expand faster — overcompensation is the risk.")
    print("  " + "─" * 58)

    print()

    # ── Big chart ────────────────────────────────────────────────────────
    # Layout: 4 rows x 3 cols
    # Row 0: Side view metrics
    # Row 1: Front view metrics
    # Row 2: Behind view metrics
    # Row 3: Hold time analysis (full width, 3 panels)

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    fig.patch.set_facecolor(DARK_BG)

    # Build title with arrow counts
    shot_count_str = " | ".join(shot_counts) if shot_counts else "No shots detected"
    fig.suptitle(f"Multi-Angle Archery Form Analysis\n{shot_count_str}",
                 fontsize=17, fontweight="bold", color=WHITE, y=1.01)

    for ax in axes.flat:
        style_ax(ax)
        ax.text(0.5, 0.5, "Not available\n(no video provided)",
                transform=ax.transAxes, ha="center", va="center",
                color="#555", fontsize=9)

    # ── Row 0: FACE ──────────────────────────────────────────────────────
    if has_face:
        snums = [s.shot_number for s in face_shots]

        # Draw elbow: release line + full-draw + follow-through
        ax = axes[0, 0]
        style_ax(ax)
        de_vals = [s.draw_elbow_angle for s in face_shots]
        ax.plot(snums, de_vals, "o-", color=C_GREEN, linewidth=2, markersize=7, label="At release")
        de_fd_pairs = [(s.shot_number, s.draw_elbow_angle_fd) for s in face_shots if s.draw_elbow_angle_fd is not None]
        if de_fd_pairs:
            fd_ns, fd_vs = zip(*de_fd_pairs)
            ax.plot(fd_ns, fd_vs, "s:", color=C_CYAN, linewidth=1.5, markersize=5, label="At full draw")
        de_ft_pairs = [(s.shot_number, s.draw_elbow_followthrough) for s in face_shots if s.draw_elbow_followthrough is not None]
        if de_ft_pairs:
            ft_ns, ft_vs = zip(*de_ft_pairs)
            ax.plot(ft_ns, ft_vs, "^--", color=C_YELLOW, linewidth=1.5, markersize=6, label="Follow-through")
        ax.axhline(np.mean(de_vals), color=C_GREEN, linestyle="--", linewidth=1.0, alpha=0.4,
                   label=f"Mean: {np.mean(de_vals):.1f}°")
        ax.set_title("[FACE] Draw Elbow Angle")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("degrees")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

        # Bow elbow: release line + follow-through
        ax = axes[0, 1]
        style_ax(ax)
        be_vals = [s.bow_elbow_angle for s in face_shots]
        ax.plot(snums, be_vals, "o-", color=C_ORANGE, linewidth=2, markersize=7, label="At release")
        be_ft_pairs = [(s.shot_number, s.bow_elbow_followthrough) for s in face_shots if s.bow_elbow_followthrough is not None]
        if be_ft_pairs:
            ft_ns, ft_vs = zip(*be_ft_pairs)
            ax.plot(ft_ns, ft_vs, "^--", color=C_YELLOW, linewidth=1.5, markersize=6, label="Follow-through")
        ax.axhline(np.mean(be_vals), color=C_ORANGE, linestyle="--", linewidth=1.0, alpha=0.4,
                   label=f"Mean: {np.mean(be_vals):.1f}°")
        ax.set_title("[FACE] Bow Elbow Angle")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("degrees")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

        # Anchor scatter (keep)
        ax = axes[0,2]
        style_ax(ax)
        axs = [s.anchor_x for s in face_shots]
        ays = [s.anchor_y for s in face_shots]
        sc = ax.scatter(axs, ays, c=snums, cmap="plasma", s=90, zorder=3)
        ax.invert_yaxis()
        mx, my = np.mean(axs), np.mean(ays)
        ax.scatter([mx], [my], marker="+", s=200, color=C_YELLOW, linewidths=2.5, zorder=4)
        sx, sy = np.std(axs), np.std(ays)
        ellipse = mpatches.Ellipse((mx, my), sx*4, sy*4,
                                   fill=False, edgecolor=C_YELLOW, linestyle="--", lw=1.5)
        ax.add_patch(ellipse)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Shot #", color=WHITE)
        cbar.ax.yaxis.set_tick_params(color=WHITE)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
        ax.set_title("[FACE] Anchor Point Scatter")
        ax.set_xlabel("X (norm)")
        ax.set_ylabel("Y (norm)")

    # ── Row 1: BACK ──────────────────────────────────────────────────────
    if has_back:
        fnums = [s.shot_number for s in back_shots]

        # Panel [1,0]: Shoulder Level + Hip Alignment on the same axes
        ax = axes[1, 0]
        style_ax(ax)
        sl_vals = [s.shoulder_level for s in back_shots]
        ha_vals = [s.hip_alignment  for s in back_shots]
        ax.plot(fnums, sl_vals, "o-", color=C_BLUE,   linewidth=2, markersize=7, label="Shoulders")
        ax.plot(fnums, ha_vals, "s-", color=C_PURPLE, linewidth=2, markersize=7, label="Hips")
        ax.axhline(0, color="#888", linestyle=":", linewidth=1.2, label="Ideal: 0°")
        ax.axhline(np.mean(sl_vals), color=C_BLUE,   linestyle="--", linewidth=1.0, alpha=0.5,
                   label=f"Sh mean: {np.mean(sl_vals):.1f}°")
        ax.axhline(np.mean(ha_vals), color=C_PURPLE, linestyle="--", linewidth=1.0, alpha=0.5,
                   label=f"Hip mean: {np.mean(ha_vals):.1f}°")
        ax.set_title("[BACK] Body Alignment (Shoulders & Hips)")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("degrees")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

        plot_metric_line(axes[1,1], fnums,
                         [s.head_lateral_tilt for s in back_shots],
                         C_CYAN, "[BACK] Head Lateral Tilt", "degrees", ideal=0)

        # Panel [1,2]: T-draw alignment
        td_nums = [s.shot_number for s in back_shots if s.t_draw_angle is not None]
        td_vals = [s.t_draw_angle for s in back_shots if s.t_draw_angle is not None]
        plot_metric_line(axes[1,2], td_nums, td_vals,
                         C_GREEN, "[BACK] T-Draw Alignment", "degrees", ideal=0)

    # ── Row 2: TARGET ─────────────────────────────────────────────────────
    if has_target:
        bnums = [s.shot_number for s in target_shots]

        # Panel [2,0]: Draw Elbow Height — release line + full-draw overlay
        ax = axes[2, 0]
        style_ax(ax)
        eh_vals = [s.draw_elbow_height for s in target_shots]
        ax.plot(bnums, eh_vals, "o-", color=C_ORANGE, linewidth=2, markersize=7, label="At release")
        eh_fd = [(s.shot_number, s.draw_elbow_height_fd)
                 for s in target_shots if s.draw_elbow_height_fd is not None]
        if eh_fd:
            fd_ns, fd_vs = zip(*eh_fd)
            ax.plot(fd_ns, fd_vs, "s--", color=C_YELLOW, linewidth=1.5, markersize=6, label="At full draw")
        ax.axhline(0, color="#888", linestyle=":", linewidth=1.2, label="Shoulder level")
        ax.axhline(np.mean(eh_vals), color=C_ORANGE, linestyle="--", linewidth=1.0, alpha=0.5,
                   label=f"Mean: {np.mean(eh_vals):.3f}")
        ax.set_title("[TARGET] Draw Elbow Height")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("norm (neg=above shoulder)")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

        # Panel [2,1]: Draw Elbow Lateral Flare (previously never charted)
        plot_metric_line(axes[2,1], bnums,
                         [s.draw_elbow_lateral for s in target_shots],
                         C_PURPLE, "[TARGET] Draw Elbow Flare", "norm X offset", ideal=0)

        # Panel [2,2]: Back Tension Proxy — release line + full-draw overlay
        ax = axes[2, 2]
        style_ax(ax)
        bt_vals = [s.back_tension_proxy for s in target_shots]
        ax.plot(bnums, bt_vals, "o-", color=C_GREEN, linewidth=2, markersize=7, label="At release")
        bt_fd = [(s.shot_number, s.back_tension_fd)
                 for s in target_shots if s.back_tension_fd is not None]
        if bt_fd:
            fd_ns, fd_vs = zip(*bt_fd)
            ax.plot(fd_ns, fd_vs, "s--", color=C_YELLOW, linewidth=1.5, markersize=6, label="At full draw")
        ax.axhline(np.mean(bt_vals), color=C_GREEN, linestyle="--", linewidth=1.0, alpha=0.5,
                   label=f"Mean: {np.mean(bt_vals):.3f}")
        ax.set_title("[TARGET] Back Tension Proxy")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("norm shoulder width")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

    # ── Row 3: HOLD TIME (all angles combined) ───────────────────────────
    # Use the view with the most hold-time data points
    hold_source_label = None
    for shots_list, label in [(face_shots, "FACE"), (back_shots, "BACK"), (target_shots, "TARGET")]:
        ht_vals = [(s.shot_number, s.hold_time_s) for s in shots_list if s.hold_time_s]
        if len(ht_vals) > 0:
            ht_nums  = [x[0] for x in ht_vals]
            ht_times = [x[1] for x in ht_vals]
            hold_source_label = label
            break

    if hold_source_label and ht_times:
        ht_arr = np.array(ht_times)

        # Panel 3,0 — Hold time per shot line chart
        ax = axes[3, 0]
        style_ax(ax)
        ax.plot(ht_nums, ht_times, "o-", color=C_YELLOW, linewidth=2, markersize=8)
        ax.axhline(ht_arr.mean(), color=C_GREEN, linestyle="--", linewidth=1.5,
                   label=f"Mean: {ht_arr.mean():.2f}s")
        ax.fill_between(ht_nums,
                        ht_arr.mean() - ht_arr.std(),
                        ht_arr.mean() + ht_arr.std(),
                        alpha=0.15, color=C_YELLOW, label=f"SD: {ht_arr.std():.2f}s")
        # Ideal zone shading (1.0 - 3.0 seconds)
        ax.axhspan(1.0, 3.0, alpha=0.08, color=C_GREEN, label="Ideal zone (1-3s)")
        ax.set_title(f"[HOLD] Hold Time per Shot ({hold_source_label} view)")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("Hold time (seconds)")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

        # Panel 3,1 — Bar chart colored by consistency
        ax = axes[3, 1]
        style_ax(ax)
        mean_ht = ht_arr.mean()
        bar_colors = []
        for t in ht_times:
            deviation = abs(t - mean_ht)
            if deviation < 0.3:
                bar_colors.append(C_GREEN)    # consistent
            elif deviation < 0.6:
                bar_colors.append(C_YELLOW)   # slight deviation
            else:
                bar_colors.append(C_ORANGE)   # large deviation
        bars = ax.bar(ht_nums, ht_times, color=bar_colors, alpha=0.85)
        ax.axhline(mean_ht, color=WHITE, linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_ht:.2f}s")
        ax.axhspan(1.0, 3.0, alpha=0.08, color=C_GREEN)
        # Legend patches
        p1 = mpatches.Patch(color=C_GREEN,  label="Within 0.3s of mean")
        p2 = mpatches.Patch(color=C_YELLOW, label="0.3-0.6s deviation")
        p3 = mpatches.Patch(color=C_ORANGE, label=">0.6s deviation")
        ax.legend(handles=[p1, p2, p3], fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)
        ax.set_title("[HOLD] Hold Time Consistency (color-coded)")
        ax.set_xlabel("Shot #")
        ax.set_ylabel("Hold time (seconds)")

        # Panel 3,2 — Distribution histogram + stats summary
        ax = axes[3, 2]
        style_ax(ax)
        ax.hist(ht_times, bins=max(5, len(ht_times)//2), color=C_PURPLE,
                alpha=0.8, edgecolor="#333")
        ax.axvline(ht_arr.mean(), color=C_YELLOW, linestyle="--", linewidth=2,
                   label=f"Mean: {ht_arr.mean():.2f}s")
        ax.axvline(ht_arr.mean() - ht_arr.std(), color=C_CYAN, linestyle=":",
                   linewidth=1.5, label=f"-1 SD: {ht_arr.mean()-ht_arr.std():.2f}s")
        ax.axvline(ht_arr.mean() + ht_arr.std(), color=C_CYAN, linestyle=":",
                   linewidth=1.5, label=f"+1 SD: {ht_arr.mean()+ht_arr.std():.2f}s")
        ax.axvspan(1.0, 3.0, alpha=0.08, color=C_GREEN)
        # Stats text box
        stats_text = (f"Mean : {ht_arr.mean():.2f}s\n"
                      f"SD   : {ht_arr.std():.2f}s\n"
                      f"Min  : {ht_arr.min():.2f}s\n"
                      f"Max  : {ht_arr.max():.2f}s\n"
                      f"Shots: {len(ht_times)}")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, color=WHITE, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="#0a0a1a", alpha=0.7))
        ax.set_title("[HOLD] Hold Time Distribution")
        ax.set_xlabel("Hold time (seconds)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DARK_BG)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "multi_angle_report.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Chart saved -> {chart_path}")
    return chart_path


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Archery Form Analyzer v2 - Multi Angle")
    parser.add_argument("--face",   default=None, help="Path to FACE view video (perpendicular to shooting line, facing archer's face/anchor side)")
    parser.add_argument("--back",   default=None, help="Path to BACK view video (perpendicular to shooting line, facing archer's spine — T-shape)")
    parser.add_argument("--target", default=None, help="Path to TARGET view video (directly behind archer along arrow line, facing target)")
    parser.add_argument("--hand",   default="right", choices=["right","left"],
                        help="Archer dominant hand (default: right)")
    parser.add_argument("--out",    default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.015,
                        help="Shot detection sensitivity - higher = less sensitive (default: 0.015)")
    parser.add_argument("--min-time", type=float, default=0,
                        help="Minimum seconds between shots (default: 0; AV detector uses 10s minimum automatically)")
    parser.add_argument("--max-shots", type=int, default=0,
                        help="Keep only the N strongest detections (0 = keep all). Ignored when AV detector is active.")
    parser.add_argument("--shots", type=int, default=0,
                        help="Expected number of arrows in the video (hint; AV detector uses this to cap results)")
    parser.add_argument("--use-ml", action="store_true",
                        help="Use ML-based shot detection instead of audio-visual detector")
    parser.add_argument("--velocity", action="store_true",
                        help="Use legacy velocity-based shot detection (not recommended)")
    parser.add_argument("--ml-model", default="models/rf_current.pkl",
                        help="Path to trained ML model (default: models/rf_current.pkl)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing annotated output videos (faster — chart + report only)")
    parser.add_argument("--pose-backend", default="yolo", choices=["yolo", "mediapipe"],
                        help="Pose estimation backend: yolo (default, YOLO11) or mediapipe (legacy)")
    parser.add_argument("--yolo-model", default=_DEFAULT_YOLO_MODEL,
                        help=f"Path to YOLO11 pose model weights (default: {_DEFAULT_YOLO_MODEL})")
    parser.add_argument("--session-json", default=None,
                        help="Write this run's detected frame numbers into metrics.verified_frames "
                             "of the given session JSON (single-view mode only — i.e. exactly one "
                             "of --face/--back/--target). Matches the --session-json convention used "
                             "by tools/analyze_{view}_yolo.py downstream. This script otherwise only "
                             "writes charts/video to --out — it does not touch session_history/ on "
                             "its own, so callers that need a session JSON (e.g. the mobile app's "
                             "upload pipeline) must pass this explicitly.")
    args = parser.parse_args()

    # Determine detection mode
    use_av  = not args.use_ml and not args.velocity
    n_shots = args.shots if args.shots > 0 else (args.max_shots if args.max_shots > 0 else None)

    if not any([args.face, args.back, args.target]):
        parser.error("Provide at least one video: --face, --back, or --target")

    print("\n" + "=" * 65)
    if args.use_ml:
        print("  ARCHERY FORM ANALYZER — ML Shot Detection")
    elif args.velocity:
        print("  ARCHERY FORM ANALYZER — Velocity Detection (legacy)")
    else:
        print("  ARCHERY FORM ANALYZER — Audio+Visual Shot Detection")
    print("=" * 65)
    if args.use_ml:
        print(f"  Using ML-based shot detection (model: {args.ml_model})")
    elif args.velocity:
        print(f"  Using legacy velocity threshold detection")
    else:
        print(f"  Using audio-visual shot detection (clicker + anchor zone)")
    if args.min_time > 0:
        print(f"  Minimum shot interval override: {args.min_time}s")
    if n_shots:
        print(f"  Expected arrows per end: {n_shots}")

    face_shots, back_shots, target_shots = [], [], []

    for angle, path in [("face", args.face), ("back", args.back), ("target", args.target)]:
        if path:
            proc = AngleProcessor(path, angle, args.hand, args.out,
                                velocity_threshold=args.threshold,
                                min_shot_time=args.min_time,
                                use_ml=args.use_ml,
                                ml_model_path=args.ml_model,
                                write_video=not args.no_video,
                                use_av=use_av,
                                n_shots=n_shots,
                                pose_backend=args.pose_backend,
                                yolo_model=args.yolo_model)
            proc.process()

            # Apply max_shots filtering only for legacy velocity detector
            if not use_av and args.max_shots > 0:
                proc.filter_strongest_shots(args.max_shots)

            if angle == "face":
                face_shots = proc.shots
            elif angle == "back":
                back_shots = proc.shots
            else:
                target_shots = proc.shots

    generate_unified_report(face_shots, back_shots, target_shots, args.out)

    if args.session_json:
        shots_by_view = {"face": face_shots, "back": back_shots, "target": target_shots}
        provided_views = [a for a in ("face", "back", "target") if getattr(args, a)]
        if len(provided_views) != 1:
            print(f"\n  --session-json given but {len(provided_views)} views were processed "
                  f"(need exactly 1) — skipping JSON write")
        else:
            frames = sorted(s.frame_index for s in shots_by_view[provided_views[0]])
            if os.path.exists(args.session_json):
                with open(args.session_json) as f:
                    session = json.load(f)
            else:
                session = {}
            session.setdefault("metrics", {})
            session["metrics"]["verified_frames"] = frames
            session["metrics"]["arrows_shot"] = len(frames)
            session.setdefault("review", {})["status"] = (
                "auto_detected" if frames else "no_shots_detected"
            )
            with open(args.session_json, "w") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
            print(f"\n  Session JSON updated -> {args.session_json}  "
                  f"({len(frames)} shot(s) detected)")

    print("\nAnalysis complete!")
    print(f"  Output folder  ->  {args.out}/")
    print(f"  Unified chart  ->  {args.out}/multi_angle_report.png")
    if args.face:   print(f"  Face video     ->  {args.out}/analyzed_face.mp4")
    if args.back:   print(f"  Back video     ->  {args.out}/analyzed_back.mp4")
    if args.target: print(f"  Target video   ->  {args.out}/analyzed_target.mp4")
    print()


if __name__ == "__main__":
    main()

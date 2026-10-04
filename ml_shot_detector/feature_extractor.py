"""
Feature extractor for ML shot detection.

Computes a 13-element feature vector from a single pose landmark frame.
These per-frame features are then aggregated into window-level features
by window_builder.py before being fed to the Random Forest.
"""
import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Optional

# ── Import geometry helpers from parent package ───────────────────────────────
# We load them from archery_analyzer_v2.py to avoid duplicating logic.
_PARENT = os.path.join(os.path.dirname(__file__), "..")
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from archery_analyzer_v2 import angle_at_joint, horizontal_angle


@dataclass
class FrameFeatures:
    """Per-frame feature vector (13 features) extracted from pose landmarks."""
    # Elbow angles
    draw_elbow_angle: float       # shoulder→elbow→wrist angle (degrees)
    bow_elbow_angle: float        # opposite arm elbow angle (degrees)

    # Draw elbow position relative to shoulder
    draw_elbow_height: float      # (elbow.y - shoulder.y) normalized; negative = above shoulder
    draw_elbow_lateral: float     # elbow.x - shoulder.x normalized

    # Draw wrist anchor position
    anchor_x: float               # draw wrist normalized X
    anchor_y: float               # draw wrist normalized Y

    # Anchor depth
    wrist_to_ear_dist: float      # euclidean dist between draw wrist and draw ear (normalized)

    # Back tension proxy
    shoulder_width: float         # |L_SH.x - R_SH.x| normalized

    # Alignment
    shoulder_tilt: float          # horizontal angle of shoulder line (degrees)
    head_tilt: float              # angle from mid-shoulder to nose (degrees)

    # Wrist velocity (frame-to-frame delta)
    wrist_velocity_x: float       # delta X from previous frame
    wrist_velocity_y: float       # delta Y from previous frame
    wrist_speed: float            # sqrt(vx^2 + vy^2)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.draw_elbow_angle,
            self.bow_elbow_angle,
            self.draw_elbow_height,
            self.draw_elbow_lateral,
            self.anchor_x,
            self.anchor_y,
            self.wrist_to_ear_dist,
            self.shoulder_width,
            self.shoulder_tilt,
            self.head_tilt,
            self.wrist_velocity_x,
            self.wrist_velocity_y,
            self.wrist_speed,
        ], dtype=np.float32)

    @staticmethod
    def feature_names():
        return [
            "draw_elbow_angle",
            "bow_elbow_angle",
            "draw_elbow_height",
            "draw_elbow_lateral",
            "anchor_x",
            "anchor_y",
            "wrist_to_ear_dist",
            "shoulder_width",
            "shoulder_tilt",
            "head_tilt",
            "wrist_velocity_x",
            "wrist_velocity_y",
            "wrist_speed",
        ]


def extract_frame_features(
    lm,                     # mediapipe landmark list (pose_landmarks.landmark)
    d_sh_idx: int,          # draw shoulder landmark index
    d_el_idx: int,          # draw elbow landmark index
    d_wr_idx: int,          # draw wrist landmark index
    b_sh_idx: int,          # bow shoulder landmark index
    b_el_idx: int,          # bow elbow landmark index
    b_wr_idx: int,          # bow wrist landmark index
    d_ear_idx: int,         # draw ear landmark index
    l_sh_idx: int,          # left shoulder landmark index
    r_sh_idx: int,          # right shoulder landmark index
    nose_idx: int,          # nose landmark index
    prev_wrist_xy: Optional[tuple] = None,  # (x, y) from previous frame
) -> FrameFeatures:
    """
    Extract per-frame features from MediaPipe pose landmarks.

    All coordinates are in normalized [0,1] space (as returned by MediaPipe).
    """
    # Convenience: normalized 2-tuples
    def xy(idx):
        return (lm[idx].x, lm[idx].y)

    d_sh = xy(d_sh_idx)
    d_el = xy(d_el_idx)
    d_wr = xy(d_wr_idx)
    b_sh = xy(b_sh_idx)
    b_el = xy(b_el_idx)
    b_wr = xy(b_wr_idx)
    d_ear = xy(d_ear_idx)
    l_sh = xy(l_sh_idx)
    r_sh = xy(r_sh_idx)
    nose = xy(nose_idx)
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)

    # Elbow angles (use normalized coords — angle_at_joint accepts (x,y) tuples)
    draw_elbow_angle = angle_at_joint(d_sh, d_el, d_wr)
    bow_elbow_angle  = angle_at_joint(b_sh, b_el, b_wr)

    # Elbow position relative to shoulder
    draw_elbow_height  = d_el[1] - d_sh[1]   # negative = above shoulder (good)
    draw_elbow_lateral = d_el[0] - d_sh[0]

    # Anchor position
    anchor_x = d_wr[0]
    anchor_y = d_wr[1]

    # Wrist-to-ear distance (key full-draw indicator)
    wrist_to_ear_dist = float(np.sqrt(
        (d_wr[0] - d_ear[0]) ** 2 + (d_wr[1] - d_ear[1]) ** 2
    ))

    # Shoulder width
    shoulder_width = abs(l_sh[0] - r_sh[0])

    # Alignment angles
    shoulder_tilt = horizontal_angle(l_sh, r_sh)
    head_tilt     = horizontal_angle(mid_sh, nose)

    # Velocity
    if prev_wrist_xy is not None:
        vx = d_wr[0] - prev_wrist_xy[0]
        vy = d_wr[1] - prev_wrist_xy[1]
    else:
        vx, vy = 0.0, 0.0
    speed = float(np.sqrt(vx ** 2 + vy ** 2))

    return FrameFeatures(
        draw_elbow_angle   = float(draw_elbow_angle),
        bow_elbow_angle    = float(bow_elbow_angle),
        draw_elbow_height  = float(draw_elbow_height),
        draw_elbow_lateral = float(draw_elbow_lateral),
        anchor_x           = float(anchor_x),
        anchor_y           = float(anchor_y),
        wrist_to_ear_dist  = float(wrist_to_ear_dist),
        shoulder_width     = float(shoulder_width),
        shoulder_tilt      = float(shoulder_tilt),
        head_tilt          = float(head_tilt),
        wrist_velocity_x   = float(vx),
        wrist_velocity_y   = float(vy),
        wrist_speed        = float(speed),
    )

"""
Sliding window aggregator for ML shot detection.

Maintains a rolling deque of FrameFeatures and, given a candidate frame index,
computes the 19 window-level aggregate features that are fed to the Random Forest.

The full feature vector fed to the classifier = 32 features:
  - 13 per-frame static features at the candidate center frame
  - 19 window-level aggregate features
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple

from .feature_extractor import FrameFeatures

WINDOW_SIZE = 30          # total frames in window
HALF_WIN    = WINDOW_SIZE // 2   # 15 frames before and after center


class WindowBuilder:
    """
    Accumulates FrameFeatures objects and builds window feature vectors.

    Usage:
        builder = WindowBuilder(fps=30.0)
        for each frame:
            builder.push(frame_features, frame_idx)
            vec, meta = builder.build_window_features(frame_idx)
            if vec is not None:
                prediction = model.predict([vec])
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Deque of (frame_idx, FrameFeatures) — keep twice the window for look-ahead
        self._buf: deque = deque(maxlen=WINDOW_SIZE * 2)

    def set_fps(self, fps: float):
        self.fps = fps

    def push(self, features: FrameFeatures, frame_idx: int):
        self._buf.append((frame_idx, features))

    def build_window_features(
        self, center_idx: int
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Build the 32-feature window vector centered on `center_idx`.

        Returns:
            (feature_vector, meta_dict) — feature_vector is None if the
            buffer doesn't have enough frames around the center yet.
        """
        # Collect frames in [center - HALF_WIN, center + HALF_WIN]
        window = [
            (fi, ff) for (fi, ff) in self._buf
            if center_idx - HALF_WIN <= fi <= center_idx + HALF_WIN
        ]

        if len(window) < 5:
            return None, {}

        # Sort by frame index
        window.sort(key=lambda x: x[0])
        frame_indices = [f for f, _ in window]
        feats         = [ff for _, ff in window]

        # Find closest frame to center
        center_pos = min(range(len(window)), key=lambda i: abs(frame_indices[i] - center_idx))

        # Split into pre-center and post-center
        pre_feats  = feats[:center_pos]        # frames before center
        post_feats = feats[center_pos + 1:]    # frames after center

        # ── Per-frame features at center ──────────────────────────────────
        center_feat = feats[center_pos]
        center_vec  = center_feat.to_array()   # 13 features

        # ── Window-level aggregates (19 features) ─────────────────────────

        # All speeds in window
        all_speeds = np.array([f.wrist_speed for f in feats], dtype=np.float32)

        # Pre-release region: frames [-15, -3] relative to center
        pre_region = [
            f for (fi, f) in window
            if (center_idx - 15) <= fi <= (center_idx - 3)
        ]
        pre_region_speeds = np.array([f.wrist_speed for f in pre_region], dtype=np.float32) \
            if pre_region else np.array([0.0], dtype=np.float32)

        # Post-release region: frames [+1, +10] relative to center
        post_region = [
            f for (fi, f) in window
            if (center_idx + 1) <= fi <= (center_idx + 10)
        ]
        post_region_speeds = np.array([f.wrist_speed for f in post_region], dtype=np.float32) \
            if post_region else np.array([0.0], dtype=np.float32)

        # 1. settle_frames — frames with speed < 0.003 before center
        settle_threshold = 0.003
        settle_frames = int(sum(1 for f in pre_feats if f.wrist_speed < settle_threshold))

        # 2. settle_duration_s
        settle_duration_s = settle_frames / self.fps

        # 3. pre_release_still
        pre_release_still = float(pre_region_speeds.mean())

        # 4. post_release_speed
        post_release_speed = float(post_region_speeds.mean())

        # 5. speed_ratio (high = snap at release)
        speed_ratio = post_release_speed / (pre_release_still + 1e-9)

        # 6. wrist_speed_max
        wrist_speed_max = float(all_speeds.max()) if len(all_speeds) > 0 else 0.0

        # 7. wrist_speed_at_center
        wrist_speed_at_center = float(center_feat.wrist_speed)

        # 8. elbow_height_mean
        elbow_heights = np.array([f.draw_elbow_height for f in feats], dtype=np.float32)
        elbow_height_mean = float(elbow_heights.mean())

        # 9. elbow_height_at_center
        elbow_height_at_center = float(center_feat.draw_elbow_height)

        # 10. anchor_stability_std — std of anchor_x among settled frames
        settled_anchor_xs = [f.anchor_x for f in pre_feats if f.wrist_speed < settle_threshold]
        anchor_stability_std = float(np.std(settled_anchor_xs)) if len(settled_anchor_xs) > 1 else 0.0

        # 11. shoulder_width_drop — shoulder_width at center vs 5 frames after
        center_sw = center_feat.shoulder_width
        post_5 = [f for (fi, f) in window if (center_idx + 1) <= fi <= (center_idx + 5)]
        post_5_sw = float(np.mean([f.shoulder_width for f in post_5])) if post_5 else center_sw
        shoulder_width_drop = center_sw - post_5_sw

        # 12. wrist_ear_ratio (scale-invariant anchor depth)
        sw_safe = center_feat.shoulder_width if center_feat.shoulder_width > 1e-6 else 1e-6
        wrist_ear_ratio = center_feat.wrist_to_ear_dist / sw_safe

        # 13. draw_angle_delta — draw elbow angle change over pre-window
        pre_15 = [f for (fi, f) in window if fi == center_idx - 15]
        draw_angle_at_minus15 = pre_15[0].draw_elbow_angle if pre_15 else center_feat.draw_elbow_angle
        draw_angle_delta = center_feat.draw_elbow_angle - draw_angle_at_minus15

        # 14. elbow_velocity_y_mean — mean vertical elbow velocity in pre window
        if len(pre_feats) >= 2:
            elbow_vel_y = [
                pre_feats[i + 1].draw_elbow_height - pre_feats[i].draw_elbow_height
                for i in range(len(pre_feats) - 1)
            ]
            elbow_velocity_y_mean = float(np.mean(elbow_vel_y))
        else:
            elbow_velocity_y_mean = 0.0

        # 15. wrist_x_range — range of draw wrist X in window (large = non-shot movement)
        wrist_xs = np.array([f.anchor_x for f in feats], dtype=np.float32)
        wrist_x_range = float(wrist_xs.max() - wrist_xs.min()) if len(wrist_xs) > 0 else 0.0

        # 16-19. Shoulder_width at center for reference, wrist_to_ear at center,
        #        draw_elbow_angle at center (duplicated from static for window context),
        #        frames_in_window
        frames_in_window = float(len(window))

        # Additional computed
        shoulder_width_at_center = float(center_feat.shoulder_width)
        wrist_to_ear_at_center   = float(center_feat.wrist_to_ear_dist)
        draw_elbow_at_center     = float(center_feat.draw_elbow_angle)

        window_vec = np.array([
            float(settle_frames),           # 0
            settle_duration_s,              # 1
            pre_release_still,              # 2
            post_release_speed,             # 3
            speed_ratio,                    # 4
            wrist_speed_max,                # 5
            wrist_speed_at_center,          # 6
            elbow_height_mean,              # 7
            elbow_height_at_center,         # 8
            anchor_stability_std,           # 9
            shoulder_width_drop,            # 10
            wrist_ear_ratio,                # 11
            draw_angle_delta,               # 12
            elbow_velocity_y_mean,          # 13
            wrist_x_range,                  # 14
            frames_in_window,               # 15
            shoulder_width_at_center,       # 16
            wrist_to_ear_at_center,         # 17
            draw_elbow_at_center,           # 18
        ], dtype=np.float32)

        # Final 32-feature vector = [13 center-frame] + [19 window-level]
        feature_vec = np.concatenate([center_vec, window_vec])

        # Replace NaN/Inf with 0
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=5.0, neginf=-5.0)

        meta = {
            "settle_frames":      settle_frames,
            "pre_release_still":  pre_release_still,
            "post_release_speed": post_release_speed,
            "speed_ratio":        speed_ratio,
            "frames_in_window":   len(window),
        }
        return feature_vec, meta

    # Temporal sequences for future LSTM (not used by RF)
    def get_temporal_sequences(self, center_idx: int) -> dict:
        window = sorted(
            [(fi, ff) for (fi, ff) in self._buf
             if center_idx - HALF_WIN <= fi <= center_idx + HALF_WIN],
            key=lambda x: x[0]
        )
        feats = [ff for _, ff in window]
        return {
            "wrist_x":       [f.anchor_x           for f in feats],
            "wrist_y":       [f.anchor_y            for f in feats],
            "wrist_speed":   [f.wrist_speed         for f in feats],
            "elbow_height":  [f.draw_elbow_height   for f in feats],
            "shoulder_width":[f.shoulder_width      for f in feats],
        }

    @staticmethod
    def feature_names():
        center_names = [f"center_{n}" for n in FrameFeatures.feature_names()]
        window_names = [
            "w_settle_frames",
            "w_settle_duration_s",
            "w_pre_release_still",
            "w_post_release_speed",
            "w_speed_ratio",
            "w_wrist_speed_max",
            "w_wrist_speed_at_center",
            "w_elbow_height_mean",
            "w_elbow_height_at_center",
            "w_anchor_stability_std",
            "w_shoulder_width_drop",
            "w_wrist_ear_ratio",
            "w_draw_angle_delta",
            "w_elbow_velocity_y_mean",
            "w_wrist_x_range",
            "w_frames_in_window",
            "w_shoulder_width_at_center",
            "w_wrist_to_ear_at_center",
            "w_draw_elbow_at_center",
        ]
        return center_names + window_names

"""
Auto-labeler for ML training data.

Given a video and a list of verified release frame indices,
this module:
  1. Runs MediaPipe pose estimation on the video
  2. Assigns multi-class labels to every frame
  3. Extracts 32-feature window vectors for all labeled frames
  4. Saves a training JSON to training_data/{date}_{session_id}.json

Label scheme (5-class):
    IDLE           — not in any shooting sequence
    DRAWING        — bow being drawn toward anchor
    FULL_DRAW      — settled at anchor, aiming
    RELEASE        — the shot itself
    FOLLOW_THROUGH — post-release
"""
import cv2
import json
import os
import sys
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional
from datetime import datetime

_PARENT = os.path.join(os.path.dirname(__file__), "..")
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from .feature_extractor import extract_frame_features, FrameFeatures
from .window_builder import WindowBuilder

# Label constants
IDLE           = "IDLE"
DRAWING        = "DRAWING"
FULL_DRAW      = "FULL_DRAW"
RELEASE        = "RELEASE"
FOLLOW_THROUGH = "FOLLOW_THROUGH"

LABEL_TO_INT = {IDLE: 0, DRAWING: 1, FULL_DRAW: 2, RELEASE: 3, FOLLOW_THROUGH: 4}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def _assign_frame_labels(
    verified_releases: List[int],
    total_frames: int,
    fps: float,
    # Settle criteria for FULL_DRAW windows
    settle_threshold: float = 0.005,
    elbow_angle_threshold: float = 40.0,
    ear_dist_threshold: float = 0.15,
) -> Dict[int, str]:
    """
    Returns {frame_idx: label_str} for all labeled frames.

    Unlabeled frames (between events) are left out and get IDLE
    treatment at sample time.
    """
    labels: Dict[int, str] = {}
    total_seconds = total_frames / fps

    for R in verified_releases:
        # RELEASE: [R-1, R+2]
        for fi in range(max(0, R - 1), min(total_frames, R + 3)):
            labels[fi] = RELEASE

        # FOLLOW_THROUGH: [R+3, R+45]
        for fi in range(max(0, R + 3), min(total_frames, R + 46)):
            if fi not in labels:
                labels[fi] = FOLLOW_THROUGH

        # FULL_DRAW: [R-45, R-2] — will be filtered by pose criteria during extraction
        for fi in range(max(0, R - 45), max(0, R - 1)):
            if fi not in labels:
                labels[fi] = FULL_DRAW

        # DRAWING: [R-120, R-46]
        for fi in range(max(0, R - 120), max(0, R - 45)):
            if fi not in labels:
                labels[fi] = DRAWING

    return labels


def extract_training_data(
    video_path: str,
    verified_releases: List[int],
    session_id: Optional[str] = None,
    archer_hand: str = "right",
    idle_sample_every: int = 30,
    output_dir: str = "training_data",
) -> str:
    """
    Run pose estimation on video, extract features, save training JSON.

    Args:
        video_path: Path to video file.
        verified_releases: List of frame indices where shots are verified.
        session_id: Identifier string (default: derived from video filename).
        archer_hand: "right" or "left".
        idle_sample_every: Sample IDLE frames every N frames.
        output_dir: Directory to save JSON.

    Returns:
        Path to saved JSON file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if session_id is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        session_id = base

    print(f"\nExtracting training data from: {video_path}")
    print(f"  {total_frames} frames @ {fps:.1f} fps, {len(verified_releases)} verified shots")
    print(f"  Verified release frames: {verified_releases}")

    # Assign labels
    frame_labels = _assign_frame_labels(verified_releases, total_frames, fps)
    label_counts = {}
    for l in frame_labels.values():
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  Labeled frames: {label_counts}")

    # Set up MediaPipe and landmark indices
    mp_pose = mp.solutions.pose
    PL      = mp_pose.PoseLandmark

    if archer_hand == "right":
        d_sh = PL.RIGHT_SHOULDER.value;  d_el = PL.RIGHT_ELBOW.value;  d_wr = PL.RIGHT_WRIST.value
        b_sh = PL.LEFT_SHOULDER.value;   b_el = PL.LEFT_ELBOW.value;   b_wr = PL.LEFT_WRIST.value
        d_ear = PL.RIGHT_EAR.value
    else:
        d_sh = PL.LEFT_SHOULDER.value;   d_el = PL.LEFT_ELBOW.value;   d_wr = PL.LEFT_WRIST.value
        b_sh = PL.RIGHT_SHOULDER.value;  b_el = PL.RIGHT_ELBOW.value;  b_wr = PL.RIGHT_WRIST.value
        d_ear = PL.LEFT_EAR.value

    l_sh  = PL.LEFT_SHOULDER.value
    r_sh  = PL.RIGHT_SHOULDER.value
    nose  = PL.NOSE.value

    # First pass: extract per-frame features for ALL frames
    print("  Running pose estimation pass 1 (all frames)...")
    per_frame_features: Dict[int, FrameFeatures] = {}
    prev_wrist_xy = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5,
                      model_complexity=1) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                ff = extract_frame_features(
                    lm, d_sh, d_el, d_wr, b_sh, b_el, b_wr, d_ear, l_sh, r_sh, nose,
                    prev_wrist_xy=prev_wrist_xy
                )
                per_frame_features[frame_idx] = ff
                prev_wrist_xy = (lm[d_wr].x, lm[d_wr].y)

            if frame_idx % 500 == 0:
                print(f"    Frame {frame_idx}/{total_frames}")
            frame_idx += 1

    cap.release()

    # Second pass: build window features for labeled frames
    print("  Building window feature vectors...")

    # Build windows using direct lookups into per_frame_features so that
    # frames anywhere in the video (not just the last 60) get full windows.
    # For each target frame we load exactly the ±HALF_WIN neighbourhood.
    from .window_builder import HALF_WIN

    def _build_window(center_fi: int):
        local = WindowBuilder(fps=fps)
        for wf in range(center_fi - HALF_WIN, center_fi + HALF_WIN + 1):
            if wf in per_frame_features:
                local.push(per_frame_features[wf], wf)
        return local.build_window_features(center_fi)

    samples = []
    sampled_idle_count = 0

    labeled_set = set(frame_labels.keys())
    all_frames_sorted = sorted(per_frame_features.keys())

    for fi in all_frames_sorted:
        if fi in labeled_set:
            label = frame_labels[fi]
            vec, meta = _build_window(fi)
            if vec is not None:
                samples.append({
                    "frame": fi,
                    "label": label,
                    "label_int": LABEL_TO_INT[label],
                    "features": vec.tolist(),
                })
        elif fi % idle_sample_every == 0:
            vec, meta = _build_window(fi)
            if vec is not None:
                samples.append({
                    "frame": fi,
                    "label": IDLE,
                    "label_int": LABEL_TO_INT[IDLE],
                    "features": vec.tolist(),
                })
                sampled_idle_count += 1

    # Post-filter FULL_DRAW: require at least 2 of 3 criteria per plan
    # (wrist_speed < 0.005 AND draw_elbow_angle < 40 AND wrist_to_ear_dist < 0.15)
    filtered_samples = []
    for s in samples:
        if s["label"] == FULL_DRAW:
            ff = per_frame_features.get(s["frame"])
            if ff is not None:
                crit1 = ff.wrist_speed < 0.005
                crit2 = ff.draw_elbow_angle < 40.0
                crit3 = ff.wrist_to_ear_dist < 0.15
                if sum([crit1, crit2, crit3]) >= 2:
                    filtered_samples.append(s)
                # else: discard ambiguous FULL_DRAW frame
            else:
                filtered_samples.append(s)  # no features → keep anyway
        else:
            filtered_samples.append(s)

    samples = filtered_samples

    # Final counts
    final_counts = {}
    for s in samples:
        final_counts[s["label"]] = final_counts.get(s["label"], 0) + 1
    print(f"  Final samples by class: {final_counts}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_file = os.path.join(output_dir, f"{date_str}_{session_id}.json")

    payload = {
        "session_id":         session_id,
        "date":               date_str,
        "video_path":         os.path.abspath(video_path),
        "fps":                fps,
        "total_frames":       total_frames,
        "archer_hand":        archer_hand,
        "verified_releases":  verified_releases,
        "feature_names":      WindowBuilder.feature_names(),
        "label_map":          LABEL_TO_INT,
        "samples":            samples,
        "sample_counts":      final_counts,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Saved {len(samples)} samples -> {out_file}")
    return out_file

#!/usr/bin/env python3
"""
Find full-draw frames in archery video using wrist/elbow stability detection.

Works across all views (face, back, target) by detecting the HOLD PHASE:
the period where the draw arm is raised AND nearly stationary.

Algorithm:
  1. Track draw elbow Y position frame-by-frame
  2. Identify frames where elbow is above shoulder (arm raised)
  3. Within those, find windows where elbow velocity is near zero (holding still)
  4. Pick the middle of each stable window as the full-draw frame

This replaces the naive "wrist at highest point" approach which fails in back view
because the draw wrist goes behind the head and is inconsistently tracked.

Usage:
  python3 tools/find_full_draw_frames.py --video path/to/video.MOV
  python3 tools/find_full_draw_frames.py --video path/to/video.MOV --view back
  python3 tools/find_full_draw_frames.py --video path/to/video.MOV --min-hold 0.3
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from yolo_pose_adapter import YoloPoseAdapter

MP_R_SH = 12   # draw shoulder
MP_R_EL = 14   # draw elbow
MP_R_WR = 16   # draw wrist
MP_NOSE = 0


def find_full_draw_frames(video_path, model_path, min_hold_s=0.3, sample_every=2):
    """
    Find full-draw frames by detecting stable hold phases.

    Returns list of (frame_number, hold_duration_s, confidence) tuples.
    """
    pose = YoloPoseAdapter(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 1: Track draw elbow position over time
    elbow_data = []  # (frame, elbow_y_norm, shoulder_y_norm, elbow_x_norm)

    for fn in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        result = pose.process(frame)
        if result.pose_landmarks is None:
            continue

        lm = result.pose_landmarks.landmark
        if lm[MP_R_EL].visibility < 0.3 or lm[MP_R_SH].visibility < 0.3:
            continue

        elbow_data.append((
            fn,
            lm[MP_R_EL].y,
            lm[MP_R_SH].y,
            lm[MP_R_EL].x,
        ))

    cap.release()

    if len(elbow_data) < 10:
        return []

    # Step 2: Find frames where elbow is above shoulder (arm raised)
    raised_frames = []
    for fn, el_y, sh_y, el_x in elbow_data:
        if el_y < sh_y:  # y increases downward, so el_y < sh_y means elbow above shoulder
            raised_frames.append((fn, el_y, el_x))

    if not raised_frames:
        return []

    # Step 3: Compute velocity (frame-to-frame movement) of elbow when raised
    velocities = []
    for i in range(1, len(raised_frames)):
        fn_prev, ey_prev, ex_prev = raised_frames[i-1]
        fn_curr, ey_curr, ex_curr = raised_frames[i]

        dt = (fn_curr - fn_prev) / fps
        if dt <= 0 or dt > 1.0:  # skip if gap too large (non-consecutive)
            velocities.append((fn_curr, 999.0))
            continue

        dy = abs(ey_curr - ey_prev)
        dx = abs(ex_curr - ex_prev)
        speed = np.sqrt(dx**2 + dy**2) / dt
        velocities.append((fn_curr, speed))

    if not velocities:
        return []

    # Step 4: Find stable windows (low velocity for consecutive frames)
    # Threshold: elbow moving less than 0.02 normalized units per second
    stability_threshold = 0.02
    min_hold_frames = int(min_hold_s * fps / sample_every)

    stable_windows = []
    current_window = []

    for fn, speed in velocities:
        if speed < stability_threshold:
            current_window.append(fn)
        else:
            if len(current_window) >= min_hold_frames:
                stable_windows.append(current_window[:])
            current_window = []

    # Don't forget the last window
    if len(current_window) >= min_hold_frames:
        stable_windows.append(current_window[:])

    # Step 5: Pick the middle frame of each stable window
    results = []
    for window in stable_windows:
        mid_idx = len(window) // 2
        mid_frame = window[mid_idx]
        hold_duration = len(window) * sample_every / fps

        # Confidence: longer hold = higher confidence
        conf = min(1.0, hold_duration / 2.0)
        results.append((mid_frame, hold_duration, conf))

    # Step 6: Merge windows that are too close (within min_shot_interval)
    min_gap_frames = 10 * fps  # 10 seconds between shots minimum
    merged = []
    for r in results:
        if merged and (r[0] - merged[-1][0]) < min_gap_frames:
            # Keep the one with longer hold
            if r[1] > merged[-1][1]:
                merged[-1] = r
        else:
            merged.append(r)

    return merged


def main():
    ap = argparse.ArgumentParser(description="Find full-draw frames using stability detection")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--view", default="auto", choices=["face", "back", "target", "auto"],
                    help="Camera view (default: auto)")
    ap.add_argument("--yolo-model", default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "models", "yolo26m-pose.pt"))
    ap.add_argument("--min-hold", type=float, default=0.3,
                    help="Minimum hold duration in seconds (default: 0.3)")
    ap.add_argument("--sample-every", type=int, default=2,
                    help="Sample every N frames (default: 2, higher=faster but less precise)")
    args = ap.parse_args()

    print(f"Video: {args.video}")
    print(f"Model: {args.yolo_model}")
    print(f"Min hold: {args.min_hold}s")
    print(f"Sampling every {args.sample_every} frames")
    print()

    results = find_full_draw_frames(
        args.video,
        args.yolo_model,
        min_hold_s=args.min_hold,
        sample_every=args.sample_every,
    )

    if not results:
        print("No full-draw frames detected.")
        return

    print(f"Found {len(results)} full-draw frame(s):")
    frames = []
    for fn, hold_dur, conf in results:
        print(f"  Frame {fn:>6d}  ({fn/30:.1f}s)  hold={hold_dur:.2f}s  confidence={conf:.2f}")
        frames.append(fn)

    print(f"\nVerified frames list: {frames}")
    print(f"  (Use with: python3 tools/analyze_back_yolo.py --video {args.video} --frames {' '.join(str(f) for f in frames)})")


if __name__ == "__main__":
    main()

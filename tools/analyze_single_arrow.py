#!/usr/bin/env python3
"""
Single Arrow Analyzer - Fully Automatic Detection for One-Arrow Videos

When video contains only ONE arrow, detection is trivial:
Just find the frame with the strongest forward wrist snap!

No false positives, no manual verification needed.
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime

def detect_single_arrow(video_path):
    """
    Find the single arrow release in a short video.

    Simply finds the frame with maximum forward wrist velocity.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_pose = mp.solutions.pose

    max_velocity = 0
    shot_frame = None
    prev_wrist_x = None

    print(f"\nAnalyzing single arrow video: {video_path}")
    print(f"Duration: {total_frames/fps:.1f}s ({total_frames} frames)")
    print("Finding release moment...\n")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                if prev_wrist_x is not None:
                    velocity = r_wrist.x - prev_wrist_x
                    if velocity > max_velocity:
                        max_velocity = velocity
                        shot_frame = frame_idx

                prev_wrist_x = r_wrist.x

            frame_idx += 1

    cap.release()

    if shot_frame is not None:
        print(f"✓ Shot detected at frame {shot_frame} ({shot_frame/fps:.1f}s)")
        print(f"  Release velocity: {max_velocity:.4f}\n")
        return shot_frame
    else:
        print("✗ Could not detect shot (no pose landmarks found)")
        return None

def analyze_frame(video_path, frame_num):
    """Analyze form metrics at the detected shot frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    h, w = frame.shape[:2]

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Get landmarks
            r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]

            # Calculate angles
            def angle_at_joint(a, b, c):
                ba = a - b
                bc = c - b
                cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

            r_sh = np.array([r_shoulder.x, r_shoulder.y])
            r_el = np.array([r_elbow.x, r_elbow.y])
            r_wr = np.array([r_wrist.x, r_wrist.y])
            l_sh = np.array([l_shoulder.x, l_shoulder.y])
            l_el = np.array([l_elbow.x, l_elbow.y])
            l_wr = np.array([l_wrist.x, l_wrist.y])

            draw_elbow_angle = angle_at_joint(r_sh, r_el, r_wr)
            bow_elbow_angle = angle_at_joint(l_sh, l_el, l_wr)
            elbow_height = r_elbow.y - r_shoulder.y

            # Shoulder width (back tension proxy)
            r_sh_px = np.array([r_shoulder.x * w, r_shoulder.y * h])
            l_sh_px = np.array([l_shoulder.x * w, l_shoulder.y * h])
            sh_dist = np.linalg.norm(l_sh_px - r_sh_px)
            frame_diag = np.sqrt(w**2 + h**2)
            back_tension = sh_dist / frame_diag

            metrics = {
                'frame': frame_num,
                'time': frame_num / fps,
                'draw_elbow_angle': draw_elbow_angle,
                'bow_elbow_angle': bow_elbow_angle,
                'elbow_height': elbow_height,
                'anchor_x': r_wrist.x,
                'anchor_y': r_wrist.y,
                'back_tension': back_tension
            }

            return metrics

    return None

def print_analysis(metrics, score=None):
    """Print form analysis."""
    print("="*80)
    print("FORM ANALYSIS - Single Arrow")
    print("="*80)

    if score is not None:
        print(f"\n  Score: {score} points")

    print(f"\n  Release Frame: {metrics['frame']} ({metrics['time']:.1f}s)")
    print(f"\n  Draw Elbow Angle:    {metrics['draw_elbow_angle']:>7.2f}°")
    print(f"  Bow Elbow Angle:     {metrics['bow_elbow_angle']:>7.2f}°")
    print(f"  Elbow Height:        {metrics['elbow_height']:>+7.3f}")

    if metrics['elbow_height'] < 0.05:
        print(f"    ✓ EXCELLENT - Elbow above shoulder")
    else:
        print(f"    ⚠️  Elbow below shoulder - raise for better back tension")

    print(f"  Anchor Position:     ({metrics['anchor_x']:.3f}, {metrics['anchor_y']:.3f})")
    print(f"  Back Tension Proxy:  {metrics['back_tension']:.3f}")

    print("="*80 + "\n")

def save_to_tracker(video_path, metrics, score=None, session_date=None,
                    arrow_number=1, view="face"):
    """Save single arrow data to session_history/arrows/ in standard format."""
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")

    filename   = os.path.basename(video_path)
    arrow_name = os.path.splitext(filename)[0]
    session_id = f"{session_date.replace('-', '')}_{view}"

    arrow_data = {
        "date":          session_date,
        "session_id":    session_id,
        "video":         video_path,
        "arrow_number":  arrow_number,
        "release_frame": metrics['frame'],
        "score":         score,
        "view":          view,
        "metrics": {
            "elbow_height":      metrics['elbow_height'],
            "draw_elbow_angle":  metrics['draw_elbow_angle'],
            "bow_elbow_angle":   metrics['bow_elbow_angle'],
            "anchor_x":          metrics['anchor_x'],
            "anchor_y":          metrics['anchor_y'],
            "back_tension":      metrics['back_tension'],
        }
    }

    os.makedirs("session_history/arrows", exist_ok=True)
    arrow_file = f"session_history/arrows/{session_date}_{session_id}_arrow{arrow_number:02d}.json"

    with open(arrow_file, 'w') as f:
        json.dump(arrow_data, f, indent=2)

    print(f"✓ Arrow data saved: {arrow_file}\n")
    return arrow_file

def main():
    parser = argparse.ArgumentParser(description="Analyze single arrow video - fully automatic!")
    parser.add_argument("video", help="Path to single arrow video file")
    parser.add_argument("--score",        type=int,   help="Arrow score (optional, for correlation analysis)")
    parser.add_argument("--date",         type=str,   help="Session date (YYYY-MM-DD, default: today)")
    parser.add_argument("--arrow-number", type=int,   default=1, help="Arrow number in session (default: 1)")
    parser.add_argument("--view",         type=str,   default="face",
                        choices=["face", "back", "target"],
                        help="Camera angle (default: face)")
    parser.add_argument("--save",         action="store_true", help="Save to session_history/arrows/")

    args = parser.parse_args()

    # Step 1: Detect shot
    shot_frame = detect_single_arrow(args.video)

    if shot_frame is None:
        print("Failed to detect shot. Make sure video shows archer shooting.")
        return

    # Step 2: Analyze form
    metrics = analyze_frame(args.video, shot_frame)

    if metrics is None:
        print("Failed to analyze form at detected frame.")
        return

    # Step 3: Print results
    print_analysis(metrics, args.score)

    # Step 4: Save if requested
    if args.save:
        save_to_tracker(args.video, metrics, args.score, args.date,
                        arrow_number=args.arrow_number, view=args.view)

if __name__ == "__main__":
    main()

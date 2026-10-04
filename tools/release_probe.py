#!/usr/bin/env python3
"""
Release-motion probe: for a full-draw hold frame, measure what happens in the
~1s AFTER it, to distinguish a SHOT (explosive release: draw hand snaps back/through,
bow reacts) from a LET-DOWN (controlled reverse: draw hand returns forward, bow lowers).

Prints per-frame kinematic features so we can pick the discriminator empirically on
labeled data, then fold it into cross_validate_shots.py as the release gate.

Usage:
  python3 tools/release_probe.py --video V.MOV --hand right --frames 660 1620 --label letdown
"""
import argparse, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(__file__))
from yolo_pose_adapter import YoloPoseAdapter

MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo26m-pose.pt")
# MediaPipe-33 indices (yolo adapter maps to these)
R_SH, R_EL, R_WR = 12, 14, 16
L_SH, L_EL, L_WR = 11, 13, 15


def track(pose, cap, f0, draw_wr, bow_wr, draw_el, pre=3, post=30):
    """Return dict of arrays over [f0-pre, f0+post]: draw wrist/elbow, bow wrist (x,y,vis)."""
    rows = {}
    for f in range(f0 - pre, f0 + post + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        r = pose.process(frame)
        if r.pose_landmarks is None:
            continue
        lm = r.pose_landmarks.landmark
        rows[f] = {
            "dwr": (lm[draw_wr].x, lm[draw_wr].y, lm[draw_wr].visibility),
            "bwr": (lm[bow_wr].x, lm[bow_wr].y, lm[bow_wr].visibility),
            "del": (lm[draw_el].x, lm[draw_el].y, lm[draw_el].visibility),
        }
    return rows


def feats(rows, f0, fps, post=30):
    """Kinematic features in the post-hold window."""
    fs = sorted(k for k in rows if f0 <= k <= f0 + post)
    if len(fs) < 3:
        return None
    def series(key, vmin=0.3):
        return [(f, rows[f][key][0], rows[f][key][1]) for f in fs if rows[f][key][2] >= vmin]
    dw = series("dwr"); bw = series("bwr"); de = series("del")

    def peak_speed(s):
        pk = 0.0
        for i in range(1, len(s)):
            dt = (s[i][0] - s[i-1][0]) / fps
            if dt <= 0:
                continue
            d = np.hypot(s[i][1]-s[i-1][1], s[i][2]-s[i-1][2]) / dt
            pk = max(pk, d)
        return pk

    def net(s):
        if len(s) < 2:
            return (0.0, 0.0)
        return (s[-1][1]-s[0][1], s[-1][2]-s[0][2])

    dw_dx, dw_dy = net(dw)
    bw_dx, bw_dy = net(bw)
    de_dx, de_dy = net(de)
    return {
        "pk_dw": peak_speed(dw),       # peak draw-wrist speed (norm/s)  -- SHOT spikes
        "dw_dx": dw_dx, "dw_dy": dw_dy, # draw-wrist net move (sign of dx ~ forward/back)
        "pk_bw": peak_speed(bw),
        "bw_dy": bw_dy,                 # bow-wrist net vertical (+down) -- LET-DOWN: bow lowers
        "de_dy": de_dy,                 # draw-elbow net vertical (+down) -- LET-DOWN: elbow drops
        "dw_vis": len(dw) / max(1, len(fs)),
    }


def snap_to_release(pose, cap, mid_frame, draw_el, fps, search=110, offset=5, thresh=0.25):
    """
    Release-anchor a full-draw frame (task #2): given the finder's MID-hold frame, scan
    forward for the release/let-down MOTION ONSET (draw-elbow speed spike — visible even in
    back view, where wrist is occluded) and return onset - offset, matching the archer's
    'settled at anchor, about to loose' convention. Consistent across hold durations (fixes
    the mid-hold-vs-release delta that scaled with hold length: 5f short holds, 54f long ones).
    Falls back to the last tracked frame if no onset within `search` (clip end / no release).
    NOTE: this localizes the frame for any DRAW (shot or let-down) — it does NOT decide which.

    Reads SEQUENTIALLY and EARLY-STOPS at the first onset (task #12): the release is usually
    well within `search`, so we rarely read the whole window — far fewer pose calls than the
    old read-all-then-scan version. Uses cap.grab()/retrieve() forward (no per-frame seek).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    prev = None          # (frame, x, y) of last visible draw-elbow
    last = mid_frame
    for f in range(mid_frame, mid_frame + search + 1):
        ret, fr = cap.read()
        if not ret:
            break
        r = pose.process(fr)
        if r.pose_landmarks is None:
            continue                      # keep prev; next valid point uses true frame gap
        lm = r.pose_landmarks.landmark
        if lm[draw_el].visibility < 0.25:
            prev = None; continue         # lost track — reset so we don't span a gap
        x, y = lm[draw_el].x, lm[draw_el].y
        last = f
        if prev is not None:
            dt = (f - prev[0]) / fps
            if dt > 0:
                spd = ((x - prev[1]) ** 2 + (y - prev[2]) ** 2) ** 0.5 / dt
                if spd > thresh:
                    return max(mid_frame, f - offset)   # release onset → anchor just before it
        prev = (f, x, y)
    return last   # no onset found in window (clip end / no clean release) — last stable frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--label", default="?")
    ap.add_argument("--post", type=int, default=30)
    args = ap.parse_args()

    if args.hand == "right":
        draw_wr, bow_wr, draw_el = R_WR, L_WR, R_EL
    else:
        draw_wr, bow_wr, draw_el = L_WR, R_WR, L_EL

    pose = YoloPoseAdapter(MODEL)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    name = os.path.basename(args.video)

    print(f"# {name}  hand={args.hand}  label={args.label}  (post-window {args.post}f={args.post/fps:.1f}s)")
    print(f"# {'frame':>6} {'pk_dw':>7} {'dw_dx':>7} {'dw_dy':>7} {'pk_bw':>7} {'bw_dy':>7} {'de_dy':>7} {'dwvis':>6}")
    for f0 in args.frames:
        rows = track(pose, cap, f0, draw_wr, bow_wr, draw_el, post=args.post)
        ft = feats(rows, f0, fps, post=args.post)
        if ft is None:
            print(f"  {f0:>6}  (insufficient pose)")
            continue
        print(f"  {f0:>6} {ft['pk_dw']:>7.3f} {ft['dw_dx']:>+7.3f} {ft['dw_dy']:>+7.3f} "
              f"{ft['pk_bw']:>7.3f} {ft['bw_dy']:>+7.3f} {ft['de_dy']:>+7.3f} {ft['dw_vis']:>6.2f}  [{args.label}]")
    cap.release()


if __name__ == "__main__":
    main()

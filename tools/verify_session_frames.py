#!/usr/bin/env python3
"""
Spot-check verification grids from SESSION-level JSONs (metrics.verified_frames).

Unlike tools/verification_grid.py (which reads per-arrow JSONs in session_history/arrows/
with a `release_frame` field), this reads the session-level files
session_history/<date>*.json and grids each clip's metrics.verified_frames so you can
eyeball that every captured frame is a genuine full draw.

Usage:
  python3 tools/verify_session_frames.py --date 2026-05-26
  python3 tools/verify_session_frames.py --date 2026-05-26 --view target
Outputs one labeled PNG per clip to output/verify_<jsonstem>.png
"""
import argparse, glob, json, os
import cv2
import numpy as np

THUMB_W = 380
PAD = 8
COLS = 4


def label_thumb(frame_bgr, idx, frame_no, fps):
    h, w = frame_bgr.shape[:2]
    scale = THUMB_W / w
    thumb = cv2.resize(frame_bgr, (THUMB_W, int(h * scale)))
    bar_h = 30
    bar = np.zeros((bar_h, THUMB_W, 3), dtype=np.uint8)
    t = frame_no / fps if fps else 0
    cv2.putText(bar, f"#{idx}  f{frame_no}  ({t:.1f}s)", (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 60), 2)
    return np.vstack([bar, thumb])


def build_grid(video_path, frames, fps):
    cap = cv2.VideoCapture(video_path)
    cells = []
    for i, fn in enumerate(frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            blank = np.zeros((int(THUMB_W * 9 / 16), THUMB_W, 3), dtype=np.uint8)
            cv2.putText(blank, f"f{fn} MISSING", (20, blank.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cells.append(label_thumb(blank, i, fn, fps))
        else:
            cells.append(label_thumb(frame, i, fn, fps))
    cap.release()
    if not cells:
        return None
    ch = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.zeros((ch - c.shape[0], c.shape[1], 3), np.uint8)])
             if c.shape[0] < ch else c for c in cells]
    rows = []
    for r in range(0, len(cells), COLS):
        row = cells[r:r + COLS]
        while len(row) < COLS and len(cells) > COLS:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    rw = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], rw - r.shape[1], 3), np.uint8)])
            if r.shape[1] < rw else r for r in rows]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--view", choices=["face", "back", "target"])
    args = ap.parse_args()
    os.makedirs("output", exist_ok=True)

    files = sorted(glob.glob(f"session_history/{args.date}*.json"))
    files = [f for f in files if "coaching" not in f]
    if args.view:
        files = [f for f in files if f"_{args.view}_" in f]

    for jf in files:
        j = json.load(open(jf))
        m = j.get("metrics", {})
        frames = m.get("verified_frames", [])
        vp = j.get("video_path", "")
        if not frames or not os.path.exists(vp):
            print(f"SKIP {os.path.basename(jf)} (frames={len(frames)} video_exists={os.path.exists(vp)})")
            continue
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        grid = build_grid(vp, frames, fps)
        stem = os.path.splitext(os.path.basename(jf))[0]
        out = f"output/verify_{stem}.png"
        cv2.imwrite(out, grid)
        print(f"{os.path.basename(jf):45s} {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()

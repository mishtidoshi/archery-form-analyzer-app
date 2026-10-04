#!/usr/bin/env python3
"""
Verification Grid — visual sanity check for batch-processed single-arrow sessions.

Reads saved arrow JSONs, extracts the detected release frame from each video,
and builds a labeled thumbnail grid so you can spot bad detections at a glance.

Suspicious frames are highlighted with a red border:
  - Detected very early (< 1.0s into the clip)
  - anchor_y outside plausible range (< 0.25 or > 0.85)
  - elbow_height > 0.15 (far below shoulder)

Usage:
    python3 tools/verification_grid.py --date 2026-03-22 --view face
    python3 tools/verification_grid.py --date 2026-03-22 --view target
    python3 tools/verification_grid.py --date 2026-03-22          # all views
"""

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

# ── layout ────────────────────────────────────────────────────────────────────
THUMB_W   = 320
THUMB_H   = 240
COLS      = 5
BORDER    = 6          # px border around each thumbnail
LABEL_H   = 52         # px reserved below each thumbnail for text
PAD       = 8          # px gap between cells
BG_COLOR  = (20, 20, 35)

# ── suspicion thresholds ─────────────────────────────────────────────────────
MIN_TIME_S        = 1.0    # detections before this are flagged
ANCHOR_Y_LOW      = 0.25
ANCHOR_Y_HIGH     = 0.85
ELBOW_HEIGHT_MAX  = 0.15   # positive = below shoulder


def load_arrows(date: str, view: str | None) -> list[dict]:
    pattern = f"session_history/arrows/{date}*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No arrow files found matching: {pattern}")
    arrows = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        if view and d.get("view") != view:
            continue
        d["_file"] = f
        arrows.append(d)
    return arrows


def is_suspicious(arrow: dict, fps: float) -> tuple[bool, list[str]]:
    reasons = []
    frame = arrow.get("release_frame", 0)
    if fps > 0 and frame / fps < MIN_TIME_S:
        reasons.append(f"early ({frame/fps:.1f}s)")
    m = arrow.get("metrics", {})
    ay = m.get("anchor_y")
    if ay is not None and (ay < ANCHOR_Y_LOW or ay > ANCHOR_Y_HIGH):
        reasons.append(f"anchor_y={ay:.2f}")
    eh = m.get("elbow_height")
    if eh is not None and eh > ELBOW_HEIGHT_MAX:
        reasons.append(f"EH=+{eh:.3f}")
    return bool(reasons), reasons


def extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def make_cell(arrow: dict) -> np.ndarray:
    """Return a (THUMB_H + LABEL_H + 2*BORDER) × (THUMB_W + 2*BORDER) BGR cell."""
    cell_w = THUMB_W + 2 * BORDER
    cell_h = THUMB_H + LABEL_H + 2 * BORDER
    cell = np.full((cell_h, cell_w, 3), BG_COLOR, dtype=np.uint8)

    video_path = arrow.get("video", "")
    release_frame = arrow.get("release_frame", 0)
    arrow_num = arrow.get("arrow_number", "?")
    view = arrow.get("view", "?")
    m = arrow.get("metrics", {})
    eh  = m.get("elbow_height")
    ay  = m.get("anchor_y")

    # get fps for suspicion check
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    suspicious, reasons = is_suspicious(arrow, fps)
    border_color = (0, 0, 220) if suspicious else (0, 180, 60)  # red / green

    # draw border
    cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), border_color, BORDER)

    # extract and resize thumbnail
    img = extract_frame(video_path, release_frame)
    if img is not None:
        thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    else:
        thumb = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        cv2.putText(thumb, "VIDEO NOT FOUND", (20, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cell[BORDER:BORDER + THUMB_H, BORDER:BORDER + THUMB_W] = thumb

    # label area
    lx = BORDER
    ly = BORDER + THUMB_H + 6

    # arrow number + frame
    time_s = release_frame / fps
    header = f"Arrow {arrow_num}  |  frame {release_frame}  ({time_s:.1f}s)"
    cv2.putText(cell, header, (lx, ly + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # metrics line
    eh_str  = (f"+{eh:.3f}" if eh >= 0 else f"{eh:.3f}") if eh is not None else "EH=n/a"
    ay_str  = f"ay={ay:.2f}" if ay is not None else "ay=n/a"
    quality = "GOOD" if eh is not None and eh < 0.05 else "LOW"
    q_color = (80, 220, 80) if quality == "GOOD" else (80, 80, 220)
    metrics_str = f"{eh_str}  {ay_str}"
    cv2.putText(cell, metrics_str, (lx, ly + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(cell, quality, (cell_w - 52, ly + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, q_color, 1, cv2.LINE_AA)

    # suspicion reasons
    if reasons:
        reason_str = "⚠ " + ", ".join(reasons)
        cv2.putText(cell, reason_str, (lx, ly + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 255), 1, cv2.LINE_AA)

    return cell


def build_grid(arrows: list[dict], title: str) -> np.ndarray:
    n = len(arrows)
    rows = math.ceil(n / COLS)
    cell_w = THUMB_W + 2 * BORDER
    cell_h = THUMB_H + LABEL_H + 2 * BORDER

    title_h = 50
    grid_w = COLS * cell_w + (COLS + 1) * PAD
    grid_h = rows * cell_h + (rows + 1) * PAD + title_h

    grid = np.full((grid_h, grid_w, 3), BG_COLOR, dtype=np.uint8)

    # title bar
    cv2.putText(grid, title, (PAD, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)

    for idx, arrow in enumerate(arrows):
        print(f"  Extracting arrow {arrow.get('arrow_number', idx+1)} "
              f"({arrow.get('view','?')}) "
              f"frame {arrow.get('release_frame','?')} ...")
        cell = make_cell(arrow)
        row = idx // COLS
        col = idx % COLS
        y0 = title_h + PAD + row * (cell_h + PAD)
        x0 = PAD + col * (cell_w + PAD)
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    return grid


def main():
    ap = argparse.ArgumentParser(
        description="Build a verification thumbnail grid for batch-processed arrows")
    ap.add_argument("--date", required=True,
                    help="Session date prefix to match (e.g. 2026-03-22)")
    ap.add_argument("--view", choices=["face", "back", "target"],
                    help="Filter by view (omit to include all views)")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: output/verification_grid_DATE_VIEW.png)")
    args = ap.parse_args()

    arrows = load_arrows(args.date, args.view)
    if not arrows:
        sys.exit(f"No arrows found for date={args.date} view={args.view}")

    n_suspicious = sum(
        1 for a in arrows
        if is_suspicious(a, 30.0)[0]
    )

    view_label = args.view if args.view else "all"
    title = (f"{args.date}  |  {view_label} view  |  "
             f"{len(arrows)} arrows  |  "
             f"{n_suspicious} flagged (red border)")

    print(f"\nBuilding verification grid: {len(arrows)} arrows, {n_suspicious} flagged")

    grid = build_grid(arrows, title)

    os.makedirs("output", exist_ok=True)
    out_path = args.out or f"output/verification_grid_{args.date}_{view_label}.png"
    cv2.imwrite(out_path, grid)
    print(f"\n✓ Saved: {out_path}")
    print(f"  Green border = clean detection")
    print(f"  Red border   = suspicious (check these manually)")


if __name__ == "__main__":
    main()

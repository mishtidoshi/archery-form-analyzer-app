#!/usr/bin/env python3
"""
Full shot-cycle frame-by-frame analysis.

Runs YOLO pose estimation across a configurable window around each anchor
frame to track biomechanical metrics through the complete shot cycle:
  SETUP → DRAWING → LOADING → ANCHOR → FOLLOW_THROUGH

Outputs:
  output/{view}_{stem}_cycle.csv   — one row per sampled frame
  output/{view}_{stem}_cycle.png   — metric trajectory chart per shot

Usage:
  python3 tools/analyze_shot_cycle.py \\
      --video data/july_2026/070826/faceview/IMG_4597.MOV \\
      --frames 135 880 1439 1979 2513 3033 \\
      --view face

  python3 tools/analyze_shot_cycle.py \\
      --video data/.../backview/IMG_XXXX.MOV \\
      --frames F1 F2 F3 \\
      --view back

Options:
  --before 3.0    seconds of window before the anchor frame (default: 3.0)
  --after  1.0    seconds of window after the anchor frame  (default: 1.0)
  --step   3      sample every N frames (default: 3 → ~10fps at 30fps video)
"""

import argparse
import os
import sys
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── MediaPipe-compatible landmark indices (COCO-17 via YoloPoseAdapter) ──────
_MP_NOSE           = 0
_MP_LEFT_EAR       = 7
_MP_RIGHT_EAR      = 8
_MP_LEFT_SHOULDER  = 11
_MP_RIGHT_SHOULDER = 12
_MP_LEFT_ELBOW     = 13
_MP_RIGHT_ELBOW    = 14
_MP_LEFT_WRIST     = 15
_MP_RIGHT_WRIST    = 16
_MP_LEFT_HIP       = 23
_MP_RIGHT_HIP      = 24

# Right-handed archer assignments
_D_SH, _D_EL, _D_WR = _MP_RIGHT_SHOULDER, _MP_RIGHT_ELBOW, _MP_RIGHT_WRIST
_B_SH, _B_EL, _B_WR = _MP_LEFT_SHOULDER,  _MP_LEFT_ELBOW,  _MP_LEFT_WRIST
_B_EAR               = _MP_LEFT_EAR

# ── Shot-cycle phase definitions (seconds relative to anchor/release) ─────────
# Anchor frame = clicker fires = t=0
PHASES = [
    (-999.0, -2.0,  "SETUP"),
    (-2.0,   -0.5,  "DRAWING"),
    (-0.5,   -0.05, "LOADING"),
    (-0.05,   0.15, "ANCHOR"),
    (0.15,   999.0, "FOLLOW_THROUGH"),
]

PHASE_COLORS = {
    "SETUP":           "#d4d4d4",
    "DRAWING":         "#a8d8f5",
    "LOADING":         "#6bb5e8",
    "ANCHOR":          "#f5d142",
    "FOLLOW_THROUGH":  "#f59442",
}


def get_phase(t_s: float) -> str:
    for lo, hi, name in PHASES:
        if lo <= t_s < hi:
            return name
    return "FOLLOW_THROUGH"


# ── Geometry helpers ──────────────────────────────────────────────────────────
def pt(lm, idx, w, h):
    return (int(lm[idx].x * w), int(lm[idx].y * h))


def angle_at_joint(a, b, c) -> float:
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    cos_v   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))


def horizontal_angle(p1, p2) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


# ── Per-frame metric computation (one function per view) ─────────────────────
def _metrics_face(lm, w, h):
    d_sh  = pt(lm, _D_SH,  w, h)
    d_el  = pt(lm, _D_EL,  w, h)
    d_wr  = pt(lm, _D_WR,  w, h)
    b_sh  = pt(lm, _B_SH,  w, h)
    b_el  = pt(lm, _B_EL,  w, h)
    b_wr  = pt(lm, _B_WR,  w, h)
    b_ear = pt(lm, _B_EAR, w, h)
    nose  = pt(lm, _MP_NOSE, w, h)

    sw = float(np.linalg.norm(np.array(b_sh) - np.array(d_sh)))

    def _sw(val):
        return round(val / sw, 4) if sw > 10 else None

    draw_elbow_angle    = angle_at_joint(d_sh, d_el, d_wr)
    bow_elbow_angle     = angle_at_joint(b_sh, b_el, b_wr)
    bse_raw             = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh)))
    anchor_y_sw         = _sw(d_wr[1] - nose[1])
    nose_string_gap     = round(float(lm[_D_WR].x - lm[_MP_NOSE].x), 4)

    return dict(
        draw_elbow_angle     = round(draw_elbow_angle, 2),
        bow_elbow_angle      = round(bow_elbow_angle, 2),
        anchor_y_sw          = anchor_y_sw,
        bow_shoulder_elev_sw = _sw(bse_raw),
        nose_y_norm          = round(float(lm[_MP_NOSE].y), 4),
        draw_wrist_y_norm    = round(float(lm[_D_WR].y), 4),
        nose_string_gap      = nose_string_gap,
        shoulder_width_px    = round(sw, 1),
    )


def _metrics_back(lm, w, h):
    l_sh  = pt(lm, _MP_LEFT_SHOULDER,  w, h)
    r_sh  = pt(lm, _MP_RIGHT_SHOULDER, w, h)
    l_ear = pt(lm, _MP_LEFT_EAR,       w, h)
    r_ear = pt(lm, _MP_RIGHT_EAR,      w, h)
    l_hip = pt(lm, _MP_LEFT_HIP,       w, h)
    r_hip = pt(lm, _MP_RIGHT_HIP,      w, h)
    b_sh  = pt(lm, _B_SH,  w, h)
    b_ear = pt(lm, _B_EAR, w, h)
    d_el  = pt(lm, _D_EL,  w, h)
    b_el  = pt(lm, _B_EL,  w, h)

    sw    = float(np.linalg.norm(np.array(l_sh) - np.array(r_sh)))
    bse_raw = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh)))

    return dict(
        shoulder_level       = round(horizontal_angle(l_sh, r_sh), 2),
        head_lateral_tilt    = round(horizontal_angle(l_ear, r_ear), 2),
        hip_alignment        = round(horizontal_angle(l_hip, r_hip), 2),
        t_draw_angle         = round(horizontal_angle(b_el, d_el), 2),
        bow_shoulder_elev_sw = round(bse_raw / sw, 4) if sw > 10 else None,
        shoulder_width_px    = round(sw, 1),
    )


def _metrics_target(lm, w, h):
    d_sh  = pt(lm, _D_SH,  w, h)
    d_el  = pt(lm, _D_EL,  w, h)
    b_sh  = pt(lm, _B_SH,  w, h)
    b_ear = pt(lm, _B_EAR, w, h)
    l_sh  = pt(lm, _MP_LEFT_SHOULDER,  w, h)
    r_sh  = pt(lm, _MP_RIGHT_SHOULDER, w, h)
    l_hip = pt(lm, _MP_LEFT_HIP,  w, h)
    r_hip = pt(lm, _MP_RIGHT_HIP, w, h)

    avg_sh_y  = (l_sh[1] + r_sh[1]) / 2
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    th = float(abs(avg_hip_y - avg_sh_y))

    def _th(val):
        return round(val / th, 4) if th > 10 else None

    bse_raw = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh)))

    return dict(
        draw_elbow_height_th  = _th(d_el[1] - d_sh[1]),
        draw_elbow_lateral_th = _th(d_el[0] - d_sh[0]),
        back_tension_proxy    = round(abs(lm[_MP_RIGHT_SHOULDER].x - lm[_MP_LEFT_SHOULDER].x), 4),
        bow_shoulder_elev_th  = _th(bse_raw),
        torso_height_px       = round(th, 1),
    )


VIEW_METRICS_FN = {
    "face":   _metrics_face,
    "back":   _metrics_back,
    "target": _metrics_target,
}

# ── Chart config: (csv_key, y-axis label, optional good-range tuple) ─────────
VIEW_PLOT_CONFIG = {
    "face": [
        ("draw_elbow_angle",     "Draw Elbow Angle (°)",      None),
        ("bow_elbow_angle",      "Bow Elbow Angle (°)",       (160, 175)),
        ("bow_shoulder_elev_sw", "Bow Shoulder Elev (sw)",    (0.55, 0.65)),
        ("anchor_y_sw",          "Draw Wrist Height (sw)",    None),
        ("nose_y_norm",          "Nose Y — head stability",   None),
    ],
    "back": [
        ("shoulder_level",       "Shoulder Level (°)",        (-3, 3)),
        ("head_lateral_tilt",    "Head Tilt (°)",             (-5, 5)),
        ("hip_alignment",        "Hip Alignment (°)",         (-5, 5)),
        ("t_draw_angle",         "T-Draw Angle (°)",          (-5, 5)),
        ("bow_shoulder_elev_sw", "Bow Shoulder Elev (sw)",    (0.55, 0.65)),
    ],
    "target": [
        ("draw_elbow_height_th",  "Elbow Height (th)",        (-1.0, 0.0)),
        ("draw_elbow_lateral_th", "Elbow Lateral (th)",       None),
        ("back_tension_proxy",    "Back Tension Proxy",       None),
        ("bow_shoulder_elev_th",  "Bow Shoulder Elev (th)",   (0.32, 0.45)),
    ],
}


def main():
    ap = argparse.ArgumentParser(
        description="Shot-cycle frame-by-frame biomechanical analysis")
    ap.add_argument("--video",      required=True,
                    help="Path to the video file")
    ap.add_argument("--frames",     type=int, nargs="+", required=True,
                    help="Verified anchor (full-draw) frame numbers")
    ap.add_argument("--view",       default="face",
                    choices=["face", "back", "target"],
                    help="Camera view (default: face)")
    ap.add_argument("--before",     type=float, default=3.0,
                    help="Seconds of window before anchor (default 3.0)")
    ap.add_argument("--after",      type=float, default=1.0,
                    help="Seconds of window after anchor (default 1.0)")
    ap.add_argument("--step",       type=int, default=3,
                    help="Sample every N frames (default 3 → ~10fps at 30fps)")
    ap.add_argument("--yolo-model", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "yolo26m-pose.pt"))
    ap.add_argument("--out-dir",    default="output")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from yolo_pose_adapter import YoloPoseAdapter
    print(f"Loading YOLO model: {args.yolo_model}")
    pose = YoloPoseAdapter(args.yolo_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}"); sys.exit(1)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps  ({total} frames total)")
    print(f"View: {args.view}  |  Anchor frames: {args.frames}")
    print(f"Window: -{args.before}s to +{args.after}s, step={args.step} frames\n")

    metrics_fn = VIEW_METRICS_FN[args.view]
    before_f   = int(args.before * fps)
    after_f    = int(args.after  * fps)
    all_rows   = []

    for shot_idx, anchor_f in enumerate(args.frames, 1):
        start_f = max(0, anchor_f - before_f)
        end_f   = min(total - 1, anchor_f + after_f)
        scan    = range(start_f, end_f + 1, args.step)
        print(f"  Shot {shot_idx}  anchor={anchor_f}  "
              f"window=[{start_f}…{end_f}]  ({len(scan)} frames)")

        shot_rows = []
        for fn in scan:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ok, frame = cap.read()
            if not ok:
                continue
            res = pose.process(frame)
            if res.pose_landmarks is None:
                continue
            lm      = res.pose_landmarks.landmark
            metrics = metrics_fn(lm, w, h)
            rel_f   = fn - anchor_f
            t_s     = rel_f / fps
            row = dict(shot=shot_idx, frame=fn, rel_frame=rel_f,
                       time_s=round(t_s, 3), phase=get_phase(t_s))
            row.update(metrics)
            shot_rows.append(row)

        print(f"    → {len(shot_rows)} frames with valid pose")
        all_rows.extend(shot_rows)

    cap.release()

    if not all_rows:
        print("No data collected — check video path and frames."); return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.out_dir, f"{args.view}_{stem}_cycle.csv")
    cols     = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nCSV written: {csv_path}  ({len(all_rows)} rows)")

    # ── Build chart ───────────────────────────────────────────────────────────
    plot_cfg  = VIEW_PLOT_CONFIG[args.view]
    n_metrics = len(plot_cfg)
    n_shots   = len(args.frames)
    shot_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_shots, 2)))

    fig, axes = plt.subplots(n_metrics, 1,
                              figsize=(14, 3.0 * n_metrics),
                              sharex=True)
    if n_metrics == 1:
        axes = [axes]

    # Group rows by shot for fast access
    by_shot = defaultdict(list)
    for row in all_rows:
        by_shot[row["shot"]].append(row)

    t_all = [r["time_s"] for r in all_rows]
    t_min, t_max = min(t_all), max(t_all)

    for ax, (metric_key, ylabel, good_range) in zip(axes, plot_cfg):
        # Phase background shading
        for lo, hi, pname in PHASES:
            lo_c = max(lo, t_min)
            hi_c = min(hi, t_max)
            if lo_c < hi_c:
                ax.axvspan(lo_c, hi_c,
                           color=PHASE_COLORS[pname], alpha=0.20, zorder=0)

        # Good-range horizontal band
        if good_range is not None:
            ax.axhspan(good_range[0], good_range[1],
                       color="green", alpha=0.10, zorder=0)

        # Release/anchor marker
        ax.axvline(0, color="crimson", linewidth=1.2,
                   linestyle="--", zorder=5)

        for s_idx, (shot_num, rows) in enumerate(sorted(by_shot.items())):
            valid = [(r["time_s"], r[metric_key])
                     for r in rows if r.get(metric_key) is not None]
            if not valid:
                continue
            ts, ys = zip(*valid)
            ax.plot(ts, ys, color=shot_colors[s_idx], linewidth=1.5,
                    alpha=0.88, label=f"Shot {shot_num}")

            # Dot at exact anchor frame
            anchor_pts = [(r["time_s"], r[metric_key])
                          for r in rows
                          if r["rel_frame"] == 0 and r.get(metric_key) is not None]
            if anchor_pts:
                ax.scatter(*zip(*anchor_pts),
                           color=shot_colors[s_idx], s=55, zorder=6)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time from release (s)", fontsize=10)

    # Legend: phase colours + shot lines
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.6, label=p.replace("_", " ").title())
        for _, _, p in PHASES
    ]
    shot_lines = [
        plt.Line2D([0], [0], color=shot_colors[i], linewidth=2.0,
                   label=f"Shot {i+1}")
        for i in range(n_shots)
    ]
    release_line = plt.Line2D([0], [0], color="crimson", linewidth=1.2,
                               linestyle="--", label="Release")
    axes[0].legend(handles=phase_patches + shot_lines + [release_line],
                   loc="upper left", fontsize=7, ncol=2, framealpha=0.8)

    title = (f"{args.view.title()} View — Full Shot Cycle  "
             f"({n_shots} shots · {args.before}s before → {args.after}s after release)")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    chart_path = os.path.join(args.out_dir, f"{args.view}_{stem}_cycle.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart written: {chart_path}")
    print(f"\nDone — {len(all_rows)} data points across "
          f"{n_shots} shots × {n_metrics} metrics.")


if __name__ == "__main__":
    main()

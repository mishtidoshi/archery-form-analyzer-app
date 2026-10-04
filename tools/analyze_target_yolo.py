#!/usr/bin/env python3
"""
Target-view YOLO analysis on manually verified full-draw frames.

Computes: draw_elbow_height, draw_elbow_lateral, back_tension_proxy,
          bow_shoulder_elevation

Usage:
  python3 tools/analyze_target_yolo.py \
      --video data/march_2026/032826/targetview/IMG_2831.MOV \
      --frames 132 661 1137 1676 2283 2811 \
      --session-json session_history/2026-03-28_practice_target_IMG2831.json \
      --yolo-model models/yolo26m-pose.pt
"""

import argparse
import json
import sys
import os
import cv2
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import metric_thresholds as MT   # noqa: E402  (canonical bands — see BB9)
import csv

# MediaPipe landmark indices
_MP_LEFT_EAR       = 7
_MP_RIGHT_EAR      = 8
_MP_LEFT_SHOULDER  = 11
_MP_RIGHT_SHOULDER = 12
_MP_LEFT_ELBOW     = 13
_MP_RIGHT_ELBOW    = 14
_MP_LEFT_HIP       = 23
_MP_RIGHT_HIP      = 24

# Right-handed archer: draw=right, bow=left
_D_SH  = _MP_RIGHT_SHOULDER
_D_EL  = _MP_RIGHT_ELBOW
_B_SH  = _MP_LEFT_SHOULDER
_B_EL  = _MP_LEFT_ELBOW
_B_EAR = _MP_LEFT_EAR



# Marker separating this run's note from the original provenance note (BB14).
_NOTE_SEP = " | original: "


def _compose_notes(old_notes: str, this_run: str) -> str:
    """Build `detection_notes` without accumulating one prefix per re-run.

    Each re-run used to prepend "Re-processed with YOLO26... Old notes: <everything
    before>", so a record re-run six times carried six nested copies and the original
    provenance note was buried at the end. This keeps exactly two parts: the current
    run and the ORIGINAL note, recovered by stripping any previously-appended chain.
    """
    original = old_notes or ""
    # Strip our own marker first (records written after this fix).
    if _NOTE_SEP in original:
        original = original.split(_NOTE_SEP, 1)[1]
    # Then unwind the legacy "Old notes: " chain to its innermost content.
    while "Old notes: " in original:
        original = original.split("Old notes: ", 1)[1]
    original = original.strip()
    return f"{this_run}{_NOTE_SEP}{original}" if original else this_run

def pt(lm, idx, w, h):
    l = lm[idx]
    return (int(l.x * w), int(l.y * h))


def analyze_frame(cap, frame_num, pose_adapter, w, h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None

    result = pose_adapter.process(frame)
    if result.pose_landmarks is None:
        return None

    lm = result.pose_landmarks.landmark

    d_sh  = pt(lm, _D_SH,  w, h)
    d_el  = pt(lm, _D_EL,  w, h)
    b_sh  = pt(lm, _B_SH,  w, h)
    b_el  = pt(lm, _B_EL,  w, h)
    b_ear = pt(lm, _B_EAR, w, h)
    l_sh  = pt(lm, _MP_LEFT_SHOULDER,  w, h)
    r_sh  = pt(lm, _MP_RIGHT_SHOULDER, w, h)

    # Negative = above shoulder (good)
    draw_elbow_height   = (lm[_D_EL].y - lm[_D_SH].y)
    # Lateral flare: positive = elbow flared out from shoulder
    draw_elbow_lateral  = (lm[_D_EL].x - lm[_D_SH].x)
    # Shoulder width proxy for back tension (normalized by width)
    back_tension_proxy  = abs(lm[_MP_RIGHT_SHOULDER].x - lm[_MP_LEFT_SHOULDER].x)
    # Bow shoulder elevation: ear-to-shoulder distance (smaller = shoulder riding up)
    bow_shoulder_elev   = np.linalg.norm(np.array(b_ear) - np.array(b_sh)) / h

    # Camera-invariant fields. Target view sees archer from behind: both shoulders
    # are visible, separation ≈ true anatomical shoulder width (with mild
    # foreshortening if torso angle varies). Scales with camera distance, so
    # ratios cancel framing changes.
    shoulder_width_px = float(np.linalg.norm(np.array(l_sh) - np.array(r_sh)))
    # Torso height: avg shoulder Y → avg hip Y. Perpendicular to the camera ray
    # in target view (vertical, in camera plane), unaffected by back tension or
    # draw mechanics — unlike shoulder_width_px, which in target view IS the
    # back_tension_proxy metric and so confounds camera distance with the
    # signal we want to measure. Preferred reference for target view.
    l_hip = pt(lm, _MP_LEFT_HIP,  w, h)
    r_hip = pt(lm, _MP_RIGHT_HIP, w, h)
    avg_sh_y  = (l_sh[1] + r_sh[1]) / 2
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    torso_height_px = float(abs(avg_hip_y - avg_sh_y))

    if shoulder_width_px < 10:
        draw_elbow_height_sw = draw_elbow_lateral_sw = bow_shoulder_elev_sw = None
    else:
        d_sh_px = pt(lm, _D_SH, w, h)
        d_el_px = pt(lm, _D_EL, w, h)
        draw_elbow_height_sw  = (d_el_px[1] - d_sh_px[1]) / shoulder_width_px
        draw_elbow_lateral_sw = (d_el_px[0] - d_sh_px[0]) / shoulder_width_px
        bow_shoulder_elev_sw  = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh))) / shoulder_width_px

    if torso_height_px < 10:
        draw_elbow_height_th = draw_elbow_lateral_th = bow_shoulder_elev_th = None
    else:
        d_sh_px = pt(lm, _D_SH, w, h)
        d_el_px = pt(lm, _D_EL, w, h)
        draw_elbow_height_th  = (d_el_px[1] - d_sh_px[1]) / torso_height_px
        draw_elbow_lateral_th = (d_el_px[0] - d_sh_px[0]) / torso_height_px
        bow_shoulder_elev_th  = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh))) / torso_height_px

    vis_d_el = lm[_D_EL].visibility
    vis_d_sh = lm[_D_SH].visibility

    return dict(
        frame               = frame_num,
        draw_elbow_height   = round(draw_elbow_height,  4),
        draw_elbow_lateral  = round(draw_elbow_lateral, 4),
        back_tension_proxy  = round(back_tension_proxy, 4),
        bow_shoulder_elev   = round(bow_shoulder_elev,  3),
        shoulder_width_px      = round(shoulder_width_px, 1),
        torso_height_px        = round(torso_height_px, 1),
        draw_elbow_height_sw   = round(draw_elbow_height_sw, 4)  if draw_elbow_height_sw  is not None else None,
        draw_elbow_lateral_sw  = round(draw_elbow_lateral_sw, 4) if draw_elbow_lateral_sw is not None else None,
        bow_shoulder_elev_sw   = round(bow_shoulder_elev_sw, 4)  if bow_shoulder_elev_sw  is not None else None,
        draw_elbow_height_th   = round(draw_elbow_height_th, 4)  if draw_elbow_height_th  is not None else None,
        draw_elbow_lateral_th  = round(draw_elbow_lateral_th, 4) if draw_elbow_lateral_th is not None else None,
        bow_shoulder_elev_th   = round(bow_shoulder_elev_th, 4)  if bow_shoulder_elev_th  is not None else None,
        vis_draw            = round((vis_d_el + vis_d_sh) / 2, 2),
    )


def _write_csv(results, video_path, out_dir="output"):
    """Write one CSV row per verified shot to output/target_{stem}_shots.csv."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_dir, f"target_{stem}_shots.csv")
    if not results:
        return csv_path
    cols = ["shot"] + list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({"shot": i, **r})
    return csv_path


def _print_feedback(results, eh_avg, el_avg, bt_avg, bse_th_avg):
    """Print a quick tabular feedback summary with strengths and areas of improvement."""
    n = len(results)
    print(f"\n{'═'*64}")
    print(f"  FEEDBACK — Target View  ({n} shot{'s' if n != 1 else ''})")
    print(f"{'═'*64}")
    print(f"  {'Metric':<32} {'Value':>10}  {'Target':<20} Status")
    print(f"  {'─'*62}")

    checks = []

    def chk(label, val, display, target_str, ok):
        symbol = "✓" if ok else "✗"
        print(f"  {label:<32} {display:>10}  {target_str:<20} {symbol}")
        checks.append((label, ok, val, display))

    chk("Draw elbow height (th)",     eh_avg,  f"{eh_avg:+.3f}",   "negative = above sh.", eh_avg < 0)
    chk("Draw elbow lateral (th)",    el_avg,  f"{el_avg:+.3f}",   "low = tucked",         abs(el_avg) <= 0.15)
    # back_tension_proxy is session-relative — flag only extreme outliers
    bt_vals = [r["back_tension_proxy"] for r in results]
    bt_std = float(np.std(bt_vals))
    chk("Back tension consistency",   bt_std,  f"std={bt_std:.4f}","std < 0.010",          bt_std < 0.010)
    if bse_th_avg is not None:
        chk("Bow shoulder elev (th)", bse_th_avg, f"{bse_th_avg:.3f}", "0.32–0.45",        0.32 <= bse_th_avg <= 0.45)

    print(f"  {'─'*62}")

    goods  = [(l, v, d) for l, ok, v, d in checks if ok]
    issues = [(l, v, d) for l, ok, v, d in checks if not ok]

    _strengths = {
        "Draw elbow height (th)":    lambda v, d: f"Draw elbow above shoulder ({d}) — good high elbow position",
        "Draw elbow lateral (th)":   lambda v, d: f"Draw elbow tucked behind body ({d})",
        "Back tension consistency":  lambda v, d: f"Back tension consistent shot-to-shot ({d})",
        "Bow shoulder elev (th)":    lambda v, d: f"Bow shoulder elevation stable ({d})",
    }
    _improvements = {
        "Draw elbow height (th)":    lambda v, d: f"Draw elbow at or below shoulder ({d}) — raise the elbow, come from below",
        "Draw elbow lateral (th)":   lambda v, d: f"Draw elbow flaring laterally ({d}) — wrap elbow behind body line",
        "Back tension consistency":  lambda v, d: f"Back tension varying shot-to-shot ({d}) — focus on consistent scapula engagement",
        "Bow shoulder elev (th)":    lambda v, d: f"Bow shoulder elevation out of range ({d}) — check shoulder position at setup",
    }

    print(f"\n  STRENGTHS")
    shown = 0
    for label, val, display in goods:
        if label in _strengths and shown < 3:
            print(f"    ✓ {_strengths[label](val, display)}")
            shown += 1
    if shown == 0:
        print("    (check raw metrics above)")

    print(f"\n  AREAS OF IMPROVEMENT")
    shown = 0
    for label, val, display in issues:
        if label in _improvements and shown < 3:
            print(f"    ✗ {_improvements[label](val, display)}")
            shown += 1
    if shown == 0:
        print("    All metrics within range this session")

    print(f"{'═'*64}\n")


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from yolo_pose_adapter import YoloPoseAdapter, resolve_pose_model_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--frames",       type=int, nargs="+", required=True,
                    help="Verified full-draw frame numbers")
    ap.add_argument("--session-json", required=False)
    ap.add_argument("--yolo-model",   default=resolve_pose_model_path())
    ap.add_argument("--hand",         default="right", choices=["right", "left"])
    args = ap.parse_args()

    print(f"Loading YOLO model: {args.yolo_model}")
    pose = YoloPoseAdapter(args.yolo_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}"); sys.exit(1)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"Verified frames: {args.frames}\n")

    results = []
    for i, fn in enumerate(args.frames, 1):
        r = analyze_frame(cap, fn, pose, w, h)
        if r is None:
            print(f"  Shot {i} frame {fn}: POSE DROPOUT")
        else:
            flag = "✓" if r["vis_draw"] > 0.4 else "⚠ low vis"
            elev_flag = "✓" if r["draw_elbow_height"] < 0 else "✗"
            print(f"  Shot {i} frame {fn:5d}  "
                  f"EH={r['draw_elbow_height']:+.4f}{elev_flag}  "
                  f"ELat={r['draw_elbow_lateral']:+.4f}  "
                  f"BT={r['back_tension_proxy']:.4f}  "
                  f"BowSh={r['bow_shoulder_elev']:.3f}  {flag}")
            results.append(r)

    cap.release()

    if not results:
        print("\nNo frames successfully analyzed.")
        return

    print(f"\n{'─'*60}")
    print(f"SUMMARY  ({len(results)}/{len(args.frames)} frames detected)")
    print(f"{'─'*60}")

    def stat(name, key, unit="", good_range=None):
        vals = [r[key] for r in results]
        avg, std = np.mean(vals), np.std(vals)
        flag = ""
        if good_range:
            flag = " ✓" if good_range[0] <= avg <= good_range[1] else " ✗"
        print(f"  {name:<28s}  avg={avg:+8.4f}{unit}  std={std:.4f}{unit}{flag}")
        return avg, std

    # Draw elbow height: canonical band is on the torso-height variant; the raw
    # image-relative value keeps its sign check only (negative = above shoulder).
    eh_avg,  eh_std  = stat("Draw Elbow Height",       "draw_elbow_height",  "", (-1.0, 0.0))
    el_avg,  el_std  = stat("Draw Elbow Lateral",      "draw_elbow_lateral")
    bt_avg,  bt_std  = stat("Back Tension Proxy",      "back_tension_proxy")
    # Legacy `/h` BSE: NO BAND (BB9) — wrong scale, failed nearly every session.
    bse_avg, bse_std = stat("Bow Shoulder Elevation",  "bow_shoulder_elev",  "")

    # Shoulder-width-normalized variants (camera-invariant)
    sw_results = [r for r in results if r.get("shoulder_width_px") is not None]
    if sw_results:
        sw_avg = float(np.mean([r["shoulder_width_px"] for r in sw_results]))
        sw_std = float(np.std([r["shoulder_width_px"] for r in sw_results]))
        print(f"\n  ── Shoulder-width-normalized (camera-invariant) ──")
        print(f"  {'Shoulder width (px)':<28s}  avg={sw_avg:.1f}px  std={sw_std:.1f}px")

        def stat_sw(name, key):
            vals = [r[key] for r in sw_results if r[key] is not None]
            if not vals:
                return None, None
            avg, std = float(np.mean(vals)), float(np.std(vals))
            print(f"  {name:<28s}  avg={avg:+8.4f}  std={std:.4f}")
            return avg, std

        eh_sw_avg, eh_sw_std   = stat_sw("Draw Elbow Height (sw)",  "draw_elbow_height_sw")
        el_sw_avg, el_sw_std   = stat_sw("Draw Elbow Lateral (sw)", "draw_elbow_lateral_sw")
        bse_sw_avg, bse_sw_std = stat_sw("Bow Shoulder Elev (sw)",  "bow_shoulder_elev_sw")
    else:
        sw_avg = sw_std = None
        eh_sw_avg = eh_sw_std = el_sw_avg = el_sw_std = None
        bse_sw_avg = bse_sw_std = None

    # Torso-height-normalized variants (preferred for target view — unaffected
    # by back tension or shoulder-axis-along-camera foreshortening)
    th_results = [r for r in results if r.get("torso_height_px") is not None and r.get("draw_elbow_height_th") is not None]
    if th_results:
        th_avg = float(np.mean([r["torso_height_px"] for r in th_results]))
        th_std = float(np.std([r["torso_height_px"] for r in th_results]))
        print(f"\n  ── Torso-height-normalized (preferred for target view) ──")
        print(f"  {'Torso height (px)':<28s}  avg={th_avg:.1f}px  std={th_std:.1f}px")

        def stat_th(name, key):
            vals = [r[key] for r in th_results if r[key] is not None]
            if not vals:
                return None, None
            avg, std = float(np.mean(vals)), float(np.std(vals))
            print(f"  {name:<28s}  avg={avg:+8.4f}  std={std:.4f}")
            return avg, std

        eh_th_avg, eh_th_std   = stat_th("Draw Elbow Height (th)",  "draw_elbow_height_th")
        el_th_avg, el_th_std   = stat_th("Draw Elbow Lateral (th)", "draw_elbow_lateral_th")
        bse_th_avg, bse_th_std = stat_th("Bow Shoulder Elev (th)",  "bow_shoulder_elev_th")
    else:
        th_avg = th_std = None
        eh_th_avg = eh_th_std = el_th_avg = el_th_std = None
        bse_th_avg = bse_th_std = None

    if args.session_json and os.path.exists(args.session_json):
        with open(args.session_json) as f:
            session = json.load(f)

        old_metrics = session.get("metrics", {})
        old_notes   = session.get("detection_notes", "")

        new_metrics = dict(old_metrics)
        # Remove old noisy fields from analyze_manual_shots.py
        # Broken target-view `_sw` fields are actively removed on re-run (BB8).
        for old_key in [
            "draw_elbow_height_sw_avg", "draw_elbow_height_sw_std",
            "draw_elbow_lateral_sw_avg", "draw_elbow_lateral_sw_std",
            "bow_shoulder_elevation_sw_avg", "bow_shoulder_elevation_sw_std",
            "draw_elbow_angle_avg", "draw_elbow_angle_std",
            "bow_elbow_angle_avg",  "bow_elbow_angle_std",
            "anchor_consistency_x_px", "anchor_consistency_y_px",
            "draw_elbow_height_range", "elbow_position_quality",
            "poses_detected",
        ]:
            new_metrics.pop(old_key, None)

        new_metrics.update({
            "view":                       "target",
            "arrows_shot":                len(args.frames),
            "arrows_with_metrics":        len(results),
            "verified_frames":            args.frames,
            "draw_elbow_height_avg":      round(eh_avg,  4),
            "draw_elbow_height_std":      round(eh_std,  4),
            "draw_elbow_lateral_avg":     round(el_avg,  4),
            "draw_elbow_lateral_std":     round(el_std,  4),
            "back_tension_proxy_avg":     round(bt_avg,  4),
            "back_tension_proxy_std":     round(bt_std,  4),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        })

        if sw_avg is not None:
            new_metrics.update({
                "shoulder_width_px_avg":         round(sw_avg, 1),
                "shoulder_width_px_std":         round(sw_std, 1),
            })
            # The `*_sw` variants are printed above for diagnostics but deliberately
            # NOT written to the session record (BB8). Shoulder width is unusable as a
            # normalization reference in target view — at full draw the shoulders lie
            # nearly along the camera axis, giving ~28 +/- 9 px (32% CV) — so these
            # values carry that noise. `*_th` (torso-height) is canonical for this
            # view. Storing them was a foot-gun: a future caller could pick `_sw` by
            # mistake and get garbage that looks like a valid camera-invariant field.

        if th_avg is not None and eh_th_avg is not None:
            new_metrics.update({
                "torso_height_px_avg":           round(th_avg, 1),
                "torso_height_px_std":           round(th_std, 1),
                "draw_elbow_height_th_avg":      round(eh_th_avg, 4),
                "draw_elbow_height_th_std":      round(eh_th_std, 4),
                "draw_elbow_lateral_th_avg":     round(el_th_avg, 4),
                "draw_elbow_lateral_th_std":     round(el_th_std, 4),
                "bow_shoulder_elevation_th_avg": round(bse_th_avg, 4),
                "bow_shoulder_elevation_th_std": round(bse_th_std, 4),
            })

        session["metrics"]         = new_metrics
        session["view"]            = "target"
        session["detection_notes"] = _compose_notes(
            old_notes,
            f"Re-processed with YOLO26 (analyze_target_yolo.py). "
            f"{len(results)}/{len(args.frames)} verified frames detected.",
        )

        with open(args.session_json, "w") as f:
            json.dump(session, f, indent=2)
        print(f"\nSession JSON updated: {args.session_json}")
    else:
        print(f"\n{'─'*60}")
        print("Metrics to paste into session JSON:")
        print(json.dumps({
            "draw_elbow_height_avg":      round(eh_avg,  4),
            "draw_elbow_height_std":      round(eh_std,  4),
            "draw_elbow_lateral_avg":     round(el_avg,  4),
            "draw_elbow_lateral_std":     round(el_std,  4),
            "back_tension_proxy_avg":     round(bt_avg,  4),
            "back_tension_proxy_std":     round(bt_std,  4),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        }, indent=4))

    # ── CSV + feedback (always, regardless of JSON flag) ─────────────────────
    csv_out = _write_csv(results, args.video)
    print(f"CSV written: {csv_out}")
    _print_feedback(results, eh_th_avg if eh_th_avg is not None else eh_avg,
                    el_th_avg if el_th_avg is not None else el_avg,
                    bt_avg, bse_th_avg)


if __name__ == "__main__":
    main()

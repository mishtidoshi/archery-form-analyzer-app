#!/usr/bin/env python3
"""
Back-view YOLO analysis on manually verified full-draw frames.

Computes: shoulder_level, hip_alignment, head_lateral_tilt,
          t_draw_angle, bow_shoulder_elevation

Usage:
  python3 tools/analyze_back_yolo.py \
      --video data/march_2026/032726/rearview_6_arrows.MOV \
      --frames 137 663 1206 1747 2267 2776 \
      --session-json session_history/2026-03-27_practice_back_rearview_6_arrows.json \
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

# ── MediaPipe landmark indices (replicated from audio_visual_shot_detector.py) ─
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

# ── Right-handed archer landmark assignments for back view ─────────────────────
# Draw arm = RIGHT, Bow arm = LEFT
_D_SH  = _MP_RIGHT_SHOULDER
_D_EL  = _MP_RIGHT_ELBOW
_B_SH  = _MP_LEFT_SHOULDER
_B_EL  = _MP_LEFT_ELBOW
_B_WR  = _MP_LEFT_WRIST
_B_EAR = _MP_LEFT_EAR
_D_EAR = _MP_RIGHT_EAR



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


def horizontal_angle(p1, p2) -> float:
    """Angle of line p1→p2 relative to horizontal, in degrees."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def angle_at_joint(a, b, c) -> float:
    """Interior angle at b, formed by the rays b→a and b→c, in degrees.

    Saturates at 180°: three collinear points give 180 regardless of ordering, so
    this cannot express "beyond straight". That is why the DFL measure below reports
    a deviation rather than attempting to detect hyperextension.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    cos_val = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def analyze_frame(cap, frame_num, pose_adapter, w, h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None

    result = pose_adapter.process(frame)
    if result.pose_landmarks is None:
        return None

    lm = result.pose_landmarks.landmark

    l_sh  = pt(lm, _MP_LEFT_SHOULDER,  w, h)
    r_sh  = pt(lm, _MP_RIGHT_SHOULDER, w, h)
    l_ear = pt(lm, _MP_LEFT_EAR,       w, h)
    r_ear = pt(lm, _MP_RIGHT_EAR,      w, h)
    l_hip = pt(lm, _MP_LEFT_HIP,       w, h)
    r_hip = pt(lm, _MP_RIGHT_HIP,      w, h)
    b_sh  = pt(lm, _B_SH,  w, h)
    b_el  = pt(lm, _B_EL,  w, h)
    b_wr  = pt(lm, _B_WR,  w, h)
    d_el  = pt(lm, _D_EL,  w, h)
    d_sh  = pt(lm, _D_SH,  w, h)
    b_ear = pt(lm, _B_EAR, w, h)

    shoulder_level     = horizontal_angle(l_sh, r_sh)
    head_lateral_tilt  = horizontal_angle(l_ear, r_ear)
    hip_alignment      = horizontal_angle(l_hip, r_hip)
    t_draw_angle       = horizontal_angle(b_el, d_el)
    bow_shoulder_elev  = np.linalg.norm(np.array(b_ear) - np.array(b_sh)) / h

    # Draw force line (DFL) angle: the interior angle at the DRAW SHOULDER between the
    # bow wrist and the draw elbow. In archery coaching the draw force line runs from
    # the bow-hand pressure point through the draw shoulder to the draw elbow, and
    # alignment is judged by how close those three points come to a straight line.
    #
    # ⚠️ 180° IS NOT THE PRACTICAL TARGET, despite being the idealised one.
    # Measured across 23 back-view sessions this reads 136–151° (mean 139°), and a
    # genuine 40° alignment fault sustained over 14 months would have been raised by a
    # coach long ago. The offset is in the measurement, not the archer: the shoulder
    # keypoint approximates the acromion rather than the joint centre the coaching
    # concept refers to, and the bow WRIST keypoint is not the bow-hand pressure point.
    # Collinearity itself survives projection, so this is an anatomical
    # landmark-definition offset rather than a perspective artifact.
    #
    # What the measurement is good for is CONSISTENCY, not absolute alignment.
    # Within-session SD is tight (median 1.54°) while between-session SD is 11.96° —
    # an 8× ratio with the same signature as draw elbow angle (see the camera-geometry
    # finding in the Limitations draft). So: compare shot-to-shot within a session,
    # never the absolute value across sessions, and do not flag against 180°.
    dfl_angle = angle_at_joint(b_wr, d_sh, d_el)
    # Deviation from the idealised straight line, recorded for interpretability only.
    dfl_deviation = 180.0 - dfl_angle

    # Camera-invariant BSE normalized by anatomical shoulder-width (L↔R shoulder
    # pixel distance at this frame). Back view sees both shoulders square-on
    # with no foreshortening, so shoulder_width_px ≈ true anatomical span and
    # scales linearly with camera distance.
    shoulder_width_px = float(np.linalg.norm(np.array(l_sh) - np.array(r_sh)))
    if shoulder_width_px < 10:
        bow_shoulder_elev_sw = None
    else:
        bow_shoulder_elev_sw = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh))) / shoulder_width_px

    # Visibility scores for key landmarks (quality check)
    vis_lsh  = lm[_MP_LEFT_SHOULDER].visibility
    vis_rsh  = lm[_MP_RIGHT_SHOULDER].visibility
    vis_lhip = lm[_MP_LEFT_HIP].visibility
    vis_rhip = lm[_MP_RIGHT_HIP].visibility

    return dict(
        frame            = frame_num,
        shoulder_level   = round(shoulder_level, 3),
        head_lateral_tilt= round(head_lateral_tilt, 3),
        hip_alignment    = round(hip_alignment, 3),
        t_draw_angle     = round(t_draw_angle, 3),
        dfl_angle        = round(dfl_angle, 3),
        dfl_deviation    = round(dfl_deviation, 3),
        bow_shoulder_elev= round(bow_shoulder_elev, 3),
        shoulder_width_px    = round(shoulder_width_px, 1),
        bow_shoulder_elev_sw = round(bow_shoulder_elev_sw, 4) if bow_shoulder_elev_sw is not None else None,
        vis_shoulders    = round((vis_lsh + vis_rsh) / 2, 2),
        vis_hips         = round((vis_lhip + vis_rhip) / 2, 2),
    )


def _write_csv(results, video_path, out_dir="output"):
    """Write one CSV row per verified shot to output/back_{stem}_shots.csv."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_dir, f"back_{stem}_shots.csv")
    if not results:
        return csv_path
    cols = ["shot"] + list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({"shot": i, **r})
    return csv_path


def _print_feedback(results, sl_avg, ht_avg, ha_avg, td_avg, bse_sw_avg):
    """Print a quick tabular feedback summary with strengths and areas of improvement."""
    n = len(results)
    print(f"\n{'═'*64}")
    print(f"  FEEDBACK — Back View  ({n} shot{'s' if n != 1 else ''})")
    print(f"{'═'*64}")
    print(f"  {'Metric':<30} {'Value':>10}  {'Target':<22} Status")
    print(f"  {'─'*62}")

    checks = []

    def chk(label, val, display, target_str, ok):
        symbol = "✓" if ok else "✗"
        print(f"  {label:<30} {display:>10}  {target_str:<22} {symbol}")
        checks.append((label, ok, val, display))

    chk("Shoulder level",            sl_avg,  f"{sl_avg:+.2f}°",    "within ±3°",          abs(sl_avg) <= 3)
    chk("Head lateral tilt",         ht_avg,  f"{ht_avg:+.2f}°",    "within ±5°",          abs(ht_avg) <= 5)
    chk("Hip alignment",             ha_avg,  f"{ha_avg:+.2f}°",    "within ±5°",          abs(ha_avg) <= 5)
    chk("T-draw angle",              td_avg,  f"{td_avg:+.2f}°",    "within ±5°",          abs(td_avg) <= 5)
    if bse_sw_avg is not None:
        chk("Bow shoulder elev (sw)",bse_sw_avg, f"{bse_sw_avg:.3f}", "0.55–0.65",          0.55 <= bse_sw_avg <= 0.65)

    print(f"  {'─'*62}")

    goods  = [(l, v, d) for l, ok, v, d in checks if ok]
    issues = [(l, v, d) for l, ok, v, d in checks if not ok]

    _strengths = {
        "Shoulder level":            lambda v, d: f"Shoulder line level ({d}) — good upper body alignment",
        "Head lateral tilt":         lambda v, d: f"Head lateral tilt within range ({d})",
        "Hip alignment":             lambda v, d: f"Hip alignment consistent ({d}) — stable stance base",
        "T-draw angle":              lambda v, d: f"T-draw angle symmetric ({d}) — arms balanced",
        "Bow shoulder elev (sw)":    lambda v, d: f"Bow shoulder elevation consistent ({d})",
    }
    _improvements = {
        "Shoulder level":            lambda v, d: f"Shoulder tilt {d} — {'left shoulder high' if v > 0 else 'right shoulder high'}; focus on leveling at setup",
        "Head lateral tilt":         lambda v, d: f"Head lateral tilt {d} — keep head vertical at anchor",
        "Hip alignment":             lambda v, d: f"Hip line tilted {d} — check foot stance width and weight distribution",
        "T-draw angle":              lambda v, d: f"T-draw asymmetric ({d}) — draw elbow needs to come more {'down' if v > 0 else 'up'} to level the T",
        "Bow shoulder elev (sw)":    lambda v, d: f"Bow shoulder elevated (sw={d}) — actively sink shoulder before and during draw",
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
    # ── Load YOLO adapter ────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from yolo_pose_adapter import YoloPoseAdapter, resolve_pose_model_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--frames",       type=int, nargs="+", required=True,
                    help="Verified full-draw frame numbers")
    ap.add_argument("--session-json", required=False,
                    help="Existing session JSON to update with YOLO metrics")
    ap.add_argument("--yolo-model",   default=resolve_pose_model_path())
    ap.add_argument("--hand",         default="right", choices=["right", "left"])
    args = ap.parse_args()

    print(f"Loading YOLO model: {args.yolo_model}")
    pose = YoloPoseAdapter(args.yolo_model)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"Verified frames: {args.frames}\n")

    # ── Analyze each verified frame ──────────────────────────────────────────
    results = []
    for i, fn in enumerate(args.frames, 1):
        r = analyze_frame(cap, fn, pose, w, h)
        if r is None:
            print(f"  Shot {i} frame {fn}: POSE DROPOUT — no landmarks detected")
        else:
            flag = "✓" if r["vis_shoulders"] > 0.4 else "⚠ low visibility"
            print(f"  Shot {i} frame {fn:5d}  "
                  f"ShouLvl={r['shoulder_level']:+7.2f}°  "
                  f"HipAlgn={r['hip_alignment']:+7.2f}°  "
                  f"HeadTilt={r['head_lateral_tilt']:+7.2f}°  "
                  f"TDraw={r['t_draw_angle']:+7.2f}°  "
                  f"BowSh={r['bow_shoulder_elev']:.3f}  {flag}")
            results.append(r)

    cap.release()

    if not results:
        print("\nNo frames successfully analyzed.")
        return

    # ── Summary statistics ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY  ({len(results)}/{len(args.frames)} frames detected)")
    print(f"{'─'*60}")

    def stat(name, key, unit="°", good_range=None):
        vals = [r[key] for r in results]
        avg, std = np.mean(vals), np.std(vals)
        flag = ""
        if good_range:
            flag = " ✓" if good_range[0] <= avg <= good_range[1] else " ✗"
        print(f"  {name:<28s}  avg={avg:+7.2f}{unit}  std={std:.2f}{unit}{flag}")
        return avg, std

    sl_avg,  sl_std  = stat("Shoulder Level",       "shoulder_level",
                            good_range=MT.good_range("back", "shoulder_level"))
    ht_avg,  ht_std  = stat("Head Lateral Tilt",    "head_lateral_tilt",
                            good_range=MT.good_range("back", "head_lateral_tilt"))
    ha_avg,  ha_std  = stat("Hip Alignment",        "hip_alignment",
                            good_range=MT.good_range("back", "hip_alignment"))
    # T-draw angle: OBSERVATION-ONLY, no pass/fail band (BB3, 2026-07-29).
    #
    # The old (-5, 5) band assumed 0 degrees is ideal ("perfect T"). It is not.
    # The metric is the tilt of the line from the BOW elbow to the DRAW elbow, and
    # in correct Olympic recurve form the draw elbow sits at or above the arrow
    # line, i.e. ABOVE the bow elbow -- which the project's own target-view cue
    # already states ("negative = above shoulder = good"). So a negative tilt is
    # correct form, not a fault. Measured: all 25 back sessions in the corpus are
    # negative (-2.3 to -17.0 deg), with within-session SD typically 0.1-0.5 deg --
    # highly repeatable, which a real 14-degree asymmetry would not be.
    #
    # The magnitude is also camera-geometry dependent, like shoulder level: the
    # backyard 18m setup reads ~-10 to -17 deg and the outdoor 60m setup ~-2 to
    # -9 deg. Cross-setting comparison of the VALUE is therefore invalid; the
    # within-session SD is the usable signal.
    #
    # A coach-confirmed target is still needed before this can be flagged at all.
    td_avg,  td_std  = stat("T-Draw Angle",         "t_draw_angle")
    # DFL: observation-only. Its within-session SD is the usable signal (median 1.54°
    # across the corpus); the absolute value is camera-dependent (between-session SD
    # 11.96°) and offset from the idealised 180° by landmark definition. See the note
    # in analyze_frame.
    dfl_avg, dfl_std = stat("DFL Angle",             "dfl_angle")
    # Legacy `/h` BSE: NO BAND (BB9) — see the note in analyze_face_yolo.py. The
    # old (0.18, 0.50) band was on the wrong scale and failed nearly every session.
    bse_avg, bse_std = stat("Bow Shoulder Elevation","bow_shoulder_elev", unit="")

    # Shoulder-width-normalized BSE (camera-invariant)
    sw_results = [r for r in results if r.get("shoulder_width_px") is not None and r.get("bow_shoulder_elev_sw") is not None]
    if sw_results:
        sw_avg = float(np.mean([r["shoulder_width_px"] for r in sw_results]))
        sw_std = float(np.std([r["shoulder_width_px"] for r in sw_results]))
        bse_sw_vals = [r["bow_shoulder_elev_sw"] for r in sw_results]
        bse_sw_avg = float(np.mean(bse_sw_vals))
        bse_sw_std = float(np.std(bse_sw_vals))
        print(f"  {'Shoulder width (px)':<28s}  avg={sw_avg:.1f}px  std={sw_std:.1f}px")
        print(f"  {'BSE (sh-widths)':<28s}  avg={bse_sw_avg:+8.4f}  std={bse_sw_std:.4f}")
    else:
        sw_avg = sw_std = None
        bse_sw_avg = bse_sw_std = None

    # ── Update session JSON if requested ─────────────────────────────────────
    if args.session_json and os.path.exists(args.session_json):
        with open(args.session_json) as f:
            session = json.load(f)

        old_metrics = session.get("metrics", {})
        old_notes   = session.get("detection_notes", "")

        new_metrics = dict(old_metrics)
        new_metrics.update({
            "arrows_shot":              len(args.frames),
            "arrows_with_metrics":      len(results),
            "verified_frames":          args.frames,
            "shoulder_level_avg":       round(sl_avg,  3),
            "shoulder_level_std":       round(sl_std,  3),
            "head_lateral_tilt_avg":    round(ht_avg,  3),
            "head_lateral_tilt_std":    round(ht_std,  3),
            "hip_alignment_avg":        round(ha_avg,  3),
            "hip_alignment_std":        round(ha_std,  3),
            "t_draw_angle_avg":         round(td_avg,  3),
            "t_draw_angle_std":         round(td_std,  3),
            "dfl_angle_avg":            round(dfl_avg, 3),
            "dfl_angle_std":            round(dfl_std, 3),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        })

        if sw_avg is not None:
            new_metrics.update({
                "shoulder_width_px_avg":         round(sw_avg, 1),
                "shoulder_width_px_std":         round(sw_std, 1),
                "bow_shoulder_elevation_sw_avg": round(bse_sw_avg, 4),
                "bow_shoulder_elevation_sw_std": round(bse_sw_std, 4),
            })

        # Remove old noisy batch-format keys
        for old_key in [
            "shoulder_level_deg_mean", "shoulder_level_deg_std",
            "shoulder_level_at_fd_deg_mean", "shoulder_level_at_fd_deg_std",
            "head_lateral_tilt_deg_mean", "head_lateral_tilt_deg_std",
            "hip_alignment_deg_mean", "hip_alignment_deg_std",
            "hip_alignment_at_fd_deg_mean", "hip_alignment_at_fd_deg_std",
            "tdraw_angle_deg_mean", "tdraw_angle_deg_std",
            "bow_shoulder_elevation_mean",
            "bow_arm_vertical_deg_mean", "bow_arm_vertical_deg_std",
        ]:
            new_metrics.pop(old_key, None)

        session["metrics"]          = new_metrics
        session["detection_notes"]  = _compose_notes(
            old_notes,
            f"Re-processed with YOLO26 (analyze_back_yolo.py). "
            f"{len(results)}/{len(args.frames)} verified frames detected. "
            f"Old MediaPipe metrics had ±60° noise (unusable).",
        )

        with open(args.session_json, "w") as f:
            json.dump(session, f, indent=2)
        print(f"\nSession JSON updated: {args.session_json}")

    else:
        print(f"\nNo session JSON updated "
              f"({'file not found' if args.session_json else 'no --session-json given'}).")
        print("\nNew metrics to paste into session JSON:")
        print(json.dumps({
            "shoulder_level_avg":        round(sl_avg,  3),
            "shoulder_level_std":        round(sl_std,  3),
            "head_lateral_tilt_avg":     round(ht_avg,  3),
            "head_lateral_tilt_std":     round(ht_std,  3),
            "hip_alignment_avg":         round(ha_avg,  3),
            "hip_alignment_std":         round(ha_std,  3),
            "t_draw_angle_avg":          round(td_avg,  3),
            "t_draw_angle_std":          round(td_std,  3),
            "dfl_angle_avg":             round(dfl_avg, 3),
            "dfl_angle_std":             round(dfl_std, 3),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        }, indent=4))

    # ── CSV + feedback (always, regardless of JSON flag) ─────────────────────
    csv_out = _write_csv(results, args.video)
    print(f"CSV written: {csv_out}")
    _print_feedback(results, sl_avg, ht_avg, ha_avg, td_avg, bse_sw_avg)


if __name__ == "__main__":
    main()

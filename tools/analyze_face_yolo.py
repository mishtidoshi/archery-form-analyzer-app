#!/usr/bin/env python3
"""
Face-view YOLO analysis on manually verified full-draw frames.

Computes: draw_elbow_angle, bow_elbow_angle, anchor_x/y, anchor_spread_px,
          nose_string_gap, bow_shoulder_elevation, nose_preanchor_drift

Usage:
  python3 tools/analyze_face_yolo.py \
      --video data/march_2026/032826/faceview/IMG_2837.MOV \
      --frames 138 678 1197 1755 2268 2846 \
      --session-json session_history/2026-03-28_practice.json \
      --yolo-model models/yolo26m-pose.pt
"""

import argparse
import json
import sys
import os
import cv2
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hold_time   # noqa: E402  (shared hold-time definition — see BB11)
import metric_thresholds as MT   # noqa: E402  (canonical bands — see BB9)
import shot_dynamics as SD       # noqa: E402  (within-shot measures — see H2)

# MediaPipe landmark indices
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

# Right-handed archer: draw=right, bow=left
_D_SH  = _MP_RIGHT_SHOULDER
_D_EL  = _MP_RIGHT_ELBOW
_D_WR  = _MP_RIGHT_WRIST
_B_SH  = _MP_LEFT_SHOULDER
_B_EL  = _MP_LEFT_ELBOW
_B_WR  = _MP_LEFT_WRIST
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


def angle_at_joint(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _read_landmarks(cap, frame_num, pose_adapter):
    """Returns (landmark_list, w, h) or (None, None, None) on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    h, w = frame.shape[:2]
    res = pose_adapter.process(frame)
    if res.pose_landmarks is None:
        return None, None, None
    return res.pose_landmarks.landmark, w, h


def compute_hold_and_followthrough(cap, fd_frame, fd_anchor_y_norm,
                                     fd_bow_wrist_x_norm, pose_adapter, w, h, fps,
                                     lookback_s: float = 6.0, lookahead_s: float = 0.67,
                                     anchor_y_tol: float = 0.025):
    """
    Hold time: delegated to `tools/hold_time.measure_hold`, which locates both the
    hold onset and the release from draw-wrist SPEED, using thresholds derived from
    each shot's own noise floor.

    History (BB1 + BB11, fixed 2026-07-28). This function used to back-scan while
    the wrist stayed within `anchor_y_tol` of its anchor Y. Two defects:
      - BB1: bounded by a 1.0s window, so the scan hit the edge on nearly every
        shot; 41 of 58 stored records are the window length, not a hold.
      - BB11: the tolerance decided the answer (.025 -> 4.77s, .010 -> 2.73s,
        .005 -> 0.67s on one shot). "Still" is not a distance — a wrist creeping
        slowly through a long aim never trips a distance tolerance, so aiming was
        counted as hold.
    The velocity formulation cut mean parameter sensitivity from ~4.1s to 0.82s.
    See tools/hold_time.py for the definition and its empirical basis.

    `anchor_y_tol` is retained in the signature for call compatibility and is no
    longer used for hold time.

    Follow-through: at fd_frame + lookahead_s, measure bow elbow angle and the
    bow wrist horizontal delta from FD.

    ⚠️ `bow_wrist_ft_dx` IS OBSERVATION-ONLY AND ITS OLD SEMANTICS WERE WRONG
    (BB5, 2026-07-29). Three separate problems, all verified on real footage:

      1. SIGN WAS INVERTED. In face view the bow arm extends toward image +x
         (measured: bow wrist x ~0.60 vs nose ~0.40), so the TARGET is image-right
         and the x axis runs ALONG the shooting line. Therefore NEGATIVE dx = the
         bow wrist moved AWAY from the target (backward), and POSITIVE = toward it.
         The old comment had this backwards ("negative = on the line = good").
      2. IT CANNOT MEASURE THE COACH'S CUE. The recurring cue is the bow arm
         drifting laterally (a common coach cue: "bow arm collapsing right"). Lateral means
         perpendicular to the shooting line — which in face view is along the CAMERA
         AXIS, i.e. depth, i.e. invisible in a 2D projection. That fault needs the
         target or back view. Face view sees only forward/backward motion.
      3. AT 0.67s IT MOSTLY MEASURES BOW-LOWERING. Across 9 shots, dx at 0.67s
         correlates -0.88 with vertical drop over the same window: ~78% of its
         variance is just how far the archer had lowered the bow by that instant,
         which is voluntary and happens on every shot. The large negative readings
         (-0.08 to -0.09) are exactly the shots that had begun lowering. Same trap
         as key_learnings §26: any window wide enough to contain the release also
         contains the lowering.

    Any real follow-through signal would live in a much shorter window (~0.2s,
    before lowering starts), where the spread across those 9 shots is only
    ±0.0065 — at or below the keypoint noise floor. Establishing whether that
    carries signal needs coach-labeled "collapsed" shots. Until then this value is
    recorded, not judged.

    Lookback/lookahead specified in seconds so they scale with fps (matters for
    slow-mo clips like IMG_3068 at 120fps).

    Returns: (hold_time_s, hold_censored, bow_elbow_ft_angle, bow_wrist_ft_dx,
              dynamics) — the last being the within-shot measures (H2), empty
              when the hold onset was censored.
    """
    lookback_frames = max(1, int(lookback_s * fps))
    lookahead_frames = max(1, int(lookahead_s * fps))

    # One extraction pass serves both hold time and the within-shot dynamics (H2).
    # Doing it twice would roughly double this function's cost, which already dominates
    # per-session runtime. hold_time works in NORMALISED units (its thresholds are
    # normalised units per frame); shot_dynamics works in PIXELS. Both are built here
    # from the same frames.
    _KP = {"d_wr": _D_WR, "b_wr": _B_WR, "b_sh": _B_SH,
           "l_hip": _MP_LEFT_HIP, "r_hip": _MP_RIGHT_HIP,
           "l_sh": _MP_LEFT_SHOULDER, "r_sh": _MP_RIGHT_SHOULDER}
    track = {}          # normalised draw wrist, for hold_time
    track_px = {}       # pixel positions of every keypoint, for shot_dynamics
    vis_px = {}
    for f in range(max(0, fd_frame - lookback_frames),
                   fd_frame + hold_time.RELEASE_SEARCH_FRAMES + 1):
        lm, fw, fh = _read_landmarks(cap, f, pose_adapter)
        if lm is None:
            continue
        track[f] = (float(lm[_D_WR].x), float(lm[_D_WR].y))
        track_px[f] = {n: (lm[i].x * fw, lm[i].y * fh) for n, i in _KP.items()}
        vis_px[f] = {n: lm[i].visibility for n, i in _KP.items()}

    hold = hold_time.measure_hold(track, fd_frame, fps)
    hold_time_s = hold["hold_time_s"]
    hold_censored = hold["censored"]

    # Within-shot dynamics need a bounded hold window. A censored onset means the draw
    # was not observed, so there is no window to measure over and the measures are
    # skipped rather than computed against a guessed boundary.
    dynamics = {}
    onset = hold.get("onset_frame")
    release = hold.get("release_frame") or fd_frame
    if onset is not None:
        dynamics = SD.measure_all(track_px, vis_px, onset, release, view="face")

    bow_elbow_ft_angle = None
    bow_wrist_ft_dx = None
    ft_frame = fd_frame + lookahead_frames
    lm_ft, _, _ = _read_landmarks(cap, ft_frame, pose_adapter)
    if lm_ft is not None:
        b_sh_ft = (int(lm_ft[_B_SH].x * w), int(lm_ft[_B_SH].y * h))
        b_el_ft = (int(lm_ft[_B_EL].x * w), int(lm_ft[_B_EL].y * h))
        b_wr_ft = (int(lm_ft[_B_WR].x * w), int(lm_ft[_B_WR].y * h))
        bow_elbow_ft_angle = angle_at_joint(b_sh_ft, b_el_ft, b_wr_ft)
        bow_wrist_ft_dx = float(lm_ft[_B_WR].x - fd_bow_wrist_x_norm)

    return hold_time_s, hold_censored, bow_elbow_ft_angle, bow_wrist_ft_dx, dynamics


# Pre-anchor drift lookback, in frames. Matches the 25-sample rolling nose-Y
# buffer in archery_analyzer_v2.py so the backfilled values line up with what
# the live per-clip analyzer produces for new sessions. Frame-based (not
# seconds-based) for that reason; on 120fps slow-mo clips this spans a shorter
# real-time window than on 30fps clips — same as the v2 definition.
PREANCHOR_LOOKBACK_FRAMES = 25


def compute_preanchor_drift(cap, fd_frame, nose_y_rel_norm, pose_adapter, h,
                            lookback_frames: int = PREANCHOR_LOOKBACK_FRAMES):
    """
    Nose-Y drift (pixels) from ~lookback_frames before the shot to the shot frame.
    Positive = chin dropped during the approach to anchor; negative = chin rose.
    Mirrors SideShot.nose_preanchor_drift in archery_analyzer_v2.py.

    Returns None if the reference frame is before the clip start or the pose
    dropped out there (the v2 path likewise skips when it lacks enough history).
    """
    ref_frame = fd_frame - lookback_frames
    if ref_frame < 0:
        return None
    lm_ref, _, _ = _read_landmarks(cap, ref_frame, pose_adapter)
    if lm_ref is None:
        return None
    nose_y_ref_norm = float(lm_ref[_MP_NOSE].y)
    return float((nose_y_rel_norm - nose_y_ref_norm) * h)


def analyze_frame(cap, frame_num, pose_adapter, w, h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return None

    result = pose_adapter.process(frame)
    if result.pose_landmarks is None:
        return None

    lm = result.pose_landmarks.landmark

    d_sh = pt(lm, _D_SH, w, h)
    d_el = pt(lm, _D_EL, w, h)
    d_wr = pt(lm, _D_WR, w, h)
    b_sh = pt(lm, _B_SH, w, h)
    b_el = pt(lm, _B_EL, w, h)
    b_wr = pt(lm, _B_WR, w, h)
    b_ear= pt(lm, _B_EAR, w, h)
    nose = pt(lm, _MP_NOSE, w, h)

    draw_elbow_angle    = angle_at_joint(d_sh, d_el, d_wr)
    bow_elbow_angle     = angle_at_joint(b_sh, b_el, b_wr)
    anchor_x            = lm[_D_WR].x
    anchor_y            = lm[_D_WR].y
    # Negative = draw wrist left of nose = string squish (good for right-handed)
    nose_string_gap     = lm[_D_WR].x - lm[_MP_NOSE].x
    bow_shoulder_elev   = np.linalg.norm(np.array(b_ear) - np.array(b_sh)) / h

    # Camera-invariant fields normalized by shoulder-width (pixel distance
    # between L and R shoulders at this frame). Adds *_sw variants alongside
    # the legacy fields so trend tools that don't know about them keep working.
    # Shoulder-width is roughly camera-distance-proportional for the same archer,
    # so dividing position offsets by it cancels camera framing changes.
    shoulder_width_px = float(np.linalg.norm(np.array(b_sh) - np.array(d_sh)))
    # Bow upper-arm length — used as the horizontal reference in face view.
    # The bow arm extends toward the target (perpendicular to camera), so the
    # upper-arm is in the camera plane and isn't foreshortened by torso rotation
    # the way apparent shoulder width is. More stable reference for horizontal
    # offsets when shoulders are seen in profile.
    bow_upper_arm_px = float(np.linalg.norm(np.array(b_el) - np.array(b_sh)))
    # Guard against degenerate frames where pose collapsed to a single point.
    if shoulder_width_px < 10:
        anchor_x_sw = anchor_y_sw = nose_string_gap_sw = bow_shoulder_elev_sw = None
    else:
        nose_px = pt(lm, _MP_NOSE, w, h)
        d_wr_px = pt(lm, _D_WR, w, h)
        anchor_x_sw          = (d_wr_px[0] - nose_px[0]) / shoulder_width_px
        anchor_y_sw          = (d_wr_px[1] - nose_px[1]) / shoulder_width_px
        nose_string_gap_sw   = (d_wr_px[0] - nose_px[0]) / shoulder_width_px
        bow_shoulder_elev_sw = float(np.linalg.norm(np.array(b_ear) - np.array(b_sh))) / shoulder_width_px

    # (The upper-arm-normalized branch that used to live here computed the `*_ua`
    # fields, dropped in BB7. `bow_upper_arm_px` itself is still recorded as a scale
    # reference.)

    vis_d_el = lm[_D_EL].visibility
    vis_b_el = lm[_B_EL].visibility

    return dict(
        frame             = frame_num,
        draw_elbow_angle  = round(draw_elbow_angle, 3),
        bow_elbow_angle   = round(bow_elbow_angle, 3),
        anchor_x          = round(anchor_x, 4),
        anchor_y          = round(anchor_y, 4),
        nose_string_gap   = round(nose_string_gap, 4),
        bow_shoulder_elev = round(bow_shoulder_elev, 3),
        anchor_x_px       = lm[_D_WR].x * w,
        anchor_y_px       = lm[_D_WR].y * h,
        shoulder_width_px    = round(shoulder_width_px, 1),
        bow_upper_arm_px     = round(bow_upper_arm_px, 1),
        anchor_x_sw          = round(anchor_x_sw, 4) if anchor_x_sw is not None else None,
        anchor_y_sw          = round(anchor_y_sw, 4) if anchor_y_sw is not None else None,
        nose_string_gap_sw   = round(nose_string_gap_sw, 4) if nose_string_gap_sw is not None else None,
        bow_shoulder_elev_sw = round(bow_shoulder_elev_sw, 4) if bow_shoulder_elev_sw is not None else None,
        bow_wrist_x_norm     = float(lm[_B_WR].x),
        nose_y_norm          = float(lm[_MP_NOSE].y),
        vis_elbows        = round((vis_d_el + vis_b_el) / 2, 2),
    )


def _write_csv(results, video_path, out_dir="output"):
    """Write one CSV row per verified shot to output/{view}_{stem}_shots.csv."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_dir, f"face_{stem}_shots.csv")
    if not results:
        return csv_path
    cols = ["shot"] + list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({"shot": i, **r})
    return csv_path


def _print_feedback(results, dea_avg, bea_avg, spread,
                     nsg_avg, bse_sw_avg, ht_avg, ft_dx_avg, pad_avg,
                     ht_n_censored=0, ht_window_s=4.0, spread_head=None,
                     creep_avg=None, sway_avg=None):
    """Print a quick tabular feedback summary with strengths and areas of improvement.

    `ht_avg` is the mean of MEASURED hold times only (None if every shot was
    censored). `ht_n_censored` is how many shots held past the scan window; their
    true hold is unknown but is >= ht_window_s. Because the hold-time target is
    one-sided (>= 1.0s), a censored value still resolves the check whenever the
    window itself clears the threshold — what it cannot do is contribute to a mean.
    """
    n = len(results)
    print(f"\n{'═'*64}")
    print(f"  FEEDBACK — Face View  ({n} shot{'s' if n != 1 else ''})")
    print(f"{'═'*64}")
    print(f"  {'Metric':<30} {'Value':>10}  {'Target':<22} Status")
    print(f"  {'─'*62}")

    checks = []  # (label, ok, val, display)

    def chk(label, val, display, target_str, ok):
        """ok=None => observation-only: report the value, no verdict (BB9)."""
        symbol = "—" if ok is None else ("✓" if ok else "✗")
        print(f"  {label:<30} {display:>10}  {target_str:<22} {symbol}")
        if ok is not None:
            checks.append((label, ok, val, display))

    def mchk(metric, val, display, label=None):
        """Check against the canonical band for a face metric (tools/metric_thresholds).

        Bands and verdicts come from one place now; previously this table carried its
        own copies, which had drifted from the rule engine (draw elbow 15–35 here vs
        20–40 there, and a 0.55–0.65 bow-shoulder pass/fail that the rule engine
        correctly treated as observation-only). See BB9.
        """
        sp = MT.spec("face", metric)
        if sp is None or val is None:
            return
        chk(label or sp["label"], val, display,
            MT.describe("face", metric), MT.verdict("face", metric, val))

    mchk("draw_elbow_angle", dea_avg, f"{dea_avg:.1f}°")
    mchk("bow_elbow_angle",  bea_avg, f"{bea_avg:.1f}°")
    # Head-relative spread is the canonical anchor-consistency measure (BB2); the
    # image-relative figure is shown for continuity but not judged.
    if spread_head is not None:
        mchk("anchor_spread_head_px", spread_head, f"{spread_head:.1f} px",
             label="Anchor spread (head)")
    print(f"  {'Anchor spread (image)':<30} {f'{spread:.1f} px':>10}  "
          f"{'not judged — see BB2':<22} —")
    mchk("nose_string_gap", nsg_avg, f"{nsg_avg:+.3f}")
    mchk("bow_shoulder_elevation_sw", bse_sw_avg,
         f"{bse_sw_avg:.3f}" if bse_sw_avg is not None else "—")
    if ht_avg is not None:
        mchk("hold_time_s", ht_avg, f"{ht_avg:.2f} s")
        if ht_n_censored:
            print(f"  {'':<30} {'':>10}  {f'({ht_n_censored} shot(s) censored, excluded)':<22}")
    elif ht_n_censored:
        # Every shot held past the window, so the true hold is >= the window. That
        # still clears the lower bound of the band, but it is a bound, not a measurement.
        lo = (MT.good_range("face", "hold_time_s") or (1.0, 3.0))[0]
        chk("Hold time", ht_window_s, f"≥ {ht_window_s:.0f} s",
            MT.describe("face", "hold_time_s"), True if ht_window_s >= lo else None)
    mchk("bow_wrist_ft_dx", ft_dx_avg,
         f"{ft_dx_avg:+.3f}" if ft_dx_avg is not None else "—")
    mchk("nose_preanchor_drift", pad_avg,
         f"{pad_avg:.1f} px" if pad_avg is not None else "—",
         label="Chin pre-anchor drift")
    # Within-shot dynamics (H2): observation-only, so mchk prints a "—" verdict.
    # Creep is signed and the sign is the point, so it is shown with an explicit reading.
    if creep_avg is not None:
        # Sign is the whole point, so it goes in the target column where there is room;
        # putting it in the value column broke the table alignment.
        sense = "− = expanding ✓" if creep_avg < 0 else "+ = creeping fwd"
        chk("Creep (draw-hand)", creep_avg, f"{creep_avg:+.3f}", sense, None)
    mchk("hip_sway_total", sway_avg,
         f"{sway_avg:.3f}" if sway_avg is not None else "—")

    print(f"  {'─'*62}")

    goods  = [(l, v, d) for l, ok, v, d in checks if ok]
    issues = [(l, v, d) for l, ok, v, d in checks if not ok]

    _strengths = {
        "Draw elbow angle":          lambda v, d: f"Draw elbow angle in range ({d}) — good back-arm engagement",
        "Bow elbow angle":           lambda v, d: f"Bow arm well extended ({d})",
        "Anchor spread":             lambda v, d: f"Anchor consistency excellent — {d} spread",
        "Nose–string gap":           lambda v, d: f"Nose-string contact in range ({d})",
        "Bow shoulder elev (sw)":    lambda v, d: f"Bow shoulder consistent ({d})",
        "Hold time":                 lambda v, d: f"Hold time solid at {d}",
        "Chin pre-anchor drift":     lambda v, d: f"Head stable during draw approach ({d})",
    }
    _improvements = {
        # Band is 1.0-3.0s (coach-stated). Direction matters: the old message
        # only ever said "hold longer", which was wrong for both reasons — it came
        # from a broken measurement, and it cannot address a too-long hold.
        "Hold time":                 lambda v, d: (
            f"Hold time {d} — under 1 s, keep expanding through the clicker rather than "
            f"releasing at it" if v < 1.0 else
            f"Hold time {d} — over 3 s, the shot is stalling at anchor; let it go while "
            f"the back is still moving"),
        "Chin pre-anchor drift":     lambda v, d: f"Chin drifting {'DOWN' if v > 0 else 'UP'} {abs(v):.1f} px — set chin at setup and carry it to anchor",
        "Draw elbow angle":          lambda v, d: f"Draw elbow angle {d} — {'too open (>35°), engage more back/scapula' if v > 35 else 'unusually bent (<15°), check drawing technique'}",
        "Bow shoulder elev (sw)":    lambda v, d: f"Bow shoulder elevated (sw={d}) — focus on sinking shoulder down",
        "Nose–string gap":           lambda v, d: f"Nose-string gap {d} out of range — check anchor depth/contact",
        "Bow elbow angle":           lambda v, d: f"Bow elbow angle {d} — check bow arm extension",
        "Anchor spread":             lambda v, d: f"Anchor spread {d} — work on consistent anchor position",
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
    ap.add_argument("--frames",       type=int, nargs="+", required=True)
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
    print(f"Video: {w}x{h} @ {fps:.1f}fps")
    print(f"Verified frames: {args.frames}\n")

    results = []
    for i, fn in enumerate(args.frames, 1):
        r = analyze_frame(cap, fn, pose, w, h)
        if r is None:
            print(f"  Shot {i} frame {fn}: POSE DROPOUT")
        else:
            # Compute hold time + bow elbow follow-through (extra ~30-40 YOLO calls per shot)
            hold_s, hold_censored, ft_angle, ft_dx, dyn = compute_hold_and_followthrough(
                cap, fn, r["anchor_y"], r["bow_wrist_x_norm"], pose, w, h, fps,
            )
            r["hold_time_s"] = round(hold_s, 3)
            # True = scan hit the lookback bound / video start / pose dropout, so the
            # real hold is >= this value. Censored values must not be averaged.
            r["hold_time_censored"] = hold_censored
            r["bow_elbow_followthrough"] = round(ft_angle, 3) if ft_angle is not None else None
            # bow_wrist_ft_dx: OBSERVATION-ONLY (BB5). negative = bow wrist moved AWAY
            # from the target; positive = toward it. Dominated by bow-lowering timing
            # at this lookahead — do not read as follow-through quality.
            r["bow_wrist_ft_dx"] = round(ft_dx, 4) if ft_dx is not None else None
            # Pre-anchor head drift: chin drop (px) from ~25 frames before the shot to the shot frame
            pad = compute_preanchor_drift(cap, fn, r["nose_y_norm"], pose, h)
            r["nose_preanchor_drift"] = round(pad, 2) if pad is not None else None
            # Within-shot dynamics (H2). Both observation-only: validated as
            # measurements on 9 shots but with no coach-confirmed band.
            #   creep < 0  = draw hand moving AWAY from the bow = expansion = correct
            #   hip_sway   = largest mid-hip separation during the hold, in shoulder widths
            r["creep"] = dyn.get("creep")
            r["hip_sway_total"] = dyn.get("hip_sway_total")
            r["hip_sway_x"] = dyn.get("hip_sway_x")

            # Per-shot biomechanical plausibility gate (BB2, 2026-07-29).
            #
            # The project already validated these constraints — they are what
            # separated YOLO26 from YOLO11 (see output/yolo_significance_test.md) —
            # but they were only ever used to compare MODELS, never applied as a QC
            # filter on production metrics. IMG_4495 shot 1 (2026-06-07) is the cost:
            # draw elbow 12.31° (outside [15,50]), visibility 0.79, shoulder width
            # +14% vs its neighbours, nose gap -0.095 vs -0.041. Every signal said
            # "bad measurement", and it still landed in the session average, where it
            # produced an apparent 44.9px anchor inconsistency.
            #
            # Flagged, not dropped: the caller decides. A flagged shot is excluded
            # from aggregates but kept in the CSV so the exclusion is auditable.
            implausible = []
            if not (15.0 <= r["draw_elbow_angle"] <= 50.0):
                implausible.append(f"draw elbow {r['draw_elbow_angle']:.1f}° outside [15,50]")
            if not (155.0 <= r["bow_elbow_angle"] <= 180.0):
                implausible.append(f"bow elbow {r['bow_elbow_angle']:.1f}° outside [155,180]")
            if r["vis_elbows"] <= 0.85:
                implausible.append(f"elbow visibility {r['vis_elbows']:.2f} ≤ 0.85")
            r["qc_implausible"] = "; ".join(implausible) if implausible else ""

            flag = "✓" if not implausible else "⚠ QC"
            ft_str = f"{r['bow_wrist_ft_dx']:+.3f}" if r['bow_wrist_ft_dx'] is not None else "n/a"
            hold_str = f"{r['hold_time_s']:.2f}s" + ("+" if r["hold_time_censored"] else " ")
            print(f"  Shot {i} frame {fn:5d}  "
                  f"DrawEl={r['draw_elbow_angle']:6.2f}°  "
                  f"BowEl={r['bow_elbow_angle']:6.2f}°  "
                  f"AncX={r['anchor_x']:.3f}  AncY={r['anchor_y']:.3f}  "
                  f"NoseGap={r['nose_string_gap']:+.3f}  "
                  f"BowSh={r['bow_shoulder_elev']:.3f}  "
                  f"Hold={hold_str} BowFT_dx={ft_str}  {flag}")
            if implausible:
                print(f"           ⚠ QC: {r['qc_implausible']} "
                      f"— excluded from aggregates, kept in CSV")
            results.append(r)

    cap.release()

    if not results:
        print("\nNo frames successfully analyzed.")
        return

    # QC split (BB2). `results` keeps every shot so the CSV stays complete and the
    # exclusion is auditable; `results_ok` is what the aggregates and feedback use.
    all_results = results
    results_ok = [r for r in results if not r.get("qc_implausible")]
    n_excluded = len(all_results) - len(results_ok)
    if n_excluded:
        print(f"\n⚠ {n_excluded} of {len(all_results)} shot(s) failed the plausibility "
              f"gate and are excluded from session aggregates.")
        if not results_ok:
            print("  No plausible shots remain — no aggregates computed.")
            _write_csv(all_results, args.video)
            return
        results = results_ok

    print(f"\n{'─'*60}")
    print(f"SUMMARY  ({len(results)}/{len(args.frames)} frames detected)")
    print(f"{'─'*60}")

    def stat(name, key, unit="", good_range=None):
        vals = [r[key] for r in results]
        avg, std = np.mean(vals), np.std(vals)
        flag = ""
        if good_range:
            flag = " ✓" if good_range[0] <= avg <= good_range[1] else " ✗"
        print(f"  {name:<28s}  avg={avg:+8.3f}{unit}  std={std:.3f}{unit}{flag}")
        return avg, std

    dea_avg, dea_std = stat("Draw Elbow Angle",       "draw_elbow_angle", "°", (15, 35))
    bea_avg, bea_std = stat("Bow Elbow Angle",        "bow_elbow_angle",  "°", (160, 175))
    ax_avg,  ax_std  = stat("Anchor X (norm)",        "anchor_x")
    ay_avg,  ay_std  = stat("Anchor Y (norm)",        "anchor_y")
    nsg_avg, nsg_std = stat("Nose–String Gap",        "nose_string_gap",  "",  (-0.12, -0.04))
    # Legacy `/h` BSE: NO BAND (BB9). The (0.18, 0.50) band applied here was
    # written for a different scale — legacy values span 0.052–0.184 across the
    # corpus, so it marked 142 of 143 sessions as failing. The canonical metric is
    # the `_sw` variant, and it is observation-only until a coach threshold exists.
    bse_avg, bse_std = stat("Bow Shoulder Elevation", "bow_shoulder_elev","")

    # Hold time and follow-through (supports "pull through the clicker" style coaching validation)
    # Censoring (2026-07-28): a shot whose backward scan hit the lookback bound has an
    # unknown true hold >= the window. Averaging censored with measured values produces a
    # number that is neither. Aggregate only the measured ones and report the censored
    # count separately so downstream consumers can tell the difference.
    ht_measured = [r["hold_time_s"] for r in results
                   if r.get("hold_time_s") is not None and not r.get("hold_time_censored")]
    ht_censored = [r["hold_time_s"] for r in results
                   if r.get("hold_time_s") is not None and r.get("hold_time_censored")]
    ht_vals = ht_measured
    ft_angle_vals = [r["bow_elbow_followthrough"] for r in results if r.get("bow_elbow_followthrough") is not None]
    ft_dx_vals = [r["bow_wrist_ft_dx"] for r in results if r.get("bow_wrist_ft_dx") is not None]
    ht_avg = float(np.mean(ht_vals)) if ht_vals else None
    ht_std = float(np.std(ht_vals)) if len(ht_vals) > 1 else 0.0
    ft_angle_avg = float(np.mean(ft_angle_vals)) if ft_angle_vals else None
    ft_angle_std = float(np.std(ft_angle_vals)) if len(ft_angle_vals) > 1 else 0.0
    ft_dx_avg = float(np.mean(ft_dx_vals)) if ft_dx_vals else None
    ft_dx_std = float(np.std(ft_dx_vals)) if len(ft_dx_vals) > 1 else 0.0
    # Within-shot dynamics aggregates (H2). Observation-only, so reported without a
    # verdict; the within-session SD is as informative as the mean for these.
    creep_vals = [r["creep"] for r in results if r.get("creep") is not None]
    sway_vals  = [r["hip_sway_total"] for r in results if r.get("hip_sway_total") is not None]
    creep_avg = float(np.mean(creep_vals)) if creep_vals else None
    creep_std = float(np.std(creep_vals)) if len(creep_vals) > 1 else 0.0
    sway_avg  = float(np.mean(sway_vals)) if sway_vals else None
    sway_std  = float(np.std(sway_vals)) if len(sway_vals) > 1 else 0.0
    pad_vals = [r["nose_preanchor_drift"] for r in results if r.get("nose_preanchor_drift") is not None]
    pad_avg = float(np.mean(pad_vals)) if pad_vals else None
    pad_std = float(np.std(pad_vals)) if len(pad_vals) > 1 else 0.0
    if ht_avg is not None:
        print(f"  {'Hold time (s)':<28s}  avg={ht_avg:+8.3f}s  std={ht_std:.3f}s"
              f"  (n={len(ht_measured)} measured)")
    if ht_censored:
        print(f"  {'Hold time — CENSORED':<28s}  {len(ht_censored)} of {len(ht_censored)+len(ht_measured)} "
              f"shots held past the {4.0:.0f}s scan window; true hold unknown (>= window). "
              f"Excluded from the average.")
    if ht_avg is None and ht_censored:
        print(f"  {'':28s}  → no measured hold times this session; hold time not reported.")
    if ft_angle_avg is not None:
        print(f"  {'Bow elbow followthrough':<28s}  avg={ft_angle_avg:+8.2f}°  std={ft_angle_std:.2f}°")
    if ft_dx_avg is not None:
        # BB5: -x is AWAY from the target in face view, +x toward it. No verdict —
        # at this lookahead the value is ~78% bow-lowering timing.
        sign = ("moved away from target" if ft_dx_avg < -0.005 else
                "moved toward target" if ft_dx_avg > 0.005 else "no net change")
        print(f"  {'Bow wrist FT dx (norm)':<28s}  avg={ft_dx_avg:+8.4f}   std={ft_dx_std:.4f}   {sign}")
    if pad_avg is not None:
        drift = "head INTO string (Judy: high arrows)" if pad_avg > 2.0 else ("chin UP" if pad_avg < -2.0 else "stable")
        print(f"  {'Nose pre-anchor drift (px)':<28s}  avg={pad_avg:+8.2f}px  std={pad_std:.2f}px  {drift}")
    if creep_avg is not None:
        sense = "expanding ✓" if creep_avg < 0 else "creeping forward"
        print(f"  {'Creep (sh-widths)':<28s}  avg={creep_avg:+8.4f}   std={creep_std:.4f}   {sense}")
    if sway_avg is not None:
        print(f"  {'Hip sway in hold (sh-w)':<28s}  avg={sway_avg:+8.4f}   std={sway_std:.4f}")

    # Anchor spread in pixels.
    #
    # Two versions, because the image-relative one is confounded (BB2, 2026-07-29).
    # `anchor_spread_px` measures scatter of the draw wrist in IMAGE coordinates, so
    # it cannot distinguish "the anchor moved on the face" from "the archer moved in
    # frame". IMG_4488 (2026-06-07) read 76.7px against 2-7px for its neighbours;
    # the nose translated by the same amount as the wrist between shots 2 and 3, so
    # the archer (or camera) shifted along the line while the anchor itself held.
    # Head-relative spread on that clip is 9.6px — normal.
    #
    # `anchor_spread_head_px` expresses the wrist relative to the nose in the same
    # frame, which is what "consistent anchor" actually means: the hand returns to
    # the same place on the FACE. Prefer it; the image-relative version is retained
    # for continuity with earlier records.
    ax_px = np.array([r["anchor_x_px"] for r in results])
    ay_px = np.array([r["anchor_y_px"] for r in results])
    spread = float(np.sqrt(np.std(ax_px)**2 + np.std(ay_px)**2))
    print(f"  {'Anchor Spread (px, image)':<28s}  {spread:.1f}px  "
          f"(X std={np.std(ax_px):.1f}px  Y std={np.std(ay_px):.1f}px)")

    head_results = [r for r in results
                    if r.get("nose_string_gap") is not None and r.get("nose_y_norm") is not None]
    spread_head = None
    if len(head_results) > 1:
        rel_x = np.array([r["nose_string_gap"] * w for r in head_results])
        rel_y = np.array([(r["anchor_y"] - r["nose_y_norm"]) * h for r in head_results])
        spread_head = float(np.sqrt(np.std(rel_x)**2 + np.std(rel_y)**2))
        flag = ""
        if spread > 20 and spread_head < 20:
            flag = "  ← image-relative figure is a body-translation artifact"
        print(f"  {'Anchor Spread (px, head-rel)':<28s}  {spread_head:.1f}px{flag}")

    # Shoulder-width-normalized stats (camera-invariant). None values appear when
    # the pose collapsed; skip those for aggregates so a single bad frame doesn't
    # poison the average.
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

        ax_sw_avg, ax_sw_std = stat_sw("Anchor X (sh-widths)",     "anchor_x_sw")
        ay_sw_avg, ay_sw_std = stat_sw("Anchor Y (sh-widths)",     "anchor_y_sw")
        nsg_sw_avg, nsg_sw_std = stat_sw("Nose-String Gap (sw)",   "nose_string_gap_sw")
        bse_sw_avg, bse_sw_std = stat_sw("Bow Shoulder Elev (sw)", "bow_shoulder_elev_sw")

        # Anchor spread in shoulder-widths (camera-invariant)
        ax_sw_vals = np.array([r["anchor_x_sw"] for r in sw_results if r["anchor_x_sw"] is not None])
        ay_sw_vals = np.array([r["anchor_y_sw"] for r in sw_results if r["anchor_y_sw"] is not None])
        spread_sw = float(np.sqrt(np.std(ax_sw_vals)**2 + np.std(ay_sw_vals)**2)) if len(ax_sw_vals) > 1 else 0.0
        print(f"  {'Anchor Spread (sw)':<28s}  {spread_sw:.4f}sw")

        # Horizontal-only metrics normalized by bow-upper-arm length.
        # In face view, shoulder-width is foreshortened by torso rotation and
        # makes horizontal offsets noisy; the bow upper-arm is in the camera plane
        # and gives a more stable horizontal reference.
        ua_results = [r for r in sw_results if r.get("bow_upper_arm_px") is not None]
        if ua_results:
            ua_avg = float(np.mean([r["bow_upper_arm_px"] for r in ua_results]))
            print(f"  {'Bow upper-arm (px)':<28s}  avg={ua_avg:.1f}px")
        else:
            ua_avg = None
    else:
        sw_avg = sw_std = None
        ax_sw_avg = ax_sw_std = ay_sw_avg = ay_sw_std = None
        nsg_sw_avg = nsg_sw_std = bse_sw_avg = bse_sw_std = None
        spread_sw = None
        ua_avg = None

    if args.session_json and os.path.exists(args.session_json):
        with open(args.session_json) as f:
            session = json.load(f)

        old_metrics = session.get("metrics", {})
        old_notes   = session.get("detection_notes", "")

        new_metrics = dict(old_metrics)
        # Remove old noisy fields
        # Deprecated fields are actively removed, not merely no longer written, so a
        # re-run cleans records that already carry them (BB7/BB8).
        for old_key in [
            "anchor_x_ua_avg", "anchor_x_ua_std",
            "nose_string_gap_ua_avg", "nose_string_gap_ua_std",
            "draw_elbow_angle_avg", "draw_elbow_angle_std",
            "bow_elbow_angle_avg",  "bow_elbow_angle_std",
            "anchor_consistency_px", "anchor_consistency_x_px", "anchor_consistency_y_px",
            "draw_elbow_height_avg", "draw_elbow_height_std",
            "elbow_position_quality",
        ]:
            new_metrics.pop(old_key, None)

        new_metrics.update({
            "view":                      "face",
            "arrows_shot":               len(args.frames),
            "arrows_with_metrics":       len(results),
            "verified_frames":           args.frames,
            "draw_elbow_angle_avg":      round(dea_avg, 3),
            "draw_elbow_angle_std":      round(dea_std, 3),
            "bow_elbow_angle_avg":       round(bea_avg, 3),
            "bow_elbow_angle_std":       round(bea_std, 3),
            "anchor_x_avg":              round(ax_avg,  4),
            "anchor_x_std":              round(ax_std,  4),
            "anchor_y_avg":              round(ay_avg,  4),
            "anchor_y_std":              round(ay_std,  4),
            "anchor_spread_px":          round(spread,  1),
            "anchor_spread_head_px":     round(spread_head, 1) if spread_head is not None else None,
            "shots_qc_excluded":         n_excluded,
            "nose_string_gap_avg":       round(nsg_avg, 4),
            "nose_string_gap_std":       round(nsg_std, 4),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        })

        if sw_avg is not None:
            new_metrics.update({
                "shoulder_width_px_avg":          round(sw_avg, 1),
                "shoulder_width_px_std":          round(sw_std, 1),
                "anchor_x_sw_avg":                round(ax_sw_avg, 4),
                "anchor_x_sw_std":                round(ax_sw_std, 4),
                "anchor_y_sw_avg":                round(ay_sw_avg, 4),
                "anchor_y_sw_std":                round(ay_sw_std, 4),
                "nose_string_gap_sw_avg":         round(nsg_sw_avg, 4),
                "nose_string_gap_sw_std":         round(nsg_sw_std, 4),
                "bow_shoulder_elevation_sw_avg":  round(bse_sw_avg, 4),
                "bow_shoulder_elevation_sw_std":  round(bse_sw_std, 4),
                "anchor_spread_sw":               round(spread_sw, 4),
            })
        # `bow_upper_arm_px` is kept as a scale reference. The `*_ua`
        # (upper-arm-normalized) metrics that used it are dropped (BB7): testing showed
        # upper-arm normalization of face-view horizontal metrics is worse than
        # image-relative (SD 0.163 vs 0.019), and nothing downstream ever read them —
        # the two aliases in trend_analysis.py were declared but never used.
        if ua_avg is not None:
            new_metrics["bow_upper_arm_px_avg"] = round(ua_avg, 1)

        if ht_avg is not None:
            new_metrics["hold_time_s_avg"] = round(ht_avg, 3)
            new_metrics["hold_time_s_std"] = round(ht_std, 3)
            new_metrics["hold_time_s_n_measured"] = len(ht_measured)
        else:
            # Every shot was censored: there is no average. Drop any value carried
            # over from a previous run rather than leaving a stale number that came
            # from a superseded definition (BB1/BB11) sitting next to a censored count.
            new_metrics.pop("hold_time_s_avg", None)
            new_metrics.pop("hold_time_s_std", None)
            new_metrics.pop("hold_time_s_n_measured", None)
        # Always record the censored count, even when it is 0 — its absence in older
        # records is what makes them indistinguishable from clean ones (see BB1).
        new_metrics["hold_time_s_n_censored"] = len(ht_censored)
        # Stamp which definition produced this value. The corpus contains values from
        # three superseded formulations (BB1 distance-scan ceiling, BB11 X-only
        # settle-truncation, and this one), and they are not comparable. Consumers
        # trust only HOLD_TIME_DEFINITION.
        new_metrics["hold_time_s_definition"] = hold_time.HOLD_TIME_DEFINITION
        if ft_angle_avg is not None:
            new_metrics["bow_elbow_followthrough_avg"] = round(ft_angle_avg, 3)
            new_metrics["bow_elbow_followthrough_std"] = round(ft_angle_std, 3)
        if ft_dx_avg is not None:
            new_metrics["bow_wrist_ft_dx_avg"] = round(ft_dx_avg, 4)
            new_metrics["bow_wrist_ft_dx_std"] = round(ft_dx_std, 4)
        if creep_avg is not None:
            new_metrics["creep_avg"] = round(creep_avg, 4)
            new_metrics["creep_std"] = round(creep_std, 4)
        if sway_avg is not None:
            new_metrics["hip_sway_total_avg"] = round(sway_avg, 4)
            new_metrics["hip_sway_total_std"] = round(sway_std, 4)
        if pad_avg is not None:
            new_metrics["nose_preanchor_drift_avg"] = round(pad_avg, 2)
            new_metrics["nose_preanchor_drift_std"] = round(pad_std, 2)

        session["metrics"]         = new_metrics
        session["view"]            = "face"
        session["detection_notes"] = _compose_notes(
            old_notes,
            f"Re-processed with YOLO26 (analyze_face_yolo.py). "
            f"{len(results)}/{len(args.frames)} verified frames detected.",
        )

        with open(args.session_json, "w") as f:
            json.dump(session, f, indent=2)
        print(f"\nSession JSON updated: {args.session_json}")
    else:
        print(f"\n{'─'*60}")
        print("Metrics to paste into session JSON:")
        print(json.dumps({
            "draw_elbow_angle_avg":      round(dea_avg, 3),
            "draw_elbow_angle_std":      round(dea_std, 3),
            "bow_elbow_angle_avg":       round(bea_avg, 3),
            "bow_elbow_angle_std":       round(bea_std, 3),
            "anchor_x_avg":              round(ax_avg,  4),
            "anchor_x_std":              round(ax_std,  4),
            "anchor_y_avg":              round(ay_avg,  4),
            "anchor_y_std":              round(ay_std,  4),
            "anchor_spread_px":          round(spread,  1),
            "anchor_spread_head_px":     round(spread_head, 1) if spread_head is not None else None,
            "shots_qc_excluded":         n_excluded,
            "nose_string_gap_avg":       round(nsg_avg, 4),
            "nose_string_gap_std":       round(nsg_std, 4),
            "bow_shoulder_elevation_avg": round(bse_avg, 3),
            "bow_shoulder_elevation_std": round(bse_std, 3),
        }, indent=4))

    # ── CSV + feedback (always, regardless of JSON flag) ─────────────────────
    # Write every shot, including any excluded by the QC gate, so the exclusion
    # is visible downstream rather than silently absent (BB2).
    csv_out = _write_csv(all_results, args.video)
    print(f"CSV written: {csv_out}")
    _print_feedback(
        results, dea_avg, bea_avg, spread, nsg_avg,
        bse_sw_avg,
        ht_avg, ft_dx_avg, pad_avg,
        ht_n_censored=len(ht_censored),
        spread_head=spread_head,
        creep_avg=creep_avg, sway_avg=sway_avg,
    )


if __name__ == "__main__":
    main()

"""
Within-shot dynamics — four measures over the hold and release window.

WHY THESE FOUR TOGETHER
    Bow arm collapse, postural sway, creeping and follow-through collapse all need
    the same thing: a per-frame keypoint trajectory spanning hold onset to shortly
    after release. Building them separately would mean paying for that extraction
    four times, so they share one pass here and one module.

THE DESIGN CONSTRAINT THAT SHAPED THEM
    Every measure below is a DIFFERENCE WITHIN A SINGLE SHOT, normalised by an
    anatomical scale measured in the same frames. None of them depends on an
    absolute reference value.

    That is deliberate, and it is the accumulated lesson of this project's
    measurement failures. Absolute-value metrics have repeatedly turned out to be
    either camera-dependent or offset by landmark definition:

      - hold time was a scan-window ceiling, not a duration
      - draw elbow angle correlates +0.56 with apparent camera scale, which itself
        varies 48% across the primary subset
      - T-draw angle's supposed 0° ideal was simply wrong: correct form is negative
      - the draw force line reads 139° where technique says 180°, because the
        shoulder keypoint is the acromion and the bow wrist is not the pressure point

    Within-shot differences are immune to all of it. The camera cannot move during a
    shot, so scale, orientation and viewing angle are constant across the window;
    they cancel in the difference and the normalisation removes what remains. This
    is the most defensible class of measurement the study's design permits.

VALIDATION OUTCOME (2026-08-01, 9 shots across 2 face-view sessions)
    Three of the four measures behave; one does not, and is disabled.

      creep                  WORKS. All 9 shots negative (-0.04 to -0.28 shoulder
                             widths), i.e. the draw hand moves AWAY from the bow
                             through the hold. That is expansion — correct form, and
                             the opposite of creeping. A unanimous sign across 9 shots
                             is what a working measure looks like.
      postural_sway          WORKS. Tight across shots (0.08-0.14 total range), and
                             mostly vertical: x range is only 0.02-0.06, so the
                             fore/aft component along the shooting line is small.
      bow_arm_collapse       PLAUSIBLE, unvalidated. -0.15 to +0.11, mixed sign, no
                             ground truth. Read with the torso-rotation caveat below.
      followthrough_collapse FAILS — DO NOT USE. See below.

WHY FOLLOW-THROUGH COLLAPSE IS DISABLED
    It reproduces a failure this project has already documented once. Across the same
    9 face-view shots it spans 0.11 to 1.79 shoulder widths — values above 1.0 are the
    archer lowering the bow, a voluntary action on every shot, so the measure is
    dominated by lowering TIMING rather than follow-through quality. That is the same
    confound that made bow-wrist follow-through drift observation-only (~78% of its
    variance was bow-lowering).

    It also fails its one piece of ground truth. On IMG_3126 shot 5 the archer
    independently reported a collapsed follow-through; this measure ranks that shot
    LOWEST of its six session-mates (+0.06 against +0.30 to +0.41). Tracing the draw
    elbow shows why: the drop begins around 20 frames after the anchor and reaches
    +468 px by frame 40, so a 20-frame window ends exactly where the collapse starts
    and a 40-frame window is deep into bow-lowering. The measure's verdict therefore
    flips with the window length — the same windowing dependence that invalidated the
    within-session fatigue finding.

    One labelled shot cannot calibrate a measure whose sign depends on an arbitrary
    window. Left in the module, returning its value, but excluded from `measure_all`
    and not registered as a metric.

VIEW AVAILABILITY IS NOT UNIFORM
    Hip visibility varies by clip, not just by view: 0.65-0.77 in one back-view
    session but 0.06-0.23 in another, which silently removes sway. Draw-hand measures
    need the face view — in back view the draw wrist sits behind the head and
    produced implausible creep values (over one shoulder width of drift). Callers
    should pass only the keypoints their view resolves and expect None elsewhere.

WHAT IS STILL NOT ESTABLISHED
    Sound measurement is not a validated threshold. None of these has a
    coach-confirmed band; all are registered observation-only.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Frames before the detected release to stop measuring creep. The release snap is a
# large forward displacement in the same direction creep would be, so including it
# would swamp a millimetre-scale drift with a centimetre-scale event.
CREEP_MARGIN_FRAMES = 3

# Frames after release at which follow-through is assessed. Matches the existing
# follow-through lookahead used elsewhere (~0.67 s at 30 fps).
FOLLOWTHROUGH_FRAMES = 20

# Minimum keypoint visibility for a frame to contribute.
MIN_VIS = 0.5


def extract_track(cap, pose_adapter, first_frame, last_frame, keypoints, step=1):
    """Per-frame keypoint positions in pixels over an inclusive frame range.

    `keypoints` maps a name to a landmark index. Returns {frame: {name: (x, y)}}
    plus a parallel {frame: {name: visibility}}. Frames where the pose is absent are
    skipped rather than interpolated, so a dropout leaves a gap instead of a
    fabricated position.

    One sequential pass; the caller is expected to reuse it for every measure.
    """
    import cv2

    track, vis = {}, {}
    for f in range(max(0, first_frame), last_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        res = pose_adapter.process(frame)
        if res.pose_landmarks is None:
            continue
        h, w = frame.shape[:2]
        lm = res.pose_landmarks.landmark
        track[f] = {name: (lm[idx].x * w, lm[idx].y * h) for name, idx in keypoints.items()}
        vis[f] = {name: lm[idx].visibility for name, idx in keypoints.items()}
    return track, vis


def _usable(track, vis, frame, names):
    """True when every named keypoint is present and visible enough at this frame."""
    if frame not in track:
        return False
    return all(vis[frame].get(n, 0.0) >= MIN_VIS for n in names)


def _nearest_usable(track, vis, target, names, search=8):
    """Frame closest to `target` where all `names` are usable, or None.

    A single dropped frame at a window boundary should not void a whole measure, so
    a short search either side is allowed. The offset used is reported by callers.
    """
    for d in range(search + 1):
        for f in (target - d, target + d):
            if _usable(track, vis, f, names):
                return f
    return None


def _scale(track, vis, frames, l_sh="l_sh", r_sh="r_sh"):
    """Median shoulder width in pixels over the window — the normalising scale.

    Median rather than mean so one bad frame cannot inflate it, and measured over
    the same window as the signal so camera distance cancels.
    """
    widths = [
        float(np.hypot(*(np.array(track[f][l_sh]) - np.array(track[f][r_sh]))))
        for f in frames
        if _usable(track, vis, f, (l_sh, r_sh))
    ]
    if not widths:
        return None
    w = float(np.median(widths))
    return w if w >= 10 else None


def bow_arm_collapse(track, vis, onset, release, scale):
    """Contraction of the bow wrist-to-shoulder distance across the hold.

    Positive = the distance shrank, i.e. the bow arm gave way under holding load.
    Normalised by shoulder width, so a value of 0.05 means the bow arm shortened by
    5% of a shoulder width between hold onset and release.

    Caveat worth stating: apparent bow-arm length also shortens if the torso rotates
    toward the camera. Within a single hold that rotation is small, but a large
    positive value should be checked against the video before being read as collapse.
    """
    names = ("b_wr", "b_sh")
    f0 = _nearest_usable(track, vis, onset, names)
    f1 = _nearest_usable(track, vis, release, names)
    if f0 is None or f1 is None or scale is None:
        return None
    d0 = np.hypot(*(np.array(track[f0]["b_wr"]) - np.array(track[f0]["b_sh"])))
    d1 = np.hypot(*(np.array(track[f1]["b_wr"]) - np.array(track[f1]["b_sh"])))
    return float((d0 - d1) / scale)


def postural_sway(track, vis, onset, release, scale):
    """Mid-hip movement during the hold, as a fraction of shoulder width.

    Returns (total_range, x_range, y_range). `total_range` is the largest distance
    between any two mid-hip positions in the window — a bound on how much the base
    moved, rather than a net drift, since sway can return to where it started.

    Axis interpretation is view-dependent and the caller must supply it: in face view
    the image x-axis runs along the shooting line, so x is fore/aft sway relative to
    the target; in back view x is side-to-side.
    """
    names = ("l_hip", "r_hip")
    pts = [
        ((track[f]["l_hip"][0] + track[f]["r_hip"][0]) / 2,
         (track[f]["l_hip"][1] + track[f]["r_hip"][1]) / 2)
        for f in sorted(track)
        if onset <= f <= release and _usable(track, vis, f, names)
    ]
    if len(pts) < 3 or scale is None:
        return None
    arr = np.array(pts)
    total = max(
        float(np.hypot(*(a - b)))
        for i, a in enumerate(arr) for b in arr[i + 1:]
    ) if len(arr) > 1 else 0.0
    return {
        "total_range": total / scale,
        # np.ptp(...) rather than arr.ptp() — the ndarray method was removed in numpy 2.
        "x_range": float(np.ptp(arr[:, 0])) / scale,
        "y_range": float(np.ptp(arr[:, 1])) / scale,
        "n_frames": len(pts),
    }


def creep(track, vis, onset, release, scale):
    """Draw-hand drift toward the bow during the hold, before the release.

    "Toward the bow" is defined by the bow-arm direction measured in the same frame
    — the unit vector from bow shoulder to bow wrist — rather than by assuming which
    way the target lies in the image. That makes the measure view-agnostic and
    removes a class of sign error: an earlier metric in this project had its sign
    convention inverted for exactly this reason.

    Positive = the draw hand crept forward. Measurement stops CREEP_MARGIN_FRAMES
    before the release, because the release snap is a large displacement in the same
    direction and would otherwise dominate.
    """
    names = ("d_wr", "b_wr", "b_sh")
    end_target = release - CREEP_MARGIN_FRAMES
    f0 = _nearest_usable(track, vis, onset, names)
    f1 = _nearest_usable(track, vis, end_target, names)
    if f0 is None or f1 is None or scale is None or f1 <= f0:
        return None
    fwd = np.array(track[f0]["b_wr"]) - np.array(track[f0]["b_sh"])
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return None
    fwd = fwd / n
    disp = np.array(track[f1]["d_wr"]) - np.array(track[f0]["d_wr"])
    return float(np.dot(disp, fwd) / scale)


def followthrough_collapse(track, vis, release, scale, frames=FOLLOWTHROUGH_FRAMES):
    """Draw-elbow vertical drop from release to release+frames.

    Positive = the elbow dropped, i.e. the line of expansion was not maintained
    through the follow-through. Image y increases downward, so a drop is a positive
    difference.

    This is the one measure here with a piece of ground truth behind it: on
    IMG_3126 shot 5 (back view) the archer independently reported the shot as having
    collapsed while its static anchor pose looked normal, and the equivalent
    displacement read roughly four times its session-mates.
    """
    names = ("d_el",)
    f0 = _nearest_usable(track, vis, release, names)
    f1 = _nearest_usable(track, vis, release + frames, names)
    if f0 is None or f1 is None or scale is None:
        return None
    return float((track[f1]["d_el"][1] - track[f0]["d_el"][1]) / scale)


def measure_all(track, vis, onset, release, view=None):
    """Every measure the available keypoints support, as a flat dict.

    Missing keypoints yield None for the affected measure rather than an error, so a
    view that cannot see the bow arm still reports sway and follow-through.
    """
    window = [f for f in sorted(track) if onset <= f <= release + FOLLOWTHROUGH_FRAMES]
    scale = _scale(track, vis, window)
    sway = postural_sway(track, vis, onset, release, scale)
    return {
        "scale_px": round(scale, 1) if scale else None,
        "bow_arm_collapse": _r(bow_arm_collapse(track, vis, onset, release, scale)),
        "hip_sway_total": _r(sway["total_range"]) if sway else None,
        "hip_sway_x": _r(sway["x_range"]) if sway else None,
        "hip_sway_y": _r(sway["y_range"]) if sway else None,
        "creep": _r(creep(track, vis, onset, release, scale)),
        # followthrough_collapse deliberately omitted — dominated by bow-lowering
        # timing and it anti-detects its one labelled case. See the module docstring.
        "followthrough_collapse": None,
        "view": view,
    }


def _r(v, nd=4):
    return None if v is None else round(v, nd)

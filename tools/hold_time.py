"""
Hold-time measurement — velocity-based, shared by all analysis paths.

WHY THIS MODULE EXISTS (BB11, 2026-07-28)
    Hold time previously had two incompatible definitions stored under one field
    name, and the primary one was dominated by an arbitrary parameter:

    1. `analyze_face_yolo.py` back-scanned from the verified frame while the draw
       wrist stayed within a fixed DISTANCE of its anchor position. The tolerance
       decided the answer: on one shot, tol .025 -> 4.77s, .010 -> 2.73s,
       .005 -> 0.67s. A sevenfold range from a number with no empirical basis.
    2. `archery_analyzer_v2.py` measured full-draw-onset -> release from wrist
       X-velocity (settle/snap). Different quantity, same `hold_time_s` field,
       plotted on the same trend axis.

    The distance formulation fails because "still" is not a distance — a wrist
    creeping 0.02 normalized units over four seconds of aiming never trips a .025
    tolerance, so the aiming phase is silently counted as hold.

THE DEFINITION USED HERE
    The hold is the interval during which the draw wrist is *not moving*, ending
    at the release. Both boundaries are found from speed, not position:

        speed(f)  = |wrist(f) - wrist(f-1)|         (normalized units per frame)
        smoothed  = 5-frame centred mean            (suppresses keypoint jitter)
        floor     = median speed over the 15 frames before the anchor frame
                    (certainly inside the hold — this is the per-shot noise level)

        hold onset  = end of the last sustained run of speed > K_ONSET * floor
                      before the anchor  (i.e. when drawing stopped)
        release     = first frame after the anchor with speed > K_RELEASE * floor
        hold_time   = (release - onset) / fps

    Deriving both thresholds from each shot's own noise floor is what makes the
    result stable: it adapts to per-session keypoint noise instead of assuming a
    global scale.

EMPIRICAL BASIS (10 shots, 2 sessions, 2026-07-28)
    Measured hold durations 1.8-4.8s. Sensitivity to each parameter, as mean
    per-shot spread over the swept range:

        distance-based (superseded) tol .005-.025 : ~4.1 s     <- parameter decides
        fixed velocity threshold    .003-.008     :  1.32 s
        adaptive K_ONSET            3-8           :  0.82 s    <- adopted
        SUSTAIN_FRAMES              2-6           :  0.07 s    <- effectively inert

    Speed separation is wide: hold median ~0.001, draw phase 0.005-0.018, release
    peak 0.014-0.040 normalized units/frame. The phases are genuinely distinct;
    the old formulation just wasn't looking at the right quantity.

HONEST LIMITS
    - 0.82s residual sensitivity to K_ONSET is real. Hold time is a
      ~+/-0.4s measurement, not a precise one. Do not report it to two decimals
      and do not read small cross-session differences as changes.
    - Not validated against hand-labeled ground truth. Parameter stability is
      not accuracy: a stable method can be consistently wrong. Establishing
      accuracy needs a labeled set (see docs/paper_todo.md BB11).
    - Censoring still applies. If the scan reaches the window bound, the start of
      video, or a pose dropout without finding onset, the hold is longer than the
      window and the value is a lower bound, flagged `censored`.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# Identifier stamped into session records as `hold_time_s_definition`. The corpus
# contains values from superseded formulations that are NOT comparable to these
# (BB1: distance-scan window ceiling, ~0.967/1.000s; BB11: X-only settle logic
# truncated by single-frame jitter, ~0.37-0.67s). Consumers must trust only values
# carrying this stamp. Bump it if the definition changes again.
HOLD_TIME_DEFINITION = "velocity_floor_v1"

# Legacy stamp applied to records whose hold time predates the velocity definition
# and cannot be recomputed in place (no verified frames, or source video absent).
HOLD_TIME_LEGACY = "legacy_pre_bb11"

# Speed smoothing half-width, in frames (5-frame centred window).
SMOOTH_HALF_WIDTH = 2

# Frames before the anchor used to estimate the per-shot noise floor. Must be
# short enough to sit entirely inside the hold on the shortest real hold observed
# (~1.8s = 54 frames at 30fps), so 15 is comfortably safe.
FLOOR_WINDOW_FRAMES = 15

# Onset: drawing is speed above this multiple of the noise floor.
K_ONSET = 5.0

# Release: the post-anchor forward snap, a much larger multiple.
K_RELEASE = 15.0

# Absolute guards, in normalized units/frame, in case a pathologically quiet floor
# would otherwise make the thresholds meaninglessly small.
MIN_ONSET_THRESHOLD = 0.0015
MIN_RELEASE_THRESHOLD = 0.008

# Consecutive above-threshold frames required to call it drawing rather than a
# jitter spike. Swept 2-6 with 0.07s effect; 3 is the middle of a flat region.
SUSTAIN_FRAMES = 3

# How far past the anchor to look for the release. Observed lag is 3-6 frames.
RELEASE_SEARCH_FRAMES = 25


def smoothed_speed(track: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
    """Per-frame wrist speed in normalized units/frame, smoothed over 5 frames.

    `track` maps frame index -> (x, y) normalized wrist position. Frames whose
    predecessor is missing are skipped rather than interpolated, so a pose dropout
    produces a gap instead of a fabricated velocity.
    """
    raw = {}
    for f, (x, y) in track.items():
        prev = track.get(f - 1)
        if prev is None:
            continue
        raw[f] = float(np.hypot(x - prev[0], y - prev[1]))
    if not raw:
        return {}
    smooth = {}
    for f in raw:
        window = [raw[g] for g in range(f - SMOOTH_HALF_WIDTH, f + SMOOTH_HALF_WIDTH + 1)
                  if g in raw]
        smooth[f] = float(np.mean(window))
    return smooth


def noise_floor(speed: Dict[int, float], anchor_frame: int) -> Optional[float]:
    """Median speed over the frames immediately before the anchor.

    That interval is inside the hold by construction, so its median is the
    shot's own keypoint-jitter level — the scale everything else is measured in.
    """
    vals = [speed[f] for f in range(anchor_frame - FLOOR_WINDOW_FRAMES, anchor_frame)
            if f in speed]
    if not vals:
        return None
    return float(np.median(vals))


def find_hold_onset(speed, anchor_frame, floor, k_onset=K_ONSET):
    """Frame at which the hold began, scanning back from the anchor.

    Returns (onset_frame, censored). `censored` is True when the scan ran out of
    data — window bound, start of video, or pose dropout — without observing the
    draw, meaning the true onset is earlier than anything measurable here and the
    resulting duration is a lower bound.
    """
    if floor is None:
        return None, True
    threshold = max(k_onset * floor, MIN_ONSET_THRESHOLD)
    earliest = min(speed) if speed else anchor_frame
    run = 0
    f = anchor_frame - 1
    while f >= earliest:
        if f not in speed:
            # Dropout: cannot tell whether the wrist was moving here.
            return None, True
        if speed[f] > threshold:
            run += 1
            if run >= SUSTAIN_FRAMES:
                # Drawing confirmed. The hold starts just after this run.
                return f + run, False
        else:
            run = 0
        f -= 1
    return None, True


def find_release(speed, anchor_frame, floor, k_release=K_RELEASE):
    """First frame after the anchor showing the release snap, or None."""
    if floor is None:
        return None
    threshold = max(k_release * floor, MIN_RELEASE_THRESHOLD)
    for f in range(anchor_frame + 1, anchor_frame + RELEASE_SEARCH_FRAMES):
        if speed.get(f, 0.0) > threshold:
            return f
    return None


def measure_hold_from_release(track, release_frame, fps):
    """Measure hold time when the RELEASE frame is already known.

    The streaming detector in `archery_analyzer_v2.py` finds the release directly
    from the forward snap, so it does not need the release search — but it must use
    the same onset rule, or the two paths drift apart again (BB11).

    The noise floor is taken from the frames immediately before the release, which
    are inside the hold for the same reason they are before the anchor.
    """
    speed = smoothed_speed(track)
    if not speed:
        return {"hold_time_s": None, "censored": True, "onset_frame": None,
                "release_frame": release_frame, "noise_floor": None}

    floor = noise_floor(speed, release_frame)
    onset, censored = find_hold_onset(speed, release_frame, floor)

    if onset is None:
        earliest = min(speed)
        return {"hold_time_s": (release_frame - earliest) / fps, "censored": True,
                "onset_frame": None, "release_frame": release_frame,
                "noise_floor": floor}

    return {"hold_time_s": (release_frame - onset) / fps, "censored": censored,
            "onset_frame": onset, "release_frame": release_frame,
            "noise_floor": floor}


def measure_hold(track, anchor_frame, fps):
    """Measure hold time for one shot.

    `track`: {frame: (x, y)} normalized draw-wrist positions spanning at least the
    intended lookback before `anchor_frame` and ~25 frames after it.

    Returns a dict:
        hold_time_s   duration, or None if onset was not found
        censored      True if the value is a lower bound, not a measurement
        onset_frame   where the hold began (None if censored)
        release_frame where the release snap was detected (None if not found)
        noise_floor   the per-shot speed floor the thresholds were derived from

    When the release is not detected the anchor frame is used as the end of the
    hold, which understates the duration by the release lag (3-6 frames observed,
    i.e. ~0.1-0.2s at 30fps). That case is not flagged as censored, since the
    error is small and bounded, unlike a missing onset.
    """
    speed = smoothed_speed(track)
    if not speed:
        return {"hold_time_s": None, "censored": True, "onset_frame": None,
                "release_frame": None, "noise_floor": None}

    floor = noise_floor(speed, anchor_frame)
    onset, censored = find_hold_onset(speed, anchor_frame, floor)
    release = find_release(speed, anchor_frame, floor)

    if onset is None:
        # Lower bound: the hold covers at least everything we could see.
        earliest = min(speed)
        end = release if release is not None else anchor_frame
        return {"hold_time_s": (end - earliest) / fps, "censored": True,
                "onset_frame": None, "release_frame": release, "noise_floor": floor}

    end = release if release is not None else anchor_frame
    return {"hold_time_s": (end - onset) / fps, "censored": censored,
            "onset_frame": onset, "release_frame": release, "noise_floor": floor}

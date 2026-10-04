#!/usr/bin/env python3
"""
Training load — per-session time under tension and within-session drift.

WHAT THIS IS FOR
    Archery is a unilateral, repetitive, isometrically-loaded activity, and the
    sports-medicine literature associates several of the loading patterns this
    pipeline already measures with overuse complaints: scapular elevation under
    load with subacromial impingement, sustained cervical flexion with neck
    loading, a locked bow elbow with joint rather than muscular loading.

    THIS IS NOT AN INJURY RISK ASSESSMENT. It reports exposure (how much loaded
    time) and degradation (whether the load-bearing metrics drift as a session
    progresses). Both are descriptive. Nothing here has been validated against a
    clinical outcome, and a correspondence with a published mechanism is not a
    prediction about this archer. Pain or suspected injury is a matter for a
    physiotherapist or physician.

WHY WITHIN-SESSION IS THE RIGHT AXIS
    The camera does not move during a session, so within-session comparison is
    unaffected by the cross-session camera-geometry confound that makes absolute
    values non-comparable between dates (apparent scale varies 48% across the
    primary subset; see the Limitations draft §N.2). Fatigue-driven form
    degradation is also the recognised proximal mechanism for overuse. The most
    reliable axis and the most relevant one coincide.

TWO MEASURES
    1. Time under tension = sum over shots of hold duration. Only shots measured
       under the current hold-time definition count; censored shots and records on
       a superseded definition are excluded and reported separately, because a
       censored value is a lower bound and mixing bounds with measurements gives a
       number that is neither.

    2. Within-session drift = change in a metric from the first half of a session's
       clips to the second half, with clips ordered by capture. Reported alongside
       the metric's noise floor, because most observed drifts are smaller than it.

METHODOLOGICAL WARNINGS, LEARNED THE HARD WAY
    - Clip *index* is not exposure. A 39-clip session and a 9-clip session are not
      comparable on clip number. Drift is therefore also reported per arrow.
    - Comparing drift BETWEEN sessions requires matched clip structure. An earlier
      project finding was retracted because a variance statistic was confounded
      with clip count (see docs/RETRACTIONS.md). Do not compare a 39-clip session's
      drift to a 7-clip session's and call the difference a change in the archer.
    - Single-clip sessions carry no within-session information at all.

USAGE
    python3 tools/training_load.py
    python3 tools/training_load.py --view face --min-clips 4
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hold_time import HOLD_TIME_DEFINITION

# Metrics whose drift is plausibly load-related, with the documented noise floor
# for each where one exists. Drift smaller than the floor is not interpretable.
DRIFT_METRICS = [
    ("bow_shoulder_elevation_sw_avg", "BSE_sw", 0.06,
     "scapular elevation under load"),
    ("bow_shoulder_elevation_th_avg", "BSE_th", 0.13,
     "scapular elevation under load (target view)"),
    ("nose_preanchor_drift_avg", "chin drift", None,
     "cervical flexion during the draw"),
    ("shoulder_level_avg", "shoulder level", None,
     "unilateral asymmetry"),
    ("hip_alignment_avg", "hip alignment", None,
     "postural stability"),
]


def clip_ordinal(record):
    """Temporal position of a clip within its session, from the filename.

    Camera filenames increment per capture, so IMG_4488 precedes IMG_4496 on the
    same date. Returns None when no ordinal can be recovered.
    """
    m = re.search(r"IMG_?(\d+)", record.get("video_path") or "")
    return int(m.group(1)) if m else None


def load_sessions(history="session_history", view=None, location=None, distance_m=None):
    sessions = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(history, "*.json"))):
        d = json.load(open(path))
        m = d.get("metrics") or {}
        if view and d.get("view") != view:
            continue
        if location is not None and d.get("location") != location:
            continue
        if distance_m is not None and d.get("distance_m") != distance_m:
            continue
        ordinal = clip_ordinal(d)
        if ordinal is None:
            continue
        sessions[(d.get("date"), d.get("view"))].append((ordinal, m, os.path.basename(path)))
    return {k: sorted(v) for k, v in sessions.items()}


def time_under_tension(clips):
    """(seconds, shots_counted, shots_excluded, reason_counts).

    Only hold times measured under the current definition contribute. Censored
    shots and superseded-definition records are counted but not summed.
    """
    total = 0.0
    counted = excluded = 0
    reasons = defaultdict(int)
    for _, m, _ in clips:
        n_meas = m.get("hold_time_s_n_measured")
        n_cens = m.get("hold_time_s_n_censored") or 0
        avg = m.get("hold_time_s_avg")
        defn = m.get("hold_time_s_definition")
        if defn != HOLD_TIME_DEFINITION:
            excluded += m.get("arrows_shot") or 0
            reasons["superseded hold-time definition"] += 1
            continue
        if n_meas and avg is not None:
            total += avg * n_meas
            counted += n_meas
        if n_cens:
            excluded += n_cens
            reasons["hold ran past the measurement window"] += n_cens
    return total, counted, excluded, dict(reasons)


def drift(clips, field):
    """First-half to second-half change, with per-arrow normalisation.

    Returns None when fewer than four clips carry the field — below that, halves
    are one or two clips each and the comparison is noise.
    """
    vals = [(m.get(field), m.get("arrows_shot") or 0) for _, m, _ in clips
            if m.get(field) is not None]
    if len(vals) < 4:
        return None
    y = np.array([v for v, _ in vals], dtype=float)
    arrows = sum(n for _, n in vals)
    half = len(y) // 2
    first, second = y[:half].mean(), y[half:].mean()
    delta = second - first
    return {
        "first_half": first,
        "second_half": second,
        "delta": delta,
        "pct": 100 * delta / abs(first) if first else float("nan"),
        "per_arrow": delta / arrows if arrows else float("nan"),
        "n_clips": len(y),
        "arrows": arrows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="session_history")
    ap.add_argument("--view", default=None, choices=["face", "back", "target"])
    ap.add_argument("--min-clips", type=int, default=4,
                    help="skip sessions with fewer clips (default 4)")
    ap.add_argument("--location", default=None,
                    help="restrict to sessions with this exact 'location' field (default: all)")
    ap.add_argument("--distance-m", type=int, default=None,
                    help="restrict to sessions shot at this distance in meters (default: all)")
    args = ap.parse_args()

    sessions = load_sessions(args.history, args.view, args.location, args.distance_m)
    if not sessions:
        print("No sessions matched.")
        return

    print("=" * 78)
    print("  TRAINING LOAD — exposure and within-session drift")
    print("  Descriptive only. Not an injury risk assessment. See the module docstring.")
    print("=" * 78)

    for (date, view), clips in sorted(sessions.items()):
        tut, counted, excluded, reasons = time_under_tension(clips)
        arrows = sum((m.get("arrows_shot") or 0) for _, m, _ in clips)
        print(f"\n  {date}  {view}  —  {len(clips)} clips, {arrows} arrows")
        if counted:
            print(f"    time under tension: {tut:6.1f}s over {counted} measured shot(s)"
                  f"   (mean {tut/counted:.2f}s/shot)")
        else:
            print(f"    time under tension: not computable")
        if excluded:
            why = "; ".join(f"{k} ({v})" for k, v in reasons.items())
            print(f"    excluded: {excluded} shot(s) — {why}")

        if len(clips) < args.min_clips:
            print(f"    drift: not computed ({len(clips)} clips < {args.min_clips})")
            continue

        shown = False
        for field, label, floor, mechanism in DRIFT_METRICS:
            r = drift(clips, field)
            if r is None:
                continue
            shown = True
            note = ""
            if floor is not None:
                note = ("  ← within noise floor, not interpretable"
                        if abs(r["pct"]) < floor * 100 else
                        f"  ← exceeds the {floor*100:.0f}% noise floor")
            print(f"    {label:14s} {r['first_half']:+.3f} → {r['second_half']:+.3f}"
                  f"  ({r['pct']:+.1f}%, {r['per_arrow']:+.5f}/arrow){note}")
        if not shown:
            print("    drift: no eligible metrics carried by enough clips")

    print("\n" + "=" * 78)
    print("  Reminder: comparing drift BETWEEN sessions needs matched clip structure.")
    print("  An earlier finding was retracted for exactly that error — docs/RETRACTIONS.md.")
    print("=" * 78)


if __name__ == "__main__":
    main()

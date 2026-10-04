#!/usr/bin/env python3
"""
Per-archer calibration — Phase 1 of the app-generalization roadmap.
(docs/APP_ROADMAP.md · development_roadmap.md Tier 4.)

WHAT / WHY
    The coaching rule engine (tools/coaching_feedback.py) can flag form against
    ABSOLUTE, coach-validated thresholds — but those numbers are tuned to whoever
    validated them, and don't generalize to a stranger's body and equipment. This
    tool makes the system calibrate itself to any archer: it reads that archer's
    own baseline sessions and derives PERSONAL RELATIVE bands (flag deviation from
    their own norm) instead of relying on someone else's absolutes. The archer
    becomes their own reference — a relative-consistency signal that tends to be
    more actionable than an absolute one for an individual archer's technique.

    Output: archers/<id>/baseline_profile.json — per view, per metric:
        {mean, sd, n_sessions, personal_range, direction, unit, cue}
    coaching_feedback.py reads this when profile.calibration.mode != "absolute".

STATUS: STUB. The statistics are real and runnable; the marked TODOs are the
    remaining Phase-1 work (handedness mirroring, discipline starter sets, and
    the coaching_feedback.py read-path). See TODOs inline.

USAGE
    python3 tools/calibrate_archer.py --archer <archer_id> --dry-run   # print, don't write
    python3 tools/calibrate_archer.py --archer <archer_id>             # write baseline_profile.json
    python3 tools/calibrate_archer.py --archer <archer_id> --sigma 2.0 --min-sessions 6

HONEST CAVEAT
    Relative mode flags "worse than YOUR normal", not "wrong in absolute terms".
    A chronic fault that shows up on every shot (e.g. a nose-to-string drift that
    sits at the archer's own median every time) sits INSIDE their personal band
    and won't flag in relative mode — that is by design. Chronic-fault thresholds
    belong in absolute/hybrid mode with a coach in the loop. Choose the mode per
    archer in profile.calibration.
"""

import argparse
import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from glob import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reuse the single-source-of-truth scope guard + loader from the rule engine.
sys.path.insert(0, os.path.join(BASE, "tools"))
from coaching_feedback import load_session, in_scope  # noqa: E402


# ── Which metrics to calibrate, and each metric's "bad direction" ─────────────
# direction:
#   "two_sided" — good is a band around the personal mean; flag either extreme
#                 (consistency metrics: elbow angle, anchor, shoulder level, hold)
#   "high_bad"  — larger is worse; only flag above mean + sigma·SD
#   "low_bad"   — smaller is worse; only flag below mean − sigma·SD
# Keep field names identical to coaching_feedback.THRESHOLDS so the two files
# reference the same JSON keys.
CALIBRATION_METRICS = {
    "face": [
        ("draw_elbow_angle_avg",         "Draw elbow angle",      "two_sided", "°"),
        ("nose_string_gap_avg",          "Nose–string gap",       "two_sided", ""),
        ("nose_preanchor_drift_avg",     "Nose pre-anchor drift", "high_bad",  " px"),
        ("hold_time_s_avg",              "Hold time",             "two_sided", "s"),
        ("bow_shoulder_elevation_sw_avg","Bow shoulder elevation","two_sided", " sh-widths"),
    ],
    "back": [
        ("shoulder_level_avg",           "Shoulder level",        "two_sided", "°"),
        ("hip_alignment_avg",            "Hip alignment",         "two_sided", "°"),
        ("bow_shoulder_elevation_sw_avg","Bow shoulder elevation","two_sided", " sh-widths"),
    ],
    "target": [
        ("draw_elbow_height_th_avg",     "Draw elbow height",     "high_bad",  " torso-heights"),
        ("bow_shoulder_elevation_th_avg","Bow shoulder elevation","two_sided", " torso-heights"),
    ],
}


def load_archer_sessions(archer, scope_only=True):
    """In-scope practice sessions belonging to `archer`."""
    out = []
    for p in sorted(glob(os.path.join(BASE, "session_history", "*.json"))):
        try:
            s = load_session(p)
        except (json.JSONDecodeError, OSError):
            continue
        if s.get("session_type") != "practice":
            continue
        if s.get("archer_id") != archer:
            continue
        if scope_only and not in_scope(s):
            continue
        out.append(s)
    return out


def personal_range(mean, sd, direction, sigma):
    """Band beyond which a value is flagged, given the metric's bad direction.
    None on a side means 'never flag on that side'."""
    lo = round(mean - sigma * sd, 4)
    hi = round(mean + sigma * sd, 4)
    if direction == "high_bad":
        return [None, hi]
    if direction == "low_bad":
        return [lo, None]
    return [lo, hi]  # two_sided


def calibrate(archer, sigma, min_sessions, scope_only=True):
    sessions = load_archer_sessions(archer, scope_only)
    views = {}
    skipped = []
    for view, specs in CALIBRATION_METRICS.items():
        vsessions = [s for s in sessions if s.get("view") == view]
        metrics = {}
        for field, label, direction, unit in specs:
            vals = [s["metrics"][field] for s in vsessions
                    if isinstance(s.get("metrics", {}).get(field), (int, float))]
            if len(vals) < min_sessions:
                skipped.append(f"{view}.{field} (n={len(vals)} < {min_sessions})")
                continue
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            metrics[field] = {
                "label": label,
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "n_sessions": len(vals),
                "direction": direction,
                "unit": unit,
                "personal_range": personal_range(mean, sd, direction, sigma),
            }
        if metrics:
            views[view] = metrics
    return views, skipped, len(sessions)


# TODO(Phase 1): handedness mirroring. For a left-handed archer, face-view
#   horizontal metrics (anchor_x, nose_string_gap) are mirrored. Read
#   profile.identity.handedness and flip sign / reference before calibrating.
# TODO(Phase 1): discipline starter sets. When n < min_sessions for a metric,
#   seed personal_range from a per-discipline default (recurve/compound/barebow)
#   so a brand-new archer still gets sane bands on session one, then converge to
#   personal as data accrues.
# TODO(Phase 1): wire coaching_feedback.py to read baseline_profile.json when
#   profile.calibration.mode in ("relative","hybrid") — use personal_range in
#   place of / alongside the hardcoded THRESHOLDS bands.


def main():
    ap = argparse.ArgumentParser(description="Derive per-archer relative form baselines.")
    ap.add_argument("--archer", required=True, help="archer_id, matching archers/<archer_id>/")
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="deviation multiplier for personal bands (default 1.5)")
    ap.add_argument("--min-sessions", type=int, default=4,
                    help="minimum sessions per metric before a band is written")
    ap.add_argument("--all-distances", action="store_true",
                    help="no-op unless you've customized in_scope() in coaching_feedback.py "
                         "to restrict sessions for your own deployment")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args()

    views, skipped, n_sessions = calibrate(
        args.archer, args.sigma, args.min_sessions, scope_only=not args.all_distances)

    if not views:
        sys.exit(f"No calibratable metrics for archer '{args.archer}' "
                 f"({n_sessions} in-scope sessions; need ≥{args.min_sessions} per metric).")

    baseline = {
        "archer_id": args.archer,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calibration": {"sigma": args.sigma, "min_sessions": args.min_sessions,
                        "scope": "backyard-18m" if not args.all_distances else "all"},
        "source_sessions": n_sessions,
        "views": views,
        "_note": "Generated by tools/calibrate_archer.py. personal_range = band beyond "
                 "which a value is flagged; None = never flag on that side. Consumed by "
                 "coaching_feedback.py when profile.calibration.mode != 'absolute'.",
    }

    print(f"Archer '{args.archer}': {n_sessions} in-scope sessions, "
          f"σ={args.sigma}, min_sessions={args.min_sessions}")
    for view, metrics in views.items():
        print(f"\n── {view} ──")
        for field, m in metrics.items():
            lo, hi = m["personal_range"]
            band = (f"≤ {hi}" if lo is None else
                    f"≥ {lo}" if hi is None else f"[{lo}, {hi}]")
            print(f"  {m['label']:24s} mean {m['mean']}{m['unit']}  "
                  f"(±{m['sd']}, n={m['n_sessions']})  flag outside {band}")
    if skipped:
        print(f"\nskipped (too few sessions): {', '.join(skipped)}")

    if args.dry_run:
        print("\n[dry-run: nothing written]")
        return

    out_path = os.path.join(BASE, "archers", args.archer, "baseline_profile.json")
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"\nWrote → {out_path}")
    print("Next: set profile.calibration.mode to 'relative' (or 'hybrid') to use it, "
          "and stamp profile.calibration.baseline_generated.")


if __name__ == "__main__":
    main()

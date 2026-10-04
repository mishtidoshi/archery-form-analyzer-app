"""
Canonical metric thresholds — the single source of truth (BB9/BB10, 2026-07-29).

WHY THIS MODULE EXISTS
    The same "good form" bands were duplicated across three places that drifted
    apart: `coaching_feedback.py` (rule engine), `trend_analysis.py` (plot shading
    and console flags), and the per-view `analyze_*_yolo.py` scripts (archer-facing
    feedback tables). Auditing them for the paper appendix found:

      - draw elbow angle:  20-40 deg in two places, 15-35 deg in the archer-facing table
      - hip alignment:     +/-5 deg in code, "+/-3 deg" in docs/WORKFLOW.md
      - hold time:         0.5-1.5s in code vs ">= 1.0s" documented, with a worked
                           example flagging a PASSING value as a failure
      - bow shoulder elev: observation-only in the rule engine but a hard 0.55-0.65
                           pass/fail in the face feedback table -- while its
                           coach-confirmed threshold is still openly TBD
      - back-view BSE:     a 0.18-0.50 band left over from the legacy `/h` scale,
                           applied to shoulder-width-normalized values

    Three metrics were also documented in WORKFLOW.md as having bands while no code
    ever flagged them, and two were flagged in code while absent from the docs.

PROVENANCE IS PART OF THE DEFINITION
    Every band carries `provenance`, because "coach-validated" was being used
    loosely for numbers with quite different standing. Two retracted findings came
    from bands fitted to broken measurements and then read back as coaching targets
    (see docs/RETRACTIONS.md), so an unsourced band is treated as a hypothesis about
    the data, not a target.

      coach_stated       an explicit number a coach gave
      coach_cue_fitted   coach named the direction; tolerance fitted to observed good sessions
      technique_derived  from Olympic recurve technique, not this archer
      descriptive        where this archer's data sits; NOT a target
      unvalidated        in use but no traceable source -- needs confirmation

    Only `coach_stated`, `coach_cue_fitted` and `technique_derived` bands should
    drive pass/fail. `descriptive` and `unvalidated` are reported without a verdict.
"""

# ── Band definitions ─────────────────────────────────────────────────────────
#
# kind:  "range"  -> (lo, hi) inclusive pass band
#        "max"    -> pass when value <= limit
#        "abs_max"-> pass when abs(value) <= limit
#        "observe"-> no verdict; `reference` is a descriptive centre if known
#
# Each entry also names the session-JSON `_avg` field and the coaching cue it maps
# to, so the rule engine, the trend tool and the per-view scripts stay in step.

THRESHOLDS = {
    "face": {
        "draw_elbow_angle": dict(
            label="Draw elbow angle", field="draw_elbow_angle_avg", unit="°",
            kind="observe", reference=None, provenance="unvalidated",
            cue="T-form / draw-side alignment",
            note="OBSERVATION-ONLY, deliberately (BB9, 2026-07-29). Three reasons, "
                 "and the third is decisive:\n"
                 "  1. The two implemented bands disagreed — 20–40° in the rule engine "
                 "and trend tool, 15–35° in the archer-facing table — and neither has "
                 "a recorded source.\n"
                 "  2. Applying 20–40° would flag 10 of 23 primary-subset sessions as "
                 "'below', i.e. tell the archer her draw elbow is too open on 43% of "
                 "sessions, on a metric no coach has ever raised as a fault. That is "
                 "the same failure mode as the old hold-time band: a threshold with no "
                 "provenance manufacturing flags against good form.\n"
                 "  3. BB6 showed this metric is camera-geometry confounded — apparent "
                 "scale varies 48% across the primary dates and the angle correlates "
                 "+0.56 with it — so the VALUE is not comparable across sessions at "
                 "all, which makes any fixed band unsound in principle.\n"
                 "The [15,50] range still applies as a per-shot PLAUSIBILITY gate in "
                 "analyze_face_yolo.py — that is a measurement-validity check, not a "
                 "coaching band, and the distinction matters. A coaching band needs "
                 "coach input plus the camera confound resolved.",
        ),
        "bow_elbow_angle": dict(
            label="Bow elbow angle", field="bow_elbow_angle_avg", unit="°",
            kind="range", band=(160, 175), provenance="technique_derived",
            cue="bow arm extension without locking",
            note="Was flagged in the face feedback table but absent from the rule "
                 "engine (BB10). WORKFLOW.md documented it inconsistently as both "
                 "~160–170° and 160–175°; 160–175° is what the code applied.",
        ),
        "anchor_spread_head_px": dict(
            label="Anchor spread (head-rel)", field="anchor_spread_head_px", unit=" px",
            kind="max", band=10.0, provenance="coach_cue_fitted",
            cue="consistent anchor — the hand returns to the same place on the face",
            note="Head-relative, not image-relative: see BB2. The image-relative "
                 "figure cannot separate 'hand moved on the face' from 'archer moved "
                 "in frame'. Was documented and flagged in the face table but absent "
                 "from the rule engine (BB10).",
        ),
        "nose_string_gap": dict(
            label="Nose–string gap", field="nose_string_gap_avg", unit="",
            kind="range", band=(-0.12, -0.04), target=-0.07,
            provenance="coach_stated",
            cue="consistent facial anchor (coach target −0.07; negative = string touching)",
        ),
        "nose_preanchor_drift": dict(
            label="Nose pre-anchor drift", field="nose_preanchor_drift_avg", unit=" px",
            kind="abs_max", band=2.0, provenance="coach_cue_fitted",
            cue="head still — pull the string to the head, don't push the head into "
                "the string; positive = chin drops during the draw",
            note="The CUE is coach-derived; the 2 px number is the dataset median "
                 "with a one-sided rationale (the chin never rises; dataset min "
                 "≈0.8 px). Not validated against per-shot arrow height.",
        ),
        "hold_time_s": dict(
            label="Hold time", field="hold_time_s_avg", unit=" s",
            kind="range", band=(1.0, 3.0), provenance="coach_stated",
            cue="expansion / pull through the clicker (ideal 1–3 s)",
            note="The old 0.5–1.5 s band had no coaching basis and was evidently "
                 "fitted to right-censored values; under correct measurement it "
                 "flagged good 2–3 s holds as failures. See docs/RETRACTIONS.md.",
        ),
        "bow_shoulder_elevation_sw": dict(
            label="Bow shoulder elevation", field="bow_shoulder_elevation_sw_avg",
            unit=" sh-widths", kind="observe", reference=0.60,
            provenance="descriptive",
            cue="sink the scapula",
            note="0.60 is where this archer sits, NOT a target. The face feedback "
                 "table applied a 0.55–0.65 pass/fail while the rule engine treated "
                 "the metric as observation-only — the pass/fail was unfounded and is "
                 "removed. Coach-confirmed threshold is open (paper_todo A5).",
        ),
        "creep": dict(
            label="Creep (draw-hand drift)", field="creep_avg", unit=" sh-widths",
            kind="observe", reference=None, provenance="unvalidated",
            cue="keep expanding through the hold — the draw hand should not drift forward",
            note="Draw-hand displacement along the bow-arm direction between hold onset "
                 "and just before release, normalised by shoulder width (paper_todo H2). "
                 "Negative = moving AWAY from the bow, i.e. expansion, which is correct.\n"
                 "  ⚠️ THIS IS NOT 'PULLING THROUGH THE CLICKER' (corrected 2026-08-02). "
                 "The two are different phenomena roughly 20-30x apart in scale. Tripping "
                 "the clicker takes 1-3 mm of additional draw; at a 250 px shoulder width "
                 "that is 0.7-2.1 px, which sits BELOW the measured YOLO26 keypoint jitter "
                 "floor of 3.56 px average and 6.24 px on the draw elbow. Clicker-scale "
                 "expansion is therefore not camera-measurable at all. What this metric "
                 "measures is GROSS draw-hand drift: the observed values of -0.13 to -0.28 "
                 "shoulder widths are 46-98 mm, well above the jitter floor and a genuinely "
                 "different quantity. Valid for what it measures; do not present it as "
                 "evidence about a 'pull through the clicker, not to it' cue.\n"
                 "  Validated as a MEASUREMENT on 9 shots across 2 sessions: all 9 "
                 "negative (−0.04 to −0.28), a unanimous sign that indicates the measure "
                 "is reading real motion rather than noise. No threshold — what counts as "
                 "insufficient expansion needs coach input.\n"
                 "  Direction is defined by the in-frame bow-arm vector rather than by "
                 "assuming which way the target lies in the image, which removes the class "
                 "of sign error that inverted bow_wrist_ft_dx.",
        ),
        "hip_sway_total": dict(
            label="Postural sway (hold)", field="hip_sway_total_avg", unit=" sh-widths",
            kind="observe", reference=None, provenance="unvalidated",
            cue="quiet base — core tight, no sway during the hold",
            note="Largest separation between any two mid-hip positions during the hold, "
                 "normalised by shoulder width. A bound on base movement rather than a net "
                 "drift, since sway can return to where it started.\n"
                 "  Consistent across 9 shots (0.08–0.14) and mostly VERTICAL: the "
                 "fore/aft component along the shooting line is only 0.02–0.06 in face "
                 "view. Availability is clip-dependent, not view-dependent — hip "
                 "visibility measured 0.65–0.77 in one back session and 0.06–0.23 in "
                 "another, which silently removes the measure.",
        ),
        "bow_arm_collapse": dict(
            label="Bow arm collapse (hold)", field="bow_arm_collapse_avg",
            unit=" sh-widths", kind="observe", reference=None, provenance="unvalidated",
            cue="hold the tricep — quiet bow arm through the shot",
            note="Contraction of the bow wrist-to-shoulder distance across the hold. "
                 "Positive = shortened. Plausible magnitudes (−0.15 to +0.11 over 9 shots) "
                 "but mixed sign and NO ground truth, so weaker than the other two.\n"
                 "  Caveat that must travel with it: apparent bow-arm length also shortens "
                 "if the torso rotates toward the camera. Within one hold that rotation is "
                 "small, but a large positive value should be checked against the video "
                 "before being read as collapse.",
        ),
        "bow_wrist_ft_dx": dict(
            label="Bow wrist follow-through", field="bow_wrist_ft_dx_avg", unit="",
            kind="observe", reference=None, provenance="unvalidated",
            cue="bow arm stays on the line through the shot",
            note="Observation-only (BB5): sign semantics were inverted, the metric "
                 "cannot see the lateral cue it was built for (that axis is the "
                 "camera axis in face view), and at 0.67 s it is ~78% bow-lowering.",
        ),
    },
    "back": {
        "shoulder_level": dict(
            label="Shoulder level", field="shoulder_level_avg", unit="°",
            kind="range", band=(-3, 3), provenance="coach_cue_fitted",
            cue="level shoulder line / open chest (0° = level)",
            note="Distance-dependent baseline: ~0° at 18 m but +3° to +5° at 60 m. "
                 "Do not compare the value across distances.",
        ),
        "hip_alignment": dict(
            label="Hip alignment", field="hip_alignment_avg", unit="°",
            kind="range", band=(-5, 5), provenance="coach_cue_fitted",
            cue="square hips / core tight (0° = level)",
            note="All three code paths used ±5°; docs/WORKFLOW.md said ±3°. Unified "
                 "on ±5° as implemented — flagged for coach confirmation, since "
                 "neither number has a recorded source.",
        ),
        "head_lateral_tilt": dict(
            label="Head lateral tilt", field="head_lateral_tilt_avg", unit="°",
            kind="range", band=(-5, 5), provenance="coach_cue_fitted",
            cue="head upright and still at anchor",
            note="Flagged in the back-view table and documented, but absent from the "
                 "rule engine (BB10).",
        ),
        "t_draw_angle": dict(
            label="T-draw angle", field="t_draw_angle_avg", unit="°",
            kind="observe", reference=None, provenance="unvalidated",
            cue="draw elbow in line with the arrow",
            note="Observation-only (BB3). 0° is NOT ideal: in correct form the draw "
                 "elbow sits above the bow elbow, so a negative tilt is correct — all "
                 "25 back sessions are negative. Magnitude is camera-geometry "
                 "dependent (18 m ≈ −13°, 60 m ≈ −5°); compare SD, not value.",
        ),
        "dfl_angle": dict(
            label="Draw force line angle", field="dfl_angle_avg", unit="°",
            kind="observe", reference=139.0, provenance="descriptive",
            cue="draw-side alignment — bow hand, draw shoulder and draw elbow on one line",
            note="Interior angle at the draw shoulder between the bow wrist and the draw "
                 "elbow. Added 2026-08-01 (paper_todo H1).\n"
                 "  **180° is the IDEALISED target but NOT the practical one.** Measured "
                 "across 23 back-view sessions the value is 136–151° (mean 139°). A real "
                 "40° alignment fault sustained over 14 months would have been raised by a "
                 "coach, so the offset sits in the measurement: the shoulder keypoint "
                 "approximates the acromion rather than the joint centre the coaching "
                 "concept refers to, and the bow WRIST keypoint is not the bow-hand "
                 "pressure point. Collinearity survives projection, so this is a landmark-"
                 "definition offset, not perspective.\n"
                 "  This was expected to be the metric with a principled, technique-derived "
                 "target — the property every band retired in the BB9 audit lacked. It is "
                 "not: the target is principled in anatomy but not in these keypoints. "
                 "Registered observation-only, reference 139° for this archer and setup.\n"
                 "  **The usable signal is within-session consistency.** Within-session SD "
                 "is 1.54° (median) against a between-session SD of 11.96° — an 8× ratio "
                 "with the same camera-geometry signature as draw elbow angle. Compare "
                 "shot-to-shot within a session; never compare the absolute value across "
                 "sessions.",
        ),
        "bow_shoulder_elevation_sw": dict(
            label="Bow shoulder elevation", field="bow_shoulder_elevation_sw_avg",
            unit=" sh-widths", kind="observe", reference=0.60,
            provenance="descriptive",
            cue="sink the scapula",
            note="Descriptive centre only. A stale 0.18–0.50 band from the legacy "
                 "`/h` scale was being applied to shoulder-width-normalized values "
                 "in analyze_back_yolo.py; removed.",
        ),
    },
    "target": {
        "draw_elbow_height_th": dict(
            label="Draw elbow height", field="draw_elbow_height_th_avg",
            unit=" torso-heights", kind="max", band=0.0,
            provenance="technique_derived",
            cue="draw elbow at or above shoulder (negative = above ✓)",
        ),
        "hold_time_s": dict(
            label="Hold time", field="hold_time_s_avg", unit=" s",
            kind="range", band=(1.0, 3.0), provenance="coach_stated",
            cue="expansion / pull through the clicker (ideal 1–3 s)",
        ),
        "back_tension_proxy": dict(
            label="Back tension proxy", field="back_tension_proxy_avg", unit="",
            kind="observe", reference=None, provenance="descriptive",
            cue="scapular engagement",
            note="Foreshortened shoulder width. Session-relative by construction — "
                 "it is the same pixel quantity that is unusable as a normalization "
                 "reference in this view, so no cross-session band is meaningful.",
        ),
        "draw_elbow_lateral_th": dict(
            label="Draw elbow lateral", field="draw_elbow_lateral_th_avg",
            unit=" torso-heights", kind="observe", reference=None,
            provenance="unvalidated",
            cue="draw elbow tucked, not flared behind the body line",
        ),
        "bow_shoulder_elevation_th": dict(
            label="Bow shoulder elevation", field="bow_shoulder_elevation_th_avg",
            unit=" torso-heights", kind="observe", reference=0.40,
            provenance="descriptive",
            cue="sink the scapula",
        ),
    },
}

# Documented in WORKFLOW.md at some point but never implemented anywhere, and
# explicitly retired rather than left implying feedback the system cannot give
# (BB10). Keys are informational only.
RETIRED = {
    "shoulder_rotation": "Documented for target view (<±8°) but no session record "
                         "carries the field — nothing ever computed it.",
    "draw_elbow_followthrough": "Documented for face view ('should increase vs "
                                "release angle') but never compared or flagged.",
}

# Provenance tiers that may drive a pass/fail verdict.
JUDGEABLE_PROVENANCE = {"coach_stated", "coach_cue_fitted", "technique_derived"}


def spec(view, metric):
    """Look up one band definition, or None."""
    return THRESHOLDS.get(view, {}).get(metric)


def specs_for(view):
    """All band definitions for a view, as a list (rule-engine iteration order).

    Each returned spec carries `_key`, its metric name, so callers can pass it back
    to verdict()/describe() without needing a reverse lookup.
    """
    return [dict(v, _key=k) for k, v in THRESHOLDS.get(view, {}).items()]


def good_range(view, metric):
    """(lo, hi) for range-kind bands, else None.

    Convenience for trend_analysis.py plot shading, which only understands ranges.
    Returns None for max/abs_max/observe kinds so nothing gets shaded against a
    band that does not exist.
    """
    s = spec(view, metric)
    if not s or s.get("kind") != "range":
        return None
    if s.get("provenance") not in JUDGEABLE_PROVENANCE:
        return None
    return tuple(s["band"])


def verdict(view, metric, value):
    """Return True (pass), False (fail), or None (no verdict possible).

    None means the metric is observation-only or its band is not judgeable — the
    caller must report the value without a pass/fail mark rather than inventing one.
    """
    s = spec(view, metric)
    if s is None or value is None:
        return None
    if s.get("kind") == "observe" or s.get("provenance") not in JUDGEABLE_PROVENANCE:
        return None
    kind = s["kind"]
    if kind == "range":
        lo, hi = s["band"]
        return lo <= value <= hi
    if kind == "max":
        return value <= s["band"]
    if kind == "abs_max":
        return abs(value) <= s["band"]
    return None


def describe(view, metric):
    """Human-readable target string for feedback tables, e.g. '20–40°'."""
    s = spec(view, metric)
    if s is None:
        return ""
    kind, unit = s.get("kind"), s.get("unit", "")
    if kind == "range":
        lo, hi = s["band"]
        return f"{lo}–{hi}{unit}"
    if kind == "max":
        return f"≤ {s['band']}{unit}"
    if kind == "abs_max":
        return f"± {s['band']}{unit}"
    ref = s.get("reference")
    return f"observation-only (ref {ref})" if ref is not None else "observation-only"

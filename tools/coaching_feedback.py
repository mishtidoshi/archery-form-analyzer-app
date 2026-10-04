#!/usr/bin/env python3
"""
Coaching-feedback synthesis — fully local, no external API.

Two-stage design, scientifically defensible for competition judges:

  1. RULE ENGINE (deterministic) — compares a session's measured metrics against
     coach-validated thresholds and flags what is out of range. This is the
     "what is wrong" layer: pure arithmetic, no model, fully auditable.

  2. LANGUAGE LAYER — a hand-built, rule-based NLG engine (tools/feedback_nlg.py:
     sentence templates + phrase banks, not a trained model) writes the feedback
     text, optionally personalized with fragments generated from the archer's own
     logged coach cues (archers/<id>/coach_voice.md) via a from-scratch n-gram
     model (tools/coach_voice_model.py). Neither ever touches the rule engine's
     findings — the language layer only ever composes sentences from a finding's
     existing fields. If the archer hasn't logged enough of their own text yet to
     personalize with ("no language provided"), `synthesize_manual()` falls back
     to a fixed bullet-point template — "basic recommendations" — instead. See
     that function's docstring for the exact threshold and dispatch logic.

Thresholds mirror the coach-validated `good_range` values in
`tools/trend_analysis.py` (single source of truth for "good form" bands).
Coaching context is read from `archers/<id>/profile.json`.

Usage:
    python3 tools/coaching_feedback.py --session session_history/2026-05-26_practice_face_IMG3121.json --archer alex
    python3 tools/coaching_feedback.py --date 2026-05-26 --archer alex   # full-day, all views
    python3 tools/coaching_feedback.py --session S.json --archer alex --save   # also write to output/

No API key, no network call, nothing external — everything here runs on-machine.
"""

import argparse
import json
import re
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hold_time as _hold_time   # noqa: E402  (definition stamp — see BB11)
import coach_voice_model as _coach_voice_model   # noqa: E402
import feedback_nlg as _feedback_nlg   # noqa: E402

BASE = os.path.dirname(os.path.dirname(__file__))

# ── Thresholds ────────────────────────────────────────────────────────────────
# Canonical definitions live in tools/metric_thresholds.py (BB9). They used to be
# duplicated here, in trend_analysis.py and in the per-view analyze_*_yolo.py
# scripts, and the three copies drifted apart — see that module's docstring for the
# audit. Each band carries provenance; only coach_stated / coach_cue_fitted /
# technique_derived bands may produce a pass/fail verdict.
import metric_thresholds as MT   # noqa: E402

THRESHOLDS = {view: MT.specs_for(view) for view in ("face", "back", "target")}


def load_session(path):
    with open(path) as f:
        return json.load(f)


def _hold_time_is_untrustworthy(val, std, n_censored, definition=None):
    """True when a hold_time_s_avg cannot be judged against a threshold.

    Three cases:
      1. `definition` is not the current one. The corpus holds values from three
         formulations (BB1 distance-scan ceiling, BB11 X-only settle-truncation,
         and the current velocity definition) which are not comparable. Anything
         not stamped `hold_time.HOLD_TIME_DEFINITION` is not judged.
      2. Some shots were censored — the hold ran past the measurement window, so
         the recorded value is a floor for those shots.
      3. Unstamped legacy records showing the BB1 saturation signature (exactly
         0.967s = 29 frames or 1.000s = 30 frames at 30fps, with std exactly 0).
         Kept as a belt-and-braces check for records the stamping pass missed.
    """
    if definition is not None and definition != _hold_time.HOLD_TIME_DEFINITION:
        return True
    if n_censored:
        return True
    if std == 0.0 and val is not None:
        if abs(val - 0.967) < 1e-6 or abs(val - 1.0) < 1e-6:
            return True
    return False


def evaluate(view, value_of, std_of=None, extra_of=None):
    """View-agnostic rule engine. `value_of(field)` returns the metric value (or
    None), `std_of(field)` the spread, `extra_of(field)` any extra context dict
    (e.g. day aggregation: n_clips / arrows / within-day spread). Returns findings."""
    specs = THRESHOLDS.get(view, [])
    findings = []
    for spec in specs:
        val = value_of(spec["field"])
        if val is None:
            continue
        std = std_of(spec["field"]) if std_of else None

        # Hold time: never flag a censored or legacy-saturated value. Reporting
        # "hold time below target" off a measurement ceiling is worse than saying
        # nothing, because the archer cannot act on it and it is not true.
        if spec["field"] == "hold_time_s_avg":
            n_cens = value_of("hold_time_s_n_censored")
            defn = value_of("hold_time_s_definition")
            if _hold_time_is_untrustworthy(val, std, n_cens, defn):
                if defn is not None and defn != _hold_time.HOLD_TIME_DEFINITION:
                    why = (f"measured under a superseded definition ({defn}) — "
                           "not comparable to current values; not judged")
                elif n_cens:
                    why = (f"hold extended past the measurement window on {n_cens} "
                           "shot(s) — true hold unknown (≥ value shown); not judged")
                else:
                    why = ("legacy record: value is the 1.0s scan ceiling, not a "
                           "measured hold — not judged")
                findings.append({
                    "metric": spec["label"], "field": spec["field"],
                    "value": round(val, 4),
                    "std": round(std, 4) if std is not None else None,
                    "unit": spec["unit"], "status": "unavailable",
                    "detail": why,
                    "cue": spec["cue"],
                    "target": spec.get("target"), "baseline": spec.get("reference"),
                    "extra": extra_of(spec["field"]) if extra_of else None,
                })
                continue

        # Verdict comes from the canonical module, which refuses to judge
        # observation-only metrics and bands whose provenance is descriptive or
        # unvalidated (BB9). A None verdict must be reported as such, never
        # silently treated as a pass.
        kind = spec.get("kind")
        band = spec.get("band")
        ok = MT.verdict(view, spec["_key"], val)

        if ok is None:
            status = "observation"
            ref = spec.get("reference")
            detail = (f"observation-only ({spec.get('provenance')})"
                      + (f"; dataset reference ≈ {ref}" if ref is not None else ""))
        elif kind == "range":
            lo, hi = band
            if val < lo:
                status, detail = "below", f"below good range [{lo}, {hi}]"
            elif val > hi:
                status, detail = "above", f"above good range [{lo}, {hi}]"
            else:
                status, detail = "in_range", f"within good range [{lo}, {hi}]"
        elif kind == "max":
            status = "in_range" if ok else "above"
            detail = (f"within target (≤ {band})" if ok
                      else f"above {band} (should be ≤ {band})")
        elif kind == "abs_max":
            status = "in_range" if ok else "above"
            detail = (f"within target (|value| ≤ {band})" if ok
                      else f"|value| above {band}")
        else:
            status, detail = "observation", "no band defined"

        findings.append({
            "metric": spec["label"], "field": spec["field"],
            "value": round(val, 4), "std": round(std, 4) if std is not None else None,
            "unit": spec["unit"], "status": status, "detail": detail,
            "cue": spec["cue"],
            "target": spec.get("target"), "baseline": spec.get("reference"),
            "provenance": spec.get("provenance"),
            "extra": extra_of(spec["field"]) if extra_of else None,
        })
    return findings


def detect_faults(session):
    """Single-clip rule engine over one session JSON's `metrics` block."""
    metrics = session.get("metrics", {})
    return evaluate(
        session.get("view", ""),
        value_of=lambda f: metrics.get(f),
        std_of=lambda f: metrics.get(f.replace("_avg", "_std")),
    )


def in_scope(session):
    """Hook for restricting coaching feedback / calibration to a subset of sessions
    (e.g. one distance, one location) if your deployment needs that. Some metrics
    are distance-dependent — shoulder level in particular drifts with shooting
    distance — so mixing distances into one archer's baseline can produce false
    flags. Returns True (no restriction) by default; customize per your own setup
    if you want to scope sessions the way the original single-archer study did
    (backyard, fixed distance only)."""
    return True


def _all_practice_sessions(scope_only=True):
    """Load every practice session JSON that carries a scorable view.
    scope_only restricts to the backyard-18m study scope (default)."""
    out = []
    for p in sorted(glob(os.path.join(BASE, "session_history", "*.json"))):
        try:
            s = load_session(p)
        except (json.JSONDecodeError, OSError):
            continue
        if s.get("session_type") == "practice" and s.get("view") in THRESHOLDS:
            if scope_only and not in_scope(s):
                continue
            out.append(s)
    return out


def _arrows(session):
    m = session.get("metrics", {})
    return m.get("arrows_with_metrics") or m.get("arrows_shot") or 1


def aggregate_view_day(clips, view):
    """Arrows-weighted day aggregation across one view's clips. Returns
    {field: {value, n_clips, arrows, spread}} where spread = max−min of the
    per-clip means (a within-day consistency signal)."""
    fields = [s["field"] for s in THRESHOLDS.get(view, [])]
    out = {}
    for f in fields:
        vals, weights = [], []
        for c in clips:
            v = c.get("metrics", {}).get(f)
            if v is not None:
                vals.append(v)
                weights.append(_arrows(c))
        if not vals:
            continue
        wmean = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        out[f] = {
            "value": wmean, "n_clips": len(vals), "arrows": sum(weights),
            "spread": round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0,
        }
    return out


def collect_day(date, sessions=None):
    """Return {view: [clip session dicts]} for practice sessions on `date`."""
    sessions = sessions if sessions is not None else _all_practice_sessions()
    by_view = {}
    for s in sessions:
        if s.get("date") == date:
            by_view.setdefault(s["view"], []).append(s)
    return by_view


def _coach_labels(coaches, hidden_only=False):
    """Names/aliases from a profile's `coaches` block (primary + guest). Each coach can
    contribute up to two labels — name AND alias (e.g. "J. Adams" and "Coach A").
    hidden_only=True restricts to entries marked `hidden_from_app` (see the note
    alongside it in profile.json) — display-only, never touches coaching_log.md,
    coach_voice.md, or session_history/*.json."""
    labels = []

    def add(entry):
        if not isinstance(entry, dict):
            return
        if hidden_only and not entry.get("hidden_from_app"):
            return
        for key in ("name", "alias"):
            label = entry.get(key)
            if label and label not in labels:
                labels.append(label)

    add(coaches.get("primary"))
    for guest in coaches.get("guest") or []:
        add(guest)
    return labels


def _hidden_coach_labels(coaches):
    """Coaches marked `hidden_from_app` — their advice is excluded entirely, not just
    their name (see load_coaching_context's active_focus filtering)."""
    return _coach_labels(coaches, hidden_only=True)


def load_coaching_context(archer_id):
    path = os.path.join(BASE, "archers", archer_id, "profile.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        prof = json.load(f)
    focus = prof.get("active_coaching_focus", {})
    coaches = prof.get("coaches", {})
    hidden = _hidden_coach_labels(coaches)
    items = focus.get("items", [])
    if hidden:
        # Each active-focus item is one coach's homework/cue, usually suffixed
        # "(Coach X ...)" — hiding a coach hides their whole item, not just their name,
        # since the item IS their advice (unlike e.g. a metric's cue text, which stays
        # even if a coach who helped validate it is hidden — see docs/WORKFLOW.md).
        items = [it for it in items if not any(h in it for h in hidden)]
    return {
        "coaches": coaches,
        "feedback_tier": prof.get("feedback_preferences", {}).get("tier", ""),
        "active_focus": items,
        "focus_as_of": focus.get("as_of", ""),
        "coach_voice_text": load_coach_voice(archer_id),
    }


def load_coach_voice(archer_id):
    """Raw text of the archer's coach-voice log, the corpus coach_voice_model trains
    on to flavor synthesize_manual()'s output. Optional — returns '' if the archer
    has no coach_voice.md yet (coach_voice_model treats '' as "not enough data")."""
    path = os.path.join(BASE, "archers", archer_id, "coach_voice.md")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


_CUE_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")


def _cue_on_or_before(item, session_date, focus_as_of):
    """Keep a cue only if it was given on/before the session under review.
    Cues embed a M/D like '(Coach A 5/17)'; year is inferred from the block's
    as_of date (nearest year that makes the cue date <= as_of). Cues with no
    parseable date fall back to comparing against the block's as_of."""
    if not session_date:
        return True
    m = _CUE_DATE.search(item)
    ref = focus_as_of or ""
    if not m:
        # no date on the cue — treat it as dated at the block's as_of
        return (not ref) or ref <= session_date
    mo, dy = int(m.group(1)), int(m.group(2))
    yr = int(ref[:4]) if len(ref) >= 4 and ref[:4].isdigit() else int(session_date[:4])
    cue_date = f"{yr:04d}-{mo:02d}-{dy:02d}"
    if ref and cue_date > ref:  # cue can't be after its own block's as_of; roll back a year
        cue_date = f"{yr-1:04d}-{mo:02d}-{dy:02d}"
    return cue_date <= session_date


# Below this many words of an archer's own logged coach-voice text, feedback_nlg's
# rich mode doesn't have enough material to personalize several sentences without
# becoming repetitive — higher than coach_voice_model's old single-line threshold
# (60) was, since rich mode draws on the chain more heavily per call.
MIN_WORDS_RICH = 150


def _compose_basic(session, findings, active_focus, focus_as_of):
    """The basic-recommendations fallback: a fixed bullet-point template, no
    randomization, no personalization. This is what synthesize_manual() returns
    whenever there isn't enough of the archer's own logged text ("language") to
    personalize with — see its docstring for the dispatch logic."""
    view = session.get("view", "session")
    m = session.get("metrics", {})
    arrows = m.get("arrows_with_metrics") or m.get("arrows_shot") or "?"

    good = [f for f in findings if f["status"] == "in_range"]
    flagged = [f for f in findings if f["status"] in ("below", "above")]

    lines = [
        f"**{view.capitalize()} view — {arrows} shot{'s' if arrows != 1 else ''}**",
        "",
    ]

    if good:
        lines.append("**What's working:**")
        for f in good:
            lines.append(f"- {f['metric']} is solid — {f['cue']}")
        lines.append("")

    if flagged:
        lines.append("**Focus for next session:**")
        for f in flagged:
            direction = "running a little high" if f["status"] == "above" else "running a little low"
            lines.append(f"- {f['metric']} is {direction} — {f['cue']}")
        lines.append("")
    else:
        lines.append("Every scorable metric was in range this session — nothing flagged to work on.")
        lines.append("")

    if active_focus:
        lines.append(f"**Active coaching focus (as of {focus_as_of}):**")
        for item in active_focus:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("_Generated from measured metrics and coach-validated thresholds — "
                  "no AI model was used._")
    return "\n".join(lines)


def synthesize_manual(session, findings, ctx=None):
    """Coaching feedback — no API key, no network call, nothing external, ever.

    Dispatches between two entirely local renderers, both built strictly from the
    rule engine's `findings` (never inventing or restating a number) and the
    archer's own `active_coaching_focus` items (filtered to cues that existed as of
    the session date, via `_cue_on_or_before`):

    - **Enough of the archer's own logged coach-voice text** (archers/<id>/coach_voice.md,
      >= MIN_WORDS_RICH words) -> `feedback_nlg.compose_rich()`: a hand-built
      rule-based NLG engine (sentence templates + phrase banks, not a trained model)
      that writes several varied, personalized sentences, splicing in short fragments
      generated from the archer's own text via `coach_voice_model.MarkovChain`.
    - **Not enough logged text** -> `_compose_basic()`, the fixed bullet-point
      template — "basic recommendations". This is also exactly today's output for
      any archer who hasn't logged much yet, so behavior never regresses or looks
      broken while their own "language" accumulates.
    """
    active_focus = (ctx or {}).get("active_focus")
    focus_as_of = (ctx or {}).get("focus_as_of", "")
    if active_focus:
        session_date = session.get("date")
        if session_date:
            active_focus = [c for c in active_focus
                            if _cue_on_or_before(c, session_date, focus_as_of)]

    coach_voice_text = (ctx or {}).get("coach_voice_text", "")
    if _coach_voice_model.enough_data(coach_voice_text, MIN_WORDS_RICH):
        return _feedback_nlg.compose_rich(session, findings, active_focus, focus_as_of,
                                          coach_voice_text)
    return _compose_basic(session, findings, active_focus, focus_as_of)


def _emit(feedback, title, save_path):
    print("=" * 70 + f"\n{title}\n" + "=" * 70 + "\n")
    print(feedback)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(f"# {title}\n\n{feedback}\n")
        print(f"\nSaved → {save_path}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", help="path to a single session_history/*.json (one view)")
    src.add_argument("--date", help="YYYY-MM-DD: full-day review across all views for that date")
    ap.add_argument("--archer", required=True, help="archer_id, matching archers/<archer_id>/")
    ap.add_argument("--save", action="store_true", help="also write feedback to output/")
    ap.add_argument("--all-distances", action="store_true",
                    help="no-op unless you've customized in_scope() to restrict sessions "
                         "(e.g. to one distance/location) for your own deployment")
    args = ap.parse_args()

    scope_only = not args.all_distances
    ctx = load_coaching_context(args.archer)

    if args.date:
        sessions = _all_practice_sessions(scope_only=scope_only)
        by_view = collect_day(args.date, sessions)
        if not by_view:
            hint = ("" if scope_only is False else
                    " (out-of-scope sessions are skipped by in_scope(), if you've "
                    "customized it — pass --all-distances to override)")
            sys.exit(f"No in-scope practice sessions found for date {args.date}.{hint}")

        order = ["face", "back", "target"]
        blocks = []
        for v in [x for x in order if x in by_view]:
            clips = by_view[v]
            agg = aggregate_view_day(clips, v)
            findings = evaluate(
                v,
                value_of=lambda fld, a=agg: a[fld]["value"] if fld in a else None,
                extra_of=lambda fld, a=agg: a.get(fld),
            )
            if not findings:
                continue
            synthetic_session = {"view": v, "date": args.date,
                                  "metrics": {"arrows_with_metrics": sum(_arrows(c) for c in clips)}}
            blocks.append(synthesize_manual(synthetic_session, findings, ctx))
        if not blocks:
            sys.exit(f"No scorable metrics found for any view on {args.date}.")
        title = f"FULL-DAY COACHING REVIEW — {args.date} ({', '.join(sorted(by_view))})"
        save_path = (os.path.join(BASE, "output", f"coaching_feedback_{args.date}_allviews.md")
                     if args.save else None)
        _emit("\n\n---\n\n".join(blocks), title, save_path)
        return

    spath = args.session if os.path.isabs(args.session) else os.path.join(BASE, args.session)
    if not os.path.exists(spath):
        sys.exit(f"Session file not found: {spath}")
    session = load_session(spath)
    if scope_only and not in_scope(session):
        sys.exit(f"{args.session} is out of scope (location={session.get('location')}, "
                 f"distance={session.get('distance_m')}m) per your customized in_scope(). "
                 "Pass --all-distances to override.")
    findings = detect_faults(session)
    if not findings:
        sys.exit(f"No scorable metrics found for view '{session.get('view')}' in {args.session}. "
                 "(Is this a practice session JSON with a 'metrics' block?)")
    feedback = synthesize_manual(session, findings, ctx)
    title = f"POST-SESSION COACHING FEEDBACK — {session.get('date')} {session.get('view')} view"
    stem = os.path.splitext(os.path.basename(spath))[0]
    save_path = os.path.join(BASE, "output", f"coaching_feedback_{stem}.md") if args.save else None
    _emit(feedback, title, save_path)


if __name__ == "__main__":
    main()

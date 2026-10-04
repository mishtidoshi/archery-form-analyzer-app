#!/usr/bin/env python3
"""
Trend analysis across all archery session history files.

Normalizes three inconsistent field naming schemes:
  - 030926 new format:     draw_elbow_height_avg, shoulder_level_avg, hold_time_s_avg
  - 032426/032726 batch:   draw_elbow_height_mean, shoulder_level_deg_mean, hold_time_s_mean
  - Old manual (030826+):  draw_elbow_height_avg, anchor_consistency_px (mixed)

Usage:
  python3 tools/trend_analysis.py
  python3 tools/trend_analysis.py --history session_history --out output/trends.png
"""

import json
import os
import glob
import sys
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hold_time import HOLD_TIME_DEFINITION as _CURRENT_HOLD_DEF  # noqa: E402
import metric_thresholds as MT   # noqa: E402  (canonical bands — BB9)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ─── Field alias lookup ───────────────────────────────────────────────────────
# canonical_name → [list of possible keys in session JSON metrics dict]
FIELD_ALIASES = {
    # Face view
    "draw_elbow_angle":      ["draw_elbow_angle_avg",       "draw_elbow_angle_deg_mean"],
    "draw_elbow_angle_std":  ["draw_elbow_angle_std",       "draw_elbow_angle_deg_std"],
    "bow_elbow_angle":       ["bow_elbow_angle_avg",        "bow_elbow_angle_deg_mean"],
    "bow_elbow_angle_std":   ["bow_elbow_angle_std",        "bow_elbow_angle_deg_std"],
    "nose_string_gap":       ["nose_string_gap_avg",        "nosestring_gap_mean"],
    "nose_string_gap_std":   ["nose_string_gap_std",        "nosestring_gap_std"],
    "bow_shoulder_elevation": ["bow_shoulder_elevation_avg", "bow_shoulder_elevation_mean"],
    "anchor_x":              ["anchor_x_avg",               "anchor_x_norm_mean"],
    "anchor_y":              ["anchor_y_avg",               "anchor_y_norm_mean"],

    # Target view
    "draw_elbow_height":     ["draw_elbow_height_avg",      "draw_elbow_height_mean"],
    "draw_elbow_height_std": ["draw_elbow_height_std"],
    "shoulder_rotation":     ["shoulder_rotation_avg",      "shoulder_rotation_deg_mean"],
    "shoulder_rotation_std": ["shoulder_rotation_std",      "shoulder_rotation_deg_std"],
    "back_tension":          ["back_tension_avg",           "back_tension_proxy_avg",
                              "back_tension_proxy_mean"],

    # Back view
    "shoulder_level":        ["shoulder_level_avg",         "shoulder_level_deg_mean"],
    "shoulder_level_std":    ["shoulder_level_std",         "shoulder_level_deg_std"],
    "hip_alignment":         ["hip_alignment_avg",          "hip_alignment_deg_mean"],
    "hip_alignment_std":     ["hip_alignment_std",          "hip_alignment_deg_std"],
    "head_lateral_tilt":     ["head_lateral_tilt_avg",      "head_lateral_tilt_deg_mean"],
    "t_draw_angle":          ["t_draw_angle_avg",           "tdraw_angle_deg_mean"],

    # Common
    "hold_time_s":           ["hold_time_s_avg",            "hold_time_s_mean",
                              "hold_time_avg"],
    "hold_time_s_std":       ["hold_time_s_std",            "hold_time_std"],
    "arrows_shot":           ["arrows_shot"],

    # Camera-invariant variants (added 2026-05-16). Per-view normalization:
    #   _sw = shoulder-width  (face, back: shoulder pixel separation)
    #   _th = torso-height    (target: shoulder Y → hip Y)
    # See analyze_face_yolo.py / analyze_back_yolo.py / analyze_target_yolo.py
    # for the per-frame definitions. Older sessions don't carry these fields;
    # callers should check None and fall back to the legacy canonical.
    "bow_shoulder_elevation_sw":  ["bow_shoulder_elevation_sw_avg"],
    "bow_shoulder_elevation_sw_std": ["bow_shoulder_elevation_sw_std"],
    "bow_shoulder_elevation_th":  ["bow_shoulder_elevation_th_avg"],
    "bow_shoulder_elevation_th_std": ["bow_shoulder_elevation_th_std"],
    "anchor_x_sw":           ["anchor_x_sw_avg"],
    "anchor_y_sw":           ["anchor_y_sw_avg"],
    "anchor_y_sw_std":       ["anchor_y_sw_std"],
    "nose_string_gap_sw":    ["nose_string_gap_sw_avg"],
    "draw_elbow_height_th":  ["draw_elbow_height_th_avg"],
    "draw_elbow_height_th_std": ["draw_elbow_height_th_std"],
    "draw_elbow_lateral_th": ["draw_elbow_lateral_th_avg"],
    "draw_elbow_lateral_th_std": ["draw_elbow_lateral_th_std"],
    "shoulder_width_px":     ["shoulder_width_px_avg"],
    "torso_height_px":       ["torso_height_px_avg"],

    # Pre-anchor head drift (face view; ported into analyze_face_yolo.py +
    # backfilled across 18m backyard sessions 2026-07-06). Pixels of nose-Y
    # travel from ~25 frames before the shot to release; positive = chin
    # dropped during the approach to anchor. Only recent face sessions carry it.
    "nose_preanchor_drift":     ["nose_preanchor_drift_avg"],
    "nose_preanchor_drift_std": ["nose_preanchor_drift_std"],
}

# Noisy-data thresholds: if std > threshold, mark session as unreliable
QUALITY_THRESHOLDS = {
    "draw_elbow_angle_std":  25.0,   # degrees
    "shoulder_level_std":    15.0,   # degrees
    "hip_alignment_std":     30.0,   # degrees
    "bow_elbow_angle_std":   25.0,   # degrees
}


def get_field(metrics: dict, canonical: str):
    """Return first non-None float found under any alias for canonical."""
    for alias in FIELD_ALIASES.get(canonical, [canonical]):
        val = metrics.get(alias)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None


def infer_view(data: dict) -> str:
    """Infer view from the 'view' field or video_path string."""
    view = (data.get("view") or "").strip().lower()
    if view in ("face", "back", "target"):
        return view
    # Fallback: parse video path
    vp = (data.get("video_path") or "").lower()
    if any(k in vp for k in ("face", "faceview")):
        return "face"
    if any(k in vp for k in ("rear", "back", "rearview")):
        return "back"
    if any(k in vp for k in ("target", "targetview")):
        return "target"
    return ""


def assess_quality(metrics: dict) -> tuple[str, list[str]]:
    """Return ('good'|'noisy', [reason strings])."""
    noisy_reasons = []
    for std_field, threshold in QUALITY_THRESHOLDS.items():
        val = get_field(metrics, std_field)
        if val is not None and val > threshold:
            noisy_reasons.append(f"{std_field}={val:.1f}° (>{threshold}°)")
    return ("noisy" if noisy_reasons else "good"), noisy_reasons


def load_sessions(history_dir: str) -> list[dict]:
    """Load and normalize all session JSON files."""
    sessions = []
    for fpath in sorted(glob.glob(os.path.join(history_dir, "*.json"))):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        if "date" not in data:
            continue

        try:
            date = datetime.strptime(str(data["date"])[:10], "%Y-%m-%d")
        except ValueError:
            continue

        metrics = data.get("metrics", {})
        view = infer_view(data)
        arrows = int(get_field(metrics, "arrows_shot") or 0)
        quality, noisy_reasons = assess_quality(metrics)

        sessions.append({
            "date":          date,
            "date_str":      data["date"],
            "view":          view,
            "arrows":        arrows,
            "quality":       quality,
            "noisy_reasons": noisy_reasons,
            "fname":         os.path.basename(fpath),
            "metrics":       metrics,
        })

    sessions.sort(key=lambda s: s["date"])
    return sessions


# ─── Plotting helpers ─────────────────────────────────────────────────────────

BLUE   = "#2196F3"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GREEN  = "#4CAF50"
RED    = "#F44336"


def _date_fmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.tick_params(axis="x", rotation=35, labelsize=7)


def plot_metric(ax, sessions, view, canonical, std_canonical=None,
                title=None, ylabel=None,
                hline=None, hline_label=None,
                good_range=None, color=BLUE):
    """
    Scatter+line plot of one metric over time for a given view.
    Clean data in `color`, noisy data as grey ×.
    """
    vsessions = [s for s in sessions if s["view"] == view]
    if not vsessions:
        ax.text(0.5, 0.5, f"No {view} sessions", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
        ax.set_title(title or canonical, fontsize=9, fontweight="bold")
        return

    good_pts, noisy_pts = [], []
    for s in vsessions:
        v = get_field(s["metrics"], canonical)
        if v is None:
            continue
        std = get_field(s["metrics"], std_canonical) if std_canonical else None
        pt = dict(date=s["date"], val=v, std=std, n=s["arrows"], fname=s["fname"])
        if s["quality"] == "good":
            good_pts.append(pt)
        else:
            noisy_pts.append(pt)

    if not good_pts and not noisy_pts:
        ax.text(0.5, 0.5, "No data for this metric", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
        ax.set_title(title or canonical, fontsize=9, fontweight="bold")
        return

    # Shaded good range
    if good_range:
        ax.axhspan(good_range[0], good_range[1], alpha=0.10, color="green",
                   label="ideal range", zorder=1)

    # Reference line
    if hline is not None:
        ax.axhline(hline, color="green", linestyle="--", alpha=0.6, linewidth=1,
                   label=hline_label or f"ideal={hline}", zorder=2)

    # Noisy data (grey ×)
    if noisy_pts:
        ax.scatter([p["date"] for p in noisy_pts],
                   [p["val"]  for p in noisy_pts],
                   s=50, color="gray", marker="x", linewidths=1.5,
                   label="noisy (excluded)", zorder=5)

    # Clean data
    if good_pts:
        dates = [p["date"] for p in good_pts]
        vals  = [p["val"]  for p in good_pts]
        stds  = [p["std"]  for p in good_pts]
        sizes = [max(40, min(180, p["n"] * 14)) for p in good_pts]

        ax.scatter(dates, vals, s=sizes, color=color, zorder=6, label="clean data")
        # Error bars
        err_vals = [s if s is not None else 0 for s in stds]
        if any(e > 0 for e in err_vals):
            ax.errorbar(dates, vals, yerr=err_vals, fmt="none",
                        color=color, alpha=0.45, capsize=3, zorder=5)
        # Trend line
        if len(dates) > 1:
            srt = sorted(zip(dates, vals), key=lambda x: x[0])
            ax.plot([x[0] for x in srt], [x[1] for x in srt],
                    color=color, alpha=0.35, linewidth=1.5, zorder=4)
        # n= annotations
        for p in good_pts:
            ax.annotate(f"n={p['n']}", (p["date"], p["val"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color="#555")

    _date_fmt(ax)
    ax.set_title(title or canonical, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel or "", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, loc="best")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(history_dir: str, output_path: str):
    sessions = load_sessions(history_dir)

    # ── Print load summary ───────────────────────────────────────────────────
    print(f"\nLoaded {len(sessions)} session files from '{history_dir}'")
    print(f"{'View':8s}  {'Total':>5s}  {'Clean':>5s}  {'Noisy':>5s}  {'Arrows':>7s}")
    print("-" * 45)
    for v in ("face", "back", "target", ""):
        vsess = [s for s in sessions if s["view"] == v]
        if not vsess:
            continue
        label = v if v else "(unknown)"
        clean  = sum(1 for s in vsess if s["quality"] == "good")
        noisy  = sum(1 for s in vsess if s["quality"] == "noisy")
        arrows = sum(s["arrows"] for s in vsess)
        print(f"{label:8s}  {len(vsess):>5d}  {clean:>5d}  {noisy:>5d}  {arrows:>7d}")

    # ── Build figure: 5 rows × 3 cols ────────────────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(17, 19))
    fig.suptitle("Archery Form Trend Analysis — Apr 2025 – Jul 2026 (camera-invariant)",
                 fontsize=14, fontweight="bold", y=0.99)

    # Row 0 — Face view
    plot_metric(axes[0][0], sessions, "face",
                "draw_elbow_angle", "draw_elbow_angle_std",
                title="Draw Elbow Angle  [face]",
                ylabel="degrees",
                good_range=MT.good_range("face", "draw_elbow_angle"),
                color=PURPLE)

    plot_metric(axes[0][1], sessions, "face",
                "nose_string_gap", "nose_string_gap_std",
                title="Nose–String Gap  [face]",
                ylabel="normalized\n(negative = squishing string)",
                hline=-0.07, hline_label="coach target (−0.07)",
                good_range=MT.good_range("face", "nose_string_gap"),
                color=PURPLE)

    plot_metric(axes[0][2], sessions, "face",
                "bow_shoulder_elevation_sw", "bow_shoulder_elevation_sw_std",
                title="Bow Shoulder Elevation  [face, sh-width-norm]",
                ylabel="shoulder-widths\n(lower = scapula sunk ✓)",
                hline=0.60, hline_label="dataset baseline (0.60)",
                color=PURPLE)

    # Row 1 — Target view
    plot_metric(axes[1][0], sessions, "target",
                "draw_elbow_height_th", "draw_elbow_height_th_std",
                title="Draw Elbow Height  [target, torso-height-norm]",
                ylabel="torso-heights\n(negative = above shoulder ✓)",
                hline=0, hline_label="shoulder level",
                color=BLUE)

    plot_metric(axes[1][1], sessions, "target",
                "bow_shoulder_elevation_th", "bow_shoulder_elevation_th_std",
                title="Bow Shoulder Elevation  [target, torso-height-norm]",
                ylabel="torso-heights",
                hline=0.40, hline_label="dataset baseline (0.40)",
                color=BLUE)

    plot_metric(axes[1][2], sessions, "target",
                "hold_time_s", "hold_time_s_std",
                title="Hold Time  [target]",
                ylabel="seconds",
                # 1.0-3.0s = coach-stated ideal. The previous (0.5, 1.5) band was
                # fitted to hold times produced by the superseded definitions and
                # is not a coaching target (BB11).
                good_range=MT.good_range("face", "hold_time_s"),
                color=BLUE)

    # Row 2 — Back view
    plot_metric(axes[2][0], sessions, "back",
                "shoulder_level", "shoulder_level_std",
                title="Shoulder Level  [back]",
                ylabel="degrees  (0° = level ✓)",
                hline=0, hline_label="ideal (0°)",
                good_range=MT.good_range("back", "shoulder_level"),
                color=ORANGE)

    plot_metric(axes[2][1], sessions, "back",
                "hip_alignment", "hip_alignment_std",
                title="Hip Alignment  [back]",
                ylabel="degrees  (0° = level ✓)",
                hline=0,
                good_range=MT.good_range("back", "hip_alignment"),
                color=ORANGE)

    plot_metric(axes[2][2], sessions, "back",
                "bow_shoulder_elevation_sw", "bow_shoulder_elevation_sw_std",
                title="Bow Shoulder Elevation  [back, sh-width-norm]",
                ylabel="shoulder-widths\n(lower = scapula sunk ✓)",
                hline=0.60, hline_label="dataset baseline (0.60)",
                color=ORANGE)

    # Row 3 — Cross-view summary panels

    # 3a: Elbow height from target view (canonical, torso-height-normalized)
    ax = axes[3][0]
    target_eh = [(s["date"], get_field(s["metrics"], "draw_elbow_height_th"),
                  get_field(s["metrics"], "draw_elbow_height_th_std"), s["arrows"])
                 for s in sessions
                 if s["view"] == "target" and s["quality"] == "good"
                 and get_field(s["metrics"], "draw_elbow_height_th") is not None]
    if target_eh:
        dates = [p[0] for p in target_eh]
        vals  = [p[1] for p in target_eh]
        stds  = [p[2] if p[2] else 0 for p in target_eh]
        sizes = [max(40, min(180, p[3] * 14)) for p in target_eh]
        ax.scatter(dates, vals, s=sizes, color=BLUE, label="target view", zorder=6)
        ax.errorbar(dates, vals, yerr=stds, fmt="none", color=BLUE, alpha=0.4, capsize=3)
        srt = sorted(zip(dates, vals))
        ax.plot([x[0] for x in srt], [x[1] for x in srt], color=BLUE, alpha=0.35, linewidth=1.5)
    ax.axhline(0, color="green", linestyle="--", alpha=0.6, linewidth=1, label="shoulder level")
    _date_fmt(ax)
    ax.set_title("Draw Elbow Height over Time  [target, torso-height-norm]",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("torso-heights", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)

    # 3b: Session volume (total arrows per day)
    ax = axes[3][1]
    date_arrows: dict = {}
    for s in sessions:
        dk = s["date"]
        date_arrows[dk] = date_arrows.get(dk, 0) + s["arrows"]
    sorted_dates = sorted(date_arrows)
    totals = [date_arrows[d] for d in sorted_dates]
    bar_colors = [GREEN if t >= 18 else BLUE if t >= 10 else ORANGE if t >= 5 else RED
                  for t in totals]
    ax.bar(range(len(sorted_dates)), totals, color=bar_colors)
    ax.set_xticks(range(len(sorted_dates)))
    ax.set_xticklabels([d.strftime("%m/%d") for d in sorted_dates], rotation=45, fontsize=8)
    ax.set_title("Arrows Tracked Per Day  (all views)", fontsize=9, fontweight="bold")
    ax.set_ylabel("arrow count", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    for i, t in enumerate(totals):
        ax.text(i, t + 0.3, str(t), ha="center", fontsize=7, color="#333")

    # 3c: Hold time from all three views on one plot
    ax = axes[3][2]
    for view, color, marker, label in [
        ("face",   PURPLE, "^", "face"),
        ("target", BLUE,   "o", "target"),
        ("back",   ORANGE, "s", "back"),
    ]:
        pts = [(s["date"], get_field(s["metrics"], "hold_time_s"))
               for s in sessions
               if s["view"] == view and s["quality"] == "good"
               and get_field(s["metrics"], "hold_time_s") is not None]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       color=color, marker=marker, s=60, label=label, zorder=5)
            srt = sorted(pts)
            ax.plot([p[0] for p in srt], [p[1] for p in srt],
                    color=color, alpha=0.3, linewidth=1.2)
    ax.axhspan(0.5, 1.5, alpha=0.08, color="green", label="ideal zone")
    _date_fmt(ax)
    ax.set_title("Hold Time — All Views", fontsize=9, fontweight="bold")
    ax.set_ylabel("seconds", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)

    # Row 4 — Face: pre-anchor head drift (chin drop during draw approach).
    # Only recent face sessions carry it (backfilled 18m backyard set onward).
    plot_metric(axes[4][0], sessions, "face",
                "nose_preanchor_drift", "nose_preanchor_drift_std",
                title="Nose Pre-Anchor Drift  [face]  (head→string = high arrows)",
                ylabel="pixels\n(positive = head pushing down into string)",
                hline=0, hline_label="no drift (0px)",
                good_range=MT.good_range("face", "nose_preanchor_drift") or (-2, 2),
                color=PURPLE)
    # Remaining row-4 cells unused for now.
    axes[4][1].axis("off")
    axes[4][2].axis("off")

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {output_path}")

    # ── Console trend summary ─────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("TREND SUMMARY")
    print("═" * 62)

    def _hold_time_censored(metrics, val, std):
        """True when a hold_time value must not be flagged or trended.

        The corpus holds values from three formulations (BB1 distance-scan ceiling,
        BB11 X-only settle-truncation, current velocity definition) which are not
        comparable. Records are stamped with `hold_time_s_definition`; anything not
        carrying the current stamp is excluded. Censored shots are excluded too,
        since their value is a floor. The saturation-signature check remains as a
        fallback for any record the stamping pass missed."""
        defn = metrics.get("hold_time_s_definition")
        if defn is not None and defn != _CURRENT_HOLD_DEF:
            return True
        n_cens = metrics.get("hold_time_s_n_censored")
        if n_cens:
            return True
        if n_cens == 0 and defn == _CURRENT_HOLD_DEF:
            return False           # explicitly clean, current definition
        if std == 0.0 and val is not None:
            return abs(val - 0.967) < 1e-6 or abs(val - 1.0) < 1e-6
        return False

    def summarize(view, metric, std_metric=None,
                  good_range=None, label=None, unit=""):
        pts = [(s["date"], get_field(s["metrics"], metric),
                get_field(s["metrics"], std_metric) if std_metric else None,
                s["arrows"], s["metrics"])
               for s in sessions
               if s["view"] == view
               and s["quality"] == "good"
               and get_field(s["metrics"], metric) is not None]
        if not pts:
            return
        pts.sort(key=lambda p: p[0])
        display = label or f"{view.upper()} | {metric}"
        print(f"\n  {display}")
        vals = []
        n_censored = 0
        for d, v, std, n, met in pts:
            std_str = f" ±{std:.3f}" if std is not None else ""
            censored = metric == "hold_time_s" and _hold_time_censored(met, v, std)
            if censored:
                # Do not flag or trend a value from a superseded definition or a
                # censored measurement.
                n_censored += 1
                defn = met.get("hold_time_s_definition")
                why = ("superseded definition" if defn and defn != _CURRENT_HOLD_DEF
                       else "scan window, not a measured hold")
                print(f"    {d.strftime('%m/%d')}: ≥{v:.3f}{unit}{std_str}  (n={n})  "
                      f"[excluded — {why}]")
                continue
            flag = ""
            if good_range is not None:
                flag = " ✓" if good_range[0] <= v <= good_range[1] else " ✗"
            print(f"    {d.strftime('%m/%d')}: {v:+.3f}{unit}{std_str}  (n={n}){flag}")
            vals.append(v)
        if n_censored:
            print(f"    ⚠ {n_censored} session(s) excluded from the trend "
                  f"(censored or superseded hold-time definition — see BB11).")
        if len(vals) > 1:
            delta = vals[-1] - vals[0]
            if abs(delta) < 0.002:
                trend = "stable"
            elif delta < 0:
                # Depends on the metric whether down = good or bad
                trend = "↓"
            else:
                trend = "↑"
            print(f"    → Δ from first to last session: {delta:+.4f}{unit}  ({trend})")

    summarize("target", "draw_elbow_height_th", "draw_elbow_height_th_std",
              label="TARGET | Draw Elbow Height  (torso-heights, negative = above shoulder ✓)")

    summarize("face", "draw_elbow_angle", "draw_elbow_angle_std",
              good_range=MT.good_range("face", "draw_elbow_angle"), unit="°",
              label="FACE   | Draw Elbow Angle")

    summarize("face", "nose_string_gap", "nose_string_gap_std",
              good_range=MT.good_range("face", "nose_string_gap"),
              label="FACE   | Nose–String Gap")

    summarize("face", "hold_time_s", "hold_time_s_std",
              good_range=MT.good_range("face", "hold_time_s"), unit="s",
              label="FACE   | Hold Time")

    summarize("face", "nose_preanchor_drift", "nose_preanchor_drift_std",
              good_range=MT.good_range("face", "nose_preanchor_drift") or (-2, 2), unit="px",
              label="FACE   | Nose Pre-Anchor Drift (positive = head into string = high arrows)")

    summarize("target", "hold_time_s", "hold_time_s_std",
              good_range=MT.good_range("face", "hold_time_s"), unit="s",
              label="TARGET | Hold Time")

    summarize("back", "shoulder_level", "shoulder_level_std",
              good_range=MT.good_range("back", "shoulder_level"), unit="°",
              label="BACK   | Shoulder Level (0° = level ✓)")

    summarize("back", "hip_alignment", "hip_alignment_std",
              good_range=MT.good_range("back", "hip_alignment"), unit="°",
              label="BACK   | Hip Alignment")

    # Camera-invariant metrics (deployed 2026-05-16, backfilled 2026-05-24).
    # All practice sessions with verified frames now carry these fields.
    print("\n" + "─" * 62)
    print("  CAMERA-INVARIANT METRICS (full history)")
    print("─" * 62)
    summarize("face", "bow_shoulder_elevation_sw", "bow_shoulder_elevation_sw_std",
              label="FACE   | Bow Shoulder Elevation (sh-widths) — baseline ~0.60")
    summarize("face", "anchor_y_sw", "anchor_y_sw_std",
              label="FACE   | Anchor Y (sh-widths) — vertical drift across sessions")
    summarize("back", "bow_shoulder_elevation_sw", "bow_shoulder_elevation_sw_std",
              label="BACK   | Bow Shoulder Elevation (sh-widths) — baseline ~0.60")
    summarize("target", "draw_elbow_height_th", "draw_elbow_height_th_std",
              label="TARGET | Draw Elbow Height (torso-heights) — negative = above shoulder ✓")
    summarize("target", "bow_shoulder_elevation_th", "bow_shoulder_elevation_th_std",
              label="TARGET | Bow Shoulder Elevation (torso-heights)")

    # Noisy sessions that were excluded
    noisy = [s for s in sessions if s["quality"] == "noisy"]
    if noisy:
        print(f"\n  ⚠  {len(noisy)} session(s) flagged as noisy/unreliable"
              f" (high landmark SD — excluded from plots):")
        for s in noisy:
            print(f"    {s['date_str']}  {s['view']:6s}  {s['fname']}")
            for r in s["noisy_reasons"]:
                print(f"             {r}")

    print("\n" + "═" * 62)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Archery session trend analysis")
    ap.add_argument("--history", default="session_history",
                    help="Directory with session JSON files (default: session_history)")
    ap.add_argument("--out", default="output/session_trends.png",
                    help="Output chart path (default: output/session_trends.png)")
    args = ap.parse_args()
    run(args.history, args.out)

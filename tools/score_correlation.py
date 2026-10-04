#!/usr/bin/env python3
"""
Score Correlation Analysis

Analyzes which form factors most strongly predict arrow score, and surfaces
tournament performance patterns:
  - Score by position in end (1st / 2nd / 3rd arrow)
  - First-end warmup penalty vs mid-session performance
  - Post-high-score overcompensation (next shot after 9 or 10)

Usage:
  python3 tools/score_correlation.py                      # all data
  python3 tools/score_correlation.py --date 2026-03-24    # one session
  python3 tools/score_correlation.py --arrows-per-end 6   # outdoor (6/end)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob
import argparse


# ── Data loading ─────────────────────────────────────────────────────────────

def load_arrow_data(session_date=None):
    """Load all arrow data from session_history/arrows/, optionally filtered by date."""
    arrows = []
    for filepath in sorted(glob.glob("session_history/arrows/*.json")):
        with open(filepath) as f:
            data = json.load(f)
        if session_date and data.get('date') != session_date:
            continue
        if data.get('score') is not None:
            arrows.append(data)
    return arrows


def derive_context(arrows, arrows_per_end=3):
    """
    Add end_number, arrow_in_end, and prev_score to each arrow dict in-place.
    Derives end/position from the stored arrow_number field and arrows_per_end.
    prev_score is the score on the immediately preceding arrow (same session,
    sorted by date + arrow_number).
    """
    arrows.sort(key=lambda a: (a.get('date', ''), a.get('arrow_number', 0)))
    for i, arrow in enumerate(arrows):
        n = arrow.get('arrow_number', i + 1)
        arrow['end_number']   = (n - 1) // arrows_per_end + 1
        arrow['arrow_in_end'] = (n - 1) % arrows_per_end + 1
        arrow['prev_score']   = arrows[i - 1]['score'] if i > 0 else None
    return arrows


# ── Form-metric correlation (original analysis) ───────────────────────────────

def calculate_correlations(arrows):
    """Calculate Pearson correlation between each form metric and score."""
    if len(arrows) < 3:
        print(f"  Need at least 3 arrows with scores (have {len(arrows)})")
        return None, None

    scores = [a['score'] for a in arrows]

    metric_defs = {
        'Elbow Height':      'elbow_height',
        'Draw Elbow Angle':  'draw_elbow_angle',
        'Bow Elbow Angle':   'bow_elbow_angle',
        'Anchor X':          'anchor_x',
        'Anchor Y':          'anchor_y',
        'Back Tension':      'back_tension',
    }

    correlations = {}
    for label, key in metric_defs.items():
        values = [a['metrics'].get(key) for a in arrows]
        # Drop any arrow where this metric is None
        pairs = [(v, s) for v, s in zip(values, scores) if v is not None]
        if len(pairs) >= 3 and len(set(v for v, _ in pairs)) > 1:
            vs, ss = zip(*pairs)
            corr, p_val = pearsonr(vs, ss)
            correlations[label] = {
                'correlation': corr,
                'p_value':     p_val,
                'values':      list(vs),
                'scores':      list(ss),
            }

    return correlations, scores


def print_correlation_report(correlations, scores, arrows):
    """Print the form-metric vs score correlation table."""
    print("\n" + "=" * 65)
    print("  FORM-METRIC CORRELATION WITH SCORE")
    print("=" * 65)
    print(f"  Arrows analyzed : {len(arrows)}")
    print(f"  Score range     : {min(scores)} – {max(scores)}")
    print(f"  Average score   : {np.mean(scores):.2f}\n")

    sorted_metrics = sorted(correlations.items(),
                            key=lambda x: abs(x[1]['correlation']),
                            reverse=True)

    print(f"  {'Metric':<22} {'r':>7}  {'Strength':<14}  p-value")
    print("  " + "-" * 56)
    for name, data in sorted_metrics:
        corr  = data['correlation']
        p_val = data['p_value']
        ac    = abs(corr)
        if ac > 0.7:
            strength = "STRONG ★★★"
        elif ac > 0.4:
            strength = "moderate ★★"
        elif ac > 0.2:
            strength = "weak ★"
        else:
            strength = "minimal"
        sig = " *" if p_val < 0.05 else ""
        print(f"  {name:<22} {corr:>+7.3f}  {strength:<14}  {p_val:.4f}{sig}")

    # Narrative for top predictors
    print()
    for name, data in sorted_metrics[:3]:
        corr = data['correlation']
        if abs(corr) < 0.2:
            break
        direction = "Higher" if corr > 0 else "Lower"
        print(f"  {name}: {direction} values → higher scores  (r = {corr:+.3f})")
        if abs(corr) > 0.7:
            print(f"    → Strong predictor. Prioritize optimizing this metric.")


# ── Tournament-pattern analysis ───────────────────────────────────────────────

def analyze_position_patterns(arrows, arrows_per_end=3):
    """
    Score by position in end (1st / 2nd / 3rd arrow).
    Flags the 3rd-arrow drop pattern observed across 3 tournaments (2024–2026).
    """
    by_pos = {}
    for a in arrows:
        pos = a.get('arrow_in_end', 1)
        by_pos.setdefault(pos, []).append(a['score'])

    print("\n  ARROW POSITION IN END")
    print("  " + "-" * 42)
    print(f"  {'Position':<14} {'n':>4}  {'Avg':>6}  {'Std':>6}  {'Min–Max'}")
    print("  " + "-" * 42)
    for pos in sorted(by_pos):
        ss = by_pos[pos]
        sfx = {1: "st", 2: "nd", 3: "rd"}.get(pos, "th")
        label = f"{pos}{sfx} arrow"
        print(f"  {label:<14} {len(ss):>4}  {np.mean(ss):>6.2f}  "
              f"{np.std(ss):>6.2f}  {min(ss)}–{max(ss)}")

    # Flag 3rd-arrow pattern
    if all(p in by_pos for p in range(1, arrows_per_end + 1)):
        avg1 = np.mean(by_pos[1])
        avg_last = np.mean(by_pos[arrows_per_end])
        if avg_last < avg1 - 0.3:
            print(f"\n  ⚠  Last arrow in end averages {avg1 - avg_last:.2f} pts lower than 1st.")
            print("     Tournament pattern: overthinking after two good arrows.")
            print("     Fix: treat every arrow as arrow #1 — same process, same focus.")
        else:
            print(f"\n  ✓  No significant last-arrow drop  "
                  f"(1st avg {avg1:.2f}  vs  last avg {avg_last:.2f}).")
    return by_pos


def analyze_warmup_penalty(arrows):
    """
    First end vs rest of session.
    Flags the first-end nervousness pattern observed in every major tournament.
    """
    end1 = [a['score'] for a in arrows if a.get('end_number') == 1]
    rest = [a['score'] for a in arrows if a.get('end_number', 1) > 1]

    print("\n  WARM-UP END vs MID-SESSION")
    print("  " + "-" * 42)
    if end1:
        print(f"  End 1 avg   : {np.mean(end1):.2f}  (n={len(end1)})")
    else:
        print("  End 1       : no data")
    if rest:
        print(f"  Ends 2+ avg : {np.mean(rest):.2f}  (n={len(rest)})")
    else:
        print("  Ends 2+     : no data")

    if end1 and rest:
        penalty = np.mean(end1) - np.mean(rest)
        if penalty < -0.3:
            print(f"\n  ⚠  First end runs {abs(penalty):.2f} pts below mid-session average.")
            print("     Tournament pattern: nervousness / cold muscles in opening ends.")
            print("     Fix: thorough warm-up routine; focus on form process in end 1,")
            print("          not the score.")
        else:
            print(f"\n  ✓  First-end warmup penalty minimal ({penalty:+.2f} pts).")
    return end1, rest


def analyze_post_high_score(arrows):
    """
    Next arrow after a 9 or 10 vs next arrow after 6/7/8.
    Flags the post-10 overcompensation pattern from Nationals Indoor 2024.
    """
    post_high  = [a['score'] for a in arrows
                  if a.get('prev_score') in (9, 10)]
    post_other = [a['score'] for a in arrows
                  if a.get('prev_score') is not None
                  and a.get('prev_score') not in (9, 10)]

    print("\n  POST-HIGH-SCORE EFFECT  (next arrow after 9 or 10)")
    print("  " + "-" * 42)
    if post_high:
        print(f"  After 9/10  : avg {np.mean(post_high):.2f}  (n={len(post_high)})")
    else:
        print("  After 9/10  : no data")
    if post_other:
        print(f"  After ≤8    : avg {np.mean(post_other):.2f}  (n={len(post_other)})")
    else:
        print("  After ≤8    : no data")

    if post_high and post_other:
        delta = np.mean(post_high) - np.mean(post_other)
        if delta < -0.3:
            print(f"\n  ⚠  Arrows after a 9/10 average {abs(delta):.2f} pts lower.")
            print("     Tournament pattern: pressure/overcompensation after a good shot.")
            print("     Fix: identical process every arrow; resist urge to grip harder")
            print("          or expand faster after a high score.")
        else:
            print(f"\n  ✓  No significant drop after high scores ({delta:+.2f} pts).")
    elif post_high:
        print("\n  (Not enough non-high-score arrows to compare.)")
    else:
        print("\n  (Need arrows with known previous-arrow scores.)")
        print("  Tip: save arrows in session order so prev_score can be derived.")
    return post_high, post_other


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_correlations(correlations, scores, arrows):
    """Scatter plots: each form metric vs score."""
    n_metrics = len(correlations)
    if n_metrics == 0:
        return
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'Form Metrics vs Score  ({len(arrows)} arrows)',
                 fontsize=16, fontweight='bold', color='white')

    axes_flat = axes.flatten() if n_metrics > 1 else [axes]
    sorted_metrics = sorted(correlations.items(),
                            key=lambda x: abs(x[1]['correlation']),
                            reverse=True)

    for idx, (name, data) in enumerate(sorted_metrics):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

        vs, ss = data['values'], data['scores']
        corr   = data['correlation']
        ax.scatter(vs, ss, s=90, alpha=0.7, c='#60a5fa', edgecolors='white', linewidths=1.5)
        z = np.polyfit(vs, ss, 1)
        x_line = np.linspace(min(vs), max(vs), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), '--', color='#00ff88', linewidth=2, alpha=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel('Score')
        ax.set_title(f'{name}\nr = {corr:+.3f}', fontweight='bold')
        ax.grid(alpha=0.2)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    out = 'score_correlation_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"\n  ✓ Form correlation chart saved: {out}")
    plt.close()


def plot_pattern_analysis(arrows, by_pos, end1_scores, rest_scores,
                          post_high, post_other):
    """
    3-panel chart visualising the three tournament performance patterns:
      Left   — score by position in end (box plots)
      Center — score by end number (bar chart, warmup penalty)
      Right  — post-high-score effect (bar chart)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Tournament Performance Patterns', fontsize=16,
                 fontweight='bold', color='white')

    DARK  = '#16213e'
    WHITE = 'white'
    BLUE  = '#60a5fa'
    GREEN = '#00ff88'
    RED   = '#ff6b6b'
    YELL  = '#ffd700'

    def style(ax):
        ax.set_facecolor(DARK)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        ax.tick_params(colors=WHITE)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        ax.grid(axis='y', alpha=0.2)

    # ── Panel 1: score by position in end ────────────────────────────────
    ax = axes[0]
    style(ax)
    positions = sorted(by_pos)
    data_for_bp = [by_pos[p] for p in positions]
    labels = [f"{p}{'st' if p==1 else 'nd' if p==2 else 'rd' if p==3 else 'th'}" for p in positions]
    bp = ax.boxplot(data_for_bp, labels=labels, patch_artist=True,
                    medianprops=dict(color=WHITE, linewidth=2))
    colors = [GREEN, BLUE, YELL, RED]
    for patch, col in zip(bp['boxes'], colors[:len(positions)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    avgs = [np.mean(by_pos[p]) for p in positions]
    ax.plot(range(1, len(positions) + 1), avgs, 'o--',
            color=WHITE, linewidth=1.5, markersize=8, label='Mean')
    ax.set_title('Score by Arrow Position in End')
    ax.set_xlabel('Arrow in End')
    ax.set_ylabel('Score')
    ax.legend(labelcolor=WHITE, facecolor=DARK, fontsize=9)

    # ── Panel 2: warmup penalty (score by end number) ────────────────────
    ax = axes[1]
    style(ax)
    end_nums  = sorted(set(a.get('end_number', 1) for a in arrows))
    end_avgs  = [np.mean([a['score'] for a in arrows if a.get('end_number') == e])
                 for e in end_nums]
    end_cols  = [RED if e == 1 else BLUE for e in end_nums]
    bars = ax.bar([str(e) for e in end_nums], end_avgs, color=end_cols, alpha=0.8, edgecolor='white')
    # Reference line = overall mean
    overall_mean = np.mean([a['score'] for a in arrows])
    ax.axhline(overall_mean, color=YELL, linestyle='--', linewidth=1.5,
               label=f'Overall avg {overall_mean:.2f}')
    ax.set_title('Score by End Number\n(red = end 1 warm-up)')
    ax.set_xlabel('End Number')
    ax.set_ylabel('Avg Score')
    ax.legend(labelcolor=WHITE, facecolor=DARK, fontsize=9)

    # ── Panel 3: post-high-score effect ──────────────────────────────────
    ax = axes[2]
    style(ax)
    groups  = []
    heights = []
    cols    = []
    if post_high:
        groups.append(f'After 9/10\n(n={len(post_high)})')
        heights.append(np.mean(post_high))
        cols.append(YELL)
    if post_other:
        groups.append(f'After ≤8\n(n={len(post_other)})')
        heights.append(np.mean(post_other))
        cols.append(BLUE)
    if groups:
        ax.bar(groups, heights, color=cols, alpha=0.8, edgecolor='white')
        ax.axhline(np.mean([a['score'] for a in arrows]),
                   color=GREEN, linestyle='--', linewidth=1.5, label='Overall avg')
        ax.legend(labelcolor=WHITE, facecolor=DARK, fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Not enough data\n(need sequential arrows)',
                ha='center', va='center', transform=ax.transAxes, color='#888')
    ax.set_title('Post-High-Score Effect\n(overcompensation check)')
    ax.set_ylabel('Avg Score')

    plt.tight_layout()
    out = 'tournament_pattern_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  ✓ Tournament pattern chart saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score correlation and tournament pattern analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", type=str,
                        help="Analyze a specific session date (YYYY-MM-DD)")
    parser.add_argument("--arrows-per-end", type=int, default=3,
                        metavar="N",
                        help="Arrows per end (3 = indoor/default, 6 = outdoor)")
    parser.add_argument("--min-arrows", type=int, default=3,
                        help="Minimum arrows required for analysis (default: 3)")
    args = parser.parse_args()

    arrows = load_arrow_data(args.date)

    if len(arrows) < args.min_arrows:
        print(f"\n  Need at least {args.min_arrows} arrows with scores "
              f"(have {len(arrows)}).")
        print("\n  To add arrows:")
        print("    python3 tools/analyze_single_arrow.py arrow.MOV --score 10 --save")
        return

    # Derive end / position / previous-score context
    arrows = derive_context(arrows, args.arrows_per_end)

    print("\n" + "=" * 65)
    print("  SCORE ANALYSIS")
    print("=" * 65)
    print(f"  Arrows          : {len(arrows)}")
    print(f"  Arrows per end  : {args.arrows_per_end}  "
          f"({'indoor' if args.arrows_per_end == 3 else 'outdoor' if args.arrows_per_end == 6 else 'custom'})")
    if args.date:
        print(f"  Session         : {args.date}")

    # ── Form-metric correlations ──────────────────────────────────────────
    correlations, scores = calculate_correlations(arrows)
    if correlations:
        print_correlation_report(correlations, scores, arrows)
        plot_correlations(correlations, scores, arrows)

    # ── Tournament patterns ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TOURNAMENT PERFORMANCE PATTERNS")
    print("=" * 65)

    by_pos  = analyze_position_patterns(arrows, args.arrows_per_end)
    e1, rst = analyze_warmup_penalty(arrows)
    ph, po  = analyze_post_high_score(arrows)

    # Chart
    plot_pattern_analysis(arrows, by_pos, e1, rst, ph, po)

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()

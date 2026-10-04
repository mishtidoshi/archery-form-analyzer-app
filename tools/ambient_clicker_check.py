#!/usr/bin/env python3
"""
Ambient clicker-strength check (T2.2 lavalier deployment aid).

Two jobs, same measurement:

  1. CHOOSE THE LAVALIER VIEW. Record one identical end on the rear-view and
     target-view phones using their AMBIENT mics (no lavalier yet), then run this
     on both clips. Mount the lavalier on the view with the WEAKER clicker — that
     is where the ambient mic is failing and the mic buys the most.

  2. VALIDATE THE MIC. After mounting, record the same view with the lavalier and
     compare against its ambient baseline. The clicker peak should be several
     times louder (the audio design expects ~5–10×: ~0.3–0.5 vs ~0.05–0.10).

It reuses the exact audio front-end the shot detector uses (Butterworth 3 kHz
high-pass → 1 ms RMS onset → peak pick), so the strengths reported here are the
same numbers detection keys off of.

Usage:
    # pick the view (weaker clicker gets the mic):
    python3 tools/ambient_clicker_check.py rear.MOV target.MOV --labels rear target

    # validate the mic against ambient baseline on the same view:
    python3 tools/ambient_clicker_check.py target_ambient.MOV target_lavalier.MOV \\
        --labels ambient lavalier
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audio_visual_shot_detector import (
    _extract_wav_mono, _read_wav, _highpass_filter, _onset_strength, _pick_peaks,
)

# Match the detector's candidate-picking defaults closely enough for a fair read.
THRESHOLD_PCT = 0.15
MIN_GAP_S = 2.0


def clip_metrics(video_path, top_k=6):
    """Return clicker-loudness metrics for one clip, or None if audio unreadable."""
    wav = _extract_wav_mono(video_path)
    if not wav:
        return None
    data, sr = _read_wav(wav)
    try:
        os.remove(wav)
    except OSError:
        pass
    data = _highpass_filter(data, sr, cutoff_hz=3000.0)
    onset, sps = _onset_strength(data, sr, window_ms=1)
    if onset.size == 0 or onset.max() <= 0:
        return {"peak": 0.0, "top_k_mean": 0.0, "n_peaks": 0, "dur_s": 0.0}
    peaks = _pick_peaks(onset, sps, THRESHOLD_PCT, MIN_GAP_S)
    strengths = sorted((s for _, s in peaks), reverse=True)
    top = strengths[:top_k]
    return {
        "peak": float(onset.max()),
        "top_k_mean": float(np.mean(top)) if top else 0.0,
        "n_peaks": len(peaks),
        "dur_s": float(onset.size * sps),
        "top_k": top_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="video clips to compare")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="labels matching the clips (default: file names)")
    ap.add_argument("--top-k", type=int, default=6,
                    help="peaks to average (≈ arrows in the test end; default 6)")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(c) for c in args.clips]
    if len(labels) != len(args.clips):
        sys.exit("--labels count must match the number of clips")

    rows = []
    for clip, label in zip(args.clips, labels):
        path = clip if os.path.isabs(clip) else os.path.join(os.getcwd(), clip)
        if not os.path.exists(path):
            print(f"  ! not found: {clip}")
            continue
        m = clip_metrics(path, top_k=args.top_k)
        if m is None:
            print(f"  ! no audio: {clip}")
            continue
        rows.append((label, m))

    if not rows:
        sys.exit("No readable clips.")

    print(f"\n{'view/label':16s} {'peak':>8s} {'top-'+str(args.top_k)+' mean':>12s} "
          f"{'#peaks':>7s} {'dur(s)':>8s}")
    print("-" * 56)
    for label, m in rows:
        print(f"{label:16s} {m['peak']:>8.3f} {m['top_k_mean']:>12.3f} "
              f"{m['n_peaks']:>7d} {m['dur_s']:>8.1f}")

    # Recommendation, keyed on top-k mean (a full end is more robust than one peak).
    if len(rows) >= 2:
        ranked = sorted(rows, key=lambda r: r[1]["top_k_mean"])
        weakest, strongest = ranked[0], ranked[-1]
        lo = weakest[1]["top_k_mean"] or 1e-9
        ratio = strongest[1]["top_k_mean"] / lo
        print()
        print(f"Weakest clicker capture: '{weakest[0]}' "
              f"(top-{args.top_k} mean {weakest[1]['top_k_mean']:.3f}).")
        print(f"Strongest: '{strongest[0]}' "
              f"({strongest[1]['top_k_mean']:.3f}) — {ratio:.1f}× louder.")
        print()
        print("→ Choosing a view: mount the lavalier on the WEAKEST ambient view "
              f"('{weakest[0]}') — that is where detection is most fragile.")
        print("→ Validating the mic: the lavalier clip should be the STRONGEST here, "
              "ideally ~5–10× its own ambient baseline.")


if __name__ == "__main__":
    main()

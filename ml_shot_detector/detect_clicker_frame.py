#!/usr/bin/env python3
"""
Detect the clicker release frame in a single-arrow archery video.

Usage:
  python3 scripts/detect_clicker_frame.py path/to/video.MOV
  python3 scripts/detect_clicker_frame.py path/to/video.MOV --preview
  python3 scripts/detect_clicker_frame.py path/to/video.MOV --min-fraction 0.15 --max-fraction 0.80
"""
import argparse
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ml_shot_detector.clicker_detector import (
    detect_clicker_frame,
    _extract_wav,
    _read_wav_mono,
    _onset_strength,
)


def preview_waveform(video_path: str, detected_frame: int = None,
                     min_fraction: float = 0.10, max_fraction: float = 0.85):
    """Show an onset-strength waveform with the detected clicker marked."""
    import tempfile
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / fps

    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        _extract_wav(video_path, tmp_wav)
        data, sr = _read_wav_mono(tmp_wav)
    finally:
        import os as _os
        if _os.path.exists(tmp_wav):
            _os.unlink(tmp_wav)

    import numpy as np
    t_audio = np.linspace(0, len(data) / sr, len(data))
    onset   = _onset_strength(data, sr)
    t_onset = np.arange(len(onset)) * 0.01   # 10 ms per sample

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    peak_norm = float(np.max(np.abs(data))) + 1e-6
    ax1.plot(t_audio, data / peak_norm, lw=0.4, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Amplitude (norm)")
    ax1.set_title(f"Waveform — {os.path.basename(video_path)}")

    ax2.plot(t_onset, onset, lw=0.8, color="darkorange")
    ax2.set_ylabel("Onset strength")
    ax2.set_xlabel("Time (s)")

    if detected_frame is not None:
        t_det = detected_frame / fps
        for ax in (ax1, ax2):
            ax.axvline(t_det, color="crimson", lw=1.8,
                       label=f"detected: frame {detected_frame} ({t_det:.2f}s)")
        ax1.legend(fontsize=9)

    # Shade excluded regions
    t_min = min_fraction * duration_s
    t_max = max_fraction * duration_s
    for ax in (ax1, ax2):
        ax.axvspan(0,       t_min,      alpha=0.10, color="gray")
        ax.axvspan(t_max,   duration_s, alpha=0.10, color="gray")
        ax.axvline(t_min, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(t_max, color="gray", lw=0.8, ls="--", alpha=0.6)

    ax2.text(t_min + 0.05, ax2.get_ylim()[1] * 0.9, "valid window →",
             color="gray", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Detect clicker release frame from archery video audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--preview", action="store_true",
                        help="Show waveform + onset plot with detected frame")
    parser.add_argument("--min-fraction", type=float, default=0.10,
                        help="Skip first N%% of video (default: 0.10)")
    parser.add_argument("--max-fraction", type=float, default=0.85,
                        help="Skip after N%% of video (default: 0.85)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Peak threshold as fraction of window max (default: 0.35)")
    args = parser.parse_args()

    frame, info = detect_clicker_frame(
        args.video,
        min_fraction=args.min_fraction,
        max_fraction=args.max_fraction,
        peak_threshold=args.threshold,
    )

    if frame is None:
        print(f"FAILED: {info}")
        sys.exit(1)

    print(f"Detected: {info}")

    if args.preview:
        preview_waveform(
            args.video,
            detected_frame=frame,
            min_fraction=args.min_fraction,
            max_fraction=args.max_fraction,
        )


if __name__ == "__main__":
    main()

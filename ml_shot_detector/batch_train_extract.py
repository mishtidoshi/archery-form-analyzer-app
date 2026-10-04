#!/usr/bin/env python3
"""
Batch-extract ML training data from a directory of single-arrow videos.

For each video:
  1. Detect the clicker sound to find the release frame (audio-based).
  2. Fall back to peak wrist velocity if audio detection fails.
  3. Run the standard training-data extraction pipeline.

Usage:
  # Process face-view videos:
  python3 scripts/batch_train_extract.py \\
      --dir data/march_2026/032226/face --view face

  # Process target-view videos:
  python3 scripts/batch_train_extract.py \\
      --dir data/march_2026/032226/target --view target

  # Dry run — detect frames only, skip pose estimation:
  python3 scripts/batch_train_extract.py \\
      --dir data/march_2026/032226/face --view face --dry-run

  # Then train:
  python3 scripts/train_model.py
  python3 scripts/evaluate_model.py
"""
import argparse
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ml_shot_detector.clicker_detector import detect_clicker_frame
from ml_shot_detector.labeler import extract_training_data

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".m4v", ".MOV", ".MP4", ".AVI", ".M4V"}


def process_directory(
    video_dir: str,
    view: str,
    hand: str,
    output_dir: str,
    dry_run: bool,
    min_fraction: float,
    max_fraction: float,
    peak_threshold: float,
):
    videos = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1] in VIDEO_EXTS
    ])
    if not videos:
        print(f"No videos found in {video_dir}")
        return 0, 0

    print(f"\n{'='*65}")
    print(f"  {len(videos)} videos  |  view={view}  hand={hand}  dir={video_dir}")
    if dry_run:
        print("  DRY RUN — frame detection only, no pose estimation")
    print(f"{'='*65}\n")

    rows    = []
    n_ok    = 0
    n_fail  = 0

    for idx, vpath in enumerate(videos, 1):
        name = os.path.basename(vpath)
        print(f"[{idx:2d}/{len(videos)}] {name}")

        # ── Step 1: detect release frame ───────────────────────────────────
        frame, info = detect_clicker_frame(
            vpath,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            peak_threshold=peak_threshold,
        )
        method = "audio_clicker"

        if frame is None:
            print(f"  ⚠  Clicker detection failed: {info}")
            print( "     Falling back to peak wrist velocity…")
            try:
                from analyze_single_arrow import detect_single_arrow
                frame  = detect_single_arrow(vpath)
                method = "wrist_velocity"
            except Exception as exc:
                print(f"  ✗  Fallback also failed: {exc}")
                rows.append((name, None, None, "both methods failed"))
                n_fail += 1
                continue

        if frame is None:
            rows.append((name, None, None, "detection failed"))
            n_fail += 1
            continue

        print(f"  ✓  {info}")

        if dry_run:
            rows.append((name, frame, method, "dry-run (skipped)"))
            n_ok += 1
            continue

        # ── Step 2: extract training data ──────────────────────────────────
        session_id = f"{os.path.splitext(name)[0]}_{view}"
        try:
            out_path = extract_training_data(
                video_path        = vpath,
                verified_releases = [frame],
                session_id        = session_id,
                archer_hand       = hand,
                output_dir        = output_dir,
            )
            print(f"  ✓  Saved: {os.path.basename(out_path)}")
            rows.append((name, frame, method, out_path))
            n_ok += 1
        except Exception as exc:
            print(f"  ✗  Training extraction failed: {exc}")
            rows.append((name, frame, method, f"extraction error: {exc}"))
            n_fail += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY — {view} view")
    print(f"{'='*65}")
    print(f"  {'Video':<20} {'Frame':>6}  {'Method':<16}  Result")
    print(f"  {'-'*60}")

    for name, frame, method, result in rows:
        is_ok = (frame is not None
                 and not str(result).startswith("extraction error")
                 and not str(result).startswith("both")
                 and not str(result).startswith("detection"))
        sym    = "✓" if is_ok else "✗"
        f_str  = str(frame)  if frame  is not None else "—"
        m_str  = method      if method is not None else "—"
        r_str  = (os.path.basename(str(result))
                  if is_ok and not str(result).startswith("dry")
                  else str(result))
        print(f"  {sym} {name:<20} {f_str:>6}  {m_str:<16}  {r_str}")

    print(f"\n  {n_ok} succeeded, {n_fail} failed")
    return n_ok, n_fail


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract ML training data from single-arrow videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir", required=True, metavar="PATH",
                        help="Directory of single-arrow videos")
    parser.add_argument("--view", default="face",
                        choices=["face", "target", "back"],
                        help="Camera angle label (default: face)")
    parser.add_argument("--hand", default="right",
                        choices=["right", "left"],
                        help="Archer dominant hand (default: right)")
    parser.add_argument("--output-dir", default="training_data",
                        help="Output directory for training JSONs (default: training_data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect frames only — skip pose estimation")
    parser.add_argument("--min-fraction", type=float, default=0.10,
                        help="Skip first N%% of video for clicker search (default: 0.10)")
    parser.add_argument("--max-fraction", type=float, default=0.85,
                        help="Skip after N%% of video for clicker search (default: 0.85)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Peak threshold as fraction of window max (default: 0.35)")

    args = parser.parse_args()

    n_ok, n_fail = process_directory(
        video_dir     = args.dir,
        view          = args.view,
        hand          = args.hand,
        output_dir    = args.output_dir,
        dry_run       = args.dry_run,
        min_fraction  = args.min_fraction,
        max_fraction  = args.max_fraction,
        peak_threshold= args.threshold,
    )

    print(f"\nTotal: {n_ok} succeeded, {n_fail} failed")

    if n_ok > 0 and not args.dry_run:
        print("\nNext steps:")
        print("  python3 scripts/train_model.py")
        print("  python3 scripts/evaluate_model.py")


if __name__ == "__main__":
    main()

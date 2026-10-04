#!/usr/bin/env python3
"""
Extract training data from a video with verified shot frames.

Usage:
  # Multi-arrow video with manually verified releases:
  python3 scripts/extract_training_data.py \
    --video data/march_2026/030926/side_6_arrows.MOV \
    --verified-frames 179 839 1439 2099 2759 3449

  # Single-arrow video (auto-detect peak velocity as release):
  python3 scripts/extract_training_data.py \
    --single-arrow data/march_2026/030926/2_arrows/arrow1.MOV

  # With custom session ID and hand setting:
  python3 scripts/extract_training_data.py \
    --video side.MOV --verified-frames 300 900 \
    --session-id 2026-03-21_practice --hand right
"""
import argparse
import sys
import os

# Allow running from project root or scripts/ directory
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ml_shot_detector.labeler import extract_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract ML training data from archery video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--video", metavar="PATH",
        help="Path to multi-arrow video (requires --verified-frames)",
    )
    mode.add_argument(
        "--single-arrow", metavar="PATH",
        help="Path to single-arrow video (auto-detects release via peak velocity)",
    )

    parser.add_argument(
        "--verified-frames", type=int, nargs="+", metavar="FRAME",
        help="Frame indices of verified release moments (required with --video)",
    )
    parser.add_argument(
        "--session-id", default=None,
        help="Identifier for this session (default: derived from video filename)",
    )
    parser.add_argument(
        "--hand", default="right", choices=["right", "left"],
        help="Archer dominant hand (default: right)",
    )
    parser.add_argument(
        "--output-dir", default="training_data",
        help="Directory to save training JSON (default: training_data)",
    )
    parser.add_argument(
        "--idle-sample-every", type=int, default=30, metavar="N",
        help="Sample IDLE frames every N frames (default: 30)",
    )

    args = parser.parse_args()

    if args.video:
        if not args.verified_frames:
            parser.error("--video requires --verified-frames")
        video_path        = args.video
        verified_releases = args.verified_frames
    else:
        # Single-arrow mode: auto-detect via peak velocity
        video_path = args.single_arrow
        print(f"\nSingle-arrow mode: detecting release in {video_path}")

        # Import detect_single_arrow from analyze_single_arrow.py
        from analyze_single_arrow import detect_single_arrow
        peak_frame = detect_single_arrow(video_path)
        if peak_frame is None:
            print("ERROR: Could not detect shot. Exiting.")
            sys.exit(1)
        verified_releases = [peak_frame]
        print(f"  Auto-detected release at frame {peak_frame}")

    # Extract and save
    out_path = extract_training_data(
        video_path        = video_path,
        verified_releases = verified_releases,
        session_id        = args.session_id,
        archer_hand       = args.hand,
        idle_sample_every = args.idle_sample_every,
        output_dir        = args.output_dir,
    )

    print(f"\nDone! Training data saved to: {out_path}")
    print("Run 'python3 scripts/train_model.py' to retrain the model.")


if __name__ == "__main__":
    main()

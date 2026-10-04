#!/usr/bin/env python3
"""
Evaluate ML shot detection model using Leave-One-Session-Out cross-validation.

Usage:
  python3 scripts/evaluate_model.py
  python3 scripts/evaluate_model.py --training-dir training_data --save-report
"""
import argparse
import sys
import os
import json

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ml_shot_detector.evaluator import run_loso_cv


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation evaluation of ML shot detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--training-dir", default="training_data",
        help="Directory containing training JSON files (default: training_data)",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save CV report to models/cv_report.json",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-fold verbose output",
    )

    args = parser.parse_args()

    results = run_loso_cv(
        training_dir = args.training_dir,
        verbose      = not args.quiet,
    )

    if args.save_report:
        os.makedirs("models", exist_ok=True)
        report_path = "models/cv_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  CV report saved -> {report_path}")

    # Summary
    agg = results.get("aggregate", {})
    rec = agg.get("release_recall", 0)
    prec = agg.get("release_precision", 0)

    print(f"\n{'='*60}")
    if rec >= 0.85 and prec >= 0.70:
        print("  RESULT: Model meets quality thresholds!")
        print(f"    RELEASE Recall={rec:.3f} >= 0.85  ✓")
        print(f"    RELEASE Precision={prec:.3f} >= 0.70  ✓")
    else:
        print("  RESULT: Model needs more training data")
        status_r  = "✓" if rec  >= 0.85 else "✗ (target > 0.85)"
        status_p  = "✓" if prec >= 0.70 else "✗ (target > 0.70)"
        print(f"    RELEASE Recall={rec:.3f}    {status_r}")
        print(f"    RELEASE Precision={prec:.3f} {status_p}")
        print("\n  Add more training data:")
        print("    python3 scripts/extract_training_data.py --video ... --verified-frames ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

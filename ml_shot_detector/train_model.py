#!/usr/bin/env python3
"""
Train the ML shot detection model from all accumulated training data.

Usage:
  python3 scripts/train_model.py
  python3 scripts/train_model.py --training-dir training_data --models-dir models
"""
import argparse
import sys
import os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ml_shot_detector.trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest shot detector from labeled training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--training-dir", default="training_data",
        help="Directory containing training JSON files (default: training_data)",
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Directory to save trained model (default: models)",
    )
    parser.add_argument(
        "--model-name", default="rf_current",
        help="Base name for saved model files (default: rf_current)",
    )

    args = parser.parse_args()

    model_path = train_model(
        training_dir = args.training_dir,
        models_dir   = args.models_dir,
        model_name   = args.model_name,
    )

    print(f"\nModel ready at: {model_path}")
    print("\nNext step — test on a video:")
    print(f"  python3 archery_analyzer_v2.py --side your_video.MOV --use-ml --ml-model {model_path}")


if __name__ == "__main__":
    main()

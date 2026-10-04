#!/usr/bin/env python3
"""
finetune_pose_model.py — continue training the stock pose model on our own dataset.

Run tools/export_pose_dataset.py first to build data/pose_finetune/ from this app's
own verified full-draw frames, and look at output/pose_finetune_preview.png before
running this — training just teaches the model to reproduce whatever is in that
dataset, pseudo-labels included.

This fine-tunes (continues training) from the pretrained yolo26m-pose.pt checkpoint —
training a pose model from random initialization needs a vastly larger dataset than a
single archer's session history will ever produce, so starting from the pretrained
weights and adapting them to our own footage is the only approach that's feasible at
this data scale.

Output:
    models/archery_pose_finetuned.pt              — the fine-tuned weights
    models/archery_pose_finetuned_metadata.json    — trained_at, base_model, data size, params

Once this file exists, tools/yolo_pose_adapter.resolve_pose_model_path() picks it up
automatically everywhere in the pipeline (archery_analyzer_v2.py, analyze_*_yolo.py) —
no further wiring needed. Delete it to fall back to the stock model again.

Usage:
    python3 tools/finetune_pose_model.py
    python3 tools/finetune_pose_model.py --epochs 30 --imgsz 640
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_pose_adapter import STOCK_MODEL_PATH, FINETUNED_MODEL_PATH  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "pose_finetune", "data.yaml")
RUNS_DIR = os.path.join(ROOT, "models", "_finetune_runs")


def _pick_device():
    """CPU by default, even on Apple Silicon. As of ultralytics 8.4.31, YOLO26's pose
    loss builds an RLE_WEIGHT tensor as float64 and moves it straight to the training
    device — MPS categorically rejects float64 (not a missing-op case, so
    PYTORCH_ENABLE_MPS_FALLBACK doesn't help), so `.train(task="pose", device="mps")`
    hard-crashes on any Mac. CUDA isn't affected by this — if it's available, use it.
    Pass --device mps yourself to retry if a future ultralytics release fixes this."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25,
                    help="small default — this dataset is small; more epochs just overfits it faster")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="override auto-detected device (mps/cuda/cpu)")
    args = ap.parse_args()

    if not os.path.exists(DATA_YAML):
        sys.exit(f"{DATA_YAML} not found. Run tools/export_pose_dataset.py first.")

    n_train = len(os.listdir(os.path.join(ROOT, "data", "pose_finetune", "images", "train")))
    print(f"Training set: {n_train} image(s). This is a small-data fine-tune, not from scratch — "
          f"starting from {STOCK_MODEL_PATH}.")

    device = args.device or _pick_device()
    print(f"Device: {device}")

    from ultralytics import YOLO
    model = YOLO(STOCK_MODEL_PATH)
    result = model.train(
        data=DATA_YAML,
        task="pose",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=RUNS_DIR,
        name="run",
        exist_ok=True,
        verbose=True,
    )

    best_weights = os.path.join(RUNS_DIR, "run", "weights", "best.pt")
    if not os.path.exists(best_weights):
        sys.exit(f"Training finished but {best_weights} wasn't produced — check the log above.")

    os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
    shutil.copy2(best_weights, FINETUNED_MODEL_PATH)
    print(f"Saved fine-tuned model: {FINETUNED_MODEL_PATH}")

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": os.path.basename(STOCK_MODEL_PATH),
        "n_train_images": n_train,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "note": "Fine-tuned (continued training) from the stock model on this app's own "
                "verified full-draw frames, using the current model's own predictions as "
                "pseudo-labels (see tools/export_pose_dataset.py) — not independently "
                "ground-truthed.",
    }
    metadata_path = FINETUNED_MODEL_PATH.replace(".pt", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {metadata_path}")
    print("\nDone. Every tool that loads a pose model will now use the fine-tuned weights "
          "automatically. Delete models/archery_pose_finetuned.pt to go back to the stock model.")


if __name__ == "__main__":
    main()

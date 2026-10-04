#!/usr/bin/env python3
"""
export_pose_dataset.py — build an Ultralytics pose fine-tuning dataset from this
app's own verified full-draw frames.

WHY THIS EXISTS
    The stock yolo26m-pose.pt is a general-purpose human pose model, never adapted to
    this app's own recording conditions (camera angles/distances, archery stance,
    bow/arrow occlusion). Every session_history/*.json already carries a
    `metrics.verified_frames` list per clip — frames a human has confirmed are genuine
    full-draw poses (the same signal tools/cross_validate_shots.py and the mobile
    backend already trust). This tool turns those into an Ultralytics pose-dataset
    directory that tools/finetune_pose_model.py can fine-tune on.

HOW LABELS ARE PRODUCED — READ THIS BEFORE TRAINING ON THE RESULT
    Each verified frame is run through the CURRENT pose model (whichever
    tools/yolo_pose_adapter.resolve_pose_model_path() resolves to) to get a COCO-17
    keypoint + bbox pseudo-label. There is no independent ground truth here — this
    fine-tunes the model to THIS APP'S recording conditions, it cannot correct a case
    where the current model is already wrong, because it doesn't know it's wrong.
    Spot-check output/pose_finetune_preview.png (a sample of labeled frames this tool
    writes) before spending time training on the result.

OUTPUT
    data/pose_finetune/
      images/train/*.jpg   images/val/*.jpg
      labels/train/*.txt   labels/val/*.txt   (YOLO pose format: one object per line)
      data.yaml
    output/pose_finetune_preview.png   (sample grid of labeled frames)

Split is by CLIP (session JSON), not by frame — frames from the same clip are highly
similar, so splitting by frame would leak near-duplicates across train/val and make
validation numbers meaningless.

Usage:
    python3 tools/export_pose_dataset.py
    python3 tools/export_pose_dataset.py --min-conf 0.6 --val-frac 0.15
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_pose_adapter import resolve_pose_model_path, YOLO_CONNECTIONS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "pose_finetune")
PREVIEW_PATH = os.path.join(ROOT, "output", "pose_finetune_preview.png")

# COCO-17 keypoint names, in COCO order (required by data.yaml / Ultralytics pose format)
COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Left/right swap indices, for Ultralytics' horizontal-flip augmentation
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _collect_clips():
    """One entry per session JSON that has verified_frames + a readable video file."""
    clips = []
    for jf in sorted(glob.glob(os.path.join(ROOT, "session_history", "*.json"))):
        try:
            s = json.load(open(jf))
        except (json.JSONDecodeError, OSError):
            continue
        frames = (s.get("metrics") or {}).get("verified_frames") or []
        video_path = s.get("video_path", "")
        if not frames or not video_path:
            continue
        abs_video = video_path if os.path.isabs(video_path) else os.path.join(ROOT, video_path)
        if not os.path.exists(abs_video):
            continue
        clips.append({"json": jf, "video": abs_video, "frames": frames})
    return clips


def _detect_pseudo_label(model, frame_bgr, min_conf):
    """Run the current pose model on one frame, return (bbox_xyxy, kpts_xy, kpts_conf)
    for the largest detected person, in raw COCO-17 order — or None if nothing clears
    min_conf. Mirrors YoloPoseAdapter.process()'s largest-bbox selection, but keeps
    COCO-17 order + bbox (YoloPoseAdapter's own .process() re-maps to a 33-slot
    MediaPipe-compatible layout, which is the wrong shape for a YOLO pose label)."""
    results = model(frame_bgr, verbose=False, conf=0.25)
    boxes = results[0].boxes
    kpts = results[0].keypoints
    if kpts is None or len(kpts.xy) == 0 or boxes is None or len(boxes.xyxy) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best = int(areas.argmax())

    xy = kpts.xy[best].cpu().numpy()
    conf = (kpts.conf[best].cpu().numpy() if kpts.conf is not None
            else np.ones(17, dtype=np.float32))
    if float(conf.mean()) < min_conf:
        return None
    return xyxy[best], xy, conf


def _write_label(path, bbox_xyxy, kpts_xy, kpts_conf, w, h, kpt_min_conf):
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    fields = [0, cx, cy, bw, bh]   # class 0 = "archer" (single-class pose dataset)
    for (px, py), c in zip(kpts_xy, kpts_conf):
        v = 2 if c >= kpt_min_conf else 0   # 2=labeled+visible, 0=not labeled (COCO convention)
        fields += [px / w, py / h, v]
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in fields) + "\n")


def _draw_preview_frame(frame_bgr, kpts_xy, kpts_conf):
    for a, b in YOLO_CONNECTIONS:
        if kpts_conf[a] < 0.2 or kpts_conf[b] < 0.2:
            continue
        pa = tuple(int(v) for v in kpts_xy[a])
        pb = tuple(int(v) for v in kpts_xy[b])
        cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2)
    for i in range(17):
        if kpts_conf[i] < 0.2:
            continue
        p = tuple(int(v) for v in kpts_xy[i])
        color = (0, 255, 255) if kpts_conf[i] > 0.6 else (0, 165, 255)
        cv2.circle(frame_bgr, p, 4, color, -1)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="minimum mean keypoint confidence to keep a frame as a label (default 0.5, "
                         "stricter than the pipeline's normal 0.30 since these become training labels)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of CLIPS (not frames) held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-n", type=int, default=12, help="frames to include in the preview sheet")
    args = ap.parse_args()

    clips = _collect_clips()
    if not clips:
        sys.exit("No session_history/*.json with verified_frames + a readable video_path found. "
                 "Nothing to export.")

    total_frames = sum(len(c["frames"]) for c in clips)
    print(f"Found {len(clips)} clip(s), {total_frames} verified frame(s) total.")

    model_path = resolve_pose_model_path()
    from ultralytics import YOLO
    model = YOLO(model_path)

    rng = random.Random(args.seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    n_val_clips = max(1, round(len(shuffled) * args.val_frac)) if len(shuffled) >= 2 else 0
    val_clips = {c["json"] for c in shuffled[:n_val_clips]}
    if n_val_clips == 0:
        print("Only one clip available — everything goes to train; no held-out validation split.")

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)
    os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)

    kept, skipped = 0, 0
    preview_frames = []
    for clip in clips:
        split = "val" if clip["json"] in val_clips else "train"
        stem = os.path.splitext(os.path.basename(clip["json"]))[0]
        cap = cv2.VideoCapture(clip["video"])
        for fn in clip["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                skipped += 1
                continue
            label = _detect_pseudo_label(model, frame, args.min_conf)
            if label is None:
                skipped += 1
                continue
            bbox_xyxy, kpts_xy, kpts_conf = label
            h, w = frame.shape[:2]
            name = f"{stem}_f{fn}"
            cv2.imwrite(os.path.join(OUT_DIR, "images", split, f"{name}.jpg"), frame)
            _write_label(os.path.join(OUT_DIR, "labels", split, f"{name}.txt"),
                         bbox_xyxy, kpts_xy, kpts_conf, w, h, args.min_conf)
            kept += 1
            if len(preview_frames) < args.preview_n:
                preview_frames.append(_draw_preview_frame(frame.copy(), kpts_xy, kpts_conf))
        cap.release()

    print(f"Wrote {kept} labeled frame(s) ({skipped} skipped: unreadable or below --min-conf) "
          f"to {OUT_DIR}")

    data_yaml = os.path.join(OUT_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images/train\n")
        f.write(f"val: images/{'val' if n_val_clips else 'train'}\n")
        f.write("kpt_shape: [17, 3]\n")
        f.write(f"flip_idx: {FLIP_IDX}\n")
        f.write("names:\n  0: archer\n")
    print(f"Wrote {data_yaml}")

    if preview_frames:
        thumbs = [cv2.resize(f, (320, int(f.shape[0] * 320 / f.shape[1]))) for f in preview_frames]
        max_h = max(t.shape[0] for t in thumbs)
        thumbs = [np.vstack([t, np.zeros((max_h - t.shape[0], t.shape[1], 3), np.uint8)])
                  if t.shape[0] < max_h else t for t in thumbs]
        cols = 4
        rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
        row_w = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], row_w - r.shape[1], 3), np.uint8)])
                if r.shape[1] < row_w else r for r in rows]
        cv2.imwrite(PREVIEW_PATH, np.vstack(rows))
        print(f"Wrote preview sheet: {PREVIEW_PATH} — LOOK AT THIS before training. "
              f"Every skeleton drawn on it is what the model will be told is correct.")

    if kept == 0:
        sys.exit("No frames cleared --min-conf — nothing to train on. Try a lower --min-conf, "
                 "or verify session_history/*.json video_path fields point at real, readable files.")


if __name__ == "__main__":
    main()

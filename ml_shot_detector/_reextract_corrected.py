#!/usr/bin/env python3
"""Re-extract training data for the 18 corrected videos."""
import subprocess
import sys
import os

SIDE   = "data/march_2026/032226/side"
BEHIND = "data/march_2026/032226/behind"

corrections = [
    (f"{SIDE}/IMG_2775.MOV",   178,  "side"),
    (f"{SIDE}/IMG_2776.MOV",   151,  "side"),
    (f"{SIDE}/IMG_2777.MOV",   144,  "side"),
    (f"{SIDE}/IMG_2778.MOV",   175,  "side"),
    (f"{SIDE}/IMG_2779.MOV",   144,  "side"),
    (f"{SIDE}/IMG_2781.MOV",   196,  "side"),
    (f"{SIDE}/IMG_2782.MOV",   193,  "side"),
    (f"{SIDE}/IMG_2785.MOV",   147,  "side"),
    (f"{SIDE}/IMG_2786.MOV",   128,  "side"),
    (f"{SIDE}/IMG_2787.MOV",   163,  "side"),
    (f"{SIDE}/IMG_2789.MOV",   153,  "side"),
    (f"{SIDE}/IMG_2790.MOV",   291,  "side"),
    (f"{SIDE}/IMG_2791.MOV",   171,  "side"),
    (f"{SIDE}/IMG_2793.MOV",   147,  "side"),
    (f"{SIDE}/IMG_2794.MOV",   153,  "side"),
    (f"{SIDE}/IMG_2795.MOV",   168,  "side"),
    (f"{SIDE}/IMG_2796.MOV",   159,  "side"),
    (f"{BEHIND}/IMG_2799.MOV", 158,  "behind"),
]

failed = []
for i, (vpath, frame, view) in enumerate(corrections, 1):
    name = os.path.splitext(os.path.basename(vpath))[0]
    sid  = f"{name}_{view}"
    print(f"[{i}/{len(corrections)}] {os.path.basename(vpath)} → frame {frame}", flush=True)
    r = subprocess.run(
        [sys.executable, "scripts/extract_training_data.py",
         "--video", vpath,
         "--verified-frames", str(frame),
         "--session-id", sid,
         "--hand", "right"],
        capture_output=False,
    )
    if r.returncode != 0:
        print(f"  FAILED")
        failed.append(os.path.basename(vpath))
    else:
        print(f"  done")

print(f"\nAll re-extractions complete. {len(corrections) - len(failed)} succeeded, {len(failed)} failed.")
if failed:
    print("Failed:", failed)

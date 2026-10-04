#!/usr/bin/env python3
"""
sort_dropbox.py — Move videos from dropbox/ into the data/ folder structure.

Usage:
    python3 tools/sort_dropbox.py            # dry run (preview only)
    python3 tools/sort_dropbox.py --move     # actually move the files

Dropbox layout:
    dropbox/
        face/       (or faceview/)
        back/       (or backview/)
        target/     (or targetview/)

    Files placed directly in dropbox/ (no view subfolder) are listed as
    unrecognized and skipped — move them into a view subfolder first.

Date detection (in order of preference):
    1. ffprobe creation_time from video metadata  (most accurate)
    2. File modification time

Destination structure:
    data/{month_name}_{year}/{MMDDYY}/{viewname}/filename.MOV
    e.g. data/july_2026/070526/faceview/IMG_5001.MOV
"""

import os
import sys
import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DROPBOX_DIR  = PROJECT_ROOT / "dropbox"
DATA_DIR     = PROJECT_ROOT / "data"

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}

# Accepted subfolder names → canonical view folder name
VIEW_MAP = {
    "face":       "faceview",
    "faceview":   "faceview",
    "back":       "backview",
    "backview":   "backview",
    "rear":       "backview",
    "rearview":   "backview",
    "target":     "targetview",
    "targetview": "targetview",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_date(path: Path) -> datetime:
    """Return recording datetime: ffprobe metadata first, mtime fallback."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format_tags=creation_time",
                str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        ts = data.get("format", {}).get("tags", {}).get("creation_time", "")
        if ts:
            # ISO format: "2026-07-05T14:32:10.000000Z"
            ts = ts.rstrip("Z").split(".")[0]
            return datetime.fromisoformat(ts)
    except Exception:
        pass
    # Fallback: file modification time
    return datetime.fromtimestamp(path.stat().st_mtime)


def date_to_paths(dt: datetime):
    """Return (month_folder, date_folder) strings for the given datetime."""
    month_folder = dt.strftime("%B").lower() + "_" + str(dt.year)   # july_2026
    date_folder  = dt.strftime("%m%d%y")                             # 070526
    return month_folder, date_folder


def scan_dropbox():
    """
    Yield (src_path, view_name, date) for each video found in view subfolders.
    Files directly in dropbox/ root are yielded with view_name=None.
    """
    if not DROPBOX_DIR.exists():
        return

    for entry in sorted(DROPBOX_DIR.iterdir()):
        if entry.is_dir():
            canonical = VIEW_MAP.get(entry.name.lower())
            for f in sorted(entry.iterdir()):
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    dt = get_video_date(f)
                    yield f, canonical, dt
        elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            # File at root level — no view info
            dt = get_video_date(entry)
            yield entry, None, dt


def destination(src: Path, view: str, dt: datetime) -> Path:
    month_folder, date_folder = date_to_paths(dt)
    return DATA_DIR / month_folder / date_folder / view / src.name


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--move" not in sys.argv

    if not DROPBOX_DIR.exists():
        print(f"dropbox/ folder not found at {DROPBOX_DIR}")
        print("Create it and add view subfolders: face/, back/, target/")
        return

    items = list(scan_dropbox())
    if not items:
        print("No video files found in dropbox/")
        return

    print(f"{'DRY RUN — ' if dry_run else ''}Sorting {len(items)} file(s) from dropbox/\n")

    moved = 0
    skipped = 0
    conflicts = 0

    for src, view, dt in items:
        rel = src.relative_to(DROPBOX_DIR)

        if view is None:
            print(f"  SKIP  {rel}")
            print(f"        No view subfolder — move into face/, back/, or target/ first.\n")
            skipped += 1
            continue

        dst = destination(src, view, dt)
        date_str = dt.strftime("%Y-%m-%d %H:%M")

        if dst.exists():
            print(f"  SKIP  {rel}")
            print(f"        Already exists: {dst.relative_to(PROJECT_ROOT)}\n")
            conflicts += 1
            continue

        print(f"  {'WOULD MOVE' if dry_run else 'MOVE'}  {rel}")
        print(f"        date    : {date_str}")
        print(f"        view    : {view}")
        print(f"        dest    : {dst.relative_to(PROJECT_ROOT)}\n")

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            # Remove empty source view subfolder if now empty
            if not any(src.parent.iterdir()):
                src.parent.rmdir()
            moved += 1

    print("─" * 60)
    if dry_run:
        ready = len(items) - skipped - conflicts
        print(f"  {ready} file(s) ready to move, {skipped} skipped (no view), "
              f"{conflicts} already exist.")
        print(f"  Run with --move to execute.")
    else:
        print(f"  Moved {moved} file(s). {skipped} skipped, {conflicts} already existed.")


if __name__ == "__main__":
    main()

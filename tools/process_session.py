#!/usr/bin/env python3
"""
process_session.py — drop a session folder, get analysis.

Implements docs/shot_detection_design.md as an orchestration layer over the
existing tools. Two phases:

  SCAN (default):   discover clips → per clip detect DRAWS (pose) + CLICKERS (audio),
                    classify each draw shot-confirmed / unresolved, write a verification
                    grid + a draft session JSON, print a review report.
  FINALIZE (--finalize): run YOLO metrics on each clip's metrics.verified_frames,
                    write final metrics into the JSONs, refresh trends.

Workflow:
  1) python3 tools/process_session.py data/May_2026/052626
  2) review the grids (output/verify_*.png) + report; edit each draft JSON's
     metrics.verified_frames to the TRUE shots (drop let-downs, add any missed);
     reconcile against the score log (scored arrows = shots).
  3) python3 tools/process_session.py data/May_2026/052626 --finalize

Pre-lavalier reality (key_learnings §24-26): the clicker is only reliable in face
view, so back/target draws mostly land in UNRESOLVED — that's expected; the score
log resolves them. Pose can under-detect target (foreshortening) — the report flags it.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2

# Unbuffered stdout so progress is visible when output is redirected to a file/log.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from cross_validate_shots import get_pose_shots, get_audio_onsets, reconcile, _DEFAULT_MODEL
from verify_session_frames import build_grid
from release_probe import snap_to_release, R_EL, L_EL
from yolo_pose_adapter import YoloPoseAdapter

ROOT = os.path.dirname(os.path.dirname(__file__))
VIEW_DIRS = {"faceview": "face", "rearview": "back", "targetview": "target"}
ANALYZE = {"face": "analyze_face_yolo.py", "back": "analyze_back_yolo.py", "target": "analyze_target_yolo.py"}


def session_date(folder):
    """data/.../052626 -> 2026-05-26 (folder basename MMDDYY)."""
    b = os.path.basename(os.path.normpath(folder))
    if len(b) == 6 and b.isdigit():
        return f"20{b[4:6]}-{b[0:2]}-{b[2:4]}"
    return b  # fallback: use as-is


def json_path(date, view, clip_stem):
    num = clip_stem.replace("IMG_", "IMG")  # IMG_3121 -> IMG3121
    return os.path.join(ROOT, "session_history", f"{date}_practice_{view}_{num}.json")


def discover(folder):
    """Yield (view, clip_path) for every MOV under known view subfolders."""
    out = []
    for sub, view in VIEW_DIRS.items():
        for clip in sorted(glob.glob(os.path.join(folder, sub, "*.MOV"))):
            out.append((view, clip))
    return out


def scan_clip(view, clip, date, sample_every, hand, tol, loc_warn, keep=False):
    fps = cv2.VideoCapture(clip).get(cv2.CAP_PROP_FPS) or 30.0
    pose_shots = get_pose_shots(clip, _DEFAULT_MODEL, 0.3, sample_every)
    audio = get_audio_onsets(clip, view, hand, fps)
    rows, unmatched = reconcile(pose_shots, audio, tol, loc_warn)

    # Release-anchor each detected draw (task #2): finder gives mid-hold; snap to ~5f before
    # the release/let-down motion onset so frames match the archer's labels regardless of hold length.
    draw_el = R_EL if hand == "right" else L_EL
    snap_pose = YoloPoseAdapter(_DEFAULT_MODEL)
    snap_cap = cv2.VideoCapture(clip)
    snap = {}
    for r in rows:
        mf = r["pose"]["frame"]
        snap[mf] = snap_to_release(snap_pose, snap_cap, mf, draw_el, fps)
    snap_cap.release()

    confirmed = sorted(snap[r["pose"]["frame"]] for r in rows if r["status"] in ("AGREE", "LOC_WARN"))
    unresolved = sorted(snap[r["pose"]["frame"]] for r in rows if r["status"] == "POSE_ONLY")
    all_draws = sorted(snap.values())
    # strong unmatched audio not explained as post-shot vibration = possible missed draw
    ftw = int(round(3.0 * fps))
    vib = [a for a in unmatched if any(0 <= (a["frame"] - p["frame"]) <= ftw for p in pose_shots)]
    matched_str = [r["audio"]["strength"] for r in rows if r["audio"]]
    floor = 0.5 * (sorted(matched_str)[len(matched_str)//2] if matched_str else 0)
    missed = sorted(a["frame"] for a in unmatched if a not in vib and floor > 0 and a["strength"] >= floor)

    stem = os.path.splitext(os.path.basename(clip))[0]

    # verification grid (all detected draws)
    grid_path = os.path.join(ROOT, "output", f"verify_{date}_practice_{view}_{stem.replace('IMG_','IMG')}.png")
    if all_draws:
        os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
        g = build_grid(clip, all_draws, fps)
        if g is not None:
            cv2.imwrite(grid_path, g)

    # draft JSON (verified_frames = all pose draws; review block for the human)
    jp = json_path(date, view, stem)
    rel = os.path.relpath(clip, ROOT)
    draft = {
        "date": date, "video_path": rel, "session_type": "practice",
        "timestamp": f"{date}T00:00:00", "view": view, "location": "", "distance_m": None,
        "metrics": {"arrows_shot": len(all_draws), "verified_frames": all_draws},
        "review": {
            "status": "DRAFT — set verified_frames to TRUE shots before --finalize",
            "clicker_confirmed": confirmed,
            "unresolved_draws": unresolved,
            "possible_missed_draws_audio_only": missed,
        },
        "detection_notes": (f"process_session.py scan: {len(all_draws)} pose draws "
                            f"({len(confirmed)} clicker-confirmed, {len(unresolved)} unresolved; "
                            f"{len(missed)} strong audio-only = possible pose misses). "
                            f"Resolve let-downs vs shots via score log / grid before finalize."),
        "notes": "Fill in location/distance/notes before finalize.",
    }
    if not (os.path.exists(jp) and keep):
        with open(jp, "w") as f:
            json.dump(draft, f, indent=2)

    return {"view": view, "clip": stem, "draws": len(all_draws), "confirmed": len(confirmed),
            "unresolved": len(unresolved), "missed": len(missed), "grid": grid_path, "json": jp}


def _scan_one(task):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    return scan_clip(*task)


def cmd_scan(args):
    date = session_date(args.folder)
    clips = discover(args.folder)
    if not clips:
        print(f"No clips found under {args.folder}/{{faceview,rearview,targetview}}/*.MOV")
        return
    print(f"\n=== SCAN {args.folder}  (date {date}, {len(clips)} clips, sample-every {args.sample_every}, jobs {args.jobs}) ===")
    print("Detecting draws (pose) + clickers (audio) per clip — this runs YOLO, give it a few min...\n")

    def run(vc):
        view, clip = vc
        return scan_clip(view, clip, date, args.sample_every, args.hand, args.tol, args.loc_warn, args.keep)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(v, c, date, args.sample_every, args.hand, args.tol, args.loc_warn, args.keep) for v, c in clips]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_scan_one, tasks))   # ordered to match `clips`
    else:
        results = [run(vc) for vc in clips]

    rows = []
    for r in results:
        rows.append(r)
        # "possible missed" is only trustworthy where pose under-detects (target / zero draws);
        # in face/back pose is the reliable recall signal, so audio-only events are just noise.
        r["flag_missed"] = r["missed"] and (r["view"] == "target" or r["draws"] == 0)
        line = (f"  [{r['view']:>6}] {r['clip']:>9}: {r['draws']} draws  "
                f"({r['confirmed']} clicker✓, {r['unresolved']} unresolved")
        line += f", ⚠{r['missed']} possible-missed" if r["flag_missed"] else ""
        print(line + ")")

    print(f"\n--- SESSION REPORT ({date}) ---")
    tot = sum(r["draws"] for r in rows)
    print(f"  Total pose-detected draws: {tot}  (shots + let-downs — confirm against score log)")
    review = [r for r in rows if r["unresolved"] or r["flag_missed"]]
    if review:
        print(f"  ⚠ REVIEW these clips (grids in output/):")
        for r in review:
            bits = []
            if r["unresolved"]: bits.append(f"{r['unresolved']} unresolved draw(s) — shot or let-down?")
            if r["flag_missed"]: bits.append(f"{r['missed']} possible missed draw(s) — check grid"
                                            + (" (target foreshortens — pose under-detects)" if r["view"] == "target" else ""))
            print(f"     {r['view']:>6} {r['clip']}: " + "; ".join(bits))
    print(f"\n  Next: review output/verify_{date}_*.png, edit each draft JSON's metrics.verified_frames")
    print(f"        to the TRUE shots (scored arrows = shots; drop let-downs), then:")
    print(f"        python3 tools/process_session.py {args.folder} --finalize")


def cmd_finalize(args):
    date = session_date(args.folder)
    drafts = sorted(glob.glob(os.path.join(ROOT, "session_history", f"{date}_practice_*.json")))
    drafts = [d for d in drafts if "coaching" not in d]
    if not drafts:
        print(f"No draft JSONs for {date} — run scan first.")
        return
    print(f"\n=== FINALIZE {date}  ({len(drafts)} clips) — running YOLO metrics ===\n")
    for jp in drafts:
        j = json.load(open(jp))
        view = j.get("view"); frames = j.get("metrics", {}).get("verified_frames", [])
        vp = os.path.join(ROOT, j["video_path"])
        if not frames or view not in ANALYZE or not os.path.exists(vp):
            print(f"  skip {os.path.basename(jp)} (frames={len(frames)}, view={view})"); continue
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), ANALYZE[view]),
               "--video", vp, "--frames", *map(str, frames), "--session-json", jp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = "✓" if r.returncode == 0 else "✗ FAILED"
        print(f"  [{view:>6}] {os.path.basename(jp)}: {len(frames)} shots -> metrics {ok}")
    print("\n  Refreshing trends...")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "trend_analysis.py")],
                   capture_output=True, text=True)
    print("  Done. output/session_trends.png updated.")


def main():
    ap = argparse.ArgumentParser(description="Process an archery session folder end-to-end.")
    ap.add_argument("folder", help="Session folder, e.g. data/May_2026/052626")
    ap.add_argument("--finalize", action="store_true", help="Run YOLO metrics on verified_frames + refresh trends")
    ap.add_argument("--sample-every", type=int, default=5, help="Pose sample rate for draw detection (default 5; validated recall on 5/26)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--jobs", type=int, default=4, help="Clips to process in parallel (default 4; set 1 to disable)")
    ap.add_argument("--tol", type=int, default=60, help="Frames within which a draw and a clicker are the same shot")
    ap.add_argument("--loc-warn", type=int, default=40)
    ap.add_argument("--keep", action="store_true", help="Don't overwrite existing session JSONs in scan")
    args = ap.parse_args()
    (cmd_finalize if args.finalize else cmd_scan)(args)


if __name__ == "__main__":
    main()

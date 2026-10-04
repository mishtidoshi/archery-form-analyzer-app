#!/usr/bin/env python3
"""
Cross-validate shot detection by reconciling two INDEPENDENT signals:

  1. POSE-PRIMARY  — find_full_draw_frames.py (sustained full-draw holds; audio-independent)
  2. AUDIO ONSETS  — AudioVisualShotDetector raw 2s-gap candidates (clicker onsets;
                     pose-independent). We use the RAW candidate list (prefer_strongest_audio
                     = False, before any NMS/cooldown) so it is a clean independent opinion and
                     not the production detector's greedy confirmed output.

The two methods fail in different ways (pose misses target foreshortening; audio misses
sub-threshold clickers), so AGREEMENT between them is strong evidence a shot is real and
correctly localized. The whole point is to AUTO-ACCEPT the agreements and surface only the
handful of disagreements for a quick human glance — instead of reviewing every frame.

Verdict per clip:
  AUTO-ACCEPT  — every pose shot has a nearby audio onset, no strong unmatched audio,
                 no large localization gaps.
  REVIEW       — otherwise; the specific conflicts are listed.

With --lavalier (T2.2 clean-clicker view): the clicker is treated as AUTHORITATIVE,
so a full-draw hold with NO clicker is reported as a LET-DOWN (design §5) rather than
UNRESOLVED. This is the whole payoff of the mic — it turns the shot-vs-let-down
ambiguity into a decision. Only use it on the lavalier-equipped view.

Usage:
  python3 tools/cross_validate_shots.py --video V.MOV --view back
  python3 tools/cross_validate_shots.py --video V.MOV --view back --session-json S.json
  python3 tools/cross_validate_shots.py --video V.MOV --view target --sample-every 3
  python3 tools/cross_validate_shots.py --video LAV.MOV --view target --lavalier
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from find_full_draw_frames import find_full_draw_frames
from audio_visual_shot_detector import AudioVisualShotDetector
from release_probe import track as _rp_track, feats as _rp_feats, R_WR, L_WR, R_EL, L_EL, MODEL as _RP_MODEL
from yolo_pose_adapter import YoloPoseAdapter
import cv2 as _cv2

# ⚠ Release gate is a NON-AUTHORITATIVE diagnostic only (see key_learnings §26).
# Pose gross-motion CANNOT reliably separate a shot from a let-down: the discriminating
# release impulse is swamped by the bow-lowering both actions share, occluded in back view,
# and under-sampled at 30fps. de_dy "worked" on n=8 calibration only by window-timing luck and
# then misclassified 2/3 on IMG_3125 ground truth. So we REPORT de_dy as a hint but do NOT use
# it to decide shot-vs-let-down. The real discriminator is the clicker (→ lavalier, T2.2).
# A pose hold with no clicker = UNRESOLVED draw (shot w/ quiet clicker OR let-down).
DE_DY_HINT = 0.045        # purely a printed hint; not a decision threshold


def release_verdict(video, hand, frames):
    """For each hold frame, classify shot/letdown/borderline by post-hold draw-elbow drop."""
    draw_wr, bow_wr, draw_el = (R_WR, L_WR, R_EL) if hand == "right" else (L_WR, R_WR, L_EL)
    pose = YoloPoseAdapter(_RP_MODEL)
    cap = _cv2.VideoCapture(video)
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    out = {}
    for f0 in frames:
        rows = _rp_track(pose, cap, f0, draw_wr, bow_wr, draw_el, post=30)
        ft = _rp_feats(rows, f0, fps, post=30)
        if ft is None:
            out[f0] = ("?", None); continue
        # Diagnostic hint only — NOT a shot/let-down decision (see §26).
        de = ft["de_dy"]
        out[f0] = ("hi" if de >= DE_DY_HINT else "lo", ft)
    cap.release()
    return out

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "models", "yolo26m-pose.pt")


def _video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def get_pose_shots(video, model, min_hold, sample_every):
    """Pose-primary detections: list of dicts {frame, hold_s, conf}."""
    res = find_full_draw_frames(video, model, min_hold_s=min_hold, sample_every=sample_every)
    return [{"frame": f, "hold_s": round(h, 2), "conf": round(c, 2)} for f, h, c in res]


def get_audio_onsets(video, view, archer_hand, fps):
    """Pose-independent audio onsets: list of dicts {frame, strength}, raw 2s-gap candidates."""
    det = AudioVisualShotDetector(
        angle=view, archer_hand=archer_hand, fps=fps,
        min_shot_interval=12.0, prefer_strongest_audio=False,
    )
    # extract_candidates prints its own candidate dump; silence stdout for a clean report
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        det.extract_candidates(video)
    cands = getattr(det, "_candidates", []) or []
    return [{"frame": int(f), "strength": round(float(s), 5)} for f, s in sorted(cands)]


def reconcile(pose_shots, audio_onsets, tol, loc_warn):
    """
    Greedily match each pose shot to the nearest unused audio onset within `tol` frames.
    Returns (rows, unmatched_audio). Each row: pose shot + matched audio (or None) + status.
    """
    used = [False] * len(audio_onsets)
    rows = []
    for ps in pose_shots:
        best_j, best_d = None, None
        for j, ao in enumerate(audio_onsets):
            if used[j]:
                continue
            d = abs(ao["frame"] - ps["frame"])
            if d <= tol and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is None:
            rows.append({"pose": ps, "audio": None, "delta": None, "status": "POSE_ONLY"})
        else:
            used[best_j] = True
            ao = audio_onsets[best_j]
            status = "AGREE" if best_d <= loc_warn else "LOC_WARN"
            rows.append({"pose": ps, "audio": ao, "delta": best_d, "status": status})

    unmatched = [audio_onsets[j] for j in range(len(audio_onsets)) if not used[j]]
    return rows, unmatched


def main():
    ap = argparse.ArgumentParser(description="Cross-validate shot detection (pose vs audio).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--view", required=True, choices=["face", "back", "target"])
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--yolo-model", default=_DEFAULT_MODEL)
    ap.add_argument("--min-hold", type=float, default=0.3)
    ap.add_argument("--sample-every", type=int, default=3,
                    help="Pose sample rate (default 3; ~2-3x faster than every-frame, still catches holds)")
    ap.add_argument("--tol", type=int, default=60,
                    help="Frames within which a pose shot and audio onset are 'the same shot' (default 60 = 2s)")
    ap.add_argument("--loc-warn", type=int, default=40,
                    help="Matched pairs farther apart than this are flagged LOC_WARN (default 40)")
    ap.add_argument("--session-json", default=None,
                    help="Optional: compare against metrics.verified_frames as ground truth")
    ap.add_argument("--lavalier", action="store_true",
                    help="This clip's audio is a clean lavalier clicker (T2.2), so the clicker "
                         "is AUTHORITATIVE: a full-draw hold with no clicker = LET-DOWN (not "
                         "unresolved). Skips the non-authoritative de_dy probe. Only use on the "
                         "lavalier-equipped view — NOT on ambient clips (quiet clickers would "
                         "misread real shots as let-downs).")
    args = ap.parse_args()

    fps = _video_fps(args.video)
    pose_shots = get_pose_shots(args.video, args.yolo_model, args.min_hold, args.sample_every)
    audio_onsets = get_audio_onsets(args.video, args.view, args.hand, fps)
    rows, unmatched = reconcile(pose_shots, audio_onsets, args.tol, args.loc_warn)

    # Release gate: NON-AUTHORITATIVE de_dy hint (see §26). Skip it entirely when the
    # lavalier makes the clicker authoritative — the hint adds nothing and costs a
    # per-hold YOLO pass.
    if args.lavalier:
        for r in rows:
            r["rel"], r["de_dy"] = "n/a", None
    else:
        rel = release_verdict(args.video, args.hand, [ps["frame"] for ps in pose_shots])
        for r in rows:
            v, ft = rel.get(r["pose"]["frame"], ("?", None))
            r["rel"] = v
            r["de_dy"] = ft["de_dy"] if ft else None

    # Strong unmatched audio = potential pose miss; weak = ignorable noise.
    # First drop unmatched onsets that are just post-release vibration / follow-through
    # of a detected shot (they fall shortly AFTER a pose full-draw). What remains is
    # audio NOT explained by any detected shot — the real "did pose miss this?" signal.
    ft_window = int(round(3.0 * fps))  # follow-through / vibration lasts up to ~3s after release
    vibration = [a for a in unmatched if any(0 <= (a["frame"] - ps["frame"]) <= ft_window for ps in pose_shots)]
    unexplained = [a for a in unmatched if a not in vibration]

    matched_str = [r["audio"]["strength"] for r in rows if r["audio"]]
    med = sorted(matched_str)[len(matched_str) // 2] if matched_str else 0.0
    strong_floor = 0.5 * med
    strong_unmatched = [a for a in unexplained if strong_floor > 0 and a["strength"] >= strong_floor]

    name = os.path.basename(args.video)
    print(f"\n{'='*68}")
    print(f"  CROSS-VALIDATION  {name}   view={args.view}  fps={fps:.0f}")
    print(f"{'='*68}")
    print(f"  Pose-primary (find_full_draw, every-{args.sample_every}): {len(pose_shots)} shot(s)")
    print(f"  Audio onsets (raw 2s-gap):                  {len(audio_onsets)} candidate(s)")
    print(f"  Match tol={args.tol}f  loc-warn>{args.loc_warn}f  strong-audio floor str>={strong_floor:.4f}")
    print(f"  {'-'*64}")
    print(f"  {'pose':>7} {'hold':>5}   {'audio':>7} {'str':>7} {'Δf':>4}   {'de_dy':>6}   status")
    for r in rows:
        ps = r["pose"]; ao = r["audio"]
        a_f = f"{ao['frame']:>7}" if ao else f"{'--':>7}"
        a_s = f"{ao['strength']:>7.4f}" if ao else f"{'--':>7}"
        d = f"{r['delta']:>4}" if r["delta"] is not None else f"{'--':>4}"
        de = f"{r['de_dy']:>6.3f}" if r["de_dy"] is not None else f"{'--':>6}"
        flag = "  ⚠" if r["status"] in ("POSE_ONLY", "LOC_WARN") else ""
        print(f"  {ps['frame']:>7} {ps['hold_s']:>5}   {a_f} {a_s} {d}   {de}   {r['status']}{flag}")
    for a in strong_unmatched:
        print(f"  {'--':>7} {'--':>5}   {a['frame']:>7} {a['strength']:>7.4f} {'--':>4}   {'--':>6}   AUDIO_ONLY  ⚠")
    if vibration:
        print(f"  ({len(vibration)} unmatched audio onset(s) suppressed as post-shot vibration/follow-through)")
    weak_unexplained = [a for a in unexplained if a not in strong_unmatched]
    if weak_unexplained:
        print(f"  ({len(weak_unexplained)} weak unexplained onset(s) below strong-audio floor — likely ambient noise)")

    # ---- shot accounting (clicker is the shot/let-down discriminator; §26) ----
    # CONFIRMED SHOT = pose hold matched to a clicker onset (AGREE/LOC_WARN).
    # Without lavalier: a clicker-less hold is UNRESOLVED (quiet-clicker shot OR
    #   let-down — pose gross-motion can't decide; §24/§26).
    # With lavalier: the clicker is authoritative, so a clicker-less hold IS a
    #   LET-DOWN (design §5). This is the whole point of the mic.
    def is_matched(r): return r["status"] in ("AGREE", "LOC_WARN")
    confirmed = [r for r in rows if is_matched(r)]
    clicker_less = [r for r in rows if not is_matched(r)]
    n_audio_only = len(strong_unmatched)

    print(f"  {'-'*64}")
    if args.lavalier:
        # Clean-clicker view: clicker-less holds are let-downs. Only a strong
        # clicker with NO full-draw hold (pose miss or stray noise) needs review.
        letdowns = clicker_less
        clean = (n_audio_only == 0)
        n_draws = len(confirmed) + len(letdowns)
        if clean:
            print(f"  VERDICT: ✅ AUTO-ACCEPT — {len(confirmed)} shots + "
                  f"{len(letdowns)} let-down(s) = {n_draws} draws (lavalier: clicker authoritative).")
        else:
            print(f"  VERDICT: 🔎 REVIEW — {n_audio_only} strong clicker(s) with no full draw "
                  f"(pose miss — esp. target foreshortening — or stray noise).")
        print(f"  SHOTS ({len(confirmed)}): {sorted(r['pose']['frame'] for r in confirmed)}")
        print(f"  LET-DOWNS ({len(letdowns)}): {sorted(r['pose']['frame'] for r in letdowns)}")
    else:
        unresolved = clicker_less
        clean = (not unresolved and n_audio_only == 0)
        if clean:
            print(f"  VERDICT: ✅ AUTO-ACCEPT — {len(confirmed)} shots (all clicker-confirmed).")
        else:
            bits = []
            if unresolved:   bits.append(f"{len(unresolved)} UNRESOLVED draw(s) {[r['pose']['frame'] for r in unresolved]} — no clicker; shot(quiet clicker) OR let-down, pose can't decide -> lavalier/manual")
            if n_audio_only: bits.append(f"{n_audio_only} strong audio-only (clicker, no full draw — noise or pose miss esp. target)")
            print(f"  VERDICT: 🔎 REVIEW — " + ("; ".join(bits) if bits else "see above"))
        print(f"  CLICKER-CONFIRMED SHOTS: {len(confirmed)}  ->  {sorted(r['pose']['frame'] for r in confirmed)}")
        if unresolved:
            hint = ", ".join(f"{r['pose']['frame']}={r['de_dy']:.3f}" for r in unresolved if r["de_dy"] is not None)
            print(f"  UNRESOLVED DRAWS (clicker-less): {sorted(r['pose']['frame'] for r in unresolved)}  (de_dy hint: {hint})")

    # ---- optional ground-truth comparison ----
    if args.session_json and os.path.exists(args.session_json):
        gt = json.load(open(args.session_json)).get("metrics", {}).get("verified_frames", [])
        if gt:
            print(f"  {'-'*64}")
            print(f"  Ground truth (verified_frames): {len(gt)} -> {gt}")
            tol = args.tol
            hits, errs = 0, []
            for g in gt:
                near = [p["frame"] for p in pose_shots if abs(p["frame"] - g) <= tol]
                if near:
                    hits += 1
                    errs.append(min(abs(n - g) for n in near))
            miss = len(gt) - hits
            extra = len(pose_shots) - hits
            print(f"  Pose-primary recall: {hits}/{len(gt)}"
                  + (f"  | missed {miss}" if miss else "")
                  + (f"  | {extra} extra pose detn(s)" if extra > 0 else ""))
            if errs:
                print(f"  Localization error vs GT: max {max(errs)}f, mean {sum(errs)//len(errs)}f")

    print()


if __name__ == "__main__":
    main()

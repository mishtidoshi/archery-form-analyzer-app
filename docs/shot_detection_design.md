# Shot Detection — Design & Architecture

**Status:** Proposed design (2026-05-31). Supersedes the implicit audio-primary design.
**Rationale sources:** audio-primary failures, let-down-vs-shot ambiguity at anchor, and the finding that pose gross-motion alone can't separate a shot from a let-down.

---

## 1. Why this redesign

The original pipeline was **audio-primary**: the clicker onset *triggered* a candidate, and a visual anchor check *gated* it. The 5/26 verification exposed three structural failures:

- **§24 — audio recall ceiling.** Sub-threshold clickers produce no candidate at all; a louder non-shot burst can suppress a real shot (NMS); a weak blip can mis-localize a shot. View-dependent: worsens the further the mic sits behind the archer (face < back < target).
- **§25 — a LET-DOWN is visually identical to a SHOT at the anchor frame.** A shot is defined by the *release*, which happens *after* the hold; the anchor pose is the same for both.
- **§26 — pose gross-motion cannot separate shot from let-down.** The discriminating release impulse is swamped by the bow-lowering both actions share, occluded in back view, and under-sampled at 30 fps. (A `de_dy` motion gate passed n=8 calibration by window-timing luck, then got 2 of 3 wrong on ground truth.)

The net lesson: **don't make one sensor do a job it's physically bad at.** Decompose the problem so each sensor does only what it's reliably good at.

---

## 2. Core principle

> **A shot is defined by *how the full-draw hold ends* (a release) — not by the hold itself. The hold is a variable-duration STATE; detect its boundaries, never assume a length.**

This single principle resolves both failure classes:
- It separates a *draw* (which pose detects well) from a *shot* (which requires the release).
- It removes any dependence on hold duration — the archer may hold 0.4 s or 3 s, or let down entirely.

---

## 3. The shot cycle as a state machine

```
 IDLE ─raise→ DRAW ─settle→ FULL-DRAW HOLD ─┬─ RELEASE  ─→ FOLLOW-THROUGH ─→ IDLE    (= SHOT)
   ▲                        (variable len)  │
   └─────────────────────────────────────── └─ LET-DOWN ─→ (controlled lower) ─→ IDLE (= NOT a shot)
```

- **HOLD** = draw elbow raised (≈ shoulder height) + low velocity, for ≥ ~0.2 s. Variable length.
- **Exit = RELEASE** → arrow leaves; clicker fires; brief explosive impulse + follow-through.
- **Exit = LET-DOWN** → string eased back to brace; bow lowered; no clicker; no arrow.

---

## 4. Sensor responsibilities (each only does what it is reliable at)

| Sensor | Reliably detects | Does NOT do | Failure mode |
|--------|------------------|-------------|--------------|
| **Pose (YOLO)** | The full-draw **HOLD** and its start/end — high recall on *draws*, any duration; a clean, stable full-draw frame | Distinguish shot from let-down (§26) | Foreshortening in target view; draw hand occluded in back view |
| **Clicker audio** | The **RELEASE** instant — the sport's purpose-built release sensor; fires only on a real loose, never on a let-down | Be heard when the mic is behind the archer | Sub-threshold when mic is far/behind (→ fixed by lavalier, §7) |

The two weaknesses **do not overlap** — that is what makes their combination robust.

---

## 5. Detection pipeline

**Stage 1 — Segment draws (pose).** Scan pose for full-draw HOLD segments → list of `(hold_start, hold_end)`. High recall on draws. Duration-agnostic. (Evolution of `find_full_draw_frames.py`, returning *boundaries* not the mid-hold frame.)

**Stage 2 — Classify each exit (clicker).**
- `SHOT` — a hold whose **end coincides with a clicker** (within ~0.3 s).
- `LET-DOWN` — a hold with **no clicker** at its end.
- `UNRESOLVED` — disagreement (hold with no clicker where one was expected, or clicker with no hold) → **flag for review; never guess.**

**Stage 3 — Canonical full-draw frame.** Anchor it to the **release** (e.g. last stable hold frame before the loose ≈ the archer's labeling convention), *not* the middle of the hold. Pose metrics are flat across the hold, so this is both robust and consistent regardless of hold length. (This fixes the localization deltas seen on 5/26, where long holds put the mid-hold frame 30–60 frames before the release.)

**The combination rule is the whole design and is intentionally simple:**
> SHOT = full-draw hold whose end coincides with a clicker. LET-DOWN = full-draw hold with no clicker. Disagreement = flag.

Tooling that implements the cross-check today: `tools/cross_validate_shots.py` (pose holds vs raw audio onsets; reports CONFIRMED / UNRESOLVED / flags). **Lavalier-ready (2026-07-04):** its `--lavalier` flag makes the clicker authoritative — a full-draw hold with no clicker is reported as a LET-DOWN (not UNRESOLVED), implementing the Stage-2 rule above for the clean-clicker view. Use it only on the lavalier-equipped view. Default (ambient) mode validated on the 5/26 face gold-standard: 6/6 pose recall, all shots clicker-confirmed at ground-truth frames.

---

## 6. The lavalier mic is the linchpin (prerequisite, not a future nice-to-have)

The entire design rests on the clicker being reliable. The 5/26 failures were fundamentally a **mic-placement artifact** (phone mic behind the archer → quiet clicker in back/target). A single wireless lavalier **on the archer** gives a loud, clean clicker regardless of camera view — making Stage 2 dependable.

It also **future-proofs** the design (see §9): with multiple archers present, only *our* archer's clicker is on *our* mic, so the lavalier disambiguates the target archer for free.

**Status:** mic ordered 2026-05-31, ETA ~1 week.

---

## 7. Interim strategy (until the lavalier arrives, ~1 week)

Pose-motion cannot resolve shot-vs-let-down (§26), so do **not** rely on it. Instead:

1. **Reconcile pose-draws against the per-arrow SCORE log (primary).** A let-down is never scored. So **scored arrows = shots**; **pose draws − scored arrows = let-downs**. The score sheet (already kept for score-correlation and coach feel-ratings) encodes the true shot count at zero extra CV cost. Resolve *which* draws were let-downs by (a) a tick on the scorecard at let-down, (b) the face-view clicker where clean, or (c) sequence position.
2. **Face view is already clicker-resolved** (mic faces the clicker) — the interim gap is only back/target, which (1) covers.
3. **Exploratory (TESTED 5/31 — mostly negative):** tested whether a let-down's hold-exit is a *gradual ramp* vs a shot's *sharp impulse* (jerk = max single-frame speed jump) on the 5/26 labeled set. **Draw-elbow & draw-wrist jerk FAIL** (let-downs are equally/more jerky — lowering the bow is itself jerky). **Bow-hand recoil jerk separates only in TARGET view** (shots ≥1.1 vs let-downs ≤0.65) and FAILS in back view (bow far from camera → recoil not captured; shots ~0.4 ≈ let-downs). Conclusion: **no robust cross-view motion discriminator at 30 fps; bow recoil is a view-specific corroborator at best.** Higher fps wouldn't fix back view (limited by *spatial* resolution, not temporal). ⇒ rely on the clicker; don't build a motion-based let-down classifier.

Honest framing: (1) is reliable and bridges the gap; (2)–(3) are supplements. The robust automated solution is the lavalier.

---

## 8. What we deliberately do NOT build (keep it simple)

Deferred unless the clicker backbone proves insufficient: ML sequence/state models, arrow-departure CV, bow-limb vibration analysis, multi-camera sync. We explicitly do **not** try to solve shot-vs-let-down with clever pose-motion features — §26 showed that is a dead end.

---

## 9. Future extensibility (beyond backyard 18 m)

The backyard constraint removes confounds *while we build and validate*; the design itself does not depend on it.

- **Other distances / indoor:** the clicker is range- and venue-independent. No change.
- **Other / multiple archers:** lavalier isolates our archer's clicker; pose largest-bounding-box selects our archer. The state machine is per-archer.
- **The core pipeline (segment draws → classify exit by release) is archer- and venue-agnostic.**

---

## 10. Validation plan

- **Ground truth:** the 5/26 session is fully archer-verified — shot frames *and* let-downs (e.g. IMG_3125: 6 shots at 615/1593/2381/2831/3303/3829, let-downs at ~247 and ~1167; IMG_3128: 5 shots, 2 let-downs at 660/1620; IMG_3129: 6 shots, 1 let-down at 2240). Use it as the labeled test set for any implementation.
- **Metrics:** per-clip recall (draws found / true draws), shot precision (clicker-confirmed shots / true shots), let-down detection, and frame-localization error vs the archer's release-anchored frames.
- **Acceptance:** ~100 % shot recall + zero let-downs miscounted as shots on the labeled set, with the lavalier in place.

---

## 11. Open questions

- Does a let-down's hold-exit differ from a shot's by *sharpness/jerk*? (§7 experiment; untested.)
  - **Frame rate is a lever here.** The release transient is ~1 frame at 30 fps (under-sampled — shape unresolved), ~2–4 at 60 fps, ~4–8 at 120 fps; a let-down ramp is ~10–15 frames at 30 fps (already resolved). So higher fps mainly sharpens the *shot* side and reduces release-frame motion blur — it would make the jerk discriminator more viable. It does NOT fix back-view occlusion or the gross-motion confound, and does not beat the clicker. Decide empirically: run the jerk test on existing 30 fps data first; if marginal, shoot a few 120 fps test ends (mixed shots + let-downs). Higher fps also benefits release/expansion kinematics (T2.4).
  - **Processing cost is flat, not 2–4×**, *if* hold detection samples to a constant temporal density (e.g. one pose per ~0.1 s → stride 3 at 30 fps, 6 at 60, 12 at 120 → same inference count); dense pose runs only in the brief release window. The 2–4× blowup only happens with the old every-frame pipeline.
  - **⚠ Audio caveat — 60 fps vs 120 fps differ critically.** 1080p**60** is a *standard* video mode: clean audio, normal container — low risk. **120/240 fps is iPhone "Slo-mo" mode**, which can time-stretch/degrade audio and produce container quirks (we've been bitten before — 5/10 session, IMG_3068/3078). Since the **clicker audio is the backbone of this design**, 120 fps must be audio-verified before it's trusted, or used only for motion-only tests. Prefer 60 fps if stepping up.
  - **Other downsides:** 2×/4× storage + per-session transfer time; mixed-fps dataset (frame rate becomes a variable to control in the longitudinal study; historical 30 fps clips can't be retro-upgraded).
- Best definition of the canonical full-draw frame relative to the release for cross-session metric consistency (fixed offset vs last-stable-frame). **RESOLVED 5/31 (task #2): snap the finder's mid-hold frame forward to the release/let-down motion onset (draw-elbow speed spike, visible even in back view) and take onset − 5 frames. `release_probe.snap_to_release()`, wired into `process_session.py`. Validated on 5/26: ±2f on all IMG_3125 shots (long-hold case fixed from −54f to −1f), exact on IMG_3123 (140→151 = archer frame). Metrics are flat across the hold so values are unchanged — this is purely label consistency.**
- Target-view pose robustness (foreshortening) — acceptable once the clicker is reliable, but worth measuring.

---

*Last updated: 2026-05-31.*
*Last updated: 2026-05-31.*

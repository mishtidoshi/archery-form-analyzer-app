# Archery Form Analyzer — Complete Workflow

**Last reviewed: 2026-08-07**

---

## Overview

Most archers get coaching once a week. They practice the other six days with no feedback, no data, and no way to know if they're reinforcing good form or bad habits.

This system changes that.

Record your practice from up to three camera angles, run the pipeline, and get quantitative biomechanical feedback on every single shot — anchor consistency, elbow position, shoulder elevation, follow-through, head stability — in minutes. No motion capture lab. No expensive equipment. Just Phones, a lavalier mic, and a 10-minute post-session recap.

**What makes it different:**
- **Automatic shot detection** — audio onset (clicker) + visual pose confirmation working together. Not just one signal, both. False positive rate: 0%.
- **Multi-view analysis** — face, back, and target cameras each capture different aspects of form. Metrics are normalized to be camera-invariant, so you can compare across sessions even if the camera moved.
- **Every shot, every session** — trends build up automatically over your own longitudinal data, revealing what coaching alone can't see: which improvements stuck, which didn't, and where form silently drifts between sessions.
- **Coach-aligned feedback** — log your coach's cues in the app, and the improvement bullets can match the language your coach already uses. When the pipeline flags a metric, it tells you what to fix in those terms.
- **Tabular, easy-to-analyze output** — one row per shot, all metrics, ready to open in any spreadsheet or feed into score correlation analysis. 
- **Training-load and form-degradation monitoring** — per-session time under tension (arrows × hold duration) plus within-session drift in the load-bearing metrics. Within-session comparison is the most reliable axis available, because the camera does not move during a session, so it is unaffected by the cross-session camera-geometry confound that cross-session comparisons need to control for.

> **This is not a medical or injury-prevention tool.** It measures joint angles and timing from video. It cannot diagnose, predict, or prevent injury, and no injury-related claim here has been validated against any clinical outcome. Several metrics correspond to loading patterns that the sports-medicine literature associates with overuse — bow-shoulder elevation with subacromial impingement, chin-drop with sustained cervical flexion under load, a locked bow elbow with joint rather than muscular loading — but a correspondence is not a risk assessment. **Pain, or any suspected injury, goes to a physiotherapist or physician, not to this pipeline.**

**Stack:** Python · YOLO26 pose · OpenCV · NumPy · Matplotlib · SciPy · FastAPI · React (mobile web app)

---

## Mobile Web App (`mobile_app/`)

A mobile-first web app for uploading videos, viewing session metrics, trend charts, and
generating AI coaching feedback — accessible from an iPhone on the same Wi-Fi as the Mac
running the backend.

### Start the app

```bash
# Terminal 1 — FastAPI backend (port 8000)
cd mobile_app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir . --reload-dir ../../tools

# Terminal 2 — React frontend (port 5173, dev only)
cd mobile_app/frontend
npm install
npm run dev
```

Then open `http://<Mac-local-IP>:5173` on your iPhone (same Wi-Fi).

**Why `--reload-dir` matters here:** plain `--reload` only watches the directory uvicorn
was launched from (`mobile_app/backend/`). `main.py` imports several modules from
`tools/` (`coaching_feedback.py`, `metric_thresholds.py`, `hold_time.py`) that live
outside that tree — without `--reload-dir ../../tools`, editing one of those files does
*not* restart the server, so the running process keeps serving old logic while `git diff`
shows the fix already landed. Found 2026-08-29 debugging a coach-hiding fix that tested
correctly in isolation but didn't show up in the live app until a manual restart.

### Production build (FastAPI serves the React app)
```bash
cd mobile_app/frontend && npm run build
# FastAPI auto-serves dist/ at port 8000 — only one process needed
```

### App screens

| Screen | What it does |
|--------|-------------|
| **Welcome** (`/welcome`) | Splash screen shown to a logged-out visitor before anything else — app name + "Get Started" button leading to the profile picker. |
| **Profiles** (`/profiles`) | Netflix-style archer picker. Pick a profile → enter its password (or set one, for a profile that's never had one) → lands on Home. A trailing "Add Archer" tile creates a brand-new profile. See **Archer accounts** below. |
| **Upload** | Pick a video from camera roll, select view type (face/back/target), enter metadata, submit. Backend runs auto-detection + YOLO analysis in background; job status is polled every 2 seconds. |
| **Sessions** | List of all sessions from `session_history/` **belonging to the signed-in archer**, grouped by date with per-view color coding. Flagged/good metric counts shown on each card. |
| **Session Detail** | All computed metrics shown as range bars (green band = good range, colored dot = your value). Tap "Generate Coaching Feedback" for a deterministic, cue-based summary — no AI model, no API key (see below). |
| **Trends** | Per-view line charts over time for every metric, scoped to the signed-in archer. Dots are green (in range) or red (flagged). Green band shows good range. Tap any point to go to that session. |
| **Coach Notes** | Single-purpose structured form to log a new coach cue (coach / date / category / context) — appends to `archers/{archer_id}/coach_voice.md`. No raw-file viewing/editing in the app (see **Coach confidentiality** below). The coach picker is per-archer — built from `profile.json`'s `coaches` block (`auth.get_archer_coaches()`), never hardcoded, so a new profile starts with none — and shows each coach's real name, since that data belongs to (and is only ever visible to) the signed-in archer. Picking "Other" and typing a name (`auth.add_archer_coach()`) saves it into that archer's `coaches.guest`, so it appears as its own button from then on; the fixed "Self observation"/"Other" options are never persisted as coaches. A coach entry with `hidden_from_app: true` doesn't appear at all. |
| **Settings** (`/settings`, gear icon on Home) | Edit display name, change password (requires current password), and a danger-zone **Delete Account** (requires current password + typing `DELETE`) — see below. |

Home has a "Switch" button that logs out and returns to the profile picker, and a gear icon that opens Settings.

### Account settings — name, password, delete

All three actions in Settings resolve the archer from the bearer token server-side
(`PUT/POST /api/archers/me...`) — the client never supplies whose account to change, so
there's no way to edit or delete anyone but yourself.

- **Change name** — updates `identity.name` in `profile.json`. Unlike the auth bootstrap
  (which never touches `profile.json`), this is an intentional edit to the file's actual
  content, so a formatting reflow on save is an accepted trade-off here.
- **Change password** — requires the current password. Revokes every existing bearer
  token for that archer (all devices/tabs get logged out) and issues a fresh one for the
  session that just made the change.
- **Delete account** — requires the current password *and* typing `DELETE` to confirm.
  Deletes `archers/<id>/` entirely (profile, auth, coaching notes) **and every
  `session_history/*.json` belonging to that archer** — this was a deliberate choice when
  the feature was built (full erasure, not just a login lockout), so treat it as
  irreversible. Revokes all tokens for that archer.

### Coach names and confidentiality in the app

A coach's real name is shown throughout the app — in the Coach Notes picker, and in
generated coaching feedback whenever a cue or active-focus item credits them (e.g.
"sink the scapula (Coach Park)"). Privacy here is about **isolation between archers**,
not about hiding a coach's identity from the archer they coach:

- Coach names live only in that archer's own `archers/<id>/profile.json`
  (`coaches.primary` / `coaches.guest`). Every read of them
  (`auth.get_archer_coaches()`, `GET /api/coach-notes`, `load_coaching_context()`) is
  scoped to the signed-in archer's own profile via their bearer token — a different
  archer's `GET /api/coach-notes` call reads a different `profile.json` and can never
  see these names. No endpoint returns another archer's coach list.
- `auth.json` (the password hash) is a separate, gitignored file — never bundled with
  `profile.json`, so cloning or inspecting the repo doesn't expose credentials even
  though coach names in `profile.json` are plain text.

**Excluding a coach's advice entirely — `hidden_from_app`.** Set
`"hidden_from_app": true` on a coach's entry in `coaches.primary` or `coaches.guest`
(see the `_hidden_from_app_note` alongside it) to drop that coach's content from
generated feedback altogether, for an archer who wants to stop seeing one coach's
input without deleting their history:
- Coach Notes' "Add Cue" picker won't offer them (`auth.get_archer_coaches()` skips
  hidden entries by default; `include_hidden=True` is used internally by
  `add_archer_coach()`'s duplicate check, so a hidden coach re-added via a raw API call
  doesn't get a second entry)
- generated feedback's "Active coaching focus" list drops any item naming them
  (`_hidden_coach_labels()` + the filter in `load_coaching_context()`) — each item is
  one coach's homework, so hiding the coach hides their whole item, not just their name

This is display-only and non-destructive either way — `coaching_log.md`,
`coach_voice.md`, and `session_history/*.json` are never edited or deleted by it.

**Coach Notes has no raw-file display.** `coaching_log.md` and `coach_voice.md` are
hand-authored, cross-referential prose — a section headed by one coach can name
another mid-paragraph. There's no reliable way to mechanically apply the
`hidden_from_app` exclusion to prose like that without either missing a mention or
mangling the other coach's text, so `GET /api/coach-notes` never returns their raw
content — not "the frontend doesn't render it," the backend genuinely never transmits
it. The Coach Notes screen is single-purpose: log a new cue via the structured
"+ Add Cue" form only (the sole write path, append-only, safe).

### Archer accounts

The app supports more than one archer profile on the same deployment — each gated by its
own password (Netflix-profile style, not full user accounts: no email, no server-side
session store beyond an in-memory bearer token that resets on backend restart). See
`mobile_app/backend/auth.py` for the full rationale.

- **New profiles** are created from the "Add Archer" tile → `POST /api/archers`, which
  writes a fresh `archers/<id>/profile.json` (see `archers/README.md`) with identity/name
  filled in and everything else left `null`/`"TBD"` for the archer or a coach to fill in later.
- **A hand-authored profile** (created by copying the schema directly instead of through
  the app) has no password yet. The picker shows "Set up password" for it instead of
  "Enter password" — this is a one-time bootstrap (`POST /api/archers/{id}/set-password`),
  not a login, and it only works once; after a password exists, the same profile requires
  the real login route.
- **Credentials live in `archers/<id>/auth.json`, not in `profile.json`.** `profile.json` is
  meant to be hand-curated (coach notes, identity, study setup); round-tripping it
  through the JSON writer on every password change would silently collapse its formatting.
  `auth.json` is gitignored — password hashes never leave the machine they were set on.
- Every session, trend, coach-note and upload endpoint is scoped to the authenticated
  archer via a bearer token (`Authorization: Bearer <token>`, issued at login/create/set-password).
  A session belonging to a different archer 404s rather than 403s, so existence isn't leaked.
- Every session JSON should carry an `archer_id` field so it's attributed correctly; the
  app sets this automatically for anything created through the upload flow or CLI tools.

### Backend endpoints (all `/api/...`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/archers` | GET | Public roster for the profile picker (id, name, has_password) — no auth |
| `/api/archers` | POST | Create a new archer profile; returns a bearer token |
| `/api/archers/{id}/set-password` | POST | One-time bootstrap for a profile with no password yet |
| `/api/auth/login` | POST | Log in to an existing profile; returns a bearer token |
| `/api/auth/logout` | POST | Invalidate the current bearer token |
| `/api/archers/me` | PUT | Change your display name |
| `/api/archers/me/change-password` | POST | Change your password (requires current password); returns a fresh token |
| `/api/archers/me/delete` | POST | Permanently delete your account **and all your sessions** (requires current password) |
| `/api/sessions` | GET | List practice sessions (cards) for the signed-in archer |
| `/api/sessions/{id}` | GET | Full session detail + enriched metrics (404 if not yours) |
| `/api/sessions/{id}/feedback` | POST | Generate coaching feedback (deterministic, no API key) |
| `/api/trends` | GET | Time-series per metric, all views, for the signed-in archer |
| `/api/upload` | POST | Upload video → background analysis job, tagged with your archer_id |
| `/api/jobs/{job_id}` | GET | Poll analysis job status (404 if not yours) |
| `/api/coach-notes` | GET | Your visible coach names only — no raw file content (see **Coach confidentiality**) |
| `/api/coach-notes/entry` | POST | Append a structured cue entry to `coach_voice.md` |

All routes except `/api/archers` (GET), `/api/archers` (POST), `/api/archers/{id}/set-password`,
and `/api/auth/login` require the `Authorization: Bearer <token>` header.

### Auto-analysis pipeline (upload flow)
1. Video saved to `data/mobile_uploads/{YYMMDD}/{view}view/`
2. Draft session JSON written to `session_history/` immediately (status: `pending_analysis`)
3. `archery_analyzer_v2.py --{view} ... --session-json <draft>` runs for shot detection —
   `--session-json` (added 2026-08-15) writes `metrics.verified_frames` straight into the
   draft; without it, the script only produces charts/video in `output/` and never touches
   `session_history/` at all, which is what it was doing here for months (see below).
4. If frames detected → `analyze_{view}_yolo.py` runs for YOLO metrics, writing into the
   same draft JSON
5. Job status updates to `done`; if no shots auto-detected, the draft JSON remains for
   manual CLI verification (`verify_session_frames.py`, then `analyze_{view}_yolo.py`).
   If detection or YOLO metrics *crash*, the job status is `failed` with the last lines of
   stderr in the message — distinct from "ran cleanly, found 0 shots."

**Bug history (2026-08-15):** two separate bugs meant every mobile-app upload silently
fell through to "no shots detected," even when detection actually worked:
- `AngleProcessor._extract_*` unconditionally called `self.detector.update(..., wrist_y=...)`
  after the BB11 hold-time fix (2026-07-28), but only the legacy `ShotDetector` class
  accepts `wrist_y` — `AudioVisualShotDetector` (the default detector) and `MLShotDetector`
  don't, so the very first confirmed shot crashed the whole script. Fixed via
  `AngleProcessor._update_detector()`, which only passes `wrist_y` to the class that
  understands it.
- Independently, `mobile_app/backend/main.py`'s upload job assumed `archery_analyzer_v2.py`
  wrote a session JSON as a side effect and globbed for one — it never did, in single-clip
  CLI mode, so this always found nothing regardless of whether detection succeeded. Fixed
  by adding `--session-json` (see step 3 above) and having the job read frames back out of
  the draft it already knows about, instead of searching for a file that was never created.

### Coaching feedback — fully local, no external API
The "Generate Coaching Feedback" button on Session Detail / Coaching calls
`coaching_feedback.synthesize_manual()`, which dispatches between two entirely local
renderers, both built strictly from the rule engine's findings and never inventing or
restating a measured number:

- **Basic recommendations** (`_compose_basic()`) — a fixed bullet-point template, used
  whenever the archer hasn't logged enough of their own coaching text yet. This is the
  default for a new archer and stays exactly this simple until there's real material
  to work with — never broken, never sparse-looking.
- **Rich mode** (`tools/feedback_nlg.py`'s `compose_rich()`) — once an archer has
  logged enough of their own coach's cues (via the app's Coach Notes screen, into
  `archers/<id>/coach_voice.md`; currently gated at `MIN_WORDS_RICH = 150` words),
  a hand-built rule-based NLG engine writes several varied, full sentences per
  finding — template banks with multiple phrasings, selected and composed in code
  (no trained model, no external API), personalized with short fragments generated
  from the archer's own text via `tools/coach_voice_model.py`'s from-scratch n-gram
  model. Hitting "Regenerate" re-rolls the phrasing (randomization is unseeded).

No network call, no API key, nothing external in either path. See
`synthesize_manual()`'s docstring for the exact dispatch logic, and
`tools/feedback_nlg.py`'s / `tools/coach_voice_model.py`'s module docstrings for why
rule-based templates (not a trained model) are the right architecture at this data
scale.

---

## 1. Camera views and what they measure

Each view is a separate camera, recording independently. You can use any combination of 1, 2, or 3 views per session.

### Face view
Camera perpendicular to the shooting line, on the archer's face/anchor side.

| Metric | What it means | Good range |
|--------|--------------|------------|
| Draw elbow angle | Angle at the draw elbow at full draw | **Observation-only.** The two bands previously in use (20–40° / 15–35°) had no recorded source, and the value is camera-geometry confounded (BB6). Coach target TBD. |
| Bow elbow angle | Angle at the bow elbow | 160–175° (technique-derived) |
| Anchor X / Anchor Y | Normalized position of draw wrist at anchor | Low spread = consistent anchor |
| Anchor spread (px) | Shot-to-shot anchor spread, **measured relative to the nose** (`anchor_spread_head_px`) | ≤ 10px. The image-relative figure is reported but not judged — it cannot separate "hand moved on the face" from "archer moved in frame" (BB2). |
| Nose-string gap | Distance between nose and string at anchor | ≈ 0 (light contact); negative = squish |
| Bow shoulder elevation | Normalized ear-to-shoulder height (`_sw` variant) | **Observation-only** — 0.60 is where this archer sits, not a target. Coach threshold TBD. |
| Hold time | Seconds from full draw to release | 1.0–3.0s (coach-stated ideal) |
| Nose pre-anchor drift | Chin drop (px) during draw approach | < 2px |
| ~~Draw elbow follow-through~~ | *Retired (BB10)* — documented but never compared or flagged | — |

### Back view
Camera perpendicular to the shooting line, facing the archer's spine (T-shape).

| Metric | What it means | Good range |
|--------|--------------|------------|
| Shoulder level | Tilt of shoulder line at full draw | Within ±3° |
| Hip alignment | Tilt of hip line | Within ±5° (as implemented; the previously documented ±3° had no source) |
| Head lateral tilt | Ear-to-ear line tilt | Within ±5° |
| T-draw angle | Tilt of the bow-elbow → draw-elbow line | Observation-only. Negative = draw elbow above the bow elbow = correct. Magnitude is camera-dependent (18m backyard ≈ −13°, 60m ≈ −5°) — compare SD, not value. Coach target TBD. |
| Bow shoulder elevation (`_sw`) | Camera-invariant shoulder depression | **Observation-only** — descriptive centre 0.60, not a target |

### Target view
Camera directly behind the archer along the arrow line, facing the target.

| Metric | What it means | Good range |
|--------|--------------|------------|
| Draw elbow height | Elbow height relative to shoulder (normalized) | Negative = above shoulder ✓ |
| Draw elbow lateral | Elbow flare behind body line | Low = tucked ✓ |
| Back tension proxy | Normalized shoulder width (scapula engagement) | **Observation-only** — session-relative by construction; no cross-session band is meaningful |
| ~~Shoulder rotation~~ | *Retired (BB10)* — documented but never computed; no session record carries the field | — |
| Bow shoulder elevation (`_th`) | Camera-invariant using torso height | **Observation-only** — descriptive centre 0.40 |

---

## 2. Hardware setup

### Cameras
- Phone(s) on tripods at **marked fixed positions** (tape marks on the ground)
- Settings: **1080p · 30fps · Landscape · Video stabilization OFF · Mono audio**
- Do not move tripods between sessions — consistency is what makes longitudinal comparison valid

### Audio
- Single wireless lavalier mic on the archer (TX clipped at bow-side collar)
- Receiver (RX) plugged into one phone **before** opening the camera app
  - iOS silently falls back to built-in mic if you plug in after — always plug in first
- The other phones use their built-in ambient mics
- The lavalier view captures the clicker much louder than ambient — this is what the detector relies on

### Why the lavalier microphone matters
The clicker sound is the most reliable signal for shot detection. From behind the archer (back/target views) the ambient phone mic is shielded by the archer's body and catches the clicker weakly. The lavalier mic is on the archer's body — it captures the clicker loudly and consistently regardless of which direction the camera faces. This makes detection reliable on all three views without any manual correction.

---

## 3. Recording protocol

**Before first arrow of each end:**
1. Press RECORD on all phones
2. Shoot the end (3–6 arrows)
3. Press STOP after the **last arrow's full follow-through** — do not cut mid-draw

**Score each arrow immediately after it lands**, before pulling arrows:
- Call out or write the score as soon as you can read it from the target face
- Do **not** wait until all arrows are pulled — once arrows are pulled you lose the per-arrow position
- Note any **let-downs** in order (drew up but did not release) — these are not scored and must be tracked so the pipeline doesn't count them as shots

**Per-arrow log format (in order of shooting):**
```
Arrow 1: 10   Arrow 2: let-down   Arrow 3: 9   Arrow 4: 10
```
- Arrow score (X, 10, 9, 8 ...)
- Let-downs marked explicitly — they consume time in the video and look like shots at anchor
- Optional: feel rating 1–10 per arrow

> **Why order matters:** The pipeline assigns metrics to shots by order of detection in the video. If you score arrows in a different order than you shot them, the metrics and scores will be mismatched. Score in shooting order, every time.

**Per-end overhead:** ~30 seconds. Recording itself is passive — just start/stop.

---

## 4. Video intake — the dropbox workflow

After a session, move video files to the computer via AirDrop or USB.

### Step 1 — Drop into `dropbox/`

Drop all video files directly into `dropbox/` — no subfolders needed. The sort script reads the filename prefix (face/back/target camera assignments are inferred from how you name the files or from metadata) and places each file in the right view folder automatically.

If you do use subfolders, any of these names work: `face`, `faceview`, `back`, `backview`, `rearview`, `rear`, `target`, `targetview`.

### Step 2 — Preview the sort

```bash
python3 tools/sort_dropbox.py
```

Shows where each file would go and what date was read from the video metadata. Nothing is moved yet.

### Step 3 — Execute

```bash
python3 tools/sort_dropbox.py --move
```

Files land in:
```
data/{month_name}_{year}/{MMDDYY}/{viewname}/IMG_NNNN.MOV
```
Example: `data/july_2026/070726/faceview/IMG_5001.MOV`

---

## 5. Shot detection

### How the detector works

The **AudioVisual shot detector** is the default. It uses two independent signals:

1. **Audio:** high-pass filtered (3 kHz+) RMS onset detects clicker candidates in the audio track
2. **Visual:** YOLO pose checks whether the draw wrist is in the anchor zone at each audio candidate

A shot is confirmed only when **both signals agree** — an audio candidate near the anchor position. This prevents false positives from noise, bystanders' clickers, or ambient sounds.

**Audio source by view:**
- **Face view:** The lavalier mic (clipped at bow-side collar) faces the clicker directly — clicker is the loudest event. Detector picks the strongest audio candidate per shot interval (`prefer_strongest_audio=True`).
- **Back / target view:** The lavalier mic still provides the clicker signal even from behind the archer. Detector iterates candidates chronologically and checks each against the anchor zone (`prefer_strongest_audio=False`), because from behind, post-release bow-arm vibration can be louder than the clicker — strength-based picking would fire on follow-through instead of full draw.

All three views rely on the lavalier mic for reliable detection. Without it, back/target clips use the ambient mic and require more care during detection.

### Running the detector

**Single clip:**
```bash
python3 archery_analyzer_v2.py --face data/july_2026/070726/faceview/IMG_5001.MOV --min-time 10
```

**Multiple views at once:**
```bash
python3 archery_analyzer_v2.py \
    --face   data/july_2026/070726/faceview/IMG_5001.MOV \
    --back   data/july_2026/070726/backview/IMG_5002.MOV \
    --target data/july_2026/070726/targetview/IMG_5003.MOV \
    --min-time 10
```

**Useful flags:**
```
--min-time 10     minimum seconds between shots (use 10 for ~3-6 arrow ends)
--no-video        skip writing annotated output video (faster)
--shots N         hint for expected arrow count (caps detections)
```

Output goes to `output/` — annotated video, multi-panel chart, text report.

### Full session orchestrator (recommended for multi-clip sessions)

```bash
# Step 1 — scan all clips in a session folder and run detection
python3 tools/process_session.py data/july_2026/070726

# Step 2 — finalize: run YOLO metrics on detected frames + refresh trends
python3 tools/process_session.py data/july_2026/070726 --finalize
```

The orchestrator processes all clips in parallel, detects shots using audio + visual confirmation, and runs YOLO metrics automatically. No manual verification step is needed under normal conditions.

---

## 6. Verification (only if detection seems off)

The detector is reliable when the lavalier mic is used and there are no aborted draws or other archers in frame. Manual verification is **not part of the standard workflow** — only do it if the shot count doesn't match what you shot, or a metric looks obviously wrong.

**When to check:**

| Symptom | Likely cause |
|---------|-------------|
| Shot count too high | Let-down counted as a shot — check your per-arrow log |
| Shot count too low | Sub-threshold clicker or ambient-only mic — check audio |
| Metric wildly off for one shot | Frame at wrong pose moment — check ±5 frames |

To inspect specific frames if needed:
```bash
python3 tools/verification_grid.py --video data/.../IMG_NNNN.MOV --frames F1 F2 F3
```

---

## 7. YOLO metrics on verified frames

Once frames are confirmed, run the view-specific YOLO tool to compute full metrics:

```bash
# Face view
python3 tools/analyze_face_yolo.py \
    --video data/july_2026/070726/faceview/IMG_5001.MOV \
    --frames 74 606 1113 \
    --session-json session_history/2026-07-07_practice_face_IMG5001.json

# Back view
python3 tools/analyze_back_yolo.py \
    --video data/july_2026/070726/backview/IMG_5002.MOV \
    --frames 81 620 1128 \
    --session-json session_history/2026-07-07_practice_back_IMG5002.json

# Target view
python3 tools/analyze_target_yolo.py \
    --video data/july_2026/070726/targetview/IMG_5003.MOV \
    --frames 79 615 1110 \
    --session-json session_history/2026-07-07_practice_target_IMG5003.json
```

Each script also writes all metrics back into the session JSON (including camera-invariant variants) and produces two additional outputs:

### Per-shot CSV (`output/{view}_{stem}_shots.csv`)

One row per verified shot. Columns vary by view:

**Face view columns:**
```
shot, frame,
draw_elbow_angle, bow_elbow_angle,
anchor_x, anchor_y,                          ← normalized image coords (0–1)
anchor_x_sw, anchor_y_sw,                    ← shoulder-width normalized
nose_string_gap, nose_string_gap_sw,
bow_shoulder_elev, bow_shoulder_elev_sw,
hold_time_s,
bow_elbow_followthrough,                      ← bow elbow angle ~20 frames post-release
bow_wrist_ft_dx,                              ← bow wrist horizontal drift after release
nose_preanchor_drift,                         ← chin drop (px) during draw approach
anchor_x_px, anchor_y_px,                    ← pixel coordinates for reference
shoulder_width_px, bow_upper_arm_px,
vis_elbows                                    ← YOLO confidence score
```

**Back view columns:**
```
shot, frame,
shoulder_level, hip_alignment,               ← degrees from horizontal (ideal: 0°)
head_lateral_tilt, t_draw_angle,             ← degrees
bow_shoulder_elev, bow_shoulder_elev_sw,
shoulder_width_px,
vis_shoulders, vis_hips                      ← YOLO confidence scores
```

**Target view columns:**
```
shot, frame,
draw_elbow_height, draw_elbow_lateral,       ← raw image-relative
draw_elbow_height_sw, draw_elbow_lateral_sw, ← shoulder-width normalized
draw_elbow_height_th, draw_elbow_lateral_th, ← torso-height normalized (most stable)
back_tension_proxy,                           ← foreshortened shoulder width / frame diagonal
bow_shoulder_elev, bow_shoulder_elev_sw, bow_shoulder_elev_th,
shoulder_width_px, torso_height_px,
vis_draw                                      ← YOLO confidence score
```

Use the `*_th` columns for target view and `*_sw` columns for face/back in any cross-session analysis — these are camera-invariant.

### Tabular feedback (terminal output)

Printed immediately after the CSV. Shows each metric with its session average, target range, and a pass/fail indicator, then 2–3 narrative bullets:

```
════════════════════════════════════════════════════════════════
  FEEDBACK — Face View  (6 shots)
════════════════════════════════════════════════════════════════
  Metric                         Value       Target                 Status
  ──────────────────────────────────────────────────────────────
  Draw elbow angle               25.8°       15–35°                 ✓
  Bow elbow angle               168.4°       160–175°               ✓
  Anchor spread                   3.8 px     < 10 px                ✓
  Nose–string gap               -0.056       −0.12 to −0.04         ✓
  Bow shoulder elev (sw)         0.622       0.55–0.65              ✓
  Hold time                      2.51 s      1.0–3.0 s              ✓
  Bow wrist follow-thru         +0.010       negative = good        ✗
  Chin pre-anchor drift           3.2 px     < 2 px                 ✗
  ──────────────────────────────────────────────────────────────

  STRENGTHS
    ✓ Draw elbow angle in range (25.8°) — good back-arm engagement
    ✓ Anchor consistency excellent — 3.8 px spread
    ✓ Nose-string contact in range (-0.056)
    ✓ Hold time solid at 2.51 s

  AREAS OF IMPROVEMENT
    ✗ Bow arm moving right after release (+0.010) — stay on the line through the clicker
    ✗ Chin drifting DOWN 3.2 px — set chin at setup and carry it to anchor
════════════════════════════════════════════════════════════════
```

The ✓/✗ flags come from `tools/metric_thresholds.py`, which is the single source of truth for every band (BB9). The rule engine, the trend tool and these per-view tables all read it, so a metric that shows green in one shows green in the others. Metrics marked observation-only print a `—` rather than a verdict: each band records its provenance, and only coach-stated, coach-cue-fitted or technique-derived bands may produce a pass/fail.

---

## 7b. Shot cycle analysis (optional but recommended)

To see how each metric changes across the **entire shot cycle** — not just at the anchor frame — run:

```bash
python3 tools/analyze_shot_cycle.py \
    --video data/july_2026/070726/faceview/IMG_5001.MOV \
    --frames 74 606 1113 \
    --view face
```

This samples every 3 frames across a ±window around each anchor, assigns a phase label, and produces:
- `output/face_IMG5001_cycle.csv` — one row per sampled frame × shot (time_s, phase, all metrics)
- `output/face_IMG5001_cycle.png` — multi-panel trajectory chart with phase shading and good-range bands

**Phase labels (mapped to KSL shot cycle steps):**

| Phase label | Time relative to anchor | KSL step(s) | What's happening |
|-------------|------------------------|-------------|-----------------|
| `SETUP` | < −2.0 s | Set / Set Up | Bow is being raised to the target line (~45° loading position). Hook, grip, and stance are already set before the window opens. |
| `DRAWING` | −2.0 s to −0.5 s | Drawing | Active draw — bow arm extends while draw arm pulls, engaging shoulder blades. Bow shoulder should be sinking, not rising. |
| `LOADING` | −0.5 s to −0.05 s | Transfer to Hold / Aim | Approaching full draw and anchor. Load transfers from arm muscles to back (scapula squeeze). Clicker is close to dropping. Head stable, aiming. |
| `ANCHOR` | −0.05 s to +0.15 s | Release | Clicker drops (t ≈ 0). Draw hand reaches face anchor. Release happens by continuing the back expansion — not a finger action. |
| `FOLLOW_THROUGH` | > +0.15 s | Follow-Through | Bow arm holds the line toward the target. Draw hand travels straight back along the jaw/neck. No collapse. |

> **Note:** The default 3-second lookback window typically captures from mid-Set-Up through Follow-Through. The Stance, Hook, and initial Set steps happen before the window opens — use `--before 5.0` if you want to capture from the very beginning of bow raise.

Default window: 3 s before and 1 s after anchor. Useful flags:
```
--before 4.0   extend lookback (e.g. to capture more setup)
--after  1.5   extend post-release window
--step 3       sample every N frames (default 3 = ~10fps at 30fps)
```

Supports all three views (`--view face|back|target`).

---

## 8. Session outputs

### Primary output — per-shot CSV
The CSV at `output/{view}_{stem}_shots.csv` is the actual deliverable. One row per shot, all metric columns. Open it in any spreadsheet app or feed it to `score_correlation.py`. This is what you read to understand how a session went.

The terminal also prints a tabular feedback summary immediately after the CSV is written (see §7 for format).

### Internal storage — session JSON
Each clip also writes a JSON to `session_history/`. This is used internally by the trend tool and orchestrator — not meant to be read directly for per-session analysis. The CSV is the human-readable output; the JSON is the machine-readable record.

**Filename:** `YYYY-MM-DD_practice_{view}_{clip}.json`
Example: `2026-07-07_practice_face_IMG5001.json`

**Structure:**
```json
{
  "date": "2026-07-07",
  "video_path": "data/july_2026/070726/faceview/IMG_5001.MOV",
  "session_type": "practice",
  "view": "face",
  "location": "backyard",
  "distance_m": 18,
  "metrics": {
    "arrows_shot": 3,
    "verified_frames": [74, 606, 1113],
    "draw_elbow_angle_avg": 18.2,
    "draw_elbow_angle_std": 1.1,
    "anchor_y_sw_avg": 0.32,
    "anchor_y_sw_std": 0.01,
    "bow_shoulder_elevation_sw_avg": 0.60,
    "hold_time_s_avg": 0.48,
    "nose_preanchor_drift_avg": 1.65,
    "..."
  },
  "review": {
    "status": "finalized",
    "clicker_confirmed": [74, 606, 1113]
  },
  "notes": "Backyard 18m. Landscape 1080p."
}
```

Add per-shot scores and feel ratings to the JSON after recording them at the target (in shooting order):
```json
"per_shot_data": [
  {"shot_index": 1, "frame": 74,   "score": "10", "feel_1_10": 8},
  {"shot_index": 2, "frame": 606,  "score": "9",  "feel_1_10": 7},
  {"shot_index": 3, "frame": 1113, "score": "9",  "feel_1_10": 6}
]
```

Once scores are added, `score_correlation.py` can match each arrow's form metrics (from the CSV) to its score.

---

## 8b. Coaching summary

**What this section is about (for non-archers):** Coaches watch an athlete and give them verbal cues — short, specific phrases about what to fix. An archery coach might say "your left shoulder is creeping up" or "don't drop your chin to the string — pull the string to your chin." These cues are based on what the coach sees in person. The pipeline independently measures the same things from video. The coaching summary is the place where those two streams of information meet: every cue the coach has ever given, mapped to the exact number the pipeline produces for it.

The goal is that when the pipeline flags a problem, it doesn't just show a raw number — it says something the archer already recognizes, in the same words they've been hearing from their coach for months.

---

### How logged coach cues turn into personalized feedback

Every cue logged through the app's Coach Notes screen ("+ Add Cue") is appended to
`archers/{archer_id}/coach_voice.md`, in the coach's own words (see the "Coach Notes"
row in **App screens**, above). Two things happen with that text, both entirely local:

1. **Structural pass (deterministic, always on).** `tools/coaching_feedback.py`'s rule
   engine flags out-of-range metrics against `tools/metric_thresholds.py`'s
   coach-attributed cues, and shows the archer's `active_coaching_focus` items
   (hand-edited in `profile.json`, filtered to cues that existed as of the session
   date — see `_cue_on_or_before()`). This is the whole output whenever there isn't
   much logged text yet — "basic recommendations".
2. **Rich NLG, personalized (only once there's enough text).** `tools/feedback_nlg.py`
   is a hand-built, rule-based natural-language-generation engine — sentence
   templates and phrase banks selected and composed in code, no trained model — that
   writes several varied full sentences per finding, splicing in short fragments
   generated by `tools/coach_voice_model.py`'s from-scratch n-gram model (trained
   fresh from `coach_voice.md`'s text on every call, no persisted model file). Below
   `MIN_WORDS_RICH` this contributes nothing, and `synthesize_manual()`'s output is
   exactly the structural pass above. See `coach_voice_model.py`'s docstring for why
   a trained model (as opposed to rule-based templates) isn't viable at this data
   scale — a single archer's coach log will never realistically contain enough text
   to teach a neural net fluent English, but the hand-built grammar in
   `feedback_nlg.py` doesn't need to learn grammar from a corpus in the first place.

Nothing here calls an external API or requires a key — the whole pipeline is local.

---

### The coaching log file (optional, hand-authored)

`archers/{archer_id}/coach_voice.md` (structured, app-written — see above) is the file
the app actually reads. `archers/{archer_id}/coaching_log.md` is a separate, optional,
entirely hand-authored file for a richer prose record if you want to keep one — no code
in this app reads or writes it; it's just disk space reserved for it (see
`archers/README.md`). A natural shape for it, if you use it:

**Active focus areas** — the top-priority cues being worked on right now, updated after each session. Each entry records:
- The cue itself, in the coach's words
- Which pipeline metric tracks it
- When it was last reinforced

**Cumulative technique tracker** — a table of every coaching topic ever raised, mapped to its pipeline metric and how often it comes up:

| Area | First introduced | Recurrence | Pipeline metric |
|------|-----------------|------------|----------------|
| Bow shoulder low (scapula down) | early sessions | Very frequent | `bow_shoulder_elevation_sw` |
| Head still at anchor | recurring | Frequent | `nose_preanchor_drift` |
| Expansion through release | recurring | Frequent | `draw_elbow_followthrough` |
| Nose contact / no squish | recurring | Frequent | `nose_string_gap` |
| Follow-through bow arm | recurring | Occasional | `bow_elbow_followthrough` |
| Both shoulder blades engaged | recurring | Moderate | `back_tension_proxy` |
| Hip / core stability | background | Background | `hip_alignment` |
| ... | | | |

This table makes clear which coaching cues have a measurable counterpart in the pipeline and which don't yet. Cues without a metric are development candidates — the next metric to build.

**Technique milestones** — key dates when something meaningfully changed: a cue clicked, a pattern corrected, a coach-observed improvement. Used to correlate trend inflection points in the data with real coaching events.

**Per-session coaching entries** — one entry per session (most recent at top): what the coach observed, cues given, homework or drills assigned, and which metrics are expected to respond.

Coaching sessions also get their own JSON in `session_history/` with `"session_type": "coaching"`:
```json
{
  "date": "2026-07-05",
  "session_type": "coaching",
  "metrics": {},
  "notes": "Focus: head stability at anchor. Homework: two process cues every arrow. Metric to watch: nose_preanchor_drift."
}
```

### How the coaching log connects to the pipeline

The feedback bullets in `_print_feedback()` use the same language as the coaching log on purpose — so when a metric flags ✗, the improvement text matches a cue the archer already knows. The cumulative tracker drives what gets built next: if a coach has given a cue repeatedly and there's no pipeline metric for it, that's the next metric to add.

---

## 9. Trend analysis

After adding one or more sessions to `session_history/`, run:

```bash
python3 tools/trend_analysis.py
```

Outputs `output/session_trends.png` — a multi-panel chart showing every metric over time across all sessions, with per-view color coding. Prints a summary table to the terminal.

The trend tool reads all JSON files in `session_history/` automatically and normalizes field names across pipeline versions.

---

## 10. Score correlation

Once per-shot score data is logged in session JSONs:

```bash
python3 tools/score_correlation.py
```

Computes Pearson/Spearman correlation between each form metric and arrow score. Helps identify which metrics actually predict where the arrow lands.

---

## 11. Tools reference

| Tool | Purpose |
|------|---------|
| `archery_analyzer_v2.py` | Main detector — audio+visual shot detection, per-shot report, annotated video |
| `tools/process_session.py` | Session orchestrator — scan all clips in a folder, then finalize |
| `tools/sort_dropbox.py` | Move videos from `dropbox/` into `data/` folder structure by date and view |
| `tools/analyze_face_yolo.py` | YOLO metrics on confirmed face-view frames; writes per-shot CSV + tabular feedback |
| `tools/analyze_back_yolo.py` | YOLO metrics on confirmed back-view frames; writes per-shot CSV + tabular feedback |
| `tools/analyze_target_yolo.py` | YOLO metrics on confirmed target-view frames; writes per-shot CSV + tabular feedback |
| `tools/analyze_shot_cycle.py` | Full shot-cycle analysis — frame-by-frame metrics across ±window, phase labels, trajectory CSV + PNG |
| `tools/trend_analysis.py` | Load all session JSONs → longitudinal trend chart |
| `tools/score_correlation.py` | Correlate form metrics with arrow scores |
| `tools/verification_grid.py` | Thumbnail grid of specific frames for manual review |
| `tools/cross_validate_shots.py` | Cross-check pose-detected draws against audio clicker onsets |
| `tools/find_full_draw_frames.py` | Stability-based full-draw detection (used internally) |
| `tools/ambient_clicker_check.py` | Compare ambient vs lavalier clicker signal strength across views |
| `tools/yolo_pose_adapter.py` | COCO-17 → MediaPipe-compatible landmark wrapper; also resolves which pose model to load (stock vs. fine-tuned) |
| `tools/export_pose_dataset.py` | Build a pose fine-tuning dataset from verified full-draw frames (see `models/README.md`) |
| `tools/finetune_pose_model.py` | Fine-tune the pose model on your own footage (see `models/README.md`) |
| `tools/coaching_feedback.py` | Rule engine; dispatches to basic recommendations or `tools/feedback_nlg.py`'s rich mode producing coach-aligned feedback for a session or day |
| `tools/feedback_nlg.py` | Rule-based NLG engine — sentence templates/phrase banks, personalized with the archer's own logged coach cues |
| `tools/coach_voice_model.py` | From-scratch n-gram model over an archer's own logged cues — the building block `feedback_nlg.py` personalizes with |
| `tools/calibrate_archer.py` | Derive an archer's personal relative baseline bands from their own session history |
| `tools/training_load.py` | Per-session time-under-tension and within-session drift monitoring |
| `tools/verify_session_frames.py` | Spot-check verification grids from session-level JSONs |
| `tools/release_probe.py` | Kinematic probe distinguishing a real release from a let-down after a full-draw hold |

---

## 12. Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| 0 shots detected | Audio candidates all rejected by anchor check | Check video has real shots; try different `--threshold`; may need manual frames |
| Wrong shot frame (too early) | Audio fired before archer reached anchor | Use verification grid; correct frame in JSON |
| Extra false shot | Another archer's clicker triggered detection | Drop that entry from `verified_frames` |
| Aborted draw flagged | Ambient noise + brief wrist near anchor zone | Remove from `verified_frames`; note as let-down |
| YOLO outlier metric (e.g. elbow angle way off) | Partial occlusion at that frame | Try ±5 frames around the detected frame |
| Slow-mo clip | Phone recorded at 120fps | Detector handles automatically; just note it in session notes |
| iOS fell back to built-in mic | Receiver plugged in after camera app opened | Always plug RX in first; re-record if clicker is missing from audio |
| Nose pre-anchor drift flagged | Chin gradually dropping during draw approach | Set chin position firmly at setup stance; carry that through the draw |

---

## 13. Post-session checklist

```
After each session:

  1.  Transfer videos from phones
  2.  Drop all files into dropbox/  (no subfolders needed)
  3.  python3 tools/sort_dropbox.py               (preview — check view assignments)
  4.  python3 tools/sort_dropbox.py --move         (sort into data/)
  5.  python3 tools/process_session.py data/.../MMDDYY
      → detects shots, runs YOLO, writes CSVs + tabular feedback to output/
  6.  python3 tools/process_session.py data/.../MMDDYY --finalize
      → refreshes session JSONs and trend data
  7.  Add per-shot scores + feel ratings to session JSONs (in shooting order)
  8.  python3 tools/trend_analysis.py              (updated longitudinal charts)
  9.  python3 tools/score_correlation.py           (if enough scored data)
 10.  (optional) python3 tools/analyze_shot_cycle.py ... --view face|back|target

  ↳ Check shot count vs your score log. If it doesn't match, see §6 (Verification).
```


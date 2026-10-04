# Archery Form Analyzer

A markerless biomechanical analysis pipeline for Olympic recurve archery. Record your
practice on ordinary phone video, and get quantitative, per-shot form feedback — no
motion capture lab, no wearables, no manual annotation once it's set up.

Most archers get coaching once a week and practice the other six days with no
feedback, no data, and no way to know if they're reinforcing good form or bad habits.
This tool closes that gap: point a phone (or up to three, for multi-angle coverage) at
practice, run the pipeline, and get anchor consistency, elbow angles, shoulder
elevation, hold time, follow-through, and head stability for every single shot.

## What it does

1. **Automatic shot detection.** Combines clicker audio onset detection with visual
   full-draw pose confirmation, so a real shot needs both signals to agree — one
   sensor's blind spot is covered by the other's strength.
2. **Biomechanical metrics across up to 3 camera views.** Face (draw/bow elbow angles,
   anchor position, nose-string gap, bow shoulder elevation), back (shoulder level,
   hip alignment, head tilt), target (draw elbow height, lateral drift, back tension
   proxy). Metrics are camera-invariant (normalized by body scale), so sessions
   recorded from different distances or camera positions are still comparable.
3. **Longitudinal trend analysis.** Per-metric trends across every session in your
   history, so you can see what's actually improving versus what only feels like
   it's improving.
4. **Per-archer calibration.** Instead of judging your form against someone else's
   absolute numbers, the system can calibrate to your own baseline and flag deviation
   from *your* normal — a much more actionable signal for an individual archer.
5. **Coaching notes integration.** Log your coach's cues in the app; generated
   feedback can echo the language your coach already uses, without ever having to
   re-type your session data into a spreadsheet.
6. **A mobile web app** for the whole loop: sign up as an archer, upload video from
   your phone, watch it auto-analyze, browse sessions/trends, and log coaching notes —
   no command line required for day-to-day use.

> **This is not a medical or injury-prevention tool.** It measures joint angles and
> timing from video. It cannot diagnose, predict, or prevent injury. Several metrics
> correspond to loading patterns the sports-medicine literature associates with
> overuse, but a correspondence is not a risk assessment. Pain, or any suspected
> injury, goes to a physiotherapist or physician, not to this pipeline.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The YOLO pose model (`yolo26m-pose.pt`) auto-downloads from Ultralytics on first use.

## Quick start (command line)

```bash
# Detect shots in a clip
python3 archery_analyzer_v2.py --face path/to/video.MOV --min-time 10 --no-video

# Once frames are verified, run YOLO analysis on them and write into a session JSON
python3 tools/analyze_face_yolo.py \
    --video path/to/video.MOV \
    --frames F1 F2 F3 \
    --session-json session_history/2026-01-01_practice_face_myvideo.json

# Refresh longitudinal trends across all sessions
python3 tools/trend_analysis.py
```

See `docs/WORKFLOW.md` for the full pipeline — hardware setup, recording protocol,
verification, per-archer calibration, and coaching-feedback synthesis.

## Quick start (mobile web app)

```bash
# Terminal 1 — FastAPI backend
cd mobile_app/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — React frontend (dev mode)
cd mobile_app/frontend
npm install
npm run dev
```

Open the printed frontend URL, create an archer profile, and upload a video. The
backend runs shot detection and YOLO metrics automatically in the background.

For a production build where FastAPI serves the built frontend directly, see
`docs/WORKFLOW.md`.

---

## Project structure

```
├── archery_analyzer_v2.py    ← main entry: audio-visual shot detection per clip
├── tools/                     ← analysis pipeline
│   ├── audio_visual_shot_detector.py   shot detector (audio onset + visual anchor check)
│   ├── analyze_face_yolo.py            face-view metrics
│   ├── analyze_back_yolo.py            back-view metrics
│   ├── analyze_target_yolo.py          target-view metrics
│   ├── analyze_shot_cycle.py           frame-by-frame shot-cycle trajectory analysis
│   ├── process_session.py              orchestrator: scan a folder of clips, then finalize
│   ├── sort_dropbox.py                 move raw videos from dropbox/ into data/
│   ├── coaching_feedback.py            rule engine; dispatches to basic or rich feedback
│   ├── feedback_nlg.py                 rule-based NLG engine for rich coaching feedback
│   ├── coach_voice_model.py            from-scratch n-gram model over your own logged cues
│   ├── calibrate_archer.py             derive a personal relative baseline
│   ├── trend_analysis.py               longitudinal trend charts
│   ├── score_correlation.py            correlate form metrics with arrow scores
│   ├── training_load.py                time-under-tension / within-session drift
│   ├── metric_thresholds.py            single source of truth for "good form" bands
│   ├── export_pose_dataset.py          build a fine-tuning dataset from your own footage
│   └── finetune_pose_model.py          fine-tune the pose model on your own footage
├── ml_shot_detector/           optional ML fallback shot-phase classifier
├── mobile_app/
│   ├── backend/                FastAPI app (auth, sessions, trends, upload, feedback)
│   └── frontend/               React mobile web UI
├── archers/                    per-archer profiles (created via the app or by hand)
├── session_history/            analyzed session JSONs — your dataset
├── data/                       raw video, organized by date (gitignored)
├── dropbox/                    drop new raw video here before sorting (gitignored)
├── output/                     generated charts/CSVs (gitignored, regeneratable)
├── models/                     pose weights + optional trained classifier (gitignored)
└── docs/
    ├── WORKFLOW.md              the complete operating manual
    └── shot_detection_design.md the shot/let-down detection design and rationale
```

---

## Tech stack

- **Pose estimation:** YOLO26m-pose (Ultralytics) — optionally fine-tuned on your own
  footage (`tools/export_pose_dataset.py` + `tools/finetune_pose_model.py`); falls back
  to the stock pretrained model automatically if you haven't
- **Person selection:** largest bounding box (avoids latching onto bystanders)
- **Audio onset:** SciPy Butterworth high-pass + RMS envelope
- **Backend:** FastAPI, per-archer bearer-token auth, background analysis jobs
- **Frontend:** React + TypeScript + Tailwind
- **Coaching language layer:** fully local, no external AI API anywhere in the app.
  Basic recommendations (rule engine, `tools/coaching_feedback.py`) by default;
  once an archer has logged enough of their own coach's cues, a hand-built,
  rule-based NLG engine (`tools/feedback_nlg.py`) writes fuller, varied sentences,
  personalized with fragments from a from-scratch n-gram model
  (`tools/coach_voice_model.py`) trained on that archer's own text.
- **Storage:** per-session JSON files, no database required at typical scale

## Getting your own data in

1. Record practice on one or more phones (face / back / target views). A lavalier
   mic on the archer dramatically improves clicker-audio shot detection quality,
   especially for rear/target camera angles.
2. Drop clips into `dropbox/`, then `python3 tools/sort_dropbox.py --move` to sort
   them into `data/` — or upload directly through the mobile app.
3. Run detection + YOLO metrics (CLI: `tools/process_session.py`; app: automatic on
   upload).
4. Spot-check with the verification grid before trusting a session's metrics —
   automatic shot detection is good but not perfect, especially on multi-arrow clips.
5. Log coaching cues (via the app's Coach Notes screen, or by hand-authoring
   `archers/<id>/coaching_log.md` / `coach_voice.md`) so generated feedback can
   reflect your coach's own language.

## Why this project

Olympic recurve archery is one of the most technically demanding sports — small,
consistent form deviations directly affect arrow scores, but they're hard for a coach
to catch by eye across hundreds of arrows in a session, and impossible to track
numerically without instrumentation. This tool turns ordinary phone video into
per-shot, per-metric measurements, so an archer can see exactly what changed between
a good session and a bad one, and validate whether a coaching cue actually stuck.

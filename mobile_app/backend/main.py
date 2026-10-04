#!/usr/bin/env python3
"""
FastAPI backend for the Archery Form Analyzer mobile web app.

Start: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # archery-form-analyzer/
SESSION_DIR = ROOT / "session_history"
ARCHERS_DIR = ROOT / "archers"
TOOLS_DIR = ROOT / "tools"
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _archer_dir(archer_id: str) -> Path:
    return ARCHERS_DIR / archer_id

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

app = FastAPI(title="Archery Form Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── in-memory job store ────────────────────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}


# ── metric metadata ────────────────────────────────────────────────────────────
#
# Derived from tools/metric_thresholds.py, which is the single source of truth for
# every band (BB9). This used to be a hardcoded copy commented "mirrors
# coaching_feedback.py thresholds" — a fourth copy alongside the rule engine, the
# trend tool and the per-view scripts. It had already drifted, and two of the
# drifted values were ones a demo would show the archer:
#
#   - hold time carried (0.5, 1.5), a band that was RETRACTED. It had no coaching
#     basis and was fitted to right-censored measurements; under correct
#     measurement it flags a good 2-3s hold as a failure. The archer's holds
#     average ~2.6s, so every session would have shown hold time as a fault.
#   - draw elbow angle carried (15, 35), now observation-only: the two bands
#     previously in use disagreed, neither had a traceable source, and the value is
#     camera-geometry confounded.
#
# See docs/RETRACTIONS.md. Deriving the table instead of copying it means the app
# cannot drift again.
#
# The shape of each entry is unchanged, so the frontend needs no modification:
# `field`, `label`, `unit`, `good_range` (None => observation-only) and `baseline`.
import metric_thresholds as _MT   # noqa: E402  (TOOLS_DIR is on sys.path above)
from hold_time import HOLD_TIME_DEFINITION as _MT_HOLD_DEF   # noqa: E402


def _band_to_good_range(spec: dict):
    """Translate a canonical band into the (lo, hi) form this API already returns.

    Returns None for anything that must not produce a verdict — observation-only
    metrics, and bands whose provenance is descriptive or unvalidated. That check is
    the point of the exercise: a metric with no traceable source should render as a
    value, never as a fault.
    """
    if spec.get("kind") == "observe":
        return None
    if spec.get("provenance") not in _MT.JUDGEABLE_PROVENANCE:
        return None
    kind, band = spec["kind"], spec["band"]
    if kind == "range":
        return tuple(band)
    if kind == "max":
        return (None, band)
    if kind == "abs_max":
        return (-band, band)
    return None


def _build_metrics_meta() -> dict[str, list[dict]]:
    meta: dict[str, list[dict]] = {}
    for view in ("face", "back", "target"):
        rows = []
        for spec in _MT.specs_for(view):
            rows.append({
                "field": spec["field"],
                "label": spec["label"],
                "unit": spec.get("unit", ""),
                "good_range": _band_to_good_range(spec),
                "baseline": spec.get("reference"),
                "provenance": spec.get("provenance"),
                "cue": spec.get("cue"),
            })
        meta[view] = rows
    # Image-relative anchor spread is retained for display on sessions predating the
    # head-relative measure, but never judged: it cannot distinguish "the hand moved
    # on the face" from "the archer moved in frame" (BB2).
    meta["face"].append({
        "field": "anchor_spread_px", "label": "Anchor Spread (image)",
        "unit": " px", "good_range": None, "baseline": None,
        "provenance": "superseded",
        "cue": "not judged — use the head-relative measure",
    })
    return meta


METRICS_META = _build_metrics_meta()


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _session_owner(s: dict) -> str:
    """Sessions should always carry an archer_id once accounts exist. This guards
    against any that somehow don't (e.g. hand-authored JSONs)."""
    return s.get("archer_id") or ""


def _metric_status(value: float, meta: dict, metrics: dict | None = None) -> str:
    """Status badge for one metric.

    `metrics` is the session's whole metrics block, needed because hold time cannot
    be judged from its value alone: the corpus contains values from three
    superseded definitions, and a censored value is a lower bound rather than a
    measurement. Judging those produces a fault the archer cannot act on and that
    is not true — which is precisely what the retracted hold-time band did.
    See docs/RETRACTIONS.md.
    """
    if metrics is not None and meta["field"] == "hold_time_s_avg":
        if metrics.get("hold_time_s_definition") != _MT_HOLD_DEF:
            return "observation"
        if metrics.get("hold_time_s_n_censored"):
            return "observation"
    gr = meta.get("good_range")
    if gr is None:
        return "observation"
    lo, hi = gr
    if lo is not None and value < lo:
        return "low"
    if hi is not None and value > hi:
        return "high"
    return "good"


def _session_card(path: Path, s: dict) -> dict:
    m = s.get("metrics", {})
    view = s.get("view", "")
    metas = METRICS_META.get(view, [])

    # Compute a quick status summary: count good / flagged metrics
    good, flagged = 0, 0
    for meta in metas:
        val = m.get(meta["field"])
        if val is not None and meta.get("good_range") is not None:
            st = _metric_status(val, meta, m)
            if st == "good":
                good += 1
            else:
                flagged += 1

    return {
        "id": path.stem,
        "date": s.get("date", ""),
        "view": view,
        "location": s.get("location", ""),
        "distance_m": s.get("distance_m"),
        "arrows": (m.get("arrows_with_metrics") or m.get("arrows_shot") or 0),
        "has_metrics": bool(m.get("draw_elbow_angle_avg") or
                            m.get("shoulder_level_avg") or
                            m.get("draw_elbow_height_th_avg")),
        "metrics_good": good,
        "metrics_flagged": flagged,
        "notes": s.get("notes", ""),
    }


# ── routes: accounts ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/archers")
def get_archers():
    """Public roster for the profile picker — no auth required, no password data."""
    return auth.list_archers()


class ArcherCreate(BaseModel):
    archer_id: str
    name: str
    password: str


@app.post("/api/archers")
def post_archer(data: ArcherCreate):
    token = auth.create_archer(data.archer_id, data.name, data.password)
    return {"token": token, "archer_id": data.archer_id.strip().lower(), "name": data.name.strip()}


class PasswordSet(BaseModel):
    password: str


@app.post("/api/archers/{archer_id}/set-password")
def post_set_password(archer_id: str, data: PasswordSet):
    token = auth.set_password(archer_id, data.password)
    return {"token": token, "archer_id": archer_id, "name": auth.get_archer_name(archer_id)}


class LoginRequest(BaseModel):
    archer_id: str
    password: str


@app.post("/api/auth/login")
def post_login(data: LoginRequest):
    token = auth.login(data.archer_id, data.password)
    return {"token": token, "archer_id": data.archer_id, "name": auth.get_archer_name(data.archer_id)}


@app.post("/api/auth/logout")
def post_logout(authorization: str = Header(default="")):
    if authorization.startswith("Bearer "):
        auth.logout(authorization.removeprefix("Bearer ").strip())
    return {"status": "logged out"}


# ── routes: account settings ────────────────────────────────────────────────────
#
# All three act on "me" — the archer resolved from the bearer token — never on an
# archer_id supplied by the client. That's what makes it safe for these to only require
# being logged in as *someone*: you cannot rename, re-password, or delete an account that
# isn't the one your token belongs to.

class NameUpdate(BaseModel):
    name: str


@app.put("/api/archers/me")
def put_my_name(data: NameUpdate, archer: str = Depends(auth.get_current_archer)):
    name = auth.update_name(archer, data.name)
    return {"archer_id": archer, "name": name}


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/archers/me/change-password")
def post_change_password(data: PasswordChange, archer: str = Depends(auth.get_current_archer)):
    auth.change_password(archer, data.current_password, data.new_password)
    # change_password revoked every token for this archer, including the one that just
    # authenticated this request — issue a fresh one so the frontend can keep the session.
    token = auth.login(archer, data.new_password)
    return {"token": token, "archer_id": archer, "name": auth.get_archer_name(archer)}


class AccountDelete(BaseModel):
    password: str


@app.post("/api/archers/me/delete")
def post_delete_account(data: AccountDelete, archer: str = Depends(auth.get_current_archer)):
    auth.delete_archer(archer, data.password)   # verifies password, wipes archers/{archer}/
    # Cascade to session_history/, per the archer's explicit choice when this feature was
    # built: deleting an account deletes their practice data too, not just the login.
    removed = 0
    for path in SESSION_DIR.glob("*.json"):
        s = _load_session(path)
        if s and _session_owner(s) == archer:
            path.unlink(missing_ok=True)
            removed += 1
    return {"status": "deleted", "sessions_removed": removed}


# ── routes: sessions ───────────────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions(archer: str = Depends(auth.get_current_archer)):
    results = []
    for path in sorted(SESSION_DIR.glob("*.json"), reverse=True):
        s = _load_session(path)
        if (s and s.get("session_type") == "practice" and s.get("view")
                and _session_owner(s) == archer):
            results.append(_session_card(path, s))
    return results


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, archer: str = Depends(auth.get_current_archer)):
    path = SESSION_DIR / f"{session_id}.json"
    s = _load_session(path)
    # 404, not 403, on someone else's session — don't leak whether it exists.
    if not s or _session_owner(s) != archer:
        raise HTTPException(404, "Session not found")

    # Attach metric metadata so the frontend can render status badges
    view = s.get("view", "")
    m = s.get("metrics", {})
    enriched_metrics = []
    for meta in METRICS_META.get(view, []):
        val = m.get(meta["field"])
        if val is None:
            continue
        enriched_metrics.append({
            **meta,
            "value": round(val, 4),
            "std": m.get(meta["field"].replace("_avg", "_std")),
            "status": _metric_status(val, meta, m),
        })

    return {**s, "enriched_metrics": enriched_metrics}


# ── routes: trends ─────────────────────────────────────────────────────────────

@app.get("/api/trends")
def get_trends(view: Optional[str] = None, archer: str = Depends(auth.get_current_archer)):
    views_to_process = ([view] if view and view in METRICS_META
                        else list(METRICS_META.keys()))
    trend_data: dict[str, Any] = {}

    for v in views_to_process:
        metas = METRICS_META[v]
        series: dict[str, list] = {m["field"]: [] for m in metas}

        for path in sorted(SESSION_DIR.glob("*.json")):
            s = _load_session(path)
            if (not s or s.get("session_type") != "practice"
                    or s.get("view") != v or _session_owner(s) != archer):
                continue
            date = s.get("date", "")
            metrics = s.get("metrics", {})
            for meta in metas:
                val = metrics.get(meta["field"])
                if val is not None:
                    series[meta["field"]].append({
                        "date": date,
                        "value": round(val, 4),
                        "session_id": path.stem,
                        "arrows": (metrics.get("arrows_with_metrics")
                                   or metrics.get("arrows_shot") or 1),
                    })

        trend_data[v] = {
            "series": {
                f: sorted(pts, key=lambda p: p["date"])
                for f, pts in series.items() if pts
            },
            "meta": {m["field"]: m for m in metas},
        }

    return trend_data


# ── routes: coaching feedback ───────────────────────────────────────────────────
#
# Fully local: a deterministic rule-engine + coach-cue template
# (tools/coaching_feedback.py's synthesize_manual()), optionally flavored by one
# closing line from tools/coach_voice_model.py — a tiny n-gram model trained fresh
# from the archer's own archers/<id>/coach_voice.md on every call. No external API,
# no API key, nothing leaves this machine. See synthesize_manual()'s docstring.

@app.post("/api/sessions/{session_id}/feedback")
def get_session_feedback(session_id: str, archer: str = Depends(auth.get_current_archer)):
    path = SESSION_DIR / f"{session_id}.json"
    existing = _load_session(path)
    if not existing or _session_owner(existing) != archer:
        raise HTTPException(404, "Session not found")
    try:
        from coaching_feedback import (  # type: ignore
            load_session, detect_faults, load_coaching_context, synthesize_manual,
        )
    except ImportError as e:
        raise HTTPException(500, f"Coaching module unavailable: {e}")

    session = load_session(str(path))
    findings = detect_faults(session)
    if not findings:
        raise HTTPException(400, "No scorable metrics in this session")

    ctx = load_coaching_context(archer)
    feedback = synthesize_manual(session, findings, ctx)

    return {"feedback": feedback, "findings_count": len(findings)}


# ── routes: coach notes ────────────────────────────────────────────────────────
#
# coach_voice.md and coaching_log.md are hand-authored, cross-referential prose — a
# section headed by one coach routinely names another mid-paragraph (see docs/WORKFLOW.md,
# "Coach names and confidentiality in the app"). There's no reliable way to mechanically
# apply the hidden_from_app exclusion to prose like that without either missing a mention
# or mangling the other coach's text, so this route never sends their raw content to the
# client at all — not "don't render it," genuinely don't transmit it. Only the coach-name
# list (already per-archer, already respects hidden_from_app) is returned. Real names are
# fine to return here: this list is scoped to the signed-in archer's own profile.json —
# see get_current_archer — never another archer's.

@app.get("/api/coach-notes")
def get_coach_notes(archer: str = Depends(auth.get_current_archer)):
    coaches = auth.get_archer_coaches(archer)
    return {"coaches": coaches}


class CoachNoteEntry(BaseModel):
    coach: str
    date: str        # YYYY-MM-DD
    cue: str
    context: str = ""
    category: str = ""   # e.g. "anchor", "back tension", "hold"


@app.post("/api/coach-notes/entry")
def add_coach_note_entry(entry: CoachNoteEntry, archer: str = Depends(auth.get_current_archer)):
    path = _archer_dir(archer) / "coach_voice.md"
    existing = path.read_text() if path.exists() else ""

    category_line = f"\n**Category:** {entry.category}\n" if entry.category else "\n"
    new_block = (
        f"\n\n---\n\n"
        f"**{entry.coach}** ({entry.date}){category_line}"
        f"- {entry.cue}"
    )
    if entry.context:
        new_block += f"\n  - _{entry.context}_"
    new_block += "\n"

    path.write_text(existing.rstrip() + new_block)
    # A coach name typed via "Other" becomes a picker option from now on.
    auth.add_archer_coach(archer, entry.coach)
    return {"status": "added"}


# ── routes: video upload + async analysis ─────────────────────────────────────

def _run_analysis_job(
    job_id: str, video_path: Path, view: str,
    date: str, location: str, distance_m: int, notes: str, archer_id: str,
) -> None:
    """Background task: create a draft session JSON, then try to run YOLO analysis."""
    _jobs[job_id]["status"] = "analyzing"
    try:
        python = sys.executable
        # ── 1. Create a draft session JSON so the app can track it ────────────
        session_stem = f"{date}_practice_{view}_mobile_{job_id}"
        draft_path = SESSION_DIR / f"{session_stem}.json"
        draft: dict[str, Any] = {
            "date": date,
            "archer_id": archer_id,
            "video_path": str(video_path.relative_to(ROOT)),
            "session_type": "practice",
            "timestamp": f"{date}T00:00:00",
            "view": view,
            "location": location,
            "distance_m": distance_m,
            "notes": notes,
            "metrics": {},
            "review": {"status": "pending_analysis"},
            "detection_notes": "Mobile app upload — auto-analysis in progress",
        }
        with open(draft_path, "w") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)

        # ── 2. Run shot-detection via archery_analyzer_v2.py ──────────────────
        # --session-json tells it to write metrics.verified_frames straight into our
        # draft — archery_analyzer_v2.py otherwise only writes charts/video to --out and
        # never touches session_history/ on its own (see its --session-json help text).
        # Previously this code searched for a session JSON the script never actually
        # produced, so this branch always silently fell through to "no shots detected"
        # even on a successful detection — fixed 2026-08-15.
        detect_cmd = [
            python, str(ROOT / "archery_analyzer_v2.py"),
            f"--{view}", str(video_path),
            "--min-time", "10",
            "--session-json", str(draft_path),
        ]
        proc = subprocess.run(
            detect_cmd,
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=600,
            input="",          # close stdin so no interactive prompts block
        )

        if proc.returncode != 0:
            # Detection crashed (as opposed to running cleanly and finding 0 shots) —
            # surface the real error instead of a generic "no shots" message that would
            # hide a code bug behind what looks like a data problem.
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            _jobs[job_id].update({
                "status": "failed",
                "message": f"Shot detection crashed:\n{stderr_tail}",
            })
            return

        s = _load_session(draft_path) or {}
        frames = s.get("metrics", {}).get("verified_frames") or []

        if frames:
            # ── 3. Run YOLO metrics on the detected frames ─────────────────────
            analyze_script = TOOLS_DIR / f"analyze_{view}_yolo.py"
            yolo_cmd = [
                python, str(analyze_script),
                "--video", str(video_path),
                "--frames", *[str(f) for f in frames],
                "--session-json", str(draft_path),
            ]
            yolo_proc = subprocess.run(
                yolo_cmd, capture_output=True, text=True,
                cwd=str(ROOT), timeout=600,
            )
            if yolo_proc.returncode != 0:
                stderr_tail = "\n".join(yolo_proc.stderr.strip().splitlines()[-15:])
                _jobs[job_id].update({
                    "status": "failed",
                    "message": f"Detected {len(frames)} shot(s) but YOLO metrics failed:\n{stderr_tail}",
                })
                return
            _jobs[job_id].update({
                "status": "done",
                "session_id": draft_path.stem,
                "message": f"Detected {len(frames)} shots. YOLO metrics computed.",
            })
        else:
            # Ran cleanly, genuinely found 0 shots — draft JSON stays available for
            # manual CLI verification.
            _jobs[job_id].update({
                "status": "done",
                "session_id": draft_path.stem,
                "message": (
                    "Shot detection did not find any shots automatically. "
                    "Use the CLI tools to manually verify frames and add metrics."
                ),
            })

    except subprocess.TimeoutExpired:
        _jobs[job_id].update({"status": "failed", "message": "Analysis timed out (>10 min)"})
    except Exception as e:
        _jobs[job_id].update({"status": "failed", "message": str(e)})


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    view: str = Form(...),
    date: str = Form(...),
    location: str = Form(default=""),
    distance_m: int = Form(default=18),
    notes: str = Form(default=""),
    archer: str = Depends(auth.get_current_archer),
):
    if view not in ("face", "back", "target"):
        raise HTTPException(400, "view must be one of: face, back, target")
    if not date:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    job_id = str(uuid.uuid4())[:8]
    date_compact = date.replace("-", "")[2:]   # 2026-07-27 → 260727

    upload_dir = ROOT / "data" / "mobile_uploads" / date_compact / f"{view}view"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"video_{job_id}.MOV"
    video_path = upload_dir / filename
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)

    _jobs[job_id] = {
        "status": "queued",
        "video": str(video_path),
        "view": view,
        "date": date,
        "archer_id": archer,
    }
    background_tasks.add_task(
        _run_analysis_job, job_id, video_path, view, date, location, distance_m, notes, archer,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, archer: str = Depends(auth.get_current_archer)):
    job = _jobs.get(job_id)
    if not job or job.get("archer_id") != archer:
        raise HTTPException(404, "Job not found")
    return job


# ── serve built React frontend in production ──────────────────────────────────
if DIST_DIR.exists():
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(str(DIST_DIR / "index.html"))

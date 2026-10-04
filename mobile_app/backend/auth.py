#!/usr/bin/env python3
"""Password-based archer accounts for the mobile app.

Local, single-deployment auth: each archer gets a sibling archers/<id>/auth.json holding
a salted password hash — deliberately kept OUT of profile.json. profile.json is a
hand-curated file (coach notes, identity, study setup) that predates accounts and is
edited by hand as much as by code; round-tripping it through json.dump on every
password change would quietly collapse its formatting (blank lines between sections,
comment ordering) on a file that has nothing to do with auth. auth.json has no such
concerns — it's wholly machine-owned.

There is no email, no OAuth, and no persistent server-side session store — bearer tokens
live in an in-memory dict and are lost on backend restart. That's intentional: this app
runs on one machine for one family, not as a hosted multi-tenant service, so a lightweight
profile-picker model (Netflix-style, one password per profile) is the right amount of
auth, not a placeholder for something bigger.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import Header, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent   # archery-form-analyzer/
ARCHERS_DIR = ROOT / "archers"

ARCHER_ID_RE = re.compile(r"^[a-z0-9_]{2,32}$")
TOKEN_TTL_SECONDS = 30 * 24 * 3600   # 30 days, sliding
MIN_PASSWORD_LEN = 4                 # a PIN is fine — this gates siblings, not strangers
_PBKDF2_ITERATIONS = 200_000

# token -> {"archer_id": str, "expires": float}
_tokens: dict[str, dict] = {}


# ── profile.json helpers (read-only from this module) ────────────────────────

def _profile_path(archer_id: str) -> Path:
    return ARCHERS_DIR / archer_id / "profile.json"


def _load_profile(archer_id: str) -> Optional[dict]:
    path = _profile_path(archer_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_profile(archer_id: str, profile: dict) -> None:
    """Only used for brand-new profiles (create_archer) — never to rewrite an existing,
    possibly hand-formatted profile.json. See module docstring."""
    path = _profile_path(archer_id)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── auth.json helpers (the actual credential store) ───────────────────────────

def _auth_path(archer_id: str) -> Path:
    return ARCHERS_DIR / archer_id / "auth.json"


def _load_auth(archer_id: str) -> Optional[dict]:
    path = _auth_path(archer_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_auth(archer_id: str, password_hash: str, salt: str) -> None:
    path = _auth_path(archer_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"password_hash": password_hash, "password_salt": salt}, f, indent=2)
        f.write("\n")


# ── password hashing ─────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS,
    )
    return digest.hex(), salt


def _issue_token(archer_id: str) -> str:
    token = secrets.token_urlsafe(32)
    _tokens[token] = {"archer_id": archer_id, "expires": time.time() + TOKEN_TTL_SECONDS}
    return token


# ── public API ────────────────────────────────────────────────────────────────

def list_archers() -> list[dict]:
    """Public-safe roster for the profile picker. Never returns password data."""
    if not ARCHERS_DIR.exists():
        return []
    out = []
    for d in sorted(ARCHERS_DIR.iterdir()):
        if not d.is_dir():
            continue
        profile = _load_profile(d.name)
        if not profile:
            continue
        creds = _load_auth(d.name)
        out.append({
            "id": d.name,
            "name": profile.get("identity", {}).get("name") or d.name,
            "has_password": bool(creds and creds.get("password_hash")),
        })
    return out


def create_archer(archer_id: str, name: str, password: str) -> str:
    """Create a new archers/<id>/profile.json (see archers/README.md), with unknown
    fields left null/TBD for the archer or a coach to fill in later. Returns a
    bearer token (auto-login)."""
    archer_id = archer_id.strip().lower()
    if not ARCHER_ID_RE.match(archer_id):
        raise HTTPException(400, "Archer ID must be 2-32 lowercase letters, digits, or underscores")
    if not name.strip():
        raise HTTPException(400, "Name is required")
    if len(password) < MIN_PASSWORD_LEN:
        raise HTTPException(400, f"Password must be at least {MIN_PASSWORD_LEN} characters")
    if _profile_path(archer_id).exists():
        raise HTTPException(409, "An archer with this ID already exists")

    password_hash, salt = _hash_password(password)
    today = time.strftime("%Y-%m-%d")
    profile = {
        "archer_id": archer_id,
        "schema_version": "2.0",
        "identity": {
            "name": name.strip(),
            "bow_hand": None,
            "handedness": None,
            "dominant_eye": None,
            "discipline": None,
            "competition_level": None,
            "approximate_age_range": None,
        },
        "anthropometrics": {
            "_status": "TBD",
            "height_cm": None, "draw_length_in": None,
            "shoulder_width_anatomical_cm": None, "arm_span_cm": None, "torso_height_cm": None,
        },
        "equipment_baseline": {
            "_status": "TBD",
            "bow_riser": None, "limbs": None, "draw_weight_lbs": None, "brace_height": None,
            "arrows": None, "arrow_spine": None, "point_weight_grain": None,
            "plunger_tension": None, "clicker_position": None, "sight": None,
            "stabilizer_setup": None,
        },
        "coaches": {
            "primary": None, "guest": [],
            "log_path": f"archers/{archer_id}/coaching_log.md",
        },
        "study_setup": {
            "_status": "TBD",
            "location": None, "distance_m": None, "distances_m": [],
        },
        "audio_capture": {"_status": "TBD", "deployed": False},
        "feedback_preferences": {"tier": "A (post-session feedback only)"},
        "active_coaching_focus": {"as_of": today, "items": []},
        "calibration": {
            # New archer, no coach-validated bands on hand yet — start in 'relative' mode
            # (deviation-from-own-norm) per the _mode_values convention in the schema.
            "mode": "relative",
            "sigma": 1.5,
            "min_sessions": 4,
            "baseline_generated": None,
        },
        "baseline_profile_path": f"archers/{archer_id}/baseline_profile.json",
        "tournament_observations_path": f"archers/{archer_id}/tournament_observations.md",
        "red_camp_notes_path": f"archers/{archer_id}/red_camp_notes.md",
        "_meta": {
            "created": today, "last_updated": today,
            "notes": "Created via the mobile app archer-account flow.",
        },
    }
    _profile_path(archer_id).parent.mkdir(parents=True, exist_ok=True)
    _save_profile(archer_id, profile)
    password_hash, salt = _hash_password(password)
    _save_auth(archer_id, password_hash, salt)
    return _issue_token(archer_id)


def set_password(archer_id: str, password: str) -> str:
    """One-time bootstrap for a profile that predates accounts (e.g. a hand-authored
    profile.json with no auth.json yet). Refuses once a password already exists —
    that's a login, not a claim."""
    if not _load_profile(archer_id):
        raise HTTPException(404, "Archer not found")
    if _load_auth(archer_id):
        raise HTTPException(409, "Password already set for this archer — use login instead")
    if len(password) < MIN_PASSWORD_LEN:
        raise HTTPException(400, f"Password must be at least {MIN_PASSWORD_LEN} characters")

    password_hash, salt = _hash_password(password)
    _save_auth(archer_id, password_hash, salt)
    return _issue_token(archer_id)


def _check_password(archer_id: str, password: str) -> None:
    """Raises if the password is wrong, unset, or the archer doesn't exist. Shared by
    login and by every settings action that re-verifies identity before a sensitive
    change (change_password, delete_archer) — one place owns "is this really you"."""
    if not _load_profile(archer_id):
        raise HTTPException(404, "Archer not found")
    creds = _load_auth(archer_id)
    stored_hash = creds and creds.get("password_hash")
    salt = creds and creds.get("password_salt")
    if not stored_hash or not salt:
        raise HTTPException(400, "No password set for this archer yet")
    candidate_hash, _ = _hash_password(password, salt)
    if not secrets.compare_digest(candidate_hash, stored_hash):
        raise HTTPException(401, "Incorrect password")


def login(archer_id: str, password: str) -> str:
    _check_password(archer_id, password)
    return _issue_token(archer_id)


def get_archer_name(archer_id: str) -> str:
    profile = _load_profile(archer_id)
    if not profile:
        return archer_id
    return profile.get("identity", {}).get("name") or archer_id


def get_archer_coaches(archer_id: str, include_hidden: bool = False) -> list[str]:
    """Coach display names from profile.json's `coaches` block (primary + guest), for
    populating per-archer coach pickers in the app. A brand-new profile has
    coaches.primary=None and coaches.guest=[] (see create_archer), so this correctly
    returns [] rather than another archer's real-world coaches — the app must never
    hardcode coach names, since they're per-archer data, not project config.

    A coach entry with `hidden_from_app: true` is skipped unless include_hidden=True.
    This is display-only — see the `_hidden_from_app_note` in profile.json's coaches
    block: it hides a coach from app pickers/feedback without deleting their history
    (coaching_log.md, coach_voice.md, session_history/*.json are untouched)."""
    profile = _load_profile(archer_id)
    if not profile:
        return []
    coaches = profile.get("coaches") or {}
    names: list[str] = []

    primary = coaches.get("primary")
    if isinstance(primary, dict) and (include_hidden or not primary.get("hidden_from_app")):
        label = primary.get("alias") or primary.get("name")
        if label:
            names.append(label)

    for guest in coaches.get("guest") or []:
        if not isinstance(guest, dict):
            continue
        if not include_hidden and guest.get("hidden_from_app"):
            continue
        label = guest.get("alias") or guest.get("name")
        if label and label not in names:
            names.append(label)

    return names


# Fixed picker options that exist in every archer's coach list but aren't real coaches —
# never persist these into profile.json even if a client sends one back as `entry.coach`.
_NON_COACH_LABELS = {"self observation", "other", "unknown", ""}


def add_archer_coach(archer_id: str, name: str) -> None:
    """Upserts a coach name into profile.json's coaches.guest, so a name typed once via
    "Other" on the coach-notes picker becomes a selectable option from then on. No-ops
    for names already present (case-sensitive match, same as get_archer_coaches) or for
    the generic picker labels a client might echo back (e.g. "Unknown", the frontend's
    fallback when "Other" is picked with no name typed)."""
    name = (name or "").strip()
    if not name or name.lower() in _NON_COACH_LABELS:
        return
    if name in get_archer_coaches(archer_id, include_hidden=True):
        return   # already on file, hidden or not — don't create a duplicate entry

    profile = _load_profile(archer_id)
    if not profile:
        return
    coaches = profile.setdefault("coaches", {})
    if coaches.get("guest") is None:
        coaches["guest"] = []
    coaches["guest"].append({"name": name, "started": time.strftime("%Y-%m-%d")})
    _save_profile(archer_id, profile)


def update_name(archer_id: str, name: str) -> str:
    """Updates identity.name in profile.json. Unlike the auth bootstrap (which
    deliberately never touches profile.json), this IS an edit to profile.json's actual
    content, made through the app on purpose — reformatting it via json.dump on save is
    an accepted trade-off here, not an incidental side effect."""
    profile = _load_profile(archer_id)
    if not profile:
        raise HTTPException(404, "Archer not found")
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required")
    profile.setdefault("identity", {})["name"] = name
    profile.setdefault("_meta", {})["last_updated"] = time.strftime("%Y-%m-%d")
    _save_profile(archer_id, profile)
    return name


def change_password(archer_id: str, current_password: str, new_password: str) -> None:
    _check_password(archer_id, current_password)
    if len(new_password) < MIN_PASSWORD_LEN:
        raise HTTPException(400, f"Password must be at least {MIN_PASSWORD_LEN} characters")
    password_hash, salt = _hash_password(new_password)
    _save_auth(archer_id, password_hash, salt)
    _revoke_all_tokens_for(archer_id)   # force re-login everywhere, including this device


def delete_archer(archer_id: str, password: str) -> None:
    """Verifies password, then deletes archers/<id>/ entirely (profile.json, auth.json,
    coaching notes, everything). Does NOT touch session_history/ — that cascade lives in
    main.py's route handler, since auth.py doesn't own SESSION_DIR."""
    _check_password(archer_id, password)
    profile_dir = _profile_path(archer_id).parent
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    _revoke_all_tokens_for(archer_id)


def _revoke_all_tokens_for(archer_id: str) -> None:
    for tok, entry in list(_tokens.items()):
        if entry["archer_id"] == archer_id:
            _tokens.pop(tok, None)


def logout(token: str) -> None:
    _tokens.pop(token, None)


def get_current_archer(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: resolves the bearer token in the Authorization header to an
    archer_id, 401ing on anything missing/malformed/expired."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    entry = _tokens.get(token)
    if not entry or entry["expires"] < time.time():
        _tokens.pop(token, None)
        raise HTTPException(401, "Invalid or expired token")
    entry["expires"] = time.time() + TOKEN_TTL_SECONDS   # sliding expiry
    return entry["archer_id"]

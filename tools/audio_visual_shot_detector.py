"""
AudioVisualShotDetector
=======================
Detects arrow releases by combining two independent signals:

  1. AUDIO  — high-pass filtered onset detection finds the metallic clicker click
              (a sharp transient distinct from broadband range noise)
  2. VISUAL — anchor zone check confirms the draw wrist was at the archer's face
              in the frames immediately before each audio candidate

Together these are highly robust:
  • Other archers' clickers fire when OUR archer's wrist is NOT at anchor → rejected
  • Bow-lowering / reaching-for-arrows movements have no audio transient → rejected
  • A real shot is the ONLY event that produces both signals simultaneously

Usage
-----
    detector = AudioVisualShotDetector(
        angle       = "face",      # "face" | "back" | "target"
        archer_hand = "right",
        fps         = 30.0,
        min_shot_interval = 8.0,   # seconds — fastest possible nock-draw-release
    )
    detector.extract_candidates(video_path)   # run once before the frame loop

    # inside the frame loop:
    detector.feed_pose(frame_idx, lm, w, h)   # every frame, even non-shot frames
    fired, hold_time_s, score = detector.update(wrist_x, frame_idx)
    if fired:
        fd_lm = detector.fd_lm   # landmark snapshot at full-draw for FD metrics
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

# MediaPipe PoseLandmark integer indices (same values used by YOLO adapter)
# Avoids runtime `import mediapipe` dependency.
_MP_NOSE           = 0
_MP_LEFT_EAR       = 7
_MP_RIGHT_EAR      = 8
_MP_LEFT_SHOULDER  = 11
_MP_RIGHT_SHOULDER = 12
_MP_LEFT_ELBOW     = 13
_MP_RIGHT_ELBOW    = 14
_MP_LEFT_WRIST     = 15
_MP_RIGHT_WRIST    = 16


# ─────────────────────────────────────────────────────────────────────────────
#  Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_wav_mono(video_path: str, target_sr: int = 44100) -> Optional[str]:
    """
    Extract mono WAV from video using afconvert (macOS) with ffmpeg fallback.
    Returns path to a temporary WAV file, or None on failure.
    Caller is responsible for deleting the file.
    """
    tmp = tempfile.mktemp(suffix=".wav")

    # Try afconvert first (no external install needed on macOS)
    r = subprocess.run(
        ["afconvert", video_path, tmp,
         "-d", f"LEI16@{target_sr}", "-f", "WAVE", "-c", "1"],
        capture_output=True,
    )
    if r.returncode == 0:
        return tmp

    # Fallback: ffmpeg
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", str(target_sr), "-vn", tmp],
        capture_output=True,
    )
    if r.returncode == 0:
        return tmp

    if os.path.exists(tmp):
        os.unlink(tmp)
    return None


def _read_wav(wav_path: str) -> Tuple[np.ndarray, int]:
    """Read WAV → (float32 mono array normalised to ±1, sample_rate)."""
    with wave.open(wav_path, "rb") as wf:
        sr  = wf.getframerate()
        n   = wf.getnframes()
        nc  = wf.getnchannels()
        raw = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nc > 1:
        data = data.reshape(-1, nc).mean(axis=1)
    peak = np.abs(data).max()
    if peak > 0:
        data /= peak
    return data, sr


def _highpass_filter(data: np.ndarray, sr: int, cutoff_hz: float = 3000.0,
                     order: int = 4) -> np.ndarray:
    """
    Butterworth high-pass filter.
    Emphasises the metallic click content (>3 kHz) while suppressing
    broadband room noise and low-frequency bow/equipment sounds.
    """
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, cutoff_hz / (sr / 2), btype="high")
        return filtfilt(b, a, data)
    except ImportError:
        # scipy not available — skip filter (degrades accuracy in noisy ranges)
        return data


def _onset_strength(data: np.ndarray, sr: int, window_ms: int = 1) -> Tuple[np.ndarray, float]:
    """
    Short-time RMS onset strength with a 1 ms window (captures fast transients).
    Returns (onset_array, seconds_per_sample).
    """
    win = max(1, int(sr * window_ms / 1000))
    hop = max(1, win // 2)
    n_frames = (len(data) - win) // hop
    rms = np.array([
        np.sqrt(np.mean(data[i * hop: i * hop + win] ** 2))
        for i in range(n_frames)
    ])
    onset = np.diff(rms, prepend=rms[0])
    return np.clip(onset, 0, None), hop / sr   # seconds per onset sample


def _pick_peaks(onset: np.ndarray, sps: float,
                threshold_pct: float, min_gap_s: float) -> List[Tuple[int, float]]:
    """
    Find local maxima above threshold_pct × global_max, separated by ≥ min_gap_s.
    Returns list of (onset_idx, strength) sorted by onset_idx.
    """
    if onset.max() <= 0:
        return []
    threshold = onset.max() * threshold_pct
    min_gap   = max(1, int(min_gap_s / sps))

    peaks: List[Tuple[int, float]] = []
    last_idx = -min_gap

    # Collect all local maxima above threshold
    candidates = [
        i for i in range(1, len(onset) - 1)
        if onset[i] > threshold and onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]
    ]

    # Apply minimum gap (greedy: keep highest-strength in each window)
    for idx in candidates:
        if idx - last_idx >= min_gap:
            peaks.append((idx, float(onset[idx])))
            last_idx = idx
        elif peaks and onset[idx] > peaks[-1][1]:
            # Replace last peak if this one is stronger within the gap window
            peaks[-1] = (idx, float(onset[idx]))
            last_idx = idx

    return peaks


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class AudioVisualShotDetector:
    """
    Drop-in replacement for ShotDetector with audio + visual confirmation.

    Interface compatibility with ShotDetector
    -----------------------------------------
    • set_fps(fps)
    • feed_pose(frame_idx, lm, w, h)   ← NEW — call every frame in the loop
    • update(wrist_x, frame_idx) → (fired, hold_time_s, velocity)
    • full_draw_just_detected          ← always False (FD handled via fd_lm)
    • fd_lm                            ← landmark snapshot at best anchor frame
    • get_current_state()              ← compatibility stub
    • _full_draw_frame                 ← compatibility stub
    """

    # How many frames before the click to look back for anchor zone
    LOOKBACK_FRAMES = 20
    # Maximum allowed shift in draw-shoulder X between confirmed shots.
    # Our archer's shoulder stays within ~0.02 units across shots.
    # Another archer standing nearby can shift shoulder X by 0.08-0.13.
    # Threshold of 0.15 is conservative — catches only extreme person-switching.
    # The primary defence against adjacent-archer FPs is the 10s minimum interval
    # floor in archery_analyzer_v2.py (adjacent archers fire their clicker <10s
    # after the first confirmed shot, while the minimum real inter-shot gap is ~14s).
    # Only applied once ≥1 shot has been confirmed (self-initialising).
    SHOULDER_STABILITY_THRESHOLD = 0.15
    # Tolerance window around the audio candidate (±frames).
    # Needs to be ≥10 to cover pose-detection dropouts at the release moment:
    # MediaPipe often loses tracking for 5-10 frames when the draw arm snaps
    # back, so update() may not be called in the ±3 window.
    # The lookback scan is bounded by cand_frame (not frame_idx), so widening
    # this window does not affect which frames contribute to the FD snapshot.
    CANDIDATE_MATCH = 10

    def __init__(
        self,
        angle: str,
        archer_hand: str,
        fps: float = 30.0,
        min_shot_interval: float = 8.0,
        n_shots: Optional[int] = None,
        audio_threshold_pct: float = 0.20,   # fraction of window max
        anchor_min_score: float = 0.10,       # minimum anchor zone score to accept
        prefer_strongest_audio: bool = True,  # see extract_candidates
    ):
        self.angle              = angle.lower()
        self.archer_hand        = archer_hand.lower()
        self._fps               = fps
        self.min_shot_interval  = min_shot_interval
        self.n_shots            = n_shots
        self.audio_threshold    = audio_threshold_pct
        self.prefer_strongest_audio = prefer_strongest_audio
        # Face view has reliable wrist/nose detection head-on; raise the floor
        # to reject borderline anchor scores (real shots score 0.52+, FPs 0.38-0.40).
        # Adjacent-archer FPs on face view score ~0.38-0.40 because their wrist is
        # partially in the jaw zone from a side angle. 0.45 sits cleanly in the gap.
        # Back/target views keep the conservative 0.10 — pose estimation is weaker.
        if anchor_min_score == 0.10 and self.angle == "face":
            self.anchor_min_score = 0.45
        else:
            self.anchor_min_score = anchor_min_score

        # Audio candidates: list of (video_frame, strength)
        self._candidates: List[Tuple[int, float]] = []

        # Circular buffer of recent pose landmarks: deque of (frame_idx, lm, w, h)
        self._pose_buffer: deque = deque(maxlen=self.LOOKBACK_FRAMES + self.CANDIDATE_MATCH + 5)

        # State
        self._last_shot_frame   = -999999
        self._fired_candidates: set = set()   # candidates already used (prevent double-fire)
        self._logged_rejections: set = set()  # track per-candidate rejection log (print once)
        self._confirmed_shoulder_xs: List[float] = []     # draw-shoulder X from confirmed shots

        # Public attributes matching ShotDetector interface
        self.full_draw_just_detected = False  # always False; FD via fd_lm
        self._full_draw_frame        = None   # compat stub
        self.fd_lm                   = None   # set to lm at best anchor frame when shot fires
        self.fd_frame_idx            = None   # frame index of fd_lm

    # ── Public interface ──────────────────────────────────────────────────────

    def set_fps(self, fps: float):
        self._fps = fps

    def extract_candidates(self, video_path: str) -> int:
        """
        Extract audio from video_path, detect onset peaks, store as candidates.
        Call once before the main frame loop.
        Returns number of audio candidates found.
        """
        wav_path = _extract_wav_mono(video_path)
        if wav_path is None:
            print(f"    [AudioVisual] Audio extraction failed — no audio track in {os.path.basename(video_path)}")
            return 0

        try:
            data, sr = _read_wav(wav_path)
        finally:
            os.unlink(wav_path)

        # High-pass filter to emphasise metallic click content
        data = _highpass_filter(data, sr, cutoff_hz=3000.0)

        onset, sps = _onset_strength(data, sr, window_ms=1)

        # Skip first 2% and last 2% of video to avoid setup/packing noise
        # (reduced from 5% so early shots in a short warmup window are captured)
        n = len(onset)
        skip_start = int(0.02 * n)
        skip_end   = max(skip_start + 1, int(0.98 * n))
        onset[:skip_start] = 0
        onset[skip_end:]   = 0

        # Audio candidate gap is shorter than the confirmed-shot gap.
        # Using min_shot_interval (8s) as the candidate gap caused louder
        # other-archer clicks to subsume our archer's quieter real shots
        # when both fell in the same 8s window.  A 2s gap lets both surface;
        # the visual anchor check rejects the other-archer event (arm down),
        # and the confirmed-shot 8s floor in update() prevents false pairs.
        AUDIO_CANDIDATE_GAP_S = 2.0
        peaks = _pick_peaks(onset, sps,
                            threshold_pct=self.audio_threshold,
                            min_gap_s=AUDIO_CANDIDATE_GAP_S)

        # Convert onset indices → video frame numbers
        # sps = seconds per onset sample
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps    = cap.get(cv2.CAP_PROP_FPS) or self._fps
        finally:
            if cap:
                cap.release()

        audio_dur_s = len(onset) * sps

        self._candidates = []
        for peak_idx, strength in peaks:
            audio_time_s = peak_idx * sps
            # Scale audio time to video time (handles minor duration mismatches)
            video_time_s = audio_time_s
            vframe = int(round(video_time_s * video_fps))
            vframe = max(0, min(vframe, total_frames - 1))
            self._candidates.append((vframe, strength))

        self._candidates.sort(key=lambda x: x[0])

        # Strength preference: when a quiet event (other-archer click, equipment
        # noise, mic bump) and our archer's louder clicker both fall within
        # min_shot_interval, the chronologically-earlier event would otherwise
        # fire first in update() and block the real shot via the gap rule.
        # Greedy non-max suppression by strength keeps only the loudest event
        # in each min_shot_interval window. Requires that our archer's clicker
        # is reliably the loudest audio event in the vicinity of each shot.
        if self.prefer_strongest_audio and self._candidates:
            gap_frames = int(self.min_shot_interval * video_fps)
            kept: List[Tuple[int, float]] = []
            for vf, st in sorted(self._candidates, key=lambda x: -x[1]):
                if all(abs(vf - kf) >= gap_frames for kf, _ in kept):
                    kept.append((vf, st))
            self._candidates = sorted(kept, key=lambda x: x[0])

        print(f"    [AudioVisual] {len(self._candidates)} audio candidate(s) "
              f"from {audio_dur_s:.1f}s audio  "
              f"(threshold={self.audio_threshold:.2f}×max, min_gap={self.min_shot_interval:.0f}s)")
        for vf, st in self._candidates:
            print(f"      candidate  frame {vf:5d}  ({vf/video_fps:.1f}s)  strength={st:.5f}")

        return len(self._candidates)

    def feed_pose(self, frame_idx: int, lm, w: int, h: int):
        """
        Store the current pose landmarks for anchor zone lookback.
        Must be called every frame BEFORE update().
        """
        self._pose_buffer.append((frame_idx, lm, w, h))

    def update(self, wrist_x: float, frame_idx: int) -> Tuple[bool, Optional[float], float]:
        """
        Check whether this frame is a confirmed shot.

        Returns (fired, hold_time_s, combined_score).
        Matches the ShotDetector.update() return type.
        """
        self.full_draw_just_detected = False  # always False for this detector
        self.fd_lm                   = None

        if not self._candidates:
            return False, None, 0.0

        min_gap_frames = int(self.min_shot_interval * self._fps)

        for cand_frame, cand_strength in self._candidates:
            if cand_frame in self._fired_candidates:
                continue
            if abs(frame_idx - cand_frame) > self.CANDIDATE_MATCH:
                continue

            # ── Anchor zone check in lookback window ──────────────────────
            best_score    = 0.0
            best_fd_frame = cand_frame
            best_fd_lm    = None
            best_fd_w     = None
            best_fd_h     = None

            for buf_frame, buf_lm, buf_w, buf_h in self._pose_buffer:
                if buf_frame < cand_frame - self.LOOKBACK_FRAMES:
                    continue
                if buf_frame > cand_frame:
                    break
                score = self._anchor_score(buf_lm, buf_w, buf_h)
                if score > best_score:
                    best_score    = score
                    best_fd_frame = buf_frame
                    best_fd_lm    = buf_lm
                    best_fd_w     = buf_w
                    best_fd_h     = buf_h

            if best_score < self.anchor_min_score:
                # ── n_shots fallback: promote if we still need more shots ──
                # When --shots N is provided and pose detection failed (back/target
                # view MediaPipe gaps), promote the strongest remaining audio candidates
                # that respect the min_gap, logging a clear warning.
                n_fired = len(self._fired_candidates)
                if (self.n_shots is not None
                        and n_fired < self.n_shots
                        and frame_idx - self._last_shot_frame >= min_gap_frames):
                    if cand_frame not in self._logged_rejections:
                        self._logged_rejections.add(cand_frame)
                        print(f"    [AudioVisual] Promoted  frame {cand_frame}  "
                              f"anchor_score={best_score:.3f}  "
                              f"audio_strength={cand_strength:.5f}  "
                              f"(n_shots fallback: pose check failed, "
                              f"{n_fired}/{self.n_shots} shots confirmed so far)")
                    # Fall through to confirmed-shot block below with score=cand_strength
                    best_score = max(best_score, 0.01)   # nonzero so combined_score is meaningful
                else:
                    if cand_frame not in self._logged_rejections:
                        self._logged_rejections.add(cand_frame)
                        print(f"    [AudioVisual] Rejected  frame {cand_frame}  "
                              f"anchor_score={best_score:.3f}  "
                              f"audio_strength={cand_strength:.5f}  "
                              f"(wrist not at anchor — other archer or non-shot noise)")
                    continue  # wrist never reached anchor zone near this click — reject

            # ── Shoulder position stability ────────────────────────────────
            # Once ≥1 shot has been confirmed, require the draw shoulder X to
            # stay within SHOULDER_STABILITY_THRESHOLD of the running mean.
            # This rejects frames where MediaPipe switched to tracking an
            # adjacent archer (their shoulder X shifts by 0.08+ in image space).
            # Skipped when best_fd_lm is None (n_shots fallback / pose dropout).
            if best_fd_lm is not None and self._confirmed_shoulder_xs:
                current_sh_x = self._get_draw_shoulder_x(best_fd_lm)
                mean_sh_x    = sum(self._confirmed_shoulder_xs) / len(self._confirmed_shoulder_xs)
                sh_drift     = abs(current_sh_x - mean_sh_x)
                if sh_drift > self.SHOULDER_STABILITY_THRESHOLD:
                    if cand_frame not in self._logged_rejections:
                        self._logged_rejections.add(cand_frame)
                        print(f"    [AudioVisual] Rejected  frame {cand_frame}  "
                              f"shoulder_x={current_sh_x:.3f}  mean={mean_sh_x:.3f}  "
                              f"drift={sh_drift:.3f}  "
                              f"(different person — shoulder shifted > {self.SHOULDER_STABILITY_THRESHOLD})")
                    continue

            # ── Minimum time gap ──────────────────────────────────────────
            if frame_idx - self._last_shot_frame < min_gap_frames:
                continue

            # ── Confirmed shot ────────────────────────────────────────────
            self._fired_candidates.add(cand_frame)
            self._last_shot_frame = frame_idx

            # Hold time = frames from best anchor frame to click
            hold_frames  = max(0, cand_frame - best_fd_frame)
            hold_time_s  = hold_frames / self._fps if hold_frames > 0 else None

            combined_score = cand_strength * best_score

            # Expose FD landmarks for FD snapshot in AngleProcessor
            self.fd_lm        = best_fd_lm
            self.fd_frame_idx = best_fd_frame
            self._full_draw_frame = best_fd_frame   # compat

            # Record draw-shoulder position for future stability checks
            if best_fd_lm is not None:
                self._confirmed_shoulder_xs.append(self._get_draw_shoulder_x(best_fd_lm))

            print(f"    [AudioVisual] Shot confirmed  frame {frame_idx}  "
                  f"anchor_score={best_score:.3f}  "
                  f"audio_strength={cand_strength:.5f}  "
                  f"hold={f'{hold_time_s:.2f}s' if hold_time_s else 'n/a'}")

            return True, hold_time_s, combined_score

        return False, None, 0.0

    # ── Compatibility stubs ───────────────────────────────────────────────────

    def get_current_state(self):
        """Compatibility stub matching MLShotDetector interface."""
        return "UNKNOWN"

    def update_full(self, landmarks_obj, frame_idx, shape):
        """
        Compatibility stub for ML path.
        Not used — the AV detector uses update() via the standard path.
        """
        return None

    # ── Anchor zone check (view-specific) ────────────────────────────────────

    def _get_draw_shoulder_x(self, lm) -> float:
        """Return the draw-side shoulder X position from a landmark set."""
        try:
            idx = _MP_RIGHT_SHOULDER if self.archer_hand == "right" else _MP_LEFT_SHOULDER
            return lm[idx].x
        except (IndexError, AttributeError):
            return 0.5   # neutral fallback — no effect on stability check

    def _anchor_score(self, lm, w: int, h: int) -> float:
        """
        Returns a 0–1 score indicating how firmly the draw wrist is in the
        anchor zone for the given camera angle.

        0   = wrist definitely not at anchor
        1   = wrist perfectly at anchor (chin level, close to face)
        """
        try:
            if self.archer_hand == "right":
                wrist    = lm[_MP_RIGHT_WRIST]
                elbow    = lm[_MP_RIGHT_ELBOW]
                shoulder = lm[_MP_RIGHT_SHOULDER]
                ear      = lm[_MP_RIGHT_EAR]
            else:
                wrist    = lm[_MP_LEFT_WRIST]
                elbow    = lm[_MP_LEFT_ELBOW]
                shoulder = lm[_MP_LEFT_SHOULDER]
                ear      = lm[_MP_LEFT_EAR]

            nose = lm[_MP_NOSE]

        except (IndexError, AttributeError):
            return 0.0

        if self.angle == "face":
            # At anchor in face view:
            #   Primary:   Wrist Y ≈ chin level, Wrist X within ±0.18 of nose X
            #   Fallback:  Draw elbow raised above shoulder and at head height
            #              (used when wrist is occluded by the face at anchor)
            if wrist.visibility >= 0.1 and nose.visibility >= 0.1:
                ideal_y = nose.y + 0.07          # chin / jaw level
                dy = abs(wrist.y - ideal_y) / 0.10
                dx = abs(wrist.x - nose.x) / 0.18
                dist = (dy ** 2 + dx ** 2) ** 0.5
                wrist_score = max(0.0, 1.0 - dist)
            else:
                wrist_score = 0.0

            # Elbow fallback: at anchor the draw elbow is elevated to ear/nose height
            if elbow.visibility >= 0.1 and shoulder.visibility >= 0.1:
                if elbow.y < shoulder.y:   # elbow above shoulder — likely at draw
                    dy_el = abs(elbow.y - nose.y) / 0.20
                    elbow_score = max(0.0, 0.6 * (1.0 - dy_el))   # cap at 0.6 (weaker signal)
                else:
                    elbow_score = 0.0
            else:
                elbow_score = 0.0

            return max(wrist_score, elbow_score)

        elif self.angle == "back":
            # At anchor in back view (T-shape, archer's spine toward camera):
            #   The draw elbow is always visible from behind and is the most
            #   reliable indicator. At full draw, MediaPipe estimates the draw
            #   elbow within ~0.10 units BELOW the shoulder (slightly lower than
            #   reality due to training-data bias toward front-facing poses).
            #   "Above shoulder" gate is too strict; use shoulder-proximity score.
            #   At rest (arm hanging down), elbow is 0.20+ below shoulder.
            #
            #   Note on lateral (X) position: at T-draw the elbow is pulled inward
            #   (~0.09 units LEFT of shoulder in image space). This varies significantly
            #   with camera distance and position across sessions, so a fixed lateral
            #   threshold is not used here. Person-switching is caught instead by the
            #   shoulder stability check in update() (SHOULDER_STABILITY_THRESHOLD).
            #
            #   Primary:  Elbow within 0.12 below shoulder → score by closeness
            #             to shoulder height (the draw-arm T-position landmark)
            #   Secondary: Wrist above shoulder (if visible) as supporting signal

            # Elbow primary signal
            if elbow.visibility >= 0.15 and shoulder.visibility >= 0.15:
                # Gate: elbow must be near shoulder height (not hanging way down)
                if elbow.y < shoulder.y + 0.14:
                    # Score: 1.0 when elbow is exactly at shoulder height; falls off
                    # as it drops below.  Tolerance = 0.12 units.
                    dy_el = max(0.0, elbow.y - shoulder.y) / 0.12
                    elbow_score = max(0.0, 1.0 - dy_el)
                else:
                    elbow_score = 0.0
            else:
                elbow_score = 0.0

            # Wrist secondary signal (if visible)
            if wrist.visibility >= 0.15 and shoulder.visibility >= 0.15:
                if wrist.y < shoulder.y + 0.05:
                    dy_wr = max(0.0, wrist.y - shoulder.y) / 0.10
                    wrist_score = max(0.0, 0.7 * (1.0 - dy_wr))   # cap at 0.7
                else:
                    wrist_score = 0.0
            else:
                wrist_score = 0.0

            return max(elbow_score, wrist_score)

        elif self.angle == "target":
            # At anchor in target view (directly behind archer along arrow line):
            #   Primary:   Draw ELBOW raised above shoulder AND laterally displaced.
            #              From directly behind, the bow elbow is forward/hidden but
            #              the draw elbow flares out to the side at full draw.
            #   Secondary: Draw wrist near head height (if visible).

            # Elbow primary signal
            if elbow.visibility >= 0.15 and shoulder.visibility >= 0.15:
                if elbow.y < shoulder.y + 0.05:   # elbow at or above shoulder — drawn
                    dy_el = abs(elbow.y - nose.y) / 0.18
                    elbow_score = max(0.0, 1.0 - dy_el)
                else:
                    elbow_score = 0.0
            else:
                elbow_score = 0.0

            # Wrist secondary signal (if visible)
            if wrist.visibility >= 0.15 and shoulder.visibility >= 0.15:
                if elbow.y > shoulder.y + 0.05:   # elbow down — definitely not drawn
                    wrist_score = 0.0
                else:
                    dy_wr = abs(wrist.y - nose.y) / 0.15
                    wrist_score = max(0.0, 0.7 * (1.0 - dy_wr))   # cap at 0.7
            else:
                wrist_score = 0.0

            return max(elbow_score, wrist_score)

        return 0.0

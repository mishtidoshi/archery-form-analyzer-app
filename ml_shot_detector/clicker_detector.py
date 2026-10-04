"""
Audio-based clicker detection for archery release frame identification.

The clicker (a small metal blade on the bow) makes a sharp click the moment
the archer's draw reaches full length.  This module extracts audio from the
video file and locates that transient, returning the corresponding video frame.

Strategy
--------
1. Extract mono 44.1 kHz PCM WAV with afconvert (macOS) or ffmpeg fallback.
2. Compute short-time onset strength in 10 ms windows.
3. Within a valid window [min_frac, max_frac] of the video duration:
      - The first ~10 % is excluded (setup / bow-raise sounds).
      - The last ~15 % is excluded (arrow hitting target, which is louder and later).
4. Find all onset peaks above 35 % of the window's own maximum.
5. Return the LAST qualifying peak — the user's clicker is always towards the end
   of the clip and is the loudest nearby sound (close microphone), so later peaks
   dominate even when other archers' clickers appear earlier in the recording.
6. Convert audio timestamp → video frame.
"""
import os
import subprocess
import tempfile
import wave

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_wav(video_path: str, out_wav: str) -> bool:
    """Extract mono 44.1 kHz 16-bit WAV.  Returns True on success."""
    r = subprocess.run(
        ["afconvert", video_path, out_wav,
         "-d", "LEI16@44100", "-f", "WAVE", "-c", "1"],
        capture_output=True,
    )
    if r.returncode == 0:
        return True
    # Fallback to ffmpeg if afconvert unavailable
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", "44100", "-vn", out_wav],
        capture_output=True,
    )
    return r.returncode == 0


def _read_wav_mono(wav_path: str):
    """Return (samples_float32, sample_rate)."""
    with wave.open(wav_path, "rb") as wf:
        sr  = wf.getframerate()
        n   = wf.getnframes()
        nc  = wf.getnchannels()
        raw = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if nc > 1:
        data = data.reshape(-1, nc).mean(axis=1)
    return data, sr


def _onset_strength(data: np.ndarray, sr: int, window_ms: int = 10) -> np.ndarray:
    """
    Short-time onset strength — positive energy derivative, one value per window.

    Index i corresponds to time  i * (window_ms / 1000)  seconds.
    """
    win       = int(sr * window_ms / 1000)
    n_windows = (len(data) - win) // win
    energy    = np.array([
        np.mean(data[i * win : i * win + win] ** 2)
        for i in range(n_windows)
    ])
    onset = np.diff(energy)
    return np.clip(onset, 0, None)   # keep only rises


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_clicker_frame(
    video_path: str,
    video_fps: float = None,
    min_fraction: float = 0.10,
    max_fraction: float = 0.85,
    peak_threshold: float = 0.35,
    window_ms: int = 10,
) -> tuple:
    """
    Detect the clicker transient and return the corresponding video frame.

    Parameters
    ----------
    video_path      : path to .MOV / .MP4 video
    video_fps       : frame rate (auto-detected when None)
    min_fraction    : skip the first N % of video (setup / bow-raise)
    max_fraction    : skip the last N % of video (arrow hitting target)
    peak_threshold  : peaks must exceed this fraction of the window's max onset
    window_ms       : onset analysis window in milliseconds

    Returns
    -------
    (frame: int | None, info: str)
        frame — 0-based video frame index, or None on failure
        info  — human-readable description of the result
    """
    # --- video metadata ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Cannot open video: {video_path}"
    if video_fps is None:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / video_fps

    # --- audio extraction ---
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        if not _extract_wav(video_path, tmp_wav):
            return None, "audio extraction failed (need afconvert or ffmpeg)"
        data, sr = _read_wav_mono(tmp_wav)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    audio_dur = len(data) / sr
    onset     = _onset_strength(data, sr, window_ms)
    dt        = window_ms / 1000.0          # seconds per onset sample

    # --- valid detection window ---
    min_idx = max(0, int(min_fraction * audio_dur / dt))
    max_idx = min(len(onset) - 1, int(max_fraction * audio_dur / dt))

    if max_idx <= min_idx:
        return None, "video too short for clicker detection"

    onset_window = onset[min_idx:max_idx]
    if onset_window.max() <= 0:
        return None, "no positive onset in valid window"

    # --- peak detection ---
    thresh = float(onset_window.max()) * peak_threshold
    peaks  = [
        min_idx + i
        for i in range(1, len(onset_window) - 1)
        if (onset_window[i] > thresh
            and onset_window[i] >= onset_window[i - 1]
            and onset_window[i] >= onset_window[i + 1])
    ]

    if not peaks:
        # Relax threshold and retry once
        thresh = float(onset_window.max()) * 0.10
        peaks  = [
            min_idx + i
            for i in range(1, len(onset_window) - 1)
            if (onset_window[i] > thresh
                and onset_window[i] >= onset_window[i - 1]
                and onset_window[i] >= onset_window[i + 1])
        ]

    if not peaks:
        return None, "no onset peaks found above threshold in valid window"

    # Take the LAST qualifying peak — the user's clicker is loud and late
    peak_idx  = peaks[-1]
    peak_time = peak_idx * dt                                    # seconds in audio
    video_time = peak_time * (duration_s / audio_dur)           # scale to video timeline
    frame      = int(round(video_time * video_fps))
    frame      = max(0, min(frame, total_frames - 1))

    confidence = float(onset[peak_idx]) / float(onset_window.max())
    info = (
        f"clicker at {peak_time:.2f}s → frame {frame}  "
        f"({len(peaks)} peak(s) in window, strength={confidence:.2f}x window-max, "
        f"window=[{min_fraction*100:.0f}%–{max_fraction*100:.0f}%] of {duration_s:.1f}s)"
    )
    return frame, info


def detect_all_clicker_frames(
    video_path: str,
    video_fps: float = None,
    min_fraction: float = 0.02,
    max_fraction: float = 0.98,
    peak_threshold: float = 0.20,
    window_ms: int = 10,
    min_spacing_s: float = 5.0,
) -> tuple:
    """
    Detect ALL clicker transients in a multi-arrow video.

    Parameters
    ----------
    video_path      : path to .MOV / .MP4 video
    video_fps       : frame rate (auto-detected when None)
    min_fraction    : skip first N % of video
    max_fraction    : skip last N % of video
    peak_threshold  : peaks must exceed this fraction of the window's max onset
    window_ms       : onset analysis window in milliseconds
    min_spacing_s   : minimum seconds between accepted peaks (prevents duplicates)

    Returns
    -------
    (frames: List[int], info: str)
        frames — list of 0-based video frame indices (sorted), empty on failure
        info   — human-readable description
    """
    # --- video metadata ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], f"Cannot open video: {video_path}"
    if video_fps is None:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / video_fps

    # --- audio extraction ---
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        if not _extract_wav(video_path, tmp_wav):
            return [], "audio extraction failed (need afconvert or ffmpeg)"
        data, sr = _read_wav_mono(tmp_wav)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    audio_dur = len(data) / sr
    onset     = _onset_strength(data, sr, window_ms)
    dt        = window_ms / 1000.0

    # --- valid detection window ---
    min_idx = max(0, int(min_fraction * audio_dur / dt))
    max_idx = min(len(onset) - 1, int(max_fraction * audio_dur / dt))

    if max_idx <= min_idx:
        return [], "video too short for clicker detection"

    onset_window = onset[min_idx:max_idx]
    if onset_window.max() <= 0:
        return [], "no positive onset in valid window"

    # --- peak detection ---
    thresh = float(onset_window.max()) * peak_threshold
    raw_peaks = [
        min_idx + i
        for i in range(1, len(onset_window) - 1)
        if (onset_window[i] > thresh
            and onset_window[i] >= onset_window[i - 1]
            and onset_window[i] >= onset_window[i + 1])
    ]

    if not raw_peaks:
        return [], "no onset peaks found above threshold"

    # --- enforce minimum spacing between accepted peaks ---
    min_spacing_samples = int(min_spacing_s / dt)
    accepted = []
    for p in raw_peaks:
        if not accepted or (p - accepted[-1]) >= min_spacing_samples:
            accepted.append(p)

    # --- convert audio peak indices → video frames ---
    frames = []
    for peak_idx in accepted:
        peak_time  = peak_idx * dt
        video_time = peak_time * (duration_s / audio_dur)
        frame      = int(round(video_time * video_fps))
        frame      = max(0, min(frame, total_frames - 1))
        frames.append(frame)

    frames.sort()
    times_str = ", ".join(f"{f/video_fps:.1f}s→f{f}" for f in frames)
    info = (
        f"{len(frames)} clicker(s) detected: [{times_str}]  "
        f"(threshold={peak_threshold:.2f}x, min_spacing={min_spacing_s:.0f}s)"
    )
    return frames, info

#!/usr/bin/env python3
"""
Clicker Frame Verification Tool

Loops through all 28 videos, shows the detected clicker frame
(with pose overlay + context frames) plus audio waveform preview.

Controls:
  y / Enter  — accept detected frame as-is
  <number>   — type a new frame number to correct
  s          — skip this video (keep current value)
  q          — quit and save progress

After reviewing, re-extracts training data only for corrected videos.
"""

import cv2
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button
import mediapipe as mp
import subprocess
import struct
import wave

# ─────────────────────────────── video list ───────────────────────────────

SIDE_DIR   = "data/march_2026/032226/side"
BEHIND_DIR = "data/march_2026/032226/behind"

DETECTED = [
    # (video_path, detected_frame, view)
    (f"{SIDE_DIR}/IMG_2773.MOV",  69,  "side"),
    (f"{SIDE_DIR}/IMG_2774.MOV", 138,  "side"),
    (f"{SIDE_DIR}/IMG_2775.MOV", 179,  "side"),
    (f"{SIDE_DIR}/IMG_2776.MOV", 152,  "side"),
    (f"{SIDE_DIR}/IMG_2777.MOV",  74,  "side"),
    (f"{SIDE_DIR}/IMG_2778.MOV", 149,  "side"),
    (f"{SIDE_DIR}/IMG_2779.MOV",  72,  "side"),
    (f"{SIDE_DIR}/IMG_2780.MOV", 153,  "side"),
    (f"{SIDE_DIR}/IMG_2781.MOV", 197,  "side"),
    (f"{SIDE_DIR}/IMG_2782.MOV", 194,  "side"),
    (f"{SIDE_DIR}/IMG_2783.MOV", 138,  "side"),
    (f"{SIDE_DIR}/IMG_2784.MOV", 138,  "side"),
    (f"{SIDE_DIR}/IMG_2785.MOV", 148,  "side"),
    (f"{SIDE_DIR}/IMG_2786.MOV", 129,  "side"),
    (f"{SIDE_DIR}/IMG_2787.MOV", 164,  "side"),
    (f"{SIDE_DIR}/IMG_2788.MOV", 157,  "side"),
    (f"{SIDE_DIR}/IMG_2789.MOV", 154,  "side"),
    (f"{SIDE_DIR}/IMG_2790.MOV", 292,  "side"),
    (f"{SIDE_DIR}/IMG_2791.MOV", 173,  "side"),
    (f"{SIDE_DIR}/IMG_2792.MOV", 152,  "side"),
    (f"{SIDE_DIR}/IMG_2793.MOV", 148,  "side"),
    (f"{SIDE_DIR}/IMG_2794.MOV", 154,  "side"),
    (f"{SIDE_DIR}/IMG_2795.MOV", 191,  "side"),
    (f"{SIDE_DIR}/IMG_2796.MOV", 161,  "side"),
    (f"{BEHIND_DIR}/IMG_2798.MOV", 124, "behind"),
    (f"{BEHIND_DIR}/IMG_2799.MOV", 266, "behind"),
    (f"{BEHIND_DIR}/IMG_2800.MOV", 177, "behind"),
    (f"{BEHIND_DIR}/IMG_2801.MOV", 157, "behind"),
]

RESULTS_FILE = "training_data/clicker_verification.json"
CONTEXT_FRAMES = 3   # show N frames before/after detected frame


# ─────────────────────────────── helpers ──────────────────────────────────

def load_progress():
    """Load previously saved verification results."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_FILE}")


def extract_frame_with_pose(video_path, frame_idx, fps, label=""):
    """Return an RGB image of the given frame with pose overlay."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )

    ts = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Frame {frame_idx} | {ts:.2f}s  {label}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_audio_waveform(video_path):
    """Extract audio and return (times, amplitude, sample_rate). Returns None on failure."""
    tmp_wav = "/tmp/_verify_clicker_tmp.wav"
    try:
        cmd = ["afconvert", video_path, tmp_wav,
               "-d", "LEI16@44100", "-f", "WAVE", "-c", "1"]
        subprocess.run(cmd, capture_output=True, check=True)
        with wave.open(tmp_wav, "rb") as wf:
            sr    = wf.getframerate()
            n     = wf.getnframes()
            raw   = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        times = np.arange(len(data)) / sr
        return times, data, sr
    except Exception:
        return None


def compute_onset(data, sr, window_ms=10):
    """Short-time positive energy derivative (onset strength)."""
    win = max(1, int(sr * window_ms / 1000))
    n_windows = len(data) // win
    energy = np.array([np.sum(data[i*win:(i+1)*win]**2) for i in range(n_windows)])
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    t_onset = (np.arange(n_windows) + 0.5) * win / sr
    return t_onset, onset


# ─────────────────────────────── main UI ──────────────────────────────────

class ClickerVerifier:
    def __init__(self, entries, results):
        self.entries    = entries          # list of (path, frame, view)
        self.results    = results          # dict {basename: {frame, status}}
        self.idx        = 0
        self.correction = None             # set by text box
        self._advance_to_next_pending()

    def _advance_to_next_pending(self):
        """Skip already-verified entries."""
        while self.idx < len(self.entries):
            path, _, _ = self.entries[self.idx]
            key = os.path.basename(path)
            if key not in self.results:
                break
            self.idx += 1

    def _current(self):
        if self.idx >= len(self.entries):
            return None, None, None
        return self.entries[self.idx]

    def run(self):
        path, det_frame, view = self._current()
        if path is None:
            print("\nAll videos already verified!")
            return

        while True:
            path, det_frame, view = self._current()
            if path is None:
                print("\n✓ All videos reviewed!")
                break

            accepted, new_frame = self._show_video(path, det_frame, view)

            key = os.path.basename(path)
            if accepted is None:   # quit
                print("\nQuitting — progress saved.")
                break
            elif accepted == "skip":
                print(f"  → Skipped {key}")
                self.results[key] = {"frame": det_frame, "status": "skipped", "view": view}
            elif accepted:
                print(f"  → Accepted {key}: frame {det_frame}")
                self.results[key] = {"frame": det_frame, "status": "accepted", "view": view}
            else:
                print(f"  → Corrected {key}: {det_frame} → {new_frame}")
                self.results[key] = {"frame": new_frame, "status": "corrected",
                                     "original": det_frame, "view": view}

            save_progress(self.results)
            self.idx += 1
            plt.close("all")

        save_progress(self.results)

    def _show_video(self, video_path, det_frame, view):
        """Show interactive verification window. Returns (accepted, corrected_frame)."""
        cap  = cv2.VideoCapture(video_path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        total = len(self.entries)
        done  = len(self.results)
        name  = os.path.basename(video_path)

        print(f"\n[{done+1}/{total}] {name}  (detected frame {det_frame} = {det_frame/fps:.2f}s)")
        print("  Extracting frames + pose overlay…")

        # Build context frames: center ± CONTEXT_FRAMES
        frame_indices = list(range(
            max(0, det_frame - CONTEXT_FRAMES),
            min(n, det_frame + CONTEXT_FRAMES + 1)
        ))

        images = []
        for fi in frame_indices:
            lbl = " ← DETECTED" if fi == det_frame else ""
            images.append(extract_frame_with_pose(video_path, fi, fps, lbl))

        # Audio waveform
        print("  Extracting audio…")
        audio_data = read_audio_waveform(video_path)

        # ── build figure ──────────────────────────────────────────────────
        n_frames = len(images)
        fig = plt.figure(figsize=(max(16, n_frames * 3.2), 9))
        fig.patch.set_facecolor("#1a1a2e")

        gs = gridspec.GridSpec(
            3, n_frames,
            figure=fig,
            height_ratios=[3, 1.2, 0.5],
            hspace=0.35, wspace=0.05
        )

        # Row 0: pose images
        ax_imgs = []
        for col, (fi, img) in enumerate(zip(frame_indices, images)):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.axis("off")
            border_color = "#ff6b35" if fi == det_frame else "#4a4a6a"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3 if fi == det_frame else 1)
            ax_imgs.append(ax)

        # Row 1: waveform spanning all columns
        ax_wave = fig.add_subplot(gs[1, :])
        ax_wave.set_facecolor("#0d0d1a")
        if audio_data is not None:
            times, amp, sr = audio_data
            dur = n / fps
            # downsample waveform for display
            step = max(1, len(amp) // 4000)
            ax_wave.plot(times[::step], amp[::step], color="#4fc3f7", lw=0.6, alpha=0.8)
            # onset strength
            t_onset, onset = compute_onset(amp, sr)
            ax_wave.plot(t_onset, onset / (onset.max() + 1e-9) * amp.max(),
                         color="#ffd54f", lw=1.2, alpha=0.7, label="onset strength")
            # vertical line at detected frame
            ax_wave.axvline(det_frame / fps, color="#ff6b35", lw=2, label=f"detected ({det_frame})")
            # shaded invalid regions
            ax_wave.axvspan(0, dur * 0.10, alpha=0.3, color="gray")
            ax_wave.axvspan(dur * 0.85, dur, alpha=0.3, color="gray")
            ax_wave.set_xlim(0, dur)
            ax_wave.set_xlabel("Time (s)", color="white")
            ax_wave.tick_params(colors="white")
            ax_wave.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
        else:
            ax_wave.text(0.5, 0.5, "Audio extraction failed", ha="center",
                         transform=ax_wave.transAxes, color="white")
        ax_wave.set_title("Audio waveform  (gray = ignored region)", color="white", fontsize=9)
        for spine in ax_wave.spines.values():
            spine.set_edgecolor("#4a4a6a")

        # Title
        fig.suptitle(
            f"[{done+1}/{total}]  {name}  |  view={view}  |  detected frame {det_frame} ({det_frame/fps:.2f}s)\n"
            f"y / Enter = accept     type frame number + Enter = correct     s = skip     q = quit",
            color="white", fontsize=11, y=0.98
        )

        # ── interaction state ─────────────────────────────────────────────
        state = {"result": None, "corrected_frame": None}

        def _accept(event=None):
            state["result"] = "accept"
            plt.close(fig)

        def _skip(event=None):
            state["result"] = "skip"
            plt.close(fig)

        def _quit(event=None):
            state["result"] = "quit"
            plt.close(fig)

        def _on_key(event):
            if event.key in ("y", "enter"):
                # only accept if text box is empty
                if not tb.text.strip():
                    _accept()
            elif event.key == "s":
                _skip()
            elif event.key == "q":
                _quit()

        # Row 2: text box for corrections + buttons
        # We'll use a wider layout: [textbox][accept][skip][quit]
        ax_tb  = fig.add_subplot(gs[2, :n_frames//2 if n_frames > 1 else 0])
        ax_tb.axis("off")
        tb_ax  = plt.axes([0.08, 0.03, 0.35, 0.055])
        tb     = TextBox(tb_ax, "Correct frame: ", color="#0d0d1a", hovercolor="#1a1a2e",
                         label_pad=0.02)
        tb.label.set_color("white")
        tb.text_disp.set_color("#ffd54f")

        btn_ax_acc  = plt.axes([0.45, 0.03, 0.14, 0.055])
        btn_ax_skp  = plt.axes([0.61, 0.03, 0.14, 0.055])
        btn_ax_qut  = plt.axes([0.77, 0.03, 0.14, 0.055])

        btn_acc = Button(btn_ax_acc, "✓ Accept (y)", color="#1b5e20", hovercolor="#2e7d32")
        btn_skp = Button(btn_ax_skp, "→ Skip (s)",   color="#37474f", hovercolor="#546e7a")
        btn_qut = Button(btn_ax_qut, "✕ Quit (q)",   color="#b71c1c", hovercolor="#c62828")

        btn_acc.label.set_color("white")
        btn_skp.label.set_color("white")
        btn_qut.label.set_color("white")

        btn_acc.on_clicked(_accept)
        btn_skp.on_clicked(_skip)
        btn_qut.on_clicked(_quit)

        def _on_submit(text):
            text = text.strip()
            if text.isdigit():
                state["corrected_frame"] = int(text)
                state["result"] = "correct"
                plt.close(fig)

        tb.on_submit(_on_submit)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        plt.show()

        r = state["result"]
        if r == "accept":
            return True, det_frame
        elif r == "correct":
            return False, state["corrected_frame"]
        elif r == "skip":
            return "skip", det_frame
        else:   # quit or window closed without action
            return None, None


# ─────────────────────────────── re-extraction ────────────────────────────

def re_extract_corrected(results):
    """Re-run extract_training_data for videos whose frame was corrected."""
    corrected = [(k, v) for k, v in results.items() if v["status"] == "corrected"]
    if not corrected:
        print("\nNo corrections — nothing to re-extract.")
        return

    print(f"\n{'='*60}")
    print(f"  Re-extracting {len(corrected)} corrected video(s)")
    print(f"{'='*60}")

    for name, info in corrected:
        frame = info["frame"]
        view  = info.get("view", "side")
        subdir = BEHIND_DIR if view == "behind" else SIDE_DIR
        video_path = os.path.join(subdir, name)

        if not os.path.exists(video_path):
            print(f"  ✗ Not found: {video_path}")
            continue

        print(f"\n  → {name}: frame {info['original']} → {frame}")
        cmd = [
            sys.executable, "scripts/extract_training_data.py",
            "--video", video_path,
            "--verified-frames", str(frame),
            "--view", view,
            "--hand", "right",
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"    ✗ Extraction failed for {name}")
        else:
            print(f"    ✓ Done")

    print(f"\n{'='*60}")
    print("  Re-extraction complete.")
    print("  Next: python3 scripts/train_model.py")
    print(f"{'='*60}")


# ─────────────────────────────── entry point ──────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify detected clicker frames for all 28 videos")
    parser.add_argument("--re-extract", action="store_true",
                        help="Skip UI, just re-extract corrected videos from saved results")
    parser.add_argument("--summary", action="store_true",
                        help="Print current verification summary and exit")
    args = parser.parse_args()

    results = load_progress()

    if args.summary:
        _print_summary(results)
        return

    if args.re_extract:
        re_extract_corrected(results)
        return

    # Show existing progress
    if results:
        done = len(results)
        corr = sum(1 for v in results.values() if v["status"] == "corrected")
        print(f"Resuming: {done}/{len(DETECTED)} done, {corr} correction(s) so far")
        print(f"(Delete {RESULTS_FILE} to restart from scratch)\n")

    verifier = ClickerVerifier(DETECTED, results)
    verifier.run()

    _print_summary(results)

    # Offer to re-extract
    corr = [k for k, v in results.items() if v["status"] == "corrected"]
    if corr:
        ans = input(f"\n{len(corr)} correction(s) found. Re-extract training data now? [y/N] ").strip().lower()
        if ans == "y":
            re_extract_corrected(results)
            ans2 = input("\nRe-train model now? [y/N] ").strip().lower()
            if ans2 == "y":
                subprocess.run([sys.executable, "scripts/train_model.py"])


def _print_summary(results):
    if not results:
        print("No verifications recorded yet.")
        return
    print(f"\n{'='*60}")
    print(f"  Verification Summary")
    print(f"{'='*60}")
    print(f"  {'Video':<25} {'Status':<12} {'Frame':<8} {'Note'}")
    print(f"  {'-'*55}")
    for name, info in sorted(results.items()):
        note = f"was {info['original']}" if info["status"] == "corrected" else ""
        print(f"  {name:<25} {info['status']:<12} {info['frame']:<8} {note}")
    total    = len(DETECTED)
    done     = len(results)
    accepted = sum(1 for v in results.values() if v["status"] == "accepted")
    corrected= sum(1 for v in results.values() if v["status"] == "corrected")
    skipped  = sum(1 for v in results.values() if v["status"] == "skipped")
    print(f"\n  {done}/{total} reviewed  |  {accepted} accepted  |  {corrected} corrected  |  {skipped} skipped")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

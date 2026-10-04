"""
Active learner — logs uncertain frames and prompts for review at session end.

Uncertainty is defined as: 0.35 < model confidence for predicted class < 0.65,
specifically focusing on frames near predicted RELEASE events.

Usage:
    learner = ActiveLearner(confidence_low=0.35, confidence_high=0.65)

    # During analysis, feed uncertainty log from MLShotDetector:
    learner.ingest(detector.get_uncertainty_log(), fps=30.0)

    # After session, show review prompt:
    learner.prompt_review(video_path)
"""
import os
import sys
import json
from typing import List, Optional


class ActiveLearner:
    """
    Collects uncertain frames from MLShotDetector and presents a minimal
    review queue at end of session.

    Phase 2 of the bootstrapping strategy: instead of reviewing all 60 timeline
    frames (15 min), the archer reviews only ~10-15 uncertain frames (~5 min).
    """

    def __init__(
        self,
        confidence_low:  float = 0.35,
        confidence_high: float = 0.65,
        context_frames:  int   = 15,   # frames around each uncertain event to cluster
    ):
        self.confidence_low    = confidence_low
        self.confidence_high   = confidence_high
        self.context_frames    = context_frames
        self._uncertain_events: List[dict] = []

    def ingest(self, uncertainty_log: List[dict], fps: float = 30.0):
        """
        Ingest uncertainty log from MLShotDetector.get_uncertainty_log().
        Clusters nearby uncertain frames into events.
        """
        if not uncertainty_log:
            return

        # Filter to frames near RELEASE uncertainty
        release_uncertain = [
            e for e in uncertainty_log
            if e["proba"].get("RELEASE", 0) > self.confidence_low
        ]

        if not release_uncertain:
            return

        # Cluster by proximity
        events = []
        current_cluster = [release_uncertain[0]]

        for entry in release_uncertain[1:]:
            if entry["frame"] - current_cluster[-1]["frame"] <= self.context_frames:
                current_cluster.append(entry)
            else:
                events.append(current_cluster)
                current_cluster = [entry]
        events.append(current_cluster)

        for cluster in events:
            center_frame = cluster[len(cluster) // 2]["frame"]
            max_release_proba = max(e["proba"].get("RELEASE", 0) for e in cluster)
            self._uncertain_events.append({
                "center_frame":    center_frame,
                "n_frames":        len(cluster),
                "max_release_prob": max_release_proba,
                "fps":             fps,
                "cluster":         cluster,
            })

    def get_uncertain_events(self) -> List[dict]:
        return self._uncertain_events

    def n_uncertain(self) -> int:
        return len(self._uncertain_events)

    def prompt_review(
        self,
        video_path: Optional[str] = None,
        auto_accept_threshold: float = 0.90,
    ) -> List[int]:
        """
        Print review prompt and return list of frame indices needing verification.

        Frames with release_probability >= auto_accept_threshold are auto-accepted.

        Returns:
            List of frame indices the archer should verify.
        """
        if not self._uncertain_events:
            print("\n  [Active Learner] No uncertain frames — model is confident. 0 frames to review.")
            return []

        auto_accepted = []
        needs_review  = []

        for event in self._uncertain_events:
            if event["max_release_prob"] >= auto_accept_threshold:
                auto_accepted.append(event["center_frame"])
            else:
                needs_review.append(event["center_frame"])

        print("\n" + "=" * 60)
        print("  ACTIVE LEARNING REVIEW")
        print("=" * 60)
        print(f"  Uncertain events detected: {len(self._uncertain_events)}")
        print(f"  Auto-accepted (>= {auto_accept_threshold:.0%} confidence): {len(auto_accepted)}")
        print(f"  Needs manual review: {len(needs_review)}")

        if needs_review:
            print(f"\n  Review these frames in: {video_path or 'your video'}")
            for i, frame in enumerate(needs_review, 1):
                event = next(e for e in self._uncertain_events if e["center_frame"] == frame)
                t     = frame / event["fps"]
                print(f"    {i:2d}. Frame {frame:5d} ({t:5.1f}s)  "
                      f"release_prob={event['max_release_prob']:.2f}")
            print(f"\n  To add these as verified shots, run:")
            frames_str = " ".join(str(f) for f in needs_review)
            if video_path:
                print(f"    python3 scripts/extract_training_data.py "
                      f"--video {video_path} --verified-frames {frames_str}")
        else:
            print("  All events auto-accepted. No manual review needed.")

        print("=" * 60)
        return needs_review

    def save_review_queue(self, output_path: str):
        """Save uncertain events to JSON for offline review."""
        with open(output_path, "w") as f:
            json.dump({
                "n_events":  len(self._uncertain_events),
                "events":    self._uncertain_events,
            }, f, indent=2)
        print(f"  Review queue saved -> {output_path}")

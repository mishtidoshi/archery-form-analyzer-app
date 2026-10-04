"""
ML Shot Detector package for archery form analysis.

Provides a Random Forest-based shot detector trained on pose landmarks,
replacing the heuristic wrist-velocity threshold with a learned model.
"""
from .ml_shot_detector import MLShotDetector, ShotEvent
from .feature_extractor import FrameFeatures, extract_frame_features
from .window_builder import WindowBuilder

__all__ = ["MLShotDetector", "ShotEvent", "FrameFeatures", "extract_frame_features", "WindowBuilder"]

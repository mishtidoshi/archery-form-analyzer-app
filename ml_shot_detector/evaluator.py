"""
Leave-One-Shot-Out / Leave-One-Session-Out cross-validation evaluator.

Measures RELEASE recall and precision as the primary metrics.
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import List, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_recall_fscore_support
    )
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

from .labeler import LABEL_TO_INT, INT_TO_LABEL
from .trainer import load_all_training_data
from .window_builder import WindowBuilder


def run_loso_cv(
    training_dir: str = "training_data",
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Session-Out cross-validation (or Leave-One-Shot-Out if only 1 session).

    Returns dict with per-fold and aggregate metrics.
    """
    X, y, session_ids = load_all_training_data(training_dir)

    unique_sessions = list(dict.fromkeys(session_ids))  # preserve order
    n_sessions      = len(unique_sessions)

    print(f"\n  {'='*60}")
    print(f"  LOSO Cross-Validation  ({n_sessions} session(s))")
    print(f"  {'='*60}")

    if n_sessions == 1:
        print("  Only 1 session — using Leave-One-Shot-Out (shot-level folds)")
        return _loso_single_session(X, y, session_ids, unique_sessions[0], verbose)

    all_y_true, all_y_pred = [], []
    fold_results = []

    for held_session in unique_sessions:
        train_mask = [s != held_session for s in session_ids]
        test_mask  = [s == held_session for s in session_ids]

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test  = X[test_mask]
        y_test  = y[test_mask]

        if len(np.unique(y_train)) < 2:
            print(f"  Skipping fold {held_session}: insufficient class diversity in training set")
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # RELEASE metrics
        release_idx = LABEL_TO_INT["RELEASE"]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[release_idx], zero_division=0
        )
        fold_results.append({
            "held_session": held_session,
            "n_test":       int(len(y_test)),
            "release_precision": float(prec[0]),
            "release_recall":    float(rec[0]),
            "release_f1":        float(f1[0]),
        })

        if verbose:
            print(f"\n  Fold: hold-out = '{held_session}' ({len(y_test)} test samples)")
            print(f"    RELEASE — Precision: {prec[0]:.3f}  Recall: {rec[0]:.3f}  F1: {f1[0]:.3f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print(f"\n  {'='*60}")
    print(f"  AGGREGATE RESULTS (all folds combined)")
    print(f"  {'='*60}")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=[INT_TO_LABEL[i] for i in range(5)],
        zero_division=0,
    ))

    release_idx = LABEL_TO_INT["RELEASE"]
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=[release_idx], zero_division=0
    )
    cm = confusion_matrix(all_y_true, all_y_pred)

    results = {
        "n_folds":           len(fold_results),
        "fold_results":      fold_results,
        "aggregate": {
            "release_precision": float(prec[0]),
            "release_recall":    float(rec[0]),
            "release_f1":        float(f1[0]),
        },
        "confusion_matrix": cm.tolist(),
        "label_names":      [INT_TO_LABEL[i] for i in range(5)],
    }

    print(f"\n  KEY METRIC — RELEASE class:")
    print(f"    Precision: {prec[0]:.3f}  (target > 0.70)")
    print(f"    Recall:    {rec[0]:.3f}  (target > 0.85)")
    print(f"    F1:        {f1[0]:.3f}")

    return results


def _loso_single_session(X, y, session_ids, session_id, verbose):
    """Leave-One-Shot-Out when only 1 session available."""
    # Use frame order to approximate shot boundaries
    # Each RELEASE sample is treated as one "shot" fold
    release_idx_label = LABEL_TO_INT["RELEASE"]
    release_positions = np.where(y == release_idx_label)[0]

    if len(release_positions) < 2:
        print("  Not enough shots for LOSO — training and evaluating on full dataset")
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, y_pred, labels=[release_idx_label], zero_division=0
        )
        print(f"  RELEASE — Precision: {prec[0]:.3f}  Recall: {rec[0]:.3f}  F1: {f1[0]:.3f}")
        return {
            "n_folds": 1,
            "aggregate": {
                "release_precision": float(prec[0]),
                "release_recall":    float(rec[0]),
                "release_f1":        float(f1[0]),
            }
        }

    # Split into N folds based on shot boundaries
    n_shots = len(release_positions)
    all_y_true, all_y_pred = [], []
    fold_results = []

    for shot_i, release_pos in enumerate(release_positions):
        # For each fold: hold out ~60 frames around this shot's release
        held_mask = np.zeros(len(y), dtype=bool)
        held_mask[max(0, release_pos - 30): min(len(y), release_pos + 31)] = True

        X_train = X[~held_mask]
        y_train = y[~held_mask]
        X_test  = X[held_mask]
        y_test  = y[held_mask]

        if len(np.unique(y_train)) < 2 or len(X_train) < 10:
            continue

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[release_idx_label], zero_division=0
        )
        fold_results.append({
            "shot_fold": shot_i + 1,
            "release_precision": float(prec[0]),
            "release_recall":    float(rec[0]),
            "release_f1":        float(f1[0]),
        })

        if verbose:
            print(f"  Fold {shot_i + 1}/{n_shots}: RELEASE P={prec[0]:.2f}  R={rec[0]:.2f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print(f"\n  {'='*60}")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=[INT_TO_LABEL[i] for i in range(5)],
        zero_division=0,
    ))

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=[release_idx_label], zero_division=0
    )
    cm = confusion_matrix(all_y_true, all_y_pred)

    print(f"\n  KEY METRIC — RELEASE:")
    print(f"    Precision: {prec[0]:.3f}  Recall: {rec[0]:.3f}  F1: {f1[0]:.3f}")

    return {
        "n_folds":      len(fold_results),
        "fold_results": fold_results,
        "aggregate": {
            "release_precision": float(prec[0]),
            "release_recall":    float(rec[0]),
            "release_f1":        float(f1[0]),
        },
        "confusion_matrix": cm.tolist(),
        "label_names": [INT_TO_LABEL[i] for i in range(5)],
    }

"""
Random Forest trainer for ML shot detection.

Loads all training JSONs from training_data/, trains a Random Forest,
and saves the model + metadata to models/.
"""
import json
import os
import sys
import pickle
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)

from .labeler import LABEL_TO_INT, INT_TO_LABEL
from .window_builder import WindowBuilder


def load_all_training_data(
    training_dir: str = "training_data",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all training JSONs and return (X, y, session_ids).

    Returns:
        X: (N, 32) feature matrix
        y: (N,) integer label vector
        session_ids: list of session_id strings (one per sample, for LOSO CV)
    """
    json_files = [
        f for f in os.listdir(training_dir)
        if f.endswith(".json") and f != "training_manifest.json"
    ]

    if not json_files:
        raise ValueError(f"No training JSON files found in {training_dir}/")

    all_X, all_y, all_sessions = [], [], []

    for jf in sorted(json_files):
        path = os.path.join(training_dir, jf)
        with open(path) as f:
            data = json.load(f)

        session_id = data.get("session_id", jf)
        samples    = data["samples"]

        for s in samples:
            all_X.append(s["features"])
            all_y.append(s["label_int"])
            all_sessions.append(session_id)

        label_counts = data.get("sample_counts", {})
        print(f"  Loaded {len(samples):4d} samples from {jf}  {label_counts}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)

    return X, y, all_sessions


def train_model(
    training_dir: str = "training_data",
    models_dir:   str = "models",
    model_name:   str = "rf_current",
) -> str:
    """
    Train Random Forest on all training data and save to models/.

    Returns path to saved model file.
    """
    print("\n" + "=" * 60)
    print("  TRAINING ML SHOT DETECTOR")
    print("=" * 60)

    X, y, session_ids = load_all_training_data(training_dir)

    n_samples  = len(y)
    label_dist = {INT_TO_LABEL[i]: int((y == i).sum()) for i in range(5)}
    print(f"\n  Total samples: {n_samples}")
    print(f"  Class distribution: {label_dist}")
    print(f"  Feature vector size: {X.shape[1]}")

    # Train
    print("\n  Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 12,
        min_samples_leaf = 3,
        class_weight  = "balanced",   # handles RELEASE scarcity
        random_state  = 42,
        n_jobs        = -1,
    )
    clf.fit(X, y)

    # Feature importances
    feat_names = WindowBuilder.feature_names()
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]

    print("\n  Top 10 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"    {rank:2d}. {name:<40} {importances[idx]:.4f}")

    # Training accuracy (in-sample)
    y_pred_train = clf.predict(X)
    from sklearn.metrics import classification_report
    print("\n  Training set classification report:")
    print(classification_report(
        y, y_pred_train,
        target_names=[INT_TO_LABEL[i] for i in range(5)],
        zero_division=0,
    ))

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f, protocol=4)

    # Save metadata
    meta = {
        "trained_at":       datetime.now().isoformat(),
        "training_dir":     os.path.abspath(training_dir),
        "n_samples":        n_samples,
        "class_distribution": label_dist,
        "label_map":        LABEL_TO_INT,
        "feature_names":    feat_names,
        "model_params":     clf.get_params(),
        "top_features":     [
            {"rank": i + 1, "name": feat_names[idx] if idx < len(feat_names) else f"feat_{idx}",
             "importance": float(importances[idx])}
            for i, idx in enumerate(top_idx)
        ],
    }
    meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved  -> {model_path}")
    print(f"  Metadata     -> {meta_path}")
    return model_path

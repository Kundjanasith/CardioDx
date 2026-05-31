from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from cardiotwin.constants import PTBXL_SUPERCLASSES


def make_baseline_model(model_type: str = "logreg", random_state: int = 42):
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=random_state)
        return clf
    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(base)),
    ])


def train_baseline(X: np.ndarray, y: np.ndarray, model_type: str = "logreg", random_state: int = 42):
    model = make_baseline_model(model_type=model_type, random_state=random_state)
    model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # OneVsRest inside pipeline returns ndarray; RF multilabel may return list
        if isinstance(probs, list):
            probs = np.vstack([p[:, 1] for p in probs]).T
        return np.asarray(probs, dtype=np.float32)
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def save_model(model, out_path: str | Path, feature_names: list[str] | None = None, label_names: list[str] | None = None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_names": label_names or PTBXL_SUPERCLASSES,
        "type": "baseline_ml",
    }
    joblib.dump(bundle, out_path)


def load_model(path: str | Path):
    bundle = joblib.load(path)
    return bundle

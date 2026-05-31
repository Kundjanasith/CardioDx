from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)

def fit_isotonic_per_class(y_true: np.ndarray, y_prob: np.ndarray):
    calibrators = []
    y_cal = np.zeros_like(y_prob, dtype=np.float32)
    for j in range(y_prob.shape[1]):
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(y_prob[:, j], y_true[:, j])
        y_cal[:, j] = cal.predict(y_prob[:, j])
        calibrators.append(cal)
    return calibrators, y_cal

def apply_isotonic(calibrators, y_prob: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y_prob, dtype=np.float32)
    for j in range(y_prob.shape[1]):
        out[:, j] = calibrators[j].predict(y_prob[:, j]) if j < len(calibrators) else y_prob[:, j]
    return np.clip(out, 0, 1)

def tune_threshold_profiles(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    profiles = {"screening": {}, "balanced": {}, "high_specificity": {}}
    rows = []
    grid = np.round(np.arange(0.03, 0.96, 0.01), 2)
    for j, label in enumerate(labels):
        yt = y_true[:, j]
        best = {"screening": (-1, 0.5), "balanced": (-1, 0.5), "high_specificity": (-1, 0.5)}
        for th in grid:
            yp = (y_prob[:, j] >= th).astype(int)
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            # Screening prioritizes sensitivity while avoiding useless precision.
            screening_score = 0.75 * rec + 0.25 * f1
            # Balanced prioritizes F1.
            balanced_score = f1
            # High-specificity prioritizes precision but not at zero recall.
            high_spec_score = 0.75 * prec + 0.25 * f1 if rec >= 0.15 else -1
            candidates = {
                "screening": screening_score,
                "balanced": balanced_score,
                "high_specificity": high_spec_score,
            }
            for profile, score in candidates.items():
                if score > best[profile][0]:
                    best[profile] = (float(score), float(th))
        for profile in profiles:
            th = best[profile][1]
            profiles[profile][label] = th
            yp = (y_prob[:, j] >= th).astype(int)
            rows.append({
                "profile": profile,
                "label": label,
                "threshold": th,
                "f1": f1_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
            })
    return profiles, pd.DataFrame(rows)

def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], n_bins: int = 10) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0, 1, n_bins + 1)
    for j, label in enumerate(labels):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (yp >= lo) & (yp < hi) if hi < 1 else (yp >= lo) & (yp <= hi)
            rows.append({
                "label": label,
                "bin": b,
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "confidence_mean": float(yp[mask].mean()) if mask.sum() else np.nan,
                "empirical_positive_rate": float(yt[mask].mean()) if mask.sum() else np.nan,
            })
    return pd.DataFrame(rows)

def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows=[]
    for j,label in enumerate(labels):
        rows.append({
            "label": label,
            "brier": brier_score_loss(y_true[:,j], y_prob[:,j]),
            "ece": expected_calibration_error(y_true[:,j], y_prob[:,j]),
            "support": int(y_true[:,j].sum()),
        })
    return pd.DataFrame(rows)

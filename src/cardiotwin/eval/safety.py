from __future__ import annotations
import numpy as np


def safety_metrics(sqi_values: list[float], y_prob, y_true=None, low_sqi_threshold: float = 0.55,
                   uncertain_prob_margin: float = 0.15, high_conf_threshold: float = 0.85) -> dict:
    sqi = np.asarray(sqi_values, dtype=float)
    probs = np.asarray(y_prob, dtype=float)
    low_sqi_rate = float(np.mean(sqi < low_sqi_threshold)) if len(sqi) else float("nan")
    # uncertain if max abnormal probability is not clearly above/below threshold
    if probs.ndim == 2:
        maxp = probs.max(axis=1)
        uncertain = np.abs(maxp - 0.5) < uncertain_prob_margin
        uncertain_rate = float(np.mean(uncertain))
    else:
        uncertain_rate = float("nan")
    false_confident = float("nan")
    if y_true is not None and probs.ndim == 2:
        yp = (probs >= 0.5).astype(int)
        wrong_any = np.any(yp != np.asarray(y_true).astype(int), axis=1)
        high_conf = np.max(np.maximum(probs, 1 - probs), axis=1) >= high_conf_threshold
        false_confident = float(np.mean(wrong_any & high_conf))
    return {
        "low_sqi_rejection_rate": low_sqi_rate,
        "uncertain_case_flag_rate": uncertain_rate,
        "false_confident_prediction_rate": false_confident,
    }

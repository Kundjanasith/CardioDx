from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

CRITICAL_LABELS = {"MI", "STTC", "CD"}

def threshold_predict(y_prob: np.ndarray, thresholds: dict, labels: list[str], profile: str = "balanced") -> np.ndarray:
    prof = thresholds.get(profile, thresholds)
    y_pred = np.zeros_like(y_prob, dtype=int)
    for j, label in enumerate(labels):
        th = float(prof.get(label, 0.5))
        y_pred[:, j] = (y_prob[:, j] >= th).astype(int)
    return y_pred

def safety_decisions(
    y_prob: np.ndarray,
    sqi: np.ndarray | None,
    thresholds: dict,
    labels: list[str],
    profile: str = "balanced",
    low_sqi_threshold: float = 0.55,
    caution_sqi_threshold: float = 0.70,
    uncertainty_margin: float = 0.08,
    high_conf_threshold: float = 0.85,
) -> list[dict]:
    y_prob = np.asarray(y_prob)
    if sqi is None:
        sqi = np.ones(len(y_prob), dtype=float)
    sqi = np.asarray(sqi, dtype=float)
    y_pred = threshold_predict(y_prob, thresholds, labels, profile=profile)
    out = []
    for i in range(len(y_prob)):
        sorted_prob = np.sort(y_prob[i])
        margin = sorted_prob[-1] - sorted_prob[-2] if y_prob.shape[1] >= 2 else 1.0
        max_prob = float(y_prob[i].max())
        positive_labels = [labels[j] for j in np.where(y_pred[i] == 1)[0]]
        if sqi[i] < low_sqi_threshold:
            status = "REJECT_LOW_SQI"
            conf = "do_not_interpret"
            reason = "Signal quality below safe interpretation threshold."
        elif margin < uncertainty_margin and max_prob < high_conf_threshold:
            status = "ABSTAIN_UNCERTAIN"
            conf = "uncertain"
            reason = "Low separation between top probabilities; requires expert review."
        elif not positive_labels and max_prob < 0.35:
            status = "ABSTAIN_LOW_EVIDENCE"
            conf = "uncertain"
            reason = "No class exceeds decision threshold with sufficient evidence."
        elif max_prob >= high_conf_threshold:
            status = "INTERPRET_WITH_CAUTION" if sqi[i] < caution_sqi_threshold else "INTERPRET"
            conf = "high"
            reason = "High model confidence; still research-use only."
        else:
            status = "INTERPRET_WITH_CAUTION"
            conf = "medium"
            reason = "Moderate confidence or signal quality caution; requires review."
        out.append({
            "status": status,
            "confidence_level": conf,
            "reason": reason,
            "max_probability": max_prob,
            "probability_margin": float(margin),
            "sqi": float(sqi[i]),
            "positive_labels": positive_labels,
        })
    return out

def safety_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], thresholds: dict, sqi: np.ndarray | None = None, profile: str = "balanced") -> tuple[dict, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = threshold_predict(y_prob, thresholds, labels, profile=profile)
    decisions = safety_decisions(y_prob, sqi, thresholds, labels, profile=profile)
    abstain = np.array([d["status"].startswith("ABSTAIN") or d["status"].startswith("REJECT") for d in decisions])
    high = np.array([d["confidence_level"] == "high" for d in decisions])
    error_any = np.any(y_pred != y_true, axis=1)
    false_confident = high & error_any & (~abstain)

    rows=[]
    critical_indices = [labels.index(x) for x in labels if x in CRITICAL_LABELS]
    critical_miss = np.zeros(len(y_true), dtype=bool)
    for j,label in enumerate(labels):
        yt = y_true[:,j]
        yp = y_pred[:,j]
        high_fp = (high & (~abstain) & (yp == 1) & (yt == 0))
        high_fn = (high & (~abstain) & (yp == 0) & (yt == 1))
        rows.append({
            "label": label,
            "profile": profile,
            "support": int(yt.sum()),
            "f1_non_abstained": f1_score(yt[~abstain], yp[~abstain], zero_division=0) if (~abstain).sum() else 0.0,
            "precision_non_abstained": precision_score(yt[~abstain], yp[~abstain], zero_division=0) if (~abstain).sum() else 0.0,
            "recall_non_abstained": recall_score(yt[~abstain], yp[~abstain], zero_division=0) if (~abstain).sum() else 0.0,
            "high_confidence_false_positive_rate": float(high_fp.mean()),
            "high_confidence_false_negative_rate": float(high_fn.mean()),
        })
        if j in critical_indices:
            critical_miss |= ((yt == 1) & (yp == 0) & (~abstain))
    overall = {
        "profile": profile,
        "abstain_or_reject_rate": float(abstain.mean()),
        "interpreted_coverage": float((~abstain).mean()),
        "false_confident_prediction_rate_v12": float(false_confident.mean()),
        "critical_miss_rate_non_abstained": float(critical_miss.mean()),
        "low_sqi_rejection_rate": float(np.mean([d["status"] == "REJECT_LOW_SQI" for d in decisions])),
        "mean_max_probability": float(np.max(y_prob, axis=1).mean()),
    }
    return overall, pd.DataFrame(rows)

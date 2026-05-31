from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, confusion_matrix)


def _safe_metric(fn, default=np.nan, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, label_names=None) -> tuple[dict, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    n_classes = y_true.shape[1]
    if label_names is None:
        label_names = [f"class_{i}" for i in range(n_classes)]
    overall = {
        "auroc_macro": float(_safe_metric(roc_auc_score, np.nan, y_true, y_prob, average="macro")),
        "auprc_macro": float(_safe_metric(average_precision_score, np.nan, y_true, y_prob, average="macro")),
        "macro_f1": float(_safe_metric(f1_score, np.nan, y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(_safe_metric(precision_score, np.nan, y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall_sensitivity": float(_safe_metric(recall_score, np.nan, y_true, y_pred, average="macro", zero_division=0)),
    }
    rows = []
    bal_accs = []
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prob = y_prob[:, i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        bal = (sensitivity + specificity) / 2
        bal_accs.append(bal)
        rows.append({
            "label": name,
            "support": int(yt.sum()),
            "auroc": float(_safe_metric(roc_auc_score, np.nan, yt, prob)),
            "auprc": float(_safe_metric(average_precision_score, np.nan, yt, prob)),
            "f1": float(_safe_metric(f1_score, np.nan, yt, yp, zero_division=0)),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "balanced_accuracy": float(bal),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    overall["balanced_accuracy_macro"] = float(np.nanmean(bal_accs))
    return overall, pd.DataFrame(rows)


def save_metrics(overall: dict, per_class: pd.DataFrame, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    per_class.to_csv(out_dir / "metrics_per_class.csv", index=False)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, brier_score_loss, precision_score, recall_score

from cardiotwin.constants import PTBXL_SUPERCLASSES
from cardiotwin.models.baseline_ml import load_model, predict_proba


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def fit_isotonic_calibrators(y_true, y_prob):
    calibrators = []
    y_cal = np.zeros_like(y_prob, dtype=float)

    for j in range(y_prob.shape[1]):
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(y_prob[:, j], y_true[:, j])
        y_cal[:, j] = cal.predict(y_prob[:, j])
        calibrators.append(cal)

    return calibrators, y_cal


def tune_thresholds(y_true, y_prob, labels):
    thresholds = {}
    rows = []

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    for j, label in enumerate(labels):
        best = {
            "threshold": 0.5,
            "f1": -1.0,
            "precision": 0.0,
            "recall": 0.0,
        }

        for th in grid:
            pred = (y_prob[:, j] >= th).astype(int)
            f1 = f1_score(y_true[:, j], pred, zero_division=0)
            prec = precision_score(y_true[:, j], pred, zero_division=0)
            rec = recall_score(y_true[:, j], pred, zero_division=0)

            if f1 > best["f1"]:
                best = {
                    "threshold": float(th),
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                }

        thresholds[label] = best["threshold"]
        rows.append({
            "label": label,
            **best,
        })

    return thresholds, pd.DataFrame(rows)


def evaluate_with_thresholds(y_true, y_prob, thresholds, labels):
    pred = np.zeros_like(y_true, dtype=int)

    for j, label in enumerate(labels):
        pred[:, j] = (y_prob[:, j] >= thresholds[label]).astype(int)

    rows = []
    for j, label in enumerate(labels):
        yt = y_true[:, j]
        yp = pred[:, j]
        pp = y_prob[:, j]

        rows.append({
            "label": label,
            "threshold": thresholds[label],
            "support": int(yt.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall_sensitivity": float(recall_score(yt, yp, zero_division=0)),
            "brier": float(brier_score_loss(yt, pp)),
            "ece": expected_calibration_error(yt, pp),
        })

    wrong_any = np.any(pred != y_true, axis=1)
    max_conf = np.max(np.maximum(y_prob, 1.0 - y_prob), axis=1)
    false_confident = float(np.mean(wrong_any & (max_conf >= 0.85)))

    margin_sorted = np.sort(y_prob, axis=1)
    if y_prob.shape[1] >= 2:
        margin = margin_sorted[:, -1] - margin_sorted[:, -2]
    else:
        margin = np.ones(len(y_prob))

    uncertain = (np.max(y_prob, axis=1) < 0.35) | (margin < 0.08)

    overall = {
        "macro_f1_thresholded": float(np.mean([r["f1"] for r in rows])),
        "macro_precision_thresholded": float(np.mean([r["precision"] for r in rows])),
        "macro_recall_thresholded": float(np.mean([r["recall_sensitivity"] for r in rows])),
        "mean_brier": float(np.mean([r["brier"] for r in rows])),
        "mean_ece": float(np.mean([r["ece"] for r in rows])),
        "uncertain_case_flag_rate_v11": float(np.mean(uncertain)),
        "false_confident_prediction_rate_v11": false_confident,
    }

    return overall, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="artifacts/processed")
    ap.add_argument("--model-path", default="artifacts/models/baseline_model.joblib")
    ap.add_argument("--out-model", default="artifacts/models/baseline_model_v11_calibrated.joblib")
    ap.add_argument("--out-dir", default="artifacts/calibration_v11")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(processed / "X_features.npy")
    Y = np.load(processed / "Y_labels.npy")
    idx = pd.read_csv(processed / "records_index.csv")

    val_mask = idx["split"].values == "val"
    test_mask = idx["split"].values == "test"

    if val_mask.sum() == 0:
        raise RuntimeError("No validation split found. Cannot calibrate thresholds.")

    bundle = load_model(args.model_path)
    model = bundle["model"]
    labels = bundle.get("label_names", PTBXL_SUPERCLASSES)

    raw_val = predict_proba(model, X[val_mask])
    raw_test = predict_proba(model, X[test_mask])

    calibrators, cal_val = fit_isotonic_calibrators(Y[val_mask], raw_val)

    cal_test = np.zeros_like(raw_test, dtype=float)
    for j, cal in enumerate(calibrators):
        cal_test[:, j] = cal.predict(raw_test[:, j])

    thresholds, threshold_df = tune_thresholds(Y[val_mask], cal_val, labels)

    overall, per_class = evaluate_with_thresholds(
        Y[test_mask],
        cal_test,
        thresholds,
        labels,
    )

    bundle["calibration"] = {
        "method": "isotonic_per_class",
        "calibrators": calibrators,
    }
    bundle["thresholds"] = thresholds
    bundle["v11_safety_gate"] = {
        "low_sqi_threshold": 0.55,
        "uncertain_max_prob_threshold": 0.35,
        "uncertain_margin_threshold": 0.08,
        "high_confidence_threshold": 0.85,
        "medium_confidence_threshold": 0.65,
        "clinical_boundary": "Research-use preliminary screening and visual explanation only; not final diagnosis.",
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_model)

    threshold_df.to_csv(out_dir / "thresholds_per_class.csv", index=False)
    per_class.to_csv(out_dir / "metrics_per_class_v11.csv", index=False)
    (out_dir / "metrics_overall_v11.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(overall, indent=2))
    print(f"Saved calibrated model: {out_model}")
    print(f"Saved v1.1 calibration outputs: {out_dir}")


if __name__ == "__main__":
    main()

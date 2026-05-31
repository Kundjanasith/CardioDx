from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression

from cardiotwin.constants import PTBXL_SUPERCLASSES


def ece_binary(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def safe_auc(y, p, kind="roc"):
    try:
        if kind == "roc":
            return float(roc_auc_score(y, p))
        return float(average_precision_score(y, p))
    except Exception:
        return None


def fit_isotonic(y_cal, p_cal, p_eval):
    p_out = np.zeros_like(p_eval, dtype=float)
    calibrators = []
    for j in range(p_cal.shape[1]):
        # If one class only, leave as identity.
        if len(np.unique(y_cal[:, j])) < 2:
            p_out[:, j] = p_eval[:, j]
            calibrators.append(None)
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_cal[:, j], y_cal[:, j])
        p_out[:, j] = iso.predict(p_eval[:, j])
        calibrators.append(iso)
    return calibrators, p_out


def tune_threshold_for_objective(y, p, label, profile):
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)

    best = {
        "threshold": 0.5,
        "score": -1,
        "f1": 0,
        "precision": 0,
        "sensitivity": 0,
    }

    for th in grid:
        pred = (p >= th).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)

        if profile == "screening":
            # prioritize sensitivity; F2-like utility
            beta2 = 4
            score = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)
            if label in {"MI", "STTC", "CD"}:
                score += 0.15 * rec
        elif profile == "balanced":
            score = f1
        elif profile == "high_specificity":
            # precision-oriented, but avoid zero recall.
            score = 0.7 * prec + 0.3 * f1
            if rec < 0.10 and y.sum() >= 20:
                score -= 0.5
        elif profile == "hyp_focus":
            if label == "HYP":
                beta2 = 4
                score = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12) + 0.2 * rec
            else:
                score = f1
        else:
            score = f1

        if score > best["score"]:
            best = {
                "threshold": float(th),
                "score": float(score),
                "f1": float(f1),
                "precision": float(prec),
                "sensitivity": float(rec),
            }

    return best


def tune_profiles(y_cal, p_cal):
    profiles = {}
    for profile in ["screening", "balanced", "high_specificity", "hyp_focus"]:
        per_class = {}
        for j, label in enumerate(PTBXL_SUPERCLASSES):
            if y_cal[:, j].sum() < 5:
                per_class[label] = {
                    "threshold": 0.5,
                    "score": None,
                    "f1": None,
                    "precision": None,
                    "sensitivity": None,
                    "note": "insufficient calibration positives; default threshold retained",
                }
            else:
                per_class[label] = tune_threshold_for_objective(
                    y_cal[:, j],
                    p_cal[:, j],
                    label,
                    profile,
                )
        profiles[profile] = per_class
    return profiles


def evaluate_profile(y, p, thresholds, profile_name):
    pred = np.zeros_like(y, dtype=int)
    rows = []

    for j, label in enumerate(PTBXL_SUPERCLASSES):
        th = float(thresholds[label]["threshold"])
        pred[:, j] = (p[:, j] >= th).astype(int)

    for j, label in enumerate(PTBXL_SUPERCLASSES):
        yt = y[:, j]
        yp = pred[:, j]
        pp = p[:, j]
        rows.append({
            "profile": profile_name,
            "label": label,
            "support": int(yt.sum()),
            "threshold": float(thresholds[label]["threshold"]),
            "auroc": safe_auc(yt, pp, "roc"),
            "auprc": safe_auc(yt, pp, "pr"),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "sensitivity": float(recall_score(yt, yp, zero_division=0)),
            "brier": float(brier_score_loss(yt, pp)) if len(np.unique(yt)) > 1 else None,
            "ece": ece_binary(yt, pp) if len(np.unique(yt)) > 1 else None,
        })

    df = pd.DataFrame(rows)
    valid = df[df["support"] >= 20].copy()

    # Safety metrics
    max_prob = p.max(axis=1)
    sorted_prob = np.sort(p, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2] if p.shape[1] >= 2 else np.ones(len(p))
    abstain = (max_prob < 0.35) | (margin < 0.05)

    # high-confidence false positive/negative
    high_conf_pos = p >= 0.85
    high_conf_neg = p <= 0.05
    y_bool = y.astype(bool)

    high_conf_fp = high_conf_pos & (~y_bool)
    high_conf_fn = high_conf_neg & y_bool

    critical_idx = [PTBXL_SUPERCLASSES.index(x) for x in ["MI", "STTC", "CD"]]
    critical_miss = (pred[:, critical_idx] == 0) & (y[:, critical_idx] == 1)
    critical_positive_rows = (y[:, critical_idx].sum(axis=1) > 0)

    if critical_positive_rows.sum() > 0:
        critical_miss_rate = float((critical_miss.any(axis=1) & critical_positive_rows).sum() / critical_positive_rows.sum())
    else:
        critical_miss_rate = None

    overall = {
        "profile": profile_name,
        "valid_labels": valid["label"].tolist(),
        "excluded_labels_support_lt_20": df[df["support"] < 20]["label"].tolist(),
        "macro_auroc_valid": float(valid["auroc"].dropna().mean()),
        "macro_auprc_valid": float(valid["auprc"].dropna().mean()),
        "macro_f1_valid": float(valid["f1"].mean()),
        "macro_precision_valid": float(valid["precision"].mean()),
        "macro_sensitivity_valid": float(valid["sensitivity"].mean()),
        "mean_brier_valid": float(valid["brier"].dropna().mean()),
        "mean_ece_valid": float(valid["ece"].dropna().mean()),
        "abstain_rate_proxy": float(abstain.mean()),
        "high_conf_false_positive_rate": float(high_conf_fp.sum() / max(high_conf_pos.sum(), 1)),
        "high_conf_false_negative_rate": float(high_conf_fn.sum() / max(high_conf_neg.sum(), 1)),
        "critical_miss_rate": critical_miss_rate,
    }

    return overall, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-path", default="artifacts/external_validation/georgia_deep_inceptiontime_v21/P_georgia_deep.npy")
    ap.add_argument("--label-path", default="artifacts/external_validation/georgia_deep_inceptiontime_v21/Y_georgia_deep.npy")
    ap.add_argument("--model-path", default="artifacts/deep_models/inceptiontime_model.pt")
    ap.add_argument("--out-dir", default="artifacts/deep_safety_v21")
    ap.add_argument("--out-model", default="artifacts/models/inceptiontime_v21_safety.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)

    P = np.load(args.pred_path)
    Y = np.load(args.label_path).astype(int)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(Y))
    rng.shuffle(idx)

    split = int(len(idx) * 0.5)
    cal_idx = idx[:split]
    eval_idx = idx[split:]

    Y_cal, P_cal_raw = Y[cal_idx], P[cal_idx]
    Y_eval, P_eval_raw = Y[eval_idx], P[eval_idx]

    calibrators, P_eval_cal = fit_isotonic(Y_cal, P_cal_raw, P_eval_raw)
    _, P_cal_cal = fit_isotonic(Y_cal, P_cal_raw, P_cal_raw)

    profiles = tune_profiles(Y_cal, P_cal_cal)

    all_overall = {}
    all_rows = []

    for profile_name, thresholds in profiles.items():
        overall, df = evaluate_profile(Y_eval, P_eval_cal, thresholds, profile_name)
        all_overall[profile_name] = overall
        all_rows.append(df)

    per_class_all = pd.concat(all_rows, ignore_index=True)
    per_class_all.to_csv(out_dir / "metrics_deep_safety_per_class_v21.csv", index=False)

    # Select recommended profile: screening for clinical screening default
    recommended = "screening"

    metrics = {
        "model_path": args.model_path,
        "pred_path": args.pred_path,
        "label_path": args.label_path,
        "calibration_method": "isotonic_per_class_split_from_georgia_external_predictions",
        "n_total": int(len(Y)),
        "n_calibration": int(len(cal_idx)),
        "n_evaluation": int(len(eval_idx)),
        "profiles": all_overall,
        "recommended_default_profile": recommended,
        "claim_boundary": "Safety thresholds are derived from Georgia external predictions and should be treated as research-stage external calibration, not clinical deployment thresholds.",
    }

    (out_dir / "metrics_deep_safety_v21.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (out_dir / "threshold_profiles_deep.json").write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Save safety wrapper model
    ckpt = torch.load(args.model_path, map_location="cpu")
    ckpt["safety_v21"] = {
        "threshold_profiles": profiles,
        "recommended_default_profile": recommended,
        "metrics": metrics,
        "label_names": PTBXL_SUPERCLASSES,
        "calibration_note": "Calibration objects are not serialized here; use thresholds and safety profiles for deployment-like dashboard gating.",
    }
    torch.save(ckpt, args.out_model)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", out_dir / "metrics_deep_safety_v21.json")
    print("Saved:", out_dir / "threshold_profiles_deep.json")
    print("Saved:", args.out_model)


if __name__ == "__main__":
    main()

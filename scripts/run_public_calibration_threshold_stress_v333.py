from pathlib import Path
import json
import math
import zipfile
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

OUT = Path("artifacts/public_multicenter_validation_v33")
RELEASE = Path("artifacts/release_rc1")
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT / "public_locked_inference_predictions_v331_full.csv"

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
GROUPS = ["cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ALL_SOURCES_STACKED_REFERENCE_ONLY"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def calc_binary_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0

    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "predicted_positives": int(y_pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        valid = np.isfinite(y_prob)
        if valid.sum() > 0:
            yt = y_true[valid]
            yp = y_prob[valid]
            if len(set(yt.tolist())) > 1:
                try:
                    out["auroc"] = float(roc_auc_score(yt, yp))
                except Exception:
                    out["auroc"] = None
            else:
                out["auroc"] = None

            if yt.sum() > 0:
                try:
                    out["auprc"] = float(average_precision_score(yt, yp))
                except Exception:
                    out["auprc"] = None
            else:
                out["auprc"] = None
        else:
            out["auroc"] = None
            out["auprc"] = None

    return out


def threshold_grid(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs)]

    base = np.linspace(0.01, 0.99, 99)
    qs = np.quantile(probs, np.linspace(0.01, 0.99, 99)) if len(probs) else np.array([])
    all_t = np.unique(np.concatenate([base, qs, np.array([0.05, 0.10, 0.13, 0.15, 0.30, 0.50, 0.70, 0.90])]))
    all_t = all_t[(all_t >= 0.0) & (all_t <= 1.0)]
    return sorted(float(x) for x in all_t)


def pick_thresholds(curve_rows):
    valid = [r for r in curve_rows if r["positives"] > 0 and r["negatives"] > 0]

    def best_by(metric):
        vals = [r for r in valid if r.get(metric) is not None]
        if not vals:
            return None
        return max(vals, key=lambda r: r[metric])

    best_f1 = best_by("f1")

    youden_rows = []
    for r in valid:
        sens = r.get("sensitivity")
        spec = r.get("specificity")
        if sens is not None and spec is not None:
            rr = dict(r)
            rr["youden_j"] = sens + spec - 1
            youden_rows.append(rr)

    best_youden = max(youden_rows, key=lambda r: r["youden_j"]) if youden_rows else None

    sens90 = [r for r in valid if r.get("sensitivity") is not None and r["sensitivity"] >= 0.90]
    sens95 = [r for r in valid if r.get("sensitivity") is not None and r["sensitivity"] >= 0.95]

    best_sens90 = max(sens90, key=lambda r: (r.get("specificity") or -1, r.get("precision") or -1)) if sens90 else None
    best_sens95 = max(sens95, key=lambda r: (r.get("specificity") or -1, r.get("precision") or -1)) if sens95 else None

    spec80 = [r for r in valid if r.get("specificity") is not None and r["specificity"] >= 0.80]
    best_spec80 = max(spec80, key=lambda r: (r.get("sensitivity") or -1, r.get("f1") or -1)) if spec80 else None

    return {
        "best_f1": best_f1,
        "best_youden": best_youden,
        "best_sensitivity_90_with_max_specificity": best_sens90,
        "best_sensitivity_95_with_max_specificity": best_sens95,
        "best_specificity_80_with_max_sensitivity": best_spec80,
    }


def run_group_label(df, group, label):
    if group == "ALL_SOURCES_STACKED_REFERENCE_ONLY":
        g = df.copy()
    else:
        g = df[df["source_id"] == group].copy()

    if len(g) == 0:
        return [], {}

    y_true = pd.to_numeric(g[f"true_{label}"], errors="coerce").fillna(0).astype(int).values
    y_prob = pd.to_numeric(g[f"prob_{label}"], errors="coerce").fillna(0).astype(float).values

    curve = []
    for t in threshold_grid(y_prob):
        y_pred = (y_prob >= t).astype(int)
        m = calc_binary_metrics(y_true, y_pred, y_prob)
        m.update({
            "group": group,
            "label": label,
            "threshold": float(t),
        })
        curve.append(m)

    picks = pick_thresholds(curve)
    return curve, picks


def main():
    created = datetime.now(timezone.utc).isoformat()

    if not PRED_CSV.exists():
        raise FileNotFoundError(PRED_CSV)

    df = pd.read_csv(PRED_CSV)
    df = df[df["status"] == "ok"].copy()

    curve_rows = []
    threshold_picks = {}

    for group in GROUPS:
        threshold_picks[group] = {}
        for label in TARGET_CLASSES:
            curve, picks = run_group_label(df, group, label)
            curve_rows.extend(curve)
            threshold_picks[group][label] = picks

    curve_csv = OUT / "public_threshold_stress_curve_v333.csv"
    picks_json = OUT / "public_threshold_stress_picks_v333.json"
    summary_json = OUT / "public_calibration_threshold_summary_v333.json"
    md_path = OUT / "PUBLIC_CALIBRATION_THRESHOLD_STRESS_v333.md"
    html_path = OUT / "public_calibration_threshold_stress_report_v333.html"

    pd.DataFrame(curve_rows).to_csv(curve_csv, index=False, encoding="utf-8")

    picks_json.write_text(json.dumps(threshold_picks, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {}
    for group, labels in threshold_picks.items():
        compact[group] = {}
        for label, picks in labels.items():
            compact[group][label] = {}
            for name, row in picks.items():
                if row is None:
                    compact[group][label][name] = None
                else:
                    compact[group][label][name] = {
                        "threshold": row.get("threshold"),
                        "f1": row.get("f1"),
                        "sensitivity": row.get("sensitivity"),
                        "specificity": row.get("specificity"),
                        "precision": row.get("precision"),
                        "positives": row.get("positives"),
                        "negatives": row.get("negatives"),
                    }

    summary = {
        "project": "CardioTwin-AI",
        "version": "v3.3.3 public calibration and threshold stress test",
        "created_at_utc": created,
        "predictions_csv": str(PRED_CSV),
        "n_ok_rows": int(len(df)),
        "groups": GROUPS,
        "target_classes": TARGET_CLASSES,
        "outputs": {
            "threshold_curve_csv": str(curve_csv),
            "threshold_picks_json": str(picks_json),
            "summary_json": str(summary_json),
            "summary_md": str(md_path),
            "summary_html": str(html_path),
        },
        "compact_threshold_recommendations": compact,
        "interpretation": [
            "Screening profile is expected to prioritize sensitivity and produce more false positives.",
            "Threshold stress results are analytical recommendations and do not modify the frozen v3.0.4.1 runtime.",
            "Final clinical thresholds require prospective/doctor-reviewed validation."
        ],
        "claim_boundary": "Calibration and threshold stress analysis only. Not clinical deployment and not final diagnostic thresholding."
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# CardioTwin-AI v3.3.3 Calibration + Threshold Stress Test")
    lines.append("")
    lines.append(f"Created: {created}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Analyze threshold trade-offs after v3.3.2 source-separated validation.")
    lines.append("")
    lines.append("## Key Interpretation")
    lines.append("")
    lines.append("- Screening thresholds prioritize sensitivity.")
    lines.append("- Lower specificity and precision are expected under screening mode.")
    lines.append("- Threshold stress does not change the frozen model/runtime.")
    lines.append("- Final clinical thresholds require prospective doctor-reviewed validation.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- {curve_csv}")
    lines.append(f"- {picks_json}")
    lines.append(f"- {summary_json}")
    lines.append("")
    lines.append("## Compact Threshold Recommendations")
    lines.append("")
    lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")

    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CardioTwin-AI v3.3.3 Calibration + Threshold Stress</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}"
        ".warning{padding:12px;background:#fff7ed;border-left:4px solid #f97316}"
        "pre{background:#f8fafc;padding:12px;overflow-x:auto}</style></head><body>"
        "<h1>CardioTwin-AI v3.3.3 Calibration + Threshold Stress</h1>"
        "<div class='warning'>Analytical threshold stress only. Does not modify frozen runtime. Not clinical deployment.</div>"
        "<h2>Summary</h2>"
        "<pre>" + json.dumps(summary, indent=2, ensure_ascii=False) + "</pre>"
        "</body></html>",
        encoding="utf-8"
    )

    zip_path = RELEASE / "cardiotwin_v3_3_3_public_calibration_threshold_pack.zip"
    manifest_path = RELEASE / "cardiotwin_v3_3_3_public_calibration_threshold_manifest.json"

    files = [
        curve_csv,
        picks_json,
        summary_json,
        md_path,
        html_path,
        OUT / "public_per_source_metrics_v332.json",
        OUT / "public_per_source_metrics_v332.csv",
    ]
    files = [p for p in files if p.exists()]

    manifest = {
        "project": "CardioTwin-AI",
        "version": "v3.3.3 Public Calibration + Threshold Stress Pack",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "files_indexed": len(files),
        "files": [
            {
                "path": p.as_posix(),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
            for p in files
        ],
        "claim_boundary": summary["claim_boundary"],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.as_posix())
        z.write(manifest_path, manifest_path.as_posix())

    print("DONE: v3.3.3 public calibration + threshold stress")
    print("CURVE_CSV:", curve_csv)
    print("PICKS_JSON:", picks_json)
    print("SUMMARY_JSON:", summary_json)
    print("MD:", md_path)
    print("HTML:", html_path)
    print("ZIP:", zip_path)
    print("MANIFEST:", manifest_path)
    print(json.dumps({
        "n_ok_rows": int(len(df)),
        "groups": GROUPS,
        "target_classes": TARGET_CLASSES,
        "files_indexed": manifest["files_indexed"],
        "claim_boundary": summary["claim_boundary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import math
import zipfile
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

OUT = Path("artifacts/public_multicenter_validation_v33")
RELEASE = Path("artifacts/release_rc1")
OUT.mkdir(parents=True, exist_ok=True)
RELEASE.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT / "public_locked_inference_predictions_v331_full.csv"
COHORT_SUMMARY = OUT / "public_locked_validation_cohort_summary_v330.json"
INFER_SUMMARY = OUT / "public_locked_inference_summary_v331_full.json"

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMAL_CLASSES = ["MI", "STTC", "CD", "HYP"]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def safe_metric(fn, y_true, y_score_or_pred, **kwargs):
    try:
        return finite_float(fn(y_true, y_score_or_pred, **kwargs))
    except Exception:
        return None


def binary_confusion(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    except Exception:
        return {
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
        }


def label_metrics(df, label, group_name):
    y_true = pd.to_numeric(df[f"true_{label}"], errors="coerce").fillna(0).astype(int).values
    y_pred = pd.to_numeric(df[f"pred_{label}"], errors="coerce").fillna(0).astype(int).values
    y_prob = pd.to_numeric(df[f"prob_{label}"], errors="coerce").values

    valid_prob = np.isfinite(y_prob)
    y_true_prob = y_true[valid_prob]
    y_prob_valid = y_prob[valid_prob]

    n = int(len(y_true))
    positives = int(y_true.sum())
    negatives = int(n - positives)
    pred_positives = int(y_pred.sum())

    has_both_classes = positives > 0 and negatives > 0 and len(y_true_prob) > 0 and len(set(y_true_prob.tolist())) > 1

    out = {
        "group": group_name,
        "label": label,
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(positives / n) if n else None,
        "predicted_positives": pred_positives,
        "predicted_positive_rate": float(pred_positives / n) if n else None,
        "threshold_median": finite_float(pd.to_numeric(df[f"threshold_{label}"], errors="coerce").median()) if f"threshold_{label}" in df.columns else None,
        "auroc": safe_metric(roc_auc_score, y_true_prob, y_prob_valid) if has_both_classes else None,
        "auprc": safe_metric(average_precision_score, y_true_prob, y_prob_valid) if positives > 0 and len(y_true_prob) > 0 else None,
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "sensitivity_recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "metric_valid_for_auroc": bool(has_both_classes),
    }

    cm = binary_confusion(y_true, y_pred)
    out.update(cm)

    if cm["tn"] is not None:
        tn, fp = cm["tn"], cm["fp"]
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else None
    else:
        out["specificity"] = None

    return out


def abnormal_any_metrics(df, group_name):
    y_true = np.zeros(len(df), dtype=int)
    y_pred = np.zeros(len(df), dtype=int)
    probs = []

    for lab in ABNORMAL_CLASSES:
        y_true = np.maximum(y_true, pd.to_numeric(df[f"true_{lab}"], errors="coerce").fillna(0).astype(int).values)
        y_pred = np.maximum(y_pred, pd.to_numeric(df[f"pred_{lab}"], errors="coerce").fillna(0).astype(int).values)
        probs.append(pd.to_numeric(df[f"prob_{lab}"], errors="coerce").fillna(0).values)

    y_prob = np.max(np.vstack(probs), axis=0)

    n = int(len(df))
    positives = int(y_true.sum())
    negatives = int(n - positives)
    pred_positives = int(y_pred.sum())
    has_both_classes = positives > 0 and negatives > 0

    out = {
        "group": group_name,
        "label": "ABNORMAL_ANY",
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(positives / n) if n else None,
        "predicted_positives": pred_positives,
        "predicted_positive_rate": float(pred_positives / n) if n else None,
        "threshold_median": None,
        "auroc": safe_metric(roc_auc_score, y_true, y_prob) if has_both_classes else None,
        "auprc": safe_metric(average_precision_score, y_true, y_prob) if positives > 0 else None,
        "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "sensitivity_recall": safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "metric_valid_for_auroc": bool(has_both_classes),
    }

    cm = binary_confusion(y_true, y_pred)
    out.update(cm)

    if cm["tn"] is not None:
        tn, fp = cm["tn"], cm["fp"]
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else None
    else:
        out["specificity"] = None

    return out


def summarize_group_metrics(rows, group_name):
    group_rows = [r for r in rows if r["group"] == group_name and r["label"] in TARGET_CLASSES]

    def mean_metric(name):
        vals = [r[name] for r in group_rows if r.get(name) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "group": group_name,
        "macro_auroc": mean_metric("auroc"),
        "macro_auprc": mean_metric("auprc"),
        "macro_f1": mean_metric("f1"),
        "macro_sensitivity": mean_metric("sensitivity_recall"),
        "macro_specificity": mean_metric("specificity"),
        "macro_precision": mean_metric("precision"),
        "labels_with_valid_auroc": int(sum(1 for r in group_rows if r.get("metric_valid_for_auroc"))),
    }


def main():
    created = datetime.now(timezone.utc).isoformat()

    if not PRED_CSV.exists():
        raise FileNotFoundError(PRED_CSV)

    df = pd.read_csv(PRED_CSV)
    df_ok = df[df["status"] == "ok"].copy()

    metrics_rows = []

    for source_id, g in df_ok.groupby("source_id"):
        for lab in TARGET_CLASSES:
            metrics_rows.append(label_metrics(g, lab, source_id))
        metrics_rows.append(abnormal_any_metrics(g, source_id))

    # Overall stacked view for reference only. Not a replacement for source-separated reporting.
    for lab in TARGET_CLASSES:
        metrics_rows.append(label_metrics(df_ok, lab, "ALL_SOURCES_STACKED_REFERENCE_ONLY"))
    metrics_rows.append(abnormal_any_metrics(df_ok, "ALL_SOURCES_STACKED_REFERENCE_ONLY"))

    metrics_csv = OUT / "public_per_source_metrics_v332.csv"
    metrics_json = OUT / "public_per_source_metrics_v332.json"
    summary_md = OUT / "PUBLIC_PER_SOURCE_METRICS_SUMMARY_v332.md"
    summary_html = OUT / "public_per_source_metrics_report_v332.html"

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")

    group_summary = {}
    for group in sorted(metrics_df["group"].unique()):
        group_summary[group] = summarize_group_metrics(metrics_rows, group)

    payload = {
        "project": "CardioTwin-AI",
        "version": "v3.3.2 public per-source metrics",
        "created_at_utc": created,
        "predictions_csv": str(PRED_CSV),
        "cohort_summary": str(COHORT_SUMMARY),
        "inference_summary": str(INFER_SUMMARY),
        "n_prediction_rows": int(len(df)),
        "n_ok_rows": int(len(df_ok)),
        "n_error_rows": int((df["status"] != "ok").sum()),
        "target_classes": TARGET_CLASSES,
        "metrics_csv": str(metrics_csv),
        "group_summary": group_summary,
        "important_interpretation": [
            "Metrics are reported by source. ALL_SOURCES_STACKED_REFERENCE_ONLY is descriptive only and must not be treated as a random-split validation claim.",
            "This uses weak/derived public challenge labels mapped to CardioTwin-AI superclasses.",
            "This is research-use external validation, not clinical deployment or final diagnosis.",
        ],
        "claim_boundary": "Public multi-center source-separated metric evaluation. Not prospective clinical validation.",
    }

    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# CardioTwin-AI v3.3.2 Public Per-source Metrics Summary")
    lines.append("")
    lines.append(f"Created: {created}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Metrics were computed from frozen v3.3.1 predictions on the locked v3.3.0 public cohort.")
    lines.append("")
    lines.append("## Source-level macro summary")
    lines.append("")
    for group, gs in group_summary.items():
        lines.append(f"### {group}")
        lines.append("")
        for k, v in gs.items():
            if k != "group":
                lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("Source-separated metrics only. Not prospective clinical validation. Do not claim doctor-level diagnosis.")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    html_table = metrics_df.to_html(index=False)
    summary_html.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.3.2 Public Per-source Metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; }}
    th {{ background: #f8fafc; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.3.2 Public Per-source Metrics</h1>
  <div class="warning">
    Metrics are source-separated. The stacked overall row is descriptive only and must not be interpreted as random-split validation.
  </div>
  <h2>Group Summary</h2>
  <pre>{json.dumps(group_summary, indent=2, ensure_ascii=False)}</pre>
  <h2>Per-label Metrics</h2>
  {html_table}
</body>
</html>
""", encoding="utf-8")

    zip_path = RELEASE / "cardiotwin_v3_3_2_public_per_source_metrics_pack.zip"
    manifest_path = RELEASE / "cardiotwin_v3_3_2_public_per_source_metrics_manifest.json"

    files = [
        metrics_csv,
        metrics_json,
        summary_md,
        summary_html,
        PRED_CSV,
        OUT / "public_locked_inference_summary_v331_full.json",
        OUT / "public_locked_validation_cohort_summary_v330.json",
    ]
    files = [p for p in files if p.exists()]

    manifest = {
        "project": "CardioTwin-AI",
        "version": "v3.3.2 Public Per-source Metrics Pack",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": payload,
        "files_indexed": len(files),
        "files": [
            {
                "path": p.as_posix(),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
            for p in files
        ],
        "claim_boundary": payload["claim_boundary"],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.as_posix())
        z.write(manifest_path, manifest_path.as_posix())

    print("DONE: v3.3.2 public per-source metrics")
    print("METRICS_CSV:", metrics_csv)
    print("METRICS_JSON:", metrics_json)
    print("SUMMARY_MD:", summary_md)
    print("HTML:", summary_html)
    print("ZIP:", zip_path)
    print("MANIFEST:", manifest_path)
    print(json.dumps({
        "n_ok_rows": int(len(df_ok)),
        "n_error_rows": int((df["status"] != "ok").sum()),
        "group_summary": group_summary,
        "files_indexed": manifest["files_indexed"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

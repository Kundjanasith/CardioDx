from pathlib import Path
import argparse
import csv
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cardiotwin.runtime.v304_real_inference_bridge import (
    LEADS,
    run_v304_real_inference,
)

OUT = Path("artifacts/locked_external_validation_v31")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def norm_id(x):
    s = str(x)
    s = s.strip()
    s = re.sub(r"^[ps]", "", s, flags=re.IGNORECASE)
    return s


def load_wfdb_12lead(hea_path):
    import wfdb

    hea_path = Path(hea_path)
    record_base = str(hea_path.with_suffix(""))
    sig, fields = wfdb.rdsamp(record_base)

    fs = float(fields.get("fs", 500))
    sig_names = list(fields.get("sig_name", []))

    arr = np.asarray(sig, dtype=np.float32)  # shape = n_samples x n_leads

    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D WFDB signal, got shape={arr.shape}")

    if len(sig_names) == arr.shape[1]:
        lead_to_idx = {name: i for i, name in enumerate(sig_names)}
        if all(lead in lead_to_idx for lead in LEADS):
            arr = np.stack([arr[:, lead_to_idx[lead]] for lead in LEADS], axis=1)
            sig_names = LEADS

    if arr.shape[1] != 12:
        raise RuntimeError(f"Expected 12 leads, got shape={arr.shape}, sig_names={sig_names}")

    x = arr.T  # 12 x n_samples

    meta = {
        "record_name": hea_path.stem,
        "header_path": str(hea_path),
        "raw_fs": fs,
        "raw_shape": list(x.shape),
        "sig_names": sig_names,
    }

    return x, fs, meta


def find_metadata_csvs(raw_dir):
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.rglob("*.csv"))


def build_metadata_lookup(raw_dir):
    """
    Try to build subject_id/study_id -> report text lookup from MIMIC metadata CSVs.
    If no usable report CSV exists, returns empty lookup.
    """
    lookup = {}
    scanned = []

    for p in find_metadata_csvs(raw_dir):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            scanned.append({"path": str(p), "status": "read_error", "error": repr(e)})
            continue

        cols = list(df.columns)
        low = {c.lower(): c for c in cols}

        subject_col = None
        study_col = None

        for cand in ["subject_id", "subject", "patient_id"]:
            if cand in low:
                subject_col = low[cand]
                break

        for cand in ["study_id", "study", "ecg_id"]:
            if cand in low:
                study_col = low[cand]
                break

        text_cols = [
            c for c in cols
            if any(k in c.lower() for k in ["report", "interpret", "statement", "diagnosis", "comment", "text"])
        ]

        scanned.append({
            "path": str(p),
            "rows": int(len(df)),
            "columns": cols,
            "subject_col": subject_col,
            "study_col": study_col,
            "text_cols": text_cols,
        })

        if subject_col is None or study_col is None or not text_cols:
            continue

        for _, row in df.iterrows():
            sid = norm_id(row.get(subject_col, ""))
            stid = norm_id(row.get(study_col, ""))
            if not sid or not stid:
                continue

            parts = []
            for c in text_cols:
                val = row.get(c, "")
                if pd.notna(val):
                    parts.append(str(val))

            text = " | ".join([p for p in parts if p.strip()])
            if text.strip():
                lookup[(sid, stid)] = text

    return lookup, scanned


def map_report_to_labels(text):
    """
    Conservative weak label mapping for dry-run audit only.
    This is not final locked full-validation mapping.
    """
    t = str(text).lower()

    labels = {k: 0 for k in TARGET_CLASSES}

    if re.search(r"\bnormal ecg\b|\bnormal sinus rhythm\b|\botherwise normal\b|\bnormal\b", t):
        labels["NORM"] = 1

    if re.search(r"myocardial infarction|\binfarct\b|\bmi\b|old infarct|acute infarct", t):
        labels["MI"] = 1

    if re.search(r"st[- ]?t|st depression|st elevation|t wave abnormal|ischemia|ischaemia|nonspecific st", t):
        labels["STTC"] = 1

    if re.search(r"bundle branch block|\blbbb\b|\brbbb\b|av block|conduction delay|intraventricular conduction", t):
        labels["CD"] = 1

    if re.search(r"hypertrophy|\blvh\b|\brvh\b|ventricular hypertrophy", t):
        labels["HYP"] = 1

    # If abnormal labels are present, do not force NORM as exclusive.
    return labels


def safe_metric_compute(y_true, y_score, y_pred):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

    result = {}

    for i, cls in enumerate(TARGET_CLASSES):
        yt = y_true[:, i]
        ys = y_score[:, i]
        yp = y_pred[:, i]

        positives = int(yt.sum())
        negatives = int(len(yt) - positives)

        item = {
            "support_positive": positives,
            "support_negative": negatives,
            "valid_for_auc": positives > 0 and negatives > 0,
        }

        if positives > 0 and negatives > 0:
            item["auroc"] = float(roc_auc_score(yt, ys))
            item["auprc"] = float(average_precision_score(yt, ys))
        else:
            item["auroc"] = None
            item["auprc"] = None

        if positives > 0:
            item["f1"] = float(f1_score(yt, yp, zero_division=0))
            item["sensitivity"] = float(recall_score(yt, yp, zero_division=0))
            item["precision"] = float(precision_score(yt, yp, zero_division=0))
        else:
            item["f1"] = None
            item["sensitivity"] = None
            item["precision"] = None

        result[cls] = item

    valid_auroc = [v["auroc"] for v in result.values() if v["auroc"] is not None]
    valid_auprc = [v["auprc"] for v in result.values() if v["auprc"] is not None]
    valid_f1 = [v["f1"] for v in result.values() if v["f1"] is not None]
    valid_sens = [v["sensitivity"] for v in result.values() if v["sensitivity"] is not None]

    macro = {
        "valid_label_count_auroc": len(valid_auroc),
        "macro_valid_auroc": float(np.mean(valid_auroc)) if valid_auroc else None,
        "macro_valid_auprc": float(np.mean(valid_auprc)) if valid_auprc else None,
        "macro_valid_f1": float(np.mean(valid_f1)) if valid_f1 else None,
        "macro_valid_sensitivity": float(np.mean(valid_sens)) if valid_sens else None,
    }

    return result, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-index", default="artifacts/locked_external_validation_v31/mimic_demo_record_index_v311.csv")
    ap.add_argument("--raw-dir", default="data/raw/mimic_iv_ecg_demo")
    ap.add_argument("--max-records", type=int, default=50)
    ap.add_argument("--profile", default="screening")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    created = datetime.now(timezone.utc).isoformat()

    record_index = Path(args.record_index)
    if not record_index.exists():
        raise FileNotFoundError(f"record index not found: {record_index}")

    df = pd.read_csv(record_index)
    df_ready = df[df["ready_for_runtime"].astype(str).str.lower() == "true"].copy()

    if not args.full:
        df_ready = df_ready.head(args.max_records)

    metadata_lookup, metadata_scanned = build_metadata_lookup(args.raw_dir)

    prediction_rows = []
    errors = []

    for idx, row in df_ready.iterrows():
        hea_path = Path(row["hea_path"])
        subject_id = norm_id(row.get("subject_id", ""))
        study_id = norm_id(row.get("study_id", ""))

        case = {
            "case_index": len(prediction_rows) + 1,
            "record_id": row.get("record_id", hea_path.stem),
            "subject_id": subject_id,
            "study_id": study_id,
            "hea_path": str(hea_path),
        }

        try:
            x, fs, meta = load_wfdb_12lead(hea_path)

            result = run_v304_real_inference(
                x_raw=x,
                fs=fs,
                model_path="artifacts/models/inceptiontime_v21_safety.pt",
                threshold_path="artifacts/deep_safety_v21/threshold_profiles_deep.json",
                profile=args.profile,
                device=args.device,
                source_meta=meta,
            )

            report_text = metadata_lookup.get((subject_id, study_id), "")
            weak_labels = map_report_to_labels(report_text)

            out = dict(case)
            out.update({
                "status": "ok",
                "inference_mode": result.get("inference_mode"),
                "model_loaded": result.get("model_meta", {}).get("loaded"),
                "threshold_source": result.get("threshold_source"),
                "region_mapper_used": result.get("region_mapper_meta", {}).get("used"),
                "sqi": result.get("sqi"),
                "positive_labels": "|".join(result.get("positive_labels", [])),
                "abnormal_positive_labels": "|".join(result.get("abnormal_positive_labels", [])),
                "recommendation": result.get("recommendation"),
                "metadata_report_found": bool(report_text),
                "report_text_preview": report_text[:300],
            })

            for cls in TARGET_CLASSES:
                out[f"prob_{cls}"] = result.get("probabilities", {}).get(cls)
                out[f"thr_{cls}"] = result.get("thresholds", {}).get(cls)
                out[f"pred_{cls}"] = int(cls in result.get("positive_labels", []))
                out[f"weak_true_{cls}"] = weak_labels[cls]

            prediction_rows.append(out)

        except Exception as e:
            err = dict(case)
            err.update({
                "status": "error",
                "error": repr(e),
            })
            errors.append(err)
            prediction_rows.append(err)

    pred_path = OUT / "mimic_demo_locked_dryrun_predictions_v312.csv"

    all_cols = []
    for r in prediction_rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        w.writerows(prediction_rows)

    ok_rows = [r for r in prediction_rows if r.get("status") == "ok"]

    metrics_payload = {
        "project": "CardioTwin-AI",
        "version": "v3.1.2 MIMIC-IV-ECG Demo locked dry-run evaluation",
        "created_at_utc": created,
        "frozen_runtime_release": "v3.0.4.1 Complete Runtime Release Bundle FINAL",
        "frozen_model": "artifacts/models/inceptiontime_v21_safety.pt",
        "threshold_profile": args.profile,
        "status": "completed_runtime_dry_run",
        "n_requested_records": int(len(df_ready)),
        "n_completed_rows": int(len(prediction_rows)),
        "n_ok": int(len(ok_rows)),
        "n_error": int(len(errors)),
        "n_metadata_report_found": int(sum(bool(r.get("metadata_report_found")) for r in ok_rows)),
        "predictions_csv": str(pred_path),
        "metadata_csv_scanned": metadata_scanned,
        "runtime_checks": {
            "all_ok_real_v2_7_torch_model": all(r.get("inference_mode") == "real_v2_7_torch_model" for r in ok_rows) if ok_rows else False,
            "all_ok_model_loaded": all(str(r.get("model_loaded")) == "True" for r in ok_rows) if ok_rows else False,
            "all_ok_region_mapper_used": all(str(r.get("region_mapper_used")) == "True" for r in ok_rows) if ok_rows else False,
        },
        "per_class_metrics": {},
        "macro_metrics": {},
        "errors": errors[:20],
        "claim_boundary": "Research-use dry-run validation only. Weak labels from metadata text require audit. Not final diagnosis.",
    }

    # Compute metrics only if weak labels exist in usable form.
    if ok_rows:
        y_true = np.array([[int(r.get(f"weak_true_{cls}", 0)) for cls in TARGET_CLASSES] for r in ok_rows], dtype=int)
        y_score = np.array([[float(r.get(f"prob_{cls}", 0.0)) for cls in TARGET_CLASSES] for r in ok_rows], dtype=float)
        y_pred = np.array([[int(r.get(f"pred_{cls}", 0)) for cls in TARGET_CLASSES] for r in ok_rows], dtype=int)

        if y_true.sum() > 0:
            per_class, macro = safe_metric_compute(y_true, y_score, y_pred)
            metrics_payload["per_class_metrics"] = per_class
            metrics_payload["macro_metrics"] = macro
            metrics_payload["status"] = "completed_runtime_dry_run_with_weak_label_metrics"
        else:
            metrics_payload["status"] = "completed_runtime_dry_run_no_usable_weak_labels"

    metrics_path = OUT / "mimic_demo_locked_dryrun_metrics_v312.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update external_metrics_v31.json
    external_metrics = OUT / "external_metrics_v31.json"
    external_metrics.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Failure/review preview
    review_path = OUT / "failure_case_review_v31.md"

    fp_candidates = []
    for r in ok_rows:
        abnormal = str(r.get("abnormal_positive_labels", "")).strip()
        if abnormal:
            fp_candidates.append(r)

    lines = []
    lines.append("# CardioTwin-AI v3.1 Failure Case Review")
    lines.append("")
    lines.append(f"Updated: {created}")
    lines.append("")
    lines.append("## MIMIC-IV-ECG Demo v3.1.2 Dry-run Review Candidates")
    lines.append("")
    lines.append("These are not confirmed failures unless label audit confirms the ground truth.")
    lines.append("")
    lines.append("| case_index | record_id | abnormal_flags | SQI | recommendation | report_found | note |")
    lines.append("|---:|---|---|---:|---|---|---|")

    for r in fp_candidates[:30]:
        lines.append(
            f"| {r.get('case_index')} | {r.get('record_id')} | {r.get('abnormal_positive_labels')} | "
            f"{float(r.get('sqi', 0)):.3f} | {r.get('recommendation')} | {r.get('metadata_report_found')} | needs label audit |"
        )

    if not fp_candidates:
        lines.append("| - | - | none | - | - | - | no abnormal screening flags in reviewed subset |")

    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("This review is for research-use safety analysis only, not clinical diagnosis.")

    review_path.write_text("\n".join(lines), encoding="utf-8")

    # HTML report
    html_path = OUT / "locked_external_validation_report_v31.html"
    html_path.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.1.2 MIMIC-IV-ECG Demo Dry-run</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.1.2 MIMIC-IV-ECG Demo Locked Dry-run Evaluation</h1>
  <div class="warning">
    Research-use dry-run validation only. Weak labels require audit. Not final diagnosis.
  </div>

  <h2>Runtime Summary</h2>
  <pre>{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}</pre>

  <h2>Outputs</h2>
  <ul>
    <li>{pred_path}</li>
    <li>{metrics_path}</li>
    <li>{review_path}</li>
  </ul>
</body>
</html>
""", encoding="utf-8")

    print("DONE: MIMIC-IV-ECG Demo locked dry-run v3.1.2")
    print("PREDICTIONS:", pred_path)
    print("METRICS:", metrics_path)
    print("EXTERNAL_METRICS:", external_metrics)
    print("FAILURE_REVIEW:", review_path)
    print("HTML_REPORT:", html_path)
    print(json.dumps({
        "status": metrics_payload["status"],
        "n_ok": metrics_payload["n_ok"],
        "n_error": metrics_payload["n_error"],
        "runtime_checks": metrics_payload["runtime_checks"],
        "n_metadata_report_found": metrics_payload["n_metadata_report_found"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

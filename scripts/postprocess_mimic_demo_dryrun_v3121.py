from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "locked_external_validation_v31"
RELEASE = ART / "release_rc1"

pred_path = OUT / "mimic_demo_locked_dryrun_predictions_v312.csv"
metrics_path = OUT / "mimic_demo_locked_dryrun_metrics_v312.json"

if not pred_path.exists():
    raise FileNotFoundError(pred_path)

df = pd.read_csv(pred_path)

metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

ok = df[df["status"].astype(str) == "ok"].copy()
err = df[df["status"].astype(str) != "ok"].copy()

def count_contains(col, token):
    if col not in ok.columns:
        return 0
    return int(ok[col].fillna("").astype(str).str.contains(token, regex=False).sum())

target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]

positive_counts = {}
abnormal_counts = {}

for cls in target_classes:
    positive_counts[cls] = count_contains("positive_labels", cls)
    if cls != "NORM":
        abnormal_counts[cls] = count_contains("abnormal_positive_labels", cls)

sqi_values = pd.to_numeric(ok.get("sqi", pd.Series(dtype=float)), errors="coerce")

n_ok = int(len(ok))
n_error = int(len(err))
n_real = int((ok.get("inference_mode", "") == "real_v2_7_torch_model").sum()) if n_ok else 0
n_loaded = int(ok.get("model_loaded", "").astype(str).eq("True").sum()) if n_ok else 0
n_region_used = int(ok.get("region_mapper_used", "").astype(str).eq("True").sum()) if n_ok else 0
n_abnormal_flagged = int(ok.get("abnormal_positive_labels", "").fillna("").astype(str).str.strip().ne("").sum()) if n_ok else 0
n_metadata_report_found = int(ok.get("metadata_report_found", "").astype(str).eq("True").sum()) if n_ok else 0

summary = {
    "project": "CardioTwin-AI",
    "version": "v3.1.2.1 MIMIC-IV-ECG Demo locked dry-run postprocess",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "completed_label_free_runtime_dry_run",
    "interpretation": (
        "MIMIC-IV-ECG Demo dry-run validates runtime compatibility of the frozen v3.0.4.1 bridge. "
        "The local demo metadata did not include diagnosis/report text, so this is not a performance metric validation."
    ),
    "n_total_rows": int(len(df)),
    "n_ok": n_ok,
    "n_error": n_error,
    "n_real_v2_7_torch_model": n_real,
    "n_model_loaded_true": n_loaded,
    "n_region_mapper_used_true": n_region_used,
    "n_abnormal_flagged": n_abnormal_flagged,
    "n_metadata_report_found": n_metadata_report_found,
    "runtime_pass": bool(n_ok > 0 and n_error == 0 and n_real == n_ok and n_loaded == n_ok),
    "region_mapper_note": (
        "region_mapper_v23 is expected to trigger mainly when abnormal positive labels are present. "
        "Therefore all_ok_region_mapper_used=false is not a runtime failure for NORM-only cases."
    ),
    "label_metric_note": (
        "No AUROC/AUPRC/F1 should be claimed from this demo run unless audited labels or diagnostic reports are available."
    ),
    "positive_label_counts": positive_counts,
    "abnormal_label_counts": abnormal_counts,
    "sqi_summary": {
        "min": float(np.nanmin(sqi_values)) if len(sqi_values.dropna()) else None,
        "median": float(np.nanmedian(sqi_values)) if len(sqi_values.dropna()) else None,
        "mean": float(np.nanmean(sqi_values)) if len(sqi_values.dropna()) else None,
        "max": float(np.nanmax(sqi_values)) if len(sqi_values.dropna()) else None,
    },
    "source_files": {
        "predictions_csv": str(pred_path),
        "metrics_json": str(metrics_path),
    },
    "previous_metrics_status": metrics.get("status"),
    "claim_boundary": "Research-use runtime dry-run only. Not final diagnosis and not clinical deployment.",
}

summary_path = OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

md_path = OUT / "MIMIC_DEMO_DRYRUN_INTERPRETATION_v3121.md"
md_path.write_text(f"""# CardioTwin-AI v3.1.2.1 MIMIC-IV-ECG Demo Dry-run Interpretation

Created: {summary["created_at_utc"]}

## Status

`{summary["status"]}`

## What passed

- Records processed: {summary["n_total_rows"]}
- OK rows: {summary["n_ok"]}
- Error rows: {summary["n_error"]}
- Real v2.7 torch model runs: {summary["n_real_v2_7_torch_model"]}
- Model loaded true: {summary["n_model_loaded_true"]}
- Runtime pass: {summary["runtime_pass"]}

## Important interpretation

This run validates that the frozen CardioTwin-AI v3.0.4.1 runtime can process MIMIC-IV-ECG Demo WFDB records.

It does not establish external AUROC/AUPRC/F1 because the local MIMIC demo metadata available to this run did not include diagnosis/report text.

## Region Mapper Note

`all_ok_region_mapper_used = false` is not necessarily a failure.

The v3.0.4.1 bridge calls Region Mapper v2.3 mainly when abnormal positive labels are present. Therefore NORM-only records may not trigger region mapping.

## Label Counts

Positive labels:

{json.dumps(summary["positive_label_counts"], indent=2, ensure_ascii=False)}

Abnormal labels:

{json.dumps(summary["abnormal_label_counts"], indent=2, ensure_ascii=False)}

## SQI Summary

{json.dumps(summary["sqi_summary"], indent=2, ensure_ascii=False)}

## Claim Boundary

Research-use runtime dry-run only. Not final diagnosis and not clinical deployment.
""", encoding="utf-8")

html_path = OUT / "locked_external_validation_report_v31.html"
html_path.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.1.2.1 MIMIC-IV-ECG Demo Runtime Dry-run</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.1.2.1 MIMIC-IV-ECG Demo Runtime Dry-run</h1>
  <div class="warning">
    Label-free runtime dry-run. No external AUROC/AUPRC/F1 should be claimed from this demo run.
  </div>

  <h2>Runtime Summary</h2>
  <pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>

  <h2>Outputs</h2>
  <ul>
    <li>{pred_path}</li>
    <li>{metrics_path}</li>
    <li>{summary_path}</li>
    <li>{md_path}</li>
  </ul>
</body>
</html>
""", encoding="utf-8")

# Package dry-run artifacts
zip_path = RELEASE / "cardiotwin_v3_1_2_1_mimic_demo_runtime_dryrun_pack.zip"
manifest_path = RELEASE / "cardiotwin_v3_1_2_1_mimic_demo_runtime_dryrun_manifest.json"

files = [
    OUT / "LOCKED_EXTERNAL_VALIDATION_PROTOCOL_v31.md",
    OUT / "dataset_readiness_report.json",
    OUT / "mimic_demo_readiness_v311.json",
    OUT / "mimic_demo_record_index_v311.csv",
    OUT / "mimic_demo_locked_dryrun_predictions_v312.csv",
    OUT / "mimic_demo_locked_dryrun_metrics_v312.json",
    OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json",
    OUT / "MIMIC_DEMO_DRYRUN_INTERPRETATION_v3121.md",
    OUT / "failure_case_review_v31.md",
    OUT / "locked_external_validation_report_v31.html",
]
files = [p for p in files if p.exists()]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.1.2.1 MIMIC-IV-ECG Demo Runtime Dry-run Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "summary": summary,
    "claim_boundary": "Research-use runtime dry-run only. Not final diagnosis and not clinical deployment.",
    "files_indexed": len(files),
    "files": [
        {
            "path": p.as_posix(),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        }
        for p in files
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.as_posix())
    z.write(manifest_path, manifest_path.as_posix())

print("DONE: v3.1.2.1 MIMIC Demo runtime dry-run postprocess")
print("SUMMARY:", summary_path)
print("INTERPRETATION:", md_path)
print("HTML:", html_path)
print("ZIP:", zip_path)
print("MANIFEST:", manifest_path)
print(json.dumps(summary, indent=2, ensure_ascii=False))

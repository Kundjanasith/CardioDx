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
RELEASE.mkdir(parents=True, exist_ok=True)

pred_path = OUT / "mimic_demo_locked_dryrun_predictions_v312.csv"
summary_path = OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json"
record_index_path = OUT / "mimic_demo_record_index_v311.csv"
readiness_path = OUT / "mimic_demo_readiness_v311.json"

if not pred_path.exists():
    raise FileNotFoundError(pred_path)

pred = pd.read_csv(pred_path)
ok = pred[pred["status"].astype(str) == "ok"].copy()

summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

readiness = {}
if readiness_path.exists():
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))

target_classes = ["NORM", "MI", "STTC", "CD", "HYP"]

def safe_num(series):
    return pd.to_numeric(series, errors="coerce")

prob_summary = {}
for cls in target_classes:
    col = f"prob_{cls}"
    if col in ok.columns:
        s = safe_num(ok[col])
        prob_summary[cls] = {
            "min": float(np.nanmin(s)) if len(s.dropna()) else None,
            "median": float(np.nanmedian(s)) if len(s.dropna()) else None,
            "mean": float(np.nanmean(s)) if len(s.dropna()) else None,
            "max": float(np.nanmax(s)) if len(s.dropna()) else None,
            "positive_count": int(ok["positive_labels"].fillna("").astype(str).str.contains(cls, regex=False).sum()),
        }

abnormal_rows = ok[ok.get("abnormal_positive_labels", "").fillna("").astype(str).str.strip().ne("")].copy()
norm_only_rows = ok[
    ok.get("positive_labels", "").fillna("").astype(str).eq("NORM")
].copy()

sqi = safe_num(ok["sqi"]) if "sqi" in ok.columns else pd.Series(dtype=float)
low_sqi_rows = ok[sqi < 0.55].copy() if len(sqi) else ok.iloc[0:0].copy()

audit = {
    "project": "CardioTwin-AI",
    "version": "v3.1.3 MIMIC-IV-ECG Demo label/report availability and screening audit",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "MIMIC-IV-ECG Demo",
    "runtime_summary": {
        "n_total_rows": int(len(pred)),
        "n_ok": int(len(ok)),
        "n_error": int(len(pred) - len(ok)),
        "runtime_pass": bool(summary.get("runtime_pass")),
        "n_real_v2_7_torch_model": int(summary.get("n_real_v2_7_torch_model", 0)),
        "n_model_loaded_true": int(summary.get("n_model_loaded_true", 0)),
        "n_region_mapper_used_true": int(summary.get("n_region_mapper_used_true", 0)),
        "n_metadata_report_found": int(summary.get("n_metadata_report_found", 0)),
    },
    "dataset_readiness": readiness.get("counts", {}),
    "label_report_audit": {
        "metadata_report_found": int(summary.get("n_metadata_report_found", 0)),
        "label_metric_available": False,
        "reason": "Local MIMIC-IV-ECG Demo files include waveform records and record_list.csv but no diagnosis/report text columns were available in this dry-run.",
        "metric_claim_allowed": False,
        "allowed_claim": "Runtime compatibility only: the frozen CardioTwin-AI v3.0.4.1 bridge processed MIMIC-IV-ECG Demo records without runtime errors.",
        "disallowed_claim": "Do not claim external AUROC, AUPRC, F1, sensitivity, or disease prevalence from this label-free demo run."
    },
    "screening_output_audit": {
        "n_abnormal_flagged": int(summary.get("n_abnormal_flagged", 0)),
        "positive_label_counts": summary.get("positive_label_counts", {}),
        "abnormal_label_counts": summary.get("abnormal_label_counts", {}),
        "probability_summary": prob_summary,
        "n_norm_only_rows": int(len(norm_only_rows)),
        "n_low_sqi_rows": int(len(low_sqi_rows)),
        "interpretation": "Screening-positive labels are review flags under the sensitive screening profile, not confirmed diagnoses."
    },
    "recommended_next_step": {
        "v3_1_4": "Package MIMIC Demo dry-run as label-free runtime validation artifact.",
        "v3_2": "Move to full MIMIC-IV-ECG or KURIAS for label/report-supported external metric validation.",
        "label_requirement": "Need diagnosis/report text, SNOMED/ontology labels, or manually audited labels before reporting AUROC/AUPRC/F1."
    },
    "claim_boundary": "Research-use runtime validation and safety audit only. Not final diagnosis and not clinical deployment."
}

audit_json = OUT / "mimic_demo_label_report_audit_v313.json"
audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

audit_md = OUT / "MIMIC_DEMO_LABEL_REPORT_AUDIT_v313.md"
audit_md.write_text(f"""# CardioTwin-AI v3.1.3 MIMIC-IV-ECG Demo Label/Report Availability Audit

Created: {audit["created_at_utc"]}

## Bottom Line

MIMIC-IV-ECG Demo dry-run passed as a runtime compatibility test.

It should not be presented as external performance validation because local metadata did not include diagnosis/report text.

## Runtime Result

- Total rows: {audit["runtime_summary"]["n_total_rows"]}
- OK rows: {audit["runtime_summary"]["n_ok"]}
- Error rows: {audit["runtime_summary"]["n_error"]}
- Runtime pass: {audit["runtime_summary"]["runtime_pass"]}
- Real v2.7 model runs: {audit["runtime_summary"]["n_real_v2_7_torch_model"]}
- Model loaded true: {audit["runtime_summary"]["n_model_loaded_true"]}
- Region mapper used true: {audit["runtime_summary"]["n_region_mapper_used_true"]}

## Label/Report Audit

- Metadata report found: {audit["label_report_audit"]["metadata_report_found"]}
- Label metric available: {audit["label_report_audit"]["label_metric_available"]}
- Metric claim allowed: {audit["label_report_audit"]["metric_claim_allowed"]}

Reason:

{audit["label_report_audit"]["reason"]}

## Screening Output Audit

Abnormal screening flags:

{json.dumps(audit["screening_output_audit"]["abnormal_label_counts"], indent=2, ensure_ascii=False)}

Positive labels:

{json.dumps(audit["screening_output_audit"]["positive_label_counts"], indent=2, ensure_ascii=False)}

Probability summary:

{json.dumps(audit["screening_output_audit"]["probability_summary"], indent=2, ensure_ascii=False)}

## Correct Claim

Allowed:

Runtime compatibility only. The frozen CardioTwin-AI v3.0.4.1 bridge processed MIMIC-IV-ECG Demo WFDB records without runtime errors.

Not allowed:

External AUROC, AUPRC, F1, sensitivity, specificity, or disease-prevalence claims from this label-free demo run.

## Next Step

Use full MIMIC-IV-ECG or KURIAS with diagnosis/report labels for true locked external performance validation.

## Claim Boundary

Research-use runtime validation and safety audit only. Not final diagnosis and not clinical deployment.
""", encoding="utf-8")

# Update HTML report with audit summary
html_path = OUT / "locked_external_validation_report_v31.html"
html_path.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.1.3 MIMIC-IV-ECG Demo Runtime + Label Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.1.3 MIMIC-IV-ECG Demo Runtime + Label Audit</h1>
  <div class="warning">
    Label-free runtime validation only. Do not claim AUROC/AUPRC/F1 from this demo run.
  </div>

  <h2>Audit Summary</h2>
  <pre>{json.dumps(audit, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
""", encoding="utf-8")

zip_path = RELEASE / "cardiotwin_v3_1_3_mimic_demo_label_report_audit_pack.zip"
manifest_path = RELEASE / "cardiotwin_v3_1_3_mimic_demo_label_report_audit_manifest.json"

files = [
    OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json",
    OUT / "MIMIC_DEMO_DRYRUN_INTERPRETATION_v3121.md",
    OUT / "mimic_demo_label_report_audit_v313.json",
    OUT / "MIMIC_DEMO_LABEL_REPORT_AUDIT_v313.md",
    OUT / "mimic_demo_locked_dryrun_predictions_v312.csv",
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
    "version": "v3.1.3 MIMIC Demo Label/Report Audit Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "audit_summary": audit,
    "claim_boundary": audit["claim_boundary"],
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

print("DONE: v3.1.3 MIMIC Demo label/report audit pack")
print("AUDIT_JSON:", audit_json)
print("AUDIT_MD:", audit_md)
print("HTML:", html_path)
print("ZIP:", zip_path)
print("MANIFEST:", manifest_path)
print(json.dumps({
    "runtime_pass": audit["runtime_summary"]["runtime_pass"],
    "n_ok": audit["runtime_summary"]["n_ok"],
    "n_error": audit["runtime_summary"]["n_error"],
    "metric_claim_allowed": audit["label_report_audit"]["metric_claim_allowed"],
    "label_metric_available": audit["label_report_audit"]["label_metric_available"],
    "n_abnormal_flagged": audit["screening_output_audit"]["n_abnormal_flagged"]
}, indent=2, ensure_ascii=False))

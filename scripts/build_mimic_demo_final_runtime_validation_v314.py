from pathlib import Path
import json
import zipfile
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "locked_external_validation_v31"
RELEASE = ART / "release_rc1"
RELEASE.mkdir(parents=True, exist_ok=True)

summary_path = OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json"
audit_path = OUT / "mimic_demo_label_report_audit_v313.json"

summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

final_report = OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.md"

final_report.write_text(f"""# CardioTwin-AI v3.1.4 MIMIC-IV-ECG Demo Label-free Runtime Validation Final Report

Created: {datetime.now(timezone.utc).isoformat()}

## Release Context

This report closes the MIMIC-IV-ECG Demo dry-run validation phase for CardioTwin-AI after the v3.0.4.1 Complete Runtime Release Bundle FINAL.

## Dataset

MIMIC-IV-ECG Demo

Prepared records:

- HEA files: {audit.get("dataset_readiness", {}).get("hea_count")}
- DAT files: {audit.get("dataset_readiness", {}).get("dat_count")}
- Ready for runtime rows: {audit.get("dataset_readiness", {}).get("ready_for_runtime_rows")}

## Runtime Result

- Total rows: {audit.get("runtime_summary", {}).get("n_total_rows")}
- OK rows: {audit.get("runtime_summary", {}).get("n_ok")}
- Error rows: {audit.get("runtime_summary", {}).get("n_error")}
- Runtime pass: {audit.get("runtime_summary", {}).get("runtime_pass")}
- Real v2.7 model runs: {audit.get("runtime_summary", {}).get("n_real_v2_7_torch_model")}
- Model loaded true: {audit.get("runtime_summary", {}).get("n_model_loaded_true")}
- Region mapper used true: {audit.get("runtime_summary", {}).get("n_region_mapper_used_true")}

## Screening Output Summary

Positive label counts:

{json.dumps(audit.get("screening_output_audit", {}).get("positive_label_counts", {}), indent=2, ensure_ascii=False)}

Abnormal screening flag counts:

{json.dumps(audit.get("screening_output_audit", {}).get("abnormal_label_counts", {}), indent=2, ensure_ascii=False)}

SQI summary:

{json.dumps(summary.get("sqi_summary", {}), indent=2, ensure_ascii=False)}

## Label/Report Audit Conclusion

Metric claim allowed: {audit.get("label_report_audit", {}).get("metric_claim_allowed")}

Label metric available: {audit.get("label_report_audit", {}).get("label_metric_available")}

Reason:

{audit.get("label_report_audit", {}).get("reason")}

## Correct Claim

Allowed claim:

The frozen CardioTwin-AI v3.0.4.1 runtime bridge processed all available MIMIC-IV-ECG Demo WFDB records without runtime errors.

Disallowed claim:

Do not claim external AUROC, AUPRC, F1, sensitivity, specificity, or disease prevalence from this label-free demo run.

## Interpretation

This is a successful label-free runtime validation, not a disease-label performance validation.

The high number of abnormal screening flags reflects the sensitive screening profile and should be interpreted as review flags, not diagnoses.

## Next Step

Proceed to label-supported external validation using one of:

1. Full MIMIC-IV-ECG with diagnostic report/label fields prepared
2. KURIAS-ECG with SNOMED/OMOP/Minnesota-style diagnosis ontology
3. Manually audited subset with expert labels

## Claim Boundary

Research-use runtime validation and safety audit only. Not final diagnosis and not clinical deployment.
""", encoding="utf-8")

html_final = OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.html"
html_final.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.1.4 MIMIC Demo Label-free Runtime Validation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.1.4 MIMIC-IV-ECG Demo Label-free Runtime Validation</h1>
  <div class="warning">
    Successful runtime validation only. No external AUROC/AUPRC/F1 should be claimed from this label-free demo run.
  </div>

  <h2>Runtime Summary</h2>
  <pre>{json.dumps(audit.get("runtime_summary", {}), indent=2, ensure_ascii=False)}</pre>

  <h2>Screening Output Audit</h2>
  <pre>{json.dumps(audit.get("screening_output_audit", {}), indent=2, ensure_ascii=False)}</pre>

  <h2>Label/Report Audit</h2>
  <pre>{json.dumps(audit.get("label_report_audit", {}), indent=2, ensure_ascii=False)}</pre>

  <h2>Claim Boundary</h2>
  <p>Research-use runtime validation and safety audit only. Not final diagnosis and not clinical deployment.</p>
</body>
</html>
""", encoding="utf-8")

files = [
    OUT / "LOCKED_EXTERNAL_VALIDATION_PROTOCOL_v31.md",
    OUT / "dataset_readiness_report.json",
    OUT / "label_mapping_v31.csv",
    OUT / "mimic_demo_readiness_v311.json",
    OUT / "mimic_demo_record_index_v311.csv",
    OUT / "mimic_demo_locked_dryrun_predictions_v312.csv",
    OUT / "mimic_demo_locked_dryrun_metrics_v312.json",
    OUT / "mimic_demo_locked_dryrun_runtime_summary_v3121.json",
    OUT / "MIMIC_DEMO_DRYRUN_INTERPRETATION_v3121.md",
    OUT / "mimic_demo_label_report_audit_v313.json",
    OUT / "MIMIC_DEMO_LABEL_REPORT_AUDIT_v313.md",
    OUT / "failure_case_review_v31.md",
    OUT / "locked_external_validation_report_v31.html",
    OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.md",
    OUT / "MIMIC_DEMO_LABEL_FREE_RUNTIME_VALIDATION_FINAL_v314.html",
]

files = [p for p in files if p.exists()]

zip_path = RELEASE / "cardiotwin_v3_1_4_mimic_demo_label_free_runtime_validation_pack.zip"
manifest_path = RELEASE / "cardiotwin_v3_1_4_mimic_demo_label_free_runtime_validation_manifest.json"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.1.4 MIMIC-IV-ECG Demo Label-free Runtime Validation Pack",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Final package for MIMIC-IV-ECG Demo label-free runtime validation after v3.0.4.1 frozen runtime release.",
    "runtime_result": audit.get("runtime_summary", {}),
    "label_report_audit": audit.get("label_report_audit", {}),
    "screening_output_audit": audit.get("screening_output_audit", {}),
    "allowed_claim": "Runtime compatibility only: frozen v3.0.4.1 bridge processed MIMIC-IV-ECG Demo WFDB records without runtime errors.",
    "disallowed_claim": "No external AUROC/AUPRC/F1/sensitivity/specificity/disease-prevalence claims from this label-free demo run.",
    "next_step": "Move to full MIMIC-IV-ECG or KURIAS with diagnosis/report labels for performance validation.",
    "claim_boundary": "Research-use runtime validation and safety audit only. Not final diagnosis and not clinical deployment.",
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

print("DONE: v3.1.4 MIMIC Demo label-free runtime validation pack")
print("FINAL_MD:", final_report)
print("FINAL_HTML:", html_final)
print("ZIP:", zip_path)
print("MANIFEST:", manifest_path)
print("files_indexed:", manifest["files_indexed"])
print(json.dumps({
    "runtime_pass": audit.get("runtime_summary", {}).get("runtime_pass"),
    "n_ok": audit.get("runtime_summary", {}).get("n_ok"),
    "n_error": audit.get("runtime_summary", {}).get("n_error"),
    "metric_claim_allowed": audit.get("label_report_audit", {}).get("metric_claim_allowed"),
    "label_metric_available": audit.get("label_report_audit", {}).get("label_metric_available"),
}, indent=2, ensure_ascii=False))

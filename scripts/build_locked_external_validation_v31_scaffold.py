from pathlib import Path
import json
import csv
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "locked_external_validation_v31"
OUT.mkdir(parents=True, exist_ok=True)

protocol = OUT / "LOCKED_EXTERNAL_VALIDATION_PROTOCOL_v31.md"
readiness = OUT / "dataset_readiness_report.json"
label_mapping = OUT / "label_mapping_v31.csv"
metrics = OUT / "external_metrics_v31.json"
failure_review = OUT / "failure_case_review_v31.md"
html_report = OUT / "locked_external_validation_report_v31.html"

created = datetime.now(timezone.utc).isoformat()

protocol.write_text(f"""# CardioTwin-AI v3.1 Locked External Validation Protocol

Created: {created}

## 1. Purpose

This protocol defines locked external validation for CardioTwin-AI after the v3.0.4.1 complete runtime release.

The goal is to evaluate the frozen CardioTwin-AI v2.7 InceptionTime safety model and v3.0.4.1 real inference bridge on a new external ECG dataset without changing model weights, thresholds, preprocessing assumptions, or label mapping after the lock point.

## 2. Frozen Runtime Release

Frozen release:

- CardioTwin-AI v3.0.4.1 Complete Runtime Release Bundle FINAL
- Frozen model: artifacts/models/inceptiontime_v21_safety.pt
- Frozen threshold profile file: artifacts/deep_safety_v21/threshold_profiles_deep.json
- Frozen runtime bridge: src/cardiotwin/runtime/v304_real_inference_bridge.py
- Frozen dashboard: apps/streamlit_cardiotwin_unified_v304_real_inference.py

## 3. Candidate External Datasets

Recommended candidates:

1. MIMIC-IV-ECG
2. MIMIC-IV-ECG Demo
3. KURIAS-ECG

Use MIMIC-IV-ECG Demo for dry-run if full MIMIC-IV-ECG access/storage is not ready.

Use KURIAS if the goal is a clean 12-lead ECG external set with standardized diagnosis ontology.

## 4. Lock Rules

After validation starts, do not change:

- model weights
- preprocessing length and sampling target
- threshold profile
- label mapping
- metric definitions
- inclusion/exclusion rules

Any change must create a new version number.

## 5. Input Requirements

Each ECG record should be converted/read as:

- 12 leads
- 10 seconds preferred
- WFDB .hea + .dat/.mat or compatible CSV
- sampling rate recorded in metadata
- diagnosis labels mapped to PTB-XL-style superclasses where possible

Target classes:

- NORM
- MI
- STTC
- CD
- HYP

## 6. Metrics

Report:

- per-class AUROC
- per-class AUPRC
- per-class F1
- per-class sensitivity/recall
- per-class specificity where available
- macro valid-label AUROC
- macro valid-label AUPRC
- macro valid-label F1
- macro valid-label sensitivity
- number of positive samples per class
- excluded classes due to insufficient support

## 7. Failure Case Review

For selected false positives, false negatives, low-SQI cases, and high-confidence disagreements:

- record ID
- dataset
- true label
- predicted labels
- probabilities
- thresholds
- SQI
- region mapper output
- reviewer note
- likely reason

## 8. Claim Boundary

This validation supports research-use preliminary screening and referral-support evaluation.

It is not clinical deployment and not final diagnosis.

## 9. Output Files

Expected outputs:

- dataset_readiness_report.json
- label_mapping_v31.csv
- external_metrics_v31.json
- failure_case_review_v31.md
- locked_external_validation_report_v31.html
""", encoding="utf-8")

dataset_candidates = {
    "mimic_iv_ecg": {
        "root": "data/raw/mimic_iv_ecg",
        "expected_format": "WFDB .hea + .dat",
        "status": "not_checked"
    },
    "mimic_iv_ecg_demo": {
        "root": "data/raw/mimic_iv_ecg_demo",
        "expected_format": "WFDB .hea + .dat",
        "status": "not_checked"
    },
    "kurias_ecg": {
        "root": "data/raw/kurias_ecg",
        "expected_format": "WFDB .hea + signal files + CSV metadata",
        "status": "not_checked"
    }
}

for name, item in dataset_candidates.items():
    root = Path(item["root"])
    hea_count = len(list(root.rglob("*.hea"))) if root.exists() else 0
    dat_count = len(list(root.rglob("*.dat"))) if root.exists() else 0
    mat_count = len(list(root.rglob("*.mat"))) if root.exists() else 0
    csv_count = len(list(root.rglob("*.csv"))) if root.exists() else 0

    item.update({
        "exists": root.exists(),
        "hea_count": hea_count,
        "dat_count": dat_count,
        "mat_count": mat_count,
        "csv_count": csv_count,
        "readiness": "ready_candidate" if hea_count > 0 and (dat_count > 0 or mat_count > 0) else "dataset_not_found_or_incomplete"
    })

readiness_payload = {
    "project": "CardioTwin-AI",
    "version": "v3.1 locked external validation",
    "created_at_utc": created,
    "frozen_release": "CardioTwin-AI v3.0.4.1 Complete Runtime Release Bundle FINAL",
    "dataset_candidates": dataset_candidates,
    "recommended_next_action": (
        "Use the first ready_candidate dataset. If none are ready, download/prepare MIMIC-IV-ECG Demo first "
        "for dry-run, then proceed to full MIMIC-IV-ECG or KURIAS."
    )
}

readiness.write_text(json.dumps(readiness_payload, indent=2, ensure_ascii=False), encoding="utf-8")

mapping_rows = [
    {
        "source_dataset": "MIMIC-IV-ECG",
        "source_label_or_text_pattern": "normal ecg|normal sinus rhythm|otherwise normal",
        "target_superclass": "NORM",
        "mapping_confidence": "medium",
        "include": "yes",
        "notes": "Report-text/machine statement mapping requires audit."
    },
    {
        "source_dataset": "MIMIC-IV-ECG",
        "source_label_or_text_pattern": "myocardial infarction|old mi|acute mi|inferior infarct|anterior infarct",
        "target_superclass": "MI",
        "mapping_confidence": "medium",
        "include": "yes",
        "notes": "Needs conservative pattern list and manual spot-check."
    },
    {
        "source_dataset": "MIMIC-IV-ECG",
        "source_label_or_text_pattern": "st-t abnormality|st depression|t wave abnormality|ischemia",
        "target_superclass": "STTC",
        "mapping_confidence": "medium",
        "include": "yes",
        "notes": "Avoid over-mapping nonspecific statements."
    },
    {
        "source_dataset": "MIMIC-IV-ECG",
        "source_label_or_text_pattern": "bundle branch block|av block|intraventricular conduction delay",
        "target_superclass": "CD",
        "mapping_confidence": "medium",
        "include": "yes",
        "notes": "Map conduction/rhythm statements carefully."
    },
    {
        "source_dataset": "MIMIC-IV-ECG",
        "source_label_or_text_pattern": "left ventricular hypertrophy|right ventricular hypertrophy|lvh|rvh",
        "target_superclass": "HYP",
        "mapping_confidence": "medium",
        "include": "yes",
        "notes": "Screening threshold can be sensitive; report as review flag."
    },
    {
        "source_dataset": "KURIAS-ECG",
        "source_label_or_text_pattern": "SNOMED/OMOP/Minnesota normal category",
        "target_superclass": "NORM",
        "mapping_confidence": "to_be_audited",
        "include": "pending",
        "notes": "Use official metadata fields after dataset preparation."
    },
    {
        "source_dataset": "KURIAS-ECG",
        "source_label_or_text_pattern": "Minnesota infarction-related category",
        "target_superclass": "MI",
        "mapping_confidence": "to_be_audited",
        "include": "pending",
        "notes": "Audit category/code mapping before locking."
    },
    {
        "source_dataset": "KURIAS-ECG",
        "source_label_or_text_pattern": "Minnesota ST-T abnormality category",
        "target_superclass": "STTC",
        "mapping_confidence": "to_be_audited",
        "include": "pending",
        "notes": "Audit category/code mapping before locking."
    },
    {
        "source_dataset": "KURIAS-ECG",
        "source_label_or_text_pattern": "Minnesota conduction/rhythm category",
        "target_superclass": "CD",
        "mapping_confidence": "to_be_audited",
        "include": "pending",
        "notes": "Audit category/code mapping before locking."
    },
    {
        "source_dataset": "KURIAS-ECG",
        "source_label_or_text_pattern": "hypertrophy-related diagnosis/category",
        "target_superclass": "HYP",
        "mapping_confidence": "to_be_audited",
        "include": "pending",
        "notes": "Audit category/code mapping before locking."
    },
]

with label_mapping.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "source_dataset",
        "source_label_or_text_pattern",
        "target_superclass",
        "mapping_confidence",
        "include",
        "notes",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(mapping_rows)

metrics_payload = {
    "project": "CardioTwin-AI",
    "version": "v3.1 locked external validation",
    "created_at_utc": created,
    "status": "not_run_yet",
    "frozen_model": "artifacts/models/inceptiontime_v21_safety.pt",
    "threshold_profile": "screening",
    "metrics": {},
    "notes": [
        "Metrics will be populated only after a dataset readiness report confirms data availability.",
        "Classes with insufficient support must be excluded from valid-label macro metrics and disclosed."
    ]
}

metrics.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

failure_review.write_text(f"""# CardioTwin-AI v3.1 Failure Case Review

Created: {created}

## Purpose

This document will record selected failure cases from locked external validation.

## Review Table Template

| case_id | dataset | true_labels | predicted_labels | probabilities | thresholds | SQI | region_output | error_type | reviewer_note |
|---|---|---|---|---|---|---:|---|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Required Failure Groups

1. False negative MI
2. False positive MI
3. False negative HYP
4. False positive HYP
5. Low-SQI cases
6. Region mapper uncertain cases
7. High-confidence disagreement cases

## Claim Boundary

Failure case review is for research improvement and safety analysis, not clinical diagnosis.
""", encoding="utf-8")

html_report.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.1 Locked External Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.1 Locked External Validation Report</h1>
  <div class="warning">
    Status: scaffold created. Metrics not run yet. Research-use only. Not final diagnosis.
  </div>

  <h2>Frozen Release</h2>
  <p>CardioTwin-AI v3.0.4.1 Complete Runtime Release Bundle FINAL</p>

  <h2>Dataset Readiness</h2>
  <pre>{json.dumps(readiness_payload, indent=2, ensure_ascii=False)}</pre>

  <h2>Next Step</h2>
  <p>Prepare one external dataset locally, then run locked evaluation without changing model, thresholds, preprocessing, or label mapping after lock.</p>
</body>
</html>
""", encoding="utf-8")

print("DONE: v3.1 locked external validation scaffold created")
print("OUT:", OUT)
print("PROTOCOL:", protocol)
print("READINESS:", readiness)
print("LABEL_MAPPING:", label_mapping)
print("METRICS:", metrics)
print("FAILURE_REVIEW:", failure_review)
print("HTML_REPORT:", html_report)

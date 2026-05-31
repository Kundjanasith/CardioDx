from pathlib import Path
import json
import csv
from datetime import datetime, timezone

ROOT = Path(".")
ART = ROOT / "artifacts"
OUT = ART / "label_supported_external_validation_v32"
OUT.mkdir(parents=True, exist_ok=True)

created = datetime.now(timezone.utc).isoformat()

protocol = OUT / "LABEL_SUPPORTED_EXTERNAL_VALIDATION_PROTOCOL_v32.md"
dataset_plan = OUT / "dataset_selection_plan_v32.json"
mapping_template = OUT / "label_mapping_template_v32.csv"
metrics_template = OUT / "external_metrics_template_v32.json"
failure_template = OUT / "failure_case_review_template_v32.md"
html = OUT / "label_supported_external_validation_report_v32.html"

protocol.write_text(f"""# CardioTwin-AI v3.2 Label-supported External Validation Protocol

Created: {created}

## Purpose

v3.2 starts true external performance validation after v3.1.4 label-free runtime validation.

The goal is to evaluate the frozen CardioTwin-AI v3.0.4.1 runtime on an external dataset that has usable diagnosis/report labels.

## Frozen Components

Do not change:

- Model: artifacts/models/inceptiontime_v21_safety.pt
- Threshold file: artifacts/deep_safety_v21/threshold_profiles_deep.json
- Runtime bridge: src/cardiotwin/runtime/v304_real_inference_bridge.py
- Preprocessing target: 12-lead, 10 seconds, 100 Hz model input
- Target classes: NORM, MI, STTC, CD, HYP

## Candidate Dataset Choice

### Option A: Full MIMIC-IV-ECG

Best when the goal is large-scale real-world external validation.

Need:
- waveform records
- report/diagnosis metadata
- subject/study linkage
- conservative report-to-superclass mapping

### Option B: KURIAS-ECG

Best when the goal is cleaner ontology-supported label mapping.

Need:
- waveform records
- diagnosis ontology / SNOMED / OMOP / Minnesota-style fields
- official label fields mapped to target superclasses

## Validation Rules

Before running metrics, freeze:

1. dataset subset
2. inclusion/exclusion criteria
3. label mapping
4. threshold profile
5. metric definitions
6. minimum support rule for each class

## Required Outputs

- dataset_readiness_v32.json
- frozen_label_mapping_v32.csv
- external_predictions_v32.csv
- external_metrics_v32.json
- failure_case_review_v32.md
- label_supported_external_validation_report_v32.html

## Claim Boundary

This is research-use validation only. Not clinical deployment and not final diagnosis.
""", encoding="utf-8")

dataset_plan_payload = {
    "project": "CardioTwin-AI",
    "version": "v3.2 label-supported external validation",
    "created_at_utc": created,
    "status": "scaffold_created",
    "previous_phase": {
        "v3_1_4": "MIMIC-IV-ECG Demo label-free runtime validation passed 659/659 records but no label metrics were allowed."
    },
    "candidate_datasets": {
        "full_mimic_iv_ecg": {
            "recommended_when": "Need large-scale real-world validation.",
            "local_root": "data/raw/mimic_iv_ecg",
            "required_assets": [
                "WFDB waveform files",
                "report/diagnosis metadata",
                "record-to-subject/study linkage"
            ],
            "expected_v32_use": "performance validation after report-label mapping audit"
        },
        "kurias_ecg": {
            "recommended_when": "Need cleaner ontology-supported label mapping.",
            "local_root": "data/raw/kurias_ecg",
            "required_assets": [
                "12-lead ECG waveform files",
                "diagnosis/ontology metadata",
                "official code or category mapping"
            ],
            "expected_v32_use": "performance validation with ontology-backed labels"
        }
    },
    "decision_recommendation": (
        "Use KURIAS first if label ontology is easier to access locally. "
        "Use full MIMIC-IV-ECG first if the report metadata is already accessible and storage is ready."
    )
}

dataset_plan.write_text(json.dumps(dataset_plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")

rows = [
    ["dataset", "source_code_or_text_pattern", "target_superclass", "include", "confidence", "notes"],
    ["FULL_MIMIC_IV_ECG", "normal ecg|normal sinus rhythm|otherwise normal", "NORM", "pending", "to_audit", "freeze only after report-field inspection"],
    ["FULL_MIMIC_IV_ECG", "myocardial infarction|old mi|acute mi|inferior infarct|anterior infarct", "MI", "pending", "to_audit", "use conservative terms only"],
    ["FULL_MIMIC_IV_ECG", "st-t abnormality|st depression|t wave abnormality|ischemia", "STTC", "pending", "to_audit", "avoid over-mapping nonspecific phrases"],
    ["FULL_MIMIC_IV_ECG", "bundle branch block|av block|conduction delay", "CD", "pending", "to_audit", "conduction only; avoid rhythm-only ambiguity"],
    ["FULL_MIMIC_IV_ECG", "lvh|rvh|ventricular hypertrophy", "HYP", "pending", "to_audit", "review profile sensitivity"],
    ["KURIAS_ECG", "official normal/ontology code", "NORM", "pending", "to_audit", "use official metadata"],
    ["KURIAS_ECG", "official MI/infarction code", "MI", "pending", "to_audit", "use official metadata"],
    ["KURIAS_ECG", "official ST-T abnormality code", "STTC", "pending", "to_audit", "use official metadata"],
    ["KURIAS_ECG", "official conduction disturbance code", "CD", "pending", "to_audit", "use official metadata"],
    ["KURIAS_ECG", "official hypertrophy code", "HYP", "pending", "to_audit", "use official metadata"],
]

with mapping_template.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerows(rows)

metrics_template.write_text(json.dumps({
    "project": "CardioTwin-AI",
    "version": "v3.2 label-supported external validation",
    "status": "not_run_yet",
    "dataset": None,
    "frozen_model": "artifacts/models/inceptiontime_v21_safety.pt",
    "threshold_profile": "screening",
    "metrics_required": [
        "per_class_AUROC",
        "per_class_AUPRC",
        "per_class_F1",
        "per_class_sensitivity",
        "per_class_specificity",
        "valid_label_macro_AUROC",
        "valid_label_macro_AUPRC",
        "valid_label_macro_F1",
        "valid_label_macro_sensitivity"
    ],
    "support_rule": "Only report AUROC/AUPRC for classes with at least one positive and one negative case.",
    "claim_boundary": "Research-use external validation only. Not final diagnosis."
}, indent=2, ensure_ascii=False), encoding="utf-8")

failure_template.write_text(f"""# CardioTwin-AI v3.2 Failure Case Review Template

Created: {created}

| case_id | dataset | true_labels | predicted_labels | probabilities | thresholds | SQI | region_mapper | error_type | reviewer_note |
|---|---|---|---|---|---|---:|---|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Required review groups

1. False negative MI
2. False positive MI
3. False negative HYP
4. False positive HYP
5. Low-SQI records
6. Region-uncertain cases
7. High-confidence disagreement cases

## Boundary

Research-use safety analysis only. Not clinical diagnosis.
""", encoding="utf-8")

html.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CardioTwin-AI v3.2 Label-supported External Validation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
    pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>CardioTwin-AI v3.2 Label-supported External Validation</h1>
  <div class="warning">Scaffold only. Metrics not run yet. Research-use only.</div>
  <h2>Dataset Plan</h2>
  <pre>{json.dumps(dataset_plan_payload, indent=2, ensure_ascii=False)}</pre>
</body>
</html>
""", encoding="utf-8")

print("DONE: v3.2 label-supported external validation scaffold created")
print("OUT:", OUT)
print("PROTOCOL:", protocol)
print("DATASET_PLAN:", dataset_plan)
print("MAPPING_TEMPLATE:", mapping_template)
print("METRICS_TEMPLATE:", metrics_template)
print("FAILURE_TEMPLATE:", failure_template)
print("HTML:", html)

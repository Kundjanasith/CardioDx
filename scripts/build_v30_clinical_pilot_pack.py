from pathlib import Path
import json
import csv
import zipfile
import hashlib
from datetime import datetime, timezone
from textwrap import dedent

ROOT = Path(".")
ART = ROOT / "artifacts"
RELEASE = ART / "release_rc1"

DIRS = {
    "locked": ART / "locked_external_validation_v30",
    "prospective": ART / "prospective_pilot_v30",
    "human": ART / "human_review_v30",
    "realtime": ART / "realtime_demo_v30",
    "risk": ART / "risk_management_v30",
    "cost": ART / "cost_effectiveness_v30",
    "pitch": ART / "pitch_pack_v30",
    "report": ART / "report_templates_v30",
    "product": ART / "product_readiness_v30",
    "apps": ROOT / "apps",
    "scripts": ROOT / "scripts",
}

for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

created = datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------
# 1) Locked external validation v3.0
# ---------------------------------------------------------------------

write(DIRS["locked"] / "LOCKED_EXTERNAL_VALIDATION_PROTOCOL_v30.md", f"""
# CardioTwin-AI v3.0 Locked External Validation Protocol

Created: `{created}`

## Purpose

This protocol defines a pre-specified locked-model external validation workflow for CardioTwin-AI.

The goal is to evaluate the frozen CardioTwin-AI v2.7 RC1 model on a new external 12-lead ECG dataset without changing model weights, thresholds, label mapping, or evaluation metrics after the protocol is frozen.

## Candidate Datasets

### Primary candidate: MIMIC-IV-ECG

Rationale: large-scale diagnostic 12-lead ECG dataset, 10-second waveform length, 500 Hz sampling, and report linkage when cardiologist reports are available.

### Secondary candidate: KURIAS-ECG

Rationale: standardized 12-lead ECG database with SNOMED CT and OMOP-CDM vocabularies, suitable for ontology-aware label mapping.

## Locked Assets

The following assets must be frozen before evaluation:

1. Model checkpoint
2. Threshold profiles
3. Label mapping
4. Inclusion/exclusion criteria
5. Signal quality criteria
6. Evaluation metrics
7. Failure-case review rules
8. Report template

## Target Labels

The initial target label set follows the CardioTwin-AI v2.7 superclass format:

- NORM
- MI
- STTC
- CD
- HYP

## Evaluation Metrics

Primary metrics:

- AUROC macro
- AUPRC macro
- Macro-F1
- Sensitivity / recall per class
- Specificity per class
- Critical miss rate
- Abstain / uncertain rate

Safety metrics:

- High-confidence false positive rate
- High-confidence false negative rate
- Low-SQI rejection rate
- Number of cases requiring doctor review

## Reporting Boundary

This is not true prospective validation. If using historical public datasets, report as:

**Pre-specified locked external validation**

or:

**Pseudo-prospective locked-model external evaluation**

Do not call it prospective validation unless cases are collected after protocol freeze in a live workflow.
""")

write(DIRS["locked"] / "MIMIC_IV_ECG_DATASET_CARD_v30.md", """
# MIMIC-IV-ECG Dataset Card v3.0

## Role in CardioTwin-AI

Candidate dataset for locked external validation.

## Access

Credentialed PhysioNet access may be required.

## Expected Structure

Place files under:

`data/raw/mimic_iv_ecg/`

Expected content may include waveform records, metadata tables, and report links.

## Intended Use

- External retrospective validation
- Locked-model external test
- Dataset shift analysis
- Report-linked ECG evaluation when reports are available

## Claim Boundary

Do not call this true prospective validation if using historical MIMIC-IV-ECG data.
""")

write(DIRS["locked"] / "KURIAS_ECG_DATASET_CARD_v30.md", """
# KURIAS-ECG Dataset Card v3.0

## Role in CardioTwin-AI

Candidate dataset for standardized external validation using ontology-aware diagnosis mapping.

## Expected Structure

Place files under:

`data/raw/kurias_ecg/`

## Intended Use

- External validation
- SNOMED CT / OMOP-CDM mapping experiment
- Generalization gap analysis
- Diagnosis ontology stress test

## Claim Boundary

Do not mix KURIAS metrics with prospective clinical pilot metrics.
""")

write_csv(
    DIRS["locked"] / "label_mapping_template_v30.csv",
    [
        {"source_dataset": "MIMIC_IV_ECG", "source_code": "", "source_text": "myocardial infarction / infarct / ischemic injury", "target_label": "MI", "mapping_strength": "review_required", "reviewer": "", "notes": ""},
        {"source_dataset": "MIMIC_IV_ECG", "source_code": "", "source_text": "ST-T abnormality / ST depression / T wave abnormality", "target_label": "STTC", "mapping_strength": "review_required", "reviewer": "", "notes": ""},
        {"source_dataset": "MIMIC_IV_ECG", "source_code": "", "source_text": "bundle branch block / AV block / conduction delay", "target_label": "CD", "mapping_strength": "review_required", "reviewer": "", "notes": ""},
        {"source_dataset": "MIMIC_IV_ECG", "source_code": "", "source_text": "left ventricular hypertrophy / right ventricular hypertrophy", "target_label": "HYP", "mapping_strength": "review_required", "reviewer": "", "notes": ""},
        {"source_dataset": "KURIAS_ECG", "source_code": "SNOMED/OMOP", "source_text": "", "target_label": "", "mapping_strength": "review_required", "reviewer": "", "notes": "Fill using standardized diagnosis statement."},
    ],
    ["source_dataset", "source_code", "source_text", "target_label", "mapping_strength", "reviewer", "notes"],
)

write_json(DIRS["locked"] / "locked_model_manifest_v30.json", {
    "version": "locked_model_manifest_v30",
    "created_at_utc": created,
    "frozen_core_release": "CardioTwin-AI v2.7 RC1",
    "model_path_expected": "artifacts/models/inceptiontime_v21_safety.pt",
    "threshold_profile_expected": "artifacts/deep_safety_v21/threshold_profiles_deep.json",
    "claim_boundary": "Locked external validation only. Do not modify model/thresholds after protocol freeze.",
    "status": "template_ready",
})

write(DIRS["locked"] / "failure_case_review_template.md", """
# Failure Case Review Template

## Case ID

`CASE_ID`

## Dataset

`MIMIC-IV-ECG / KURIAS / Other`

## AI Output

- Model:
- Safety profile:
- Positive classes:
- Probabilities:
- Thresholds:
- Uncertain / abstain:

## Reference Interpretation

- Reference label:
- Doctor/report text:
- Adjudicated label:

## Failure Type

- False negative
- False positive
- Low-SQI rejection
- Uncertain but clinically important
- Label mapping ambiguity
- Dataset artifact

## Root Cause Hypothesis

- Signal quality
- Label mapping
- Domain shift
- Missing lead / lead reversal
- Threshold issue
- Model limitation

## Action

- Keep as known limitation
- Update label mapping only
- Add risk warning
- Add to prospective review list
- Do not retrain during locked evaluation
""")

write(DIRS["locked"] / "external_validation_report_template.html", """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CardioTwin-AI v3.0 Locked External Validation Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 36px; line-height: 1.45; }
h1, h2 { color: #1f2937; }
.warning { padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #ddd; padding: 8px; }
th { background: #f3f4f6; }
</style>
</head>
<body>
<h1>CardioTwin-AI v3.0 Locked External Validation Report</h1>
<div class="warning">
Research-use locked external validation. Not prospective clinical deployment and not final diagnosis.
</div>
<h2>Dataset</h2>
<p>Fill after MIMIC-IV-ECG or KURIAS adapter run.</p>
<h2>Locked Assets</h2>
<p>Model, thresholds, label mapping, and metrics must be frozen before evaluation.</p>
<h2>Metrics</h2>
<p>Insert AUROC, AUPRC, Macro-F1, sensitivity, specificity, critical miss rate, and abstain rate.</p>
<h2>Failure Case Review</h2>
<p>Summarize false negatives, false positives, low-SQI cases, and uncertain cases.</p>
</body>
</html>
""")

# ---------------------------------------------------------------------
# 2) Prospective pilot v3.0
# ---------------------------------------------------------------------

write(DIRS["prospective"] / "PROSPECTIVE_PILOT_PROTOCOL_v30.md", """
# CardioTwin-AI v3.0 Prospective Pilot Protocol

## Title

Prospective pilot evaluation of a low-cost, safety-calibrated, multi-scale ECG intelligence platform for preliminary screening, visual explanation, and evidence-based referral support.

## Intended Use in Pilot

CardioTwin-AI is used as a research-stage preliminary ECG screening support tool. It does not provide final diagnosis.

## Study Type

Prospective observational pilot.

## Target Sample Size

Initial pilot: 50–100 de-identified ECG cases.

Expansion: 100–300 cases after workflow stability is confirmed.

## Workflow

1. Acquire 12-lead ECG.
2. Assign anonymized case ID.
3. Run CardioTwin-AI using frozen model and frozen thresholds.
4. Store AI prediction, safety flags, SQI status, and report.
5. Human reviewer independently reviews the ECG.
6. Compare AI output with reference interpretation.
7. Adjudicate disagreement cases.
8. Analyze false negatives, false positives, abstain cases, and uncertain cases.

## Primary Outcomes

- Feasibility of running AI screening on new ECG cases.
- Sensitivity for clinically important abnormal patterns.
- Safety-gate behavior.
- Doctor-review workload.
- Report usability.

## Safety Boundary

Any high-risk, uncertain, or low-SQI case must be reviewed by a qualified clinician or domain expert.

## Exclusion Criteria

- Missing required leads
- Unreadable waveform
- Non-ECG data
- Non-de-identified patient information
- Hardware-acquired ECG without confirmed electrical safety

## Data Privacy

Only anonymized case IDs may be stored. Do not store names, national IDs, phone numbers, addresses, or identifiable notes.
""")

write_csv(
    DIRS["prospective"] / "prospective_case_registry.csv",
    [
        {"case_id": "PILOT-0001", "collection_date": "", "source": "", "age_group": "", "sex": "", "ecg_file_path": "", "sampling_rate_hz": "", "duration_sec": "", "deidentified": "yes", "ai_run_status": "pending", "review_status": "pending", "notes": ""},
    ],
    ["case_id", "collection_date", "source", "age_group", "sex", "ecg_file_path", "sampling_rate_hz", "duration_sec", "deidentified", "ai_run_status", "review_status", "notes"],
)

write_csv(
    DIRS["prospective"] / "doctor_ai_comparison.csv",
    [
        {"case_id": "PILOT-0001", "ai_profile": "screening", "ai_positive_labels": "", "ai_uncertain": "", "ai_low_sqi": "", "doctor_reference_labels": "", "doctor_urgent_flag": "", "agreement": "", "disagreement_type": "", "adjudication": "", "final_notes": ""},
    ],
    ["case_id", "ai_profile", "ai_positive_labels", "ai_uncertain", "ai_low_sqi", "doctor_reference_labels", "doctor_urgent_flag", "agreement", "disagreement_type", "adjudication", "final_notes"],
)

write_json(DIRS["prospective"] / "prospective_metrics_template.json", {
    "n_cases": 0,
    "n_reviewed": 0,
    "sensitivity_by_label": {},
    "specificity_by_label": {},
    "macro_f1": None,
    "critical_miss_rate": None,
    "uncertain_rate": None,
    "low_sqi_rejection_rate": None,
    "doctor_review_required_rate": None,
    "claim_boundary": "Template only. Fill after true prospective pilot data collection."
})

write(DIRS["prospective"] / "false_negative_review.md", """
# False Negative Review

## Purpose

Review all cases where the reference interpretation indicates clinically important abnormality but AI did not flag it.

## Priority Labels

- MI
- STTC
- CD
- HYP
- Any urgent reviewer flag

## Review Table

| case_id | reference_label | ai_probability | ai_threshold | SQI | possible_reason | mitigation |
|---|---|---:|---:|---|---|---|
| | | | | | | |

## Policy

False negatives for MI-like or urgent patterns must be treated as high-priority safety review cases.
""")

write(DIRS["prospective"] / "uncertain_case_review.md", """
# Uncertain Case Review

## Purpose

Analyze cases where the AI abstain/uncertain gate triggered.

## Review Questions

1. Was the ECG low quality?
2. Was the probability near threshold?
3. Was the label mapping ambiguous?
4. Did the reviewer agree that human review was needed?
5. Should this case be flagged as repeat ECG / urgent review / routine review?

## Summary Table

| case_id | uncertain_reason | reviewer_action | final_interpretation | workflow_recommendation |
|---|---|---|---|---|
| | | | | |
""")

write(DIRS["prospective"] / "prospective_validation_report.html", """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Prospective Pilot Report</title></head>
<body>
<h1>CardioTwin-AI v3.0 Prospective Pilot Report</h1>
<p>Status: template ready. Fill after 50–100 new ECG cases are collected and reviewed.</p>
<h2>Primary Metrics</h2>
<ul>
<li>Sensitivity</li>
<li>Specificity</li>
<li>Macro-F1</li>
<li>Critical miss rate</li>
<li>Uncertain/abstain rate</li>
<li>Doctor review rate</li>
</ul>
</body>
</html>
""")

# ---------------------------------------------------------------------
# 3) Human-in-the-loop review system
# ---------------------------------------------------------------------

write(DIRS["human"] / "doctor_in_the_loop_workflow.md", """
# Doctor-in-the-loop Review Workflow v3.0

## Workflow

1. AI generates screening result.
2. AI displays probability, threshold, profile, SQI, uncertainty, and region explanation.
3. Case is routed:
   - Low risk and high SQI: routine review
   - Positive abnormal class: doctor review
   - Uncertain: doctor review
   - Low SQI: repeat ECG or review
   - MI-like/high-risk: urgent review
4. Reviewer records agreement, override, and final interpretation.
5. Disagreements are adjudicated.

## Reviewer Actions

- Agree with AI
- Override AI
- Mark uncertain
- Request repeat ECG
- Urgent referral
- Routine follow-up
""")

write_csv(
    DIRS["human"] / "reviewer_decisions.csv",
    [
        {"case_id": "PILOT-0001", "reviewer_id": "", "review_datetime": "", "ai_positive_labels": "", "reviewer_labels": "", "reviewer_action": "", "override": "", "urgent_referral": "", "repeat_ecg": "", "comments": ""},
    ],
    ["case_id", "reviewer_id", "review_datetime", "ai_positive_labels", "reviewer_labels", "reviewer_action", "override", "urgent_referral", "repeat_ecg", "comments"],
)

write_csv(
    DIRS["human"] / "ai_vs_reviewer_disagreement.csv",
    [
        {"case_id": "PILOT-0001", "ai_label": "", "reviewer_label": "", "disagreement_type": "", "severity": "", "root_cause": "", "adjudication_result": "", "mitigation": ""},
    ],
    ["case_id", "ai_label", "reviewer_label", "disagreement_type", "severity", "root_cause", "adjudication_result", "mitigation"],
)

write(DIRS["human"] / "adjudication_summary.md", """
# Adjudication Summary

## Purpose

Summarize cases where AI and reviewer disagreed.

## Categories

- AI false positive
- AI false negative
- Reviewer uncertainty
- Label mapping ambiguity
- Signal quality issue
- Clinically acceptable difference

## Summary

| category | count | example_case | mitigation |
|---|---:|---|---|
| | | | |
""")

# ---------------------------------------------------------------------
# 4) Real-time replay dashboard app
# ---------------------------------------------------------------------

write(DIRS["apps"] / "streamlit_realtime_replay_v30.py", r"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_READY = True
except Exception:
    PLOTLY_READY = False

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

st.set_page_config(page_title="CardioTwin-AI v3.0 Real-time ECG Replay", layout="wide")

st.title("CardioTwin-AI v3.0 Real-time ECG Replay")
st.caption("Research-use replay dashboard. This is not live clinical diagnosis.")

def synthetic_ecg(fs=500, seconds=10):
    t = np.arange(0, seconds, 1 / fs)
    data = {}
    for i, lead in enumerate(LEADS):
        base = 0.08 * np.sin(2 * np.pi * 1.2 * t + i * 0.1)
        qrs = np.zeros_like(t)
        for beat in np.arange(0.6, seconds, 0.85):
            qrs += 0.8 * np.exp(-0.5 * ((t - beat) / 0.025) ** 2)
            qrs -= 0.25 * np.exp(-0.5 * ((t - beat + 0.035) / 0.015) ** 2)
        data[lead] = (1 - i * 0.03) * qrs + base + 0.01 * np.random.randn(len(t))
    df = pd.DataFrame(data)
    df.insert(0, "time_sec", t)
    return df

def load_uploaded_csv(file):
    df = pd.read_csv(file)
    missing = [c for c in LEADS if c not in df.columns]
    if missing:
        st.warning(f"Uploaded CSV missing leads: {missing}. Using synthetic demo ECG instead.")
        return synthetic_ecg()
    if "time_sec" not in df.columns:
        df.insert(0, "time_sec", np.arange(len(df)) / 500.0)
    return df[["time_sec"] + LEADS]

def plot_window(df, start_idx, end_idx):
    win = df.iloc[start_idx:end_idx]
    if PLOTLY_READY:
        fig = go.Figure()
        offset = 0
        for lead in LEADS:
            fig.add_trace(go.Scatter(
                x=win["time_sec"],
                y=win[lead] + offset,
                mode="lines",
                name=lead,
                line=dict(width=1),
            ))
            offset += 1.5
        fig.update_layout(
            height=700,
            title="Scrolling 12-lead ECG replay",
            xaxis_title="Time (s)",
            yaxis_title="Lead offset",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(win.set_index("time_sec")[LEADS])

def pseudo_ai_panel(df_window):
    # Placeholder safety demo: replace with CardioTwin inference call when connected.
    amp = float(np.nanmean(np.abs(df_window[LEADS].values)))
    sqi = max(0.0, min(1.0, 1.0 - abs(amp - 0.35)))
    uncertain = sqi < 0.55
    possible_abnormal = amp > 0.45

    return {
        "signal_quality_index": sqi,
        "possible_abnormal_pattern": possible_abnormal,
        "uncertain_or_review_required": uncertain or possible_abnormal,
        "recommendation": "Doctor review / repeat ECG if low SQI" if uncertain else ("Doctor review" if possible_abnormal else "Routine review"),
    }

uploaded = st.sidebar.file_uploader("Optional: upload 12-lead ECG CSV", type=["csv"])
fs = st.sidebar.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=500, step=50)
window_sec = st.sidebar.slider("Replay window seconds", 2, 10, 6)
step_sec = st.sidebar.slider("Step seconds", 1, 5, 1)
max_steps = st.sidebar.slider("Max replay steps", 5, 60, 20)
speed = st.sidebar.slider("Replay delay seconds", 0.0, 2.0, 0.3, 0.1)

if uploaded:
    df = load_uploaded_csv(uploaded)
else:
    df = synthetic_ecg(fs=fs, seconds=20)

st.sidebar.write(f"Rows: {len(df)}")
st.sidebar.write(f"Duration: {df['time_sec'].max():.2f} sec")

run = st.sidebar.button("Start replay")

plot_slot = st.empty()
panel_slot = st.empty()

if not run:
    with plot_slot.container():
        plot_window(df, 0, min(len(df), int(window_sec * fs)))
    with panel_slot.container():
        st.info("Press Start replay to simulate real-time ECG streaming.")
else:
    n_window = int(window_sec * fs)
    n_step = int(step_sec * fs)

    for k, start in enumerate(range(0, max(1, len(df) - n_window), n_step)):
        if k >= max_steps:
            break
        end = min(len(df), start + n_window)
        win = df.iloc[start:end]

        with plot_slot.container():
            plot_window(df, start, end)

        result = pseudo_ai_panel(win)
        with panel_slot.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SQI", f"{result['signal_quality_index']:.3f}")
            c2.metric("Possible abnormal", str(result["possible_abnormal_pattern"]))
            c3.metric("Review required", str(result["uncertain_or_review_required"]))
            c4.metric("Recommendation", result["recommendation"])
            st.caption("Pseudo-AI panel for replay demonstration. Connect to CardioTwin inference for production research runs.")

        time.sleep(speed)
""")

write(DIRS["realtime"] / "demo_video_script.md", """
# 90-second Real-time Replay Demo Script

## 0–15 sec

Introduce CardioTwin-AI as a low-cost, safety-calibrated, multi-scale ECG intelligence platform.

## 15–35 sec

Show scrolling 12-lead ECG replay. Explain that this simulates live acquisition without connecting to a human subject.

## 35–55 sec

Show live signal quality index, review flag, and recommendation panel.

## 55–75 sec

Explain how this connects to frozen v2.7 AI inference, 3D/4D explanation, and exportable reports.

## 75–90 sec

Close with claim boundary: preliminary screening and referral support, not final diagnosis.
""")

write_json(DIRS["realtime"] / "latency_report_template.json", {
    "version": "realtime_latency_template_v30",
    "ecg_input": "file replay / hardware stream",
    "buffer_seconds": 10,
    "target_dashboard_response_ms": 500,
    "target_inference_ms": 50,
    "measured": [],
    "claim_boundary": "Template only until actual replay/hardware run is measured."
})

write(DIRS["realtime"] / "demo_cases/README.md", """
# Demo Cases

Place de-identified ECG demo cases here.

Recommended formats:

- WFDB `.hea/.mat`
- CSV with columns: time_sec, I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

Do not store identifiable patient information.
""")

# ---------------------------------------------------------------------
# 5) Risk management
# ---------------------------------------------------------------------

write_csv(
    DIRS["risk"] / "risk_register.csv",
    [
        {"hazard": "False negative MI", "cause": "Model misses MI-like pattern or threshold too high", "possible_harm": "Delayed referral", "severity": "high", "mitigation": "Screening threshold, uncertainty gate, doctor review, urgent referral policy", "residual_risk": "medium", "verification_evidence": "False negative review + prospective pilot"},
        {"hazard": "False positive abnormality", "cause": "Noise, domain shift, label mismatch", "possible_harm": "Unnecessary anxiety/review", "severity": "medium", "mitigation": "Doctor-in-loop review, report wording as screening positive, not diagnosis", "residual_risk": "low-medium", "verification_evidence": "Specificity + FP review"},
        {"hazard": "Low signal quality", "cause": "Lead-off, noise, baseline wander", "possible_harm": "Incorrect AI output", "severity": "high", "mitigation": "SQI gate, repeat ECG recommendation, low-SQI rejection", "residual_risk": "medium", "verification_evidence": "SQI rejection metrics"},
        {"hazard": "Overclaiming diagnosis", "cause": "User interprets AI output as final diagnosis", "possible_harm": "Clinical misuse", "severity": "high", "mitigation": "Use warning card, intended-use statement, report language", "residual_risk": "medium", "verification_evidence": "Report template review"},
        {"hazard": "Privacy breach", "cause": "Identifiable ECG metadata stored", "possible_harm": "Patient privacy harm", "severity": "high", "mitigation": "Anonymized case ID, no identifiers, access control", "residual_risk": "medium", "verification_evidence": "Case registry audit"},
    ],
    ["hazard", "cause", "possible_harm", "severity", "mitigation", "residual_risk", "verification_evidence"],
)

write(DIRS["risk"] / "false_negative_policy.md", """
# False Negative Policy

## Critical Labels

- MI
- STTC with urgent reviewer concern
- CD with urgent reviewer concern
- Any case flagged urgent by reviewer

## Policy

Any clinically important false negative must trigger:

1. Immediate failure case review.
2. Check SQI and lead completeness.
3. Check threshold and probability.
4. Check reference interpretation.
5. Add to risk register if not already covered.
6. Do not retrain during locked/prospective evaluation.
""")

write(DIRS["risk"] / "use_warning_card.md", """
# Use Warning Card

CardioTwin-AI is a research-stage preliminary ECG screening and visual explanation tool.

It is not a final diagnostic system.

Do not use CardioTwin-AI output to delay urgent clinical care.

High-risk, uncertain, or low-signal-quality cases require qualified human review.
""")

write(DIRS["risk"] / "intended_use_statement.md", """
# Intended Use Statement v3.0

CardioTwin-AI is intended for research-use preliminary screening support of 12-lead ECG records. It provides safety-calibrated AI predictions, uncertainty flags, visual explanation, and report export to support evidence-based review and referral decisions.

It is not intended to provide final diagnosis or replace qualified clinical interpretation.
""")

write(DIRS["risk"] / "contraindication_statement.md", """
# Contraindication Statement

Do not use CardioTwin-AI as the sole basis for diagnosis, treatment, emergency triage, or ruling out acute cardiac events.

Do not use hardware-connected ECG acquisition on humans without electrical safety review, isolation, ethics approval where required, and qualified supervision.
""")

# ---------------------------------------------------------------------
# 6) Cost-effectiveness
# ---------------------------------------------------------------------

write_csv(
    DIRS["cost"] / "bill_of_materials.csv",
    [
        {"item": "Laptop / mini PC", "role": "AI dashboard and inference", "unit_cost_thb_low": "", "unit_cost_thb_high": "", "notes": "Use existing device if available."},
        {"item": "ECG acquisition hardware", "role": "Future real-time ECG input", "unit_cost_thb_low": "", "unit_cost_thb_high": "", "notes": "Must confirm electrical safety before human use."},
        {"item": "Electrodes/leads", "role": "Signal acquisition", "unit_cost_thb_low": "", "unit_cost_thb_high": "", "notes": "Consumable; estimate per case separately."},
        {"item": "Cloud/storage optional", "role": "Backup and evidence export", "unit_cost_thb_low": "", "unit_cost_thb_high": "", "notes": "Avoid storing identifiers."},
    ],
    ["item", "role", "unit_cost_thb_low", "unit_cost_thb_high", "notes"],
)

write(DIRS["cost"] / "standard_workflow_vs_ai_workflow.md", """
# Standard Workflow vs AI-supported Workflow

## Standard Workflow

1. ECG acquired.
2. Expert reviews ECG.
3. Report generated.
4. Referral decision made.

## AI-supported Preliminary Screening Workflow

1. ECG acquired.
2. CardioTwin-AI generates screening result, SQI, uncertainty flag, and visual explanation.
3. High-risk/uncertain/low-SQI cases are routed for review.
4. Human reviewer confirms or overrides.
5. Evidence report is exported.

## Correct Cost Claim

CardioTwin-AI may reduce screening workload and support triage in low-resource settings.

## Claims to Avoid

Do not claim replacement of cardiologists, echocardiography, CT, MRI, or emergency clinical judgment.
""")

write_json(DIRS["cost"] / "cost_assumptions_template.json", {
    "hardware_cost_thb": None,
    "software_cost_thb": 0,
    "consumable_cost_per_case_thb": None,
    "staff_time_standard_min": None,
    "staff_time_ai_supported_min": None,
    "report_generation_time_sec": None,
    "abstain_rate": None,
    "doctor_review_rate": None,
    "notes": "Fill with real pilot data before making cost-effectiveness claims."
})

write(DIRS["cost"] / "cost_effectiveness_report.html", """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Cost-effectiveness Report</title></head>
<body>
<h1>CardioTwin-AI v3.0 Cost-effectiveness Report</h1>
<p>Status: template ready.</p>
<p>Use this report to compare preliminary screening workflow time, cost per case, doctor-review rate, and report generation time.</p>
<p>Do not claim replacement of standard diagnostic methods.</p>
</body>
</html>
""")

# Try to create xlsx if openpyxl is available.
try:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Cost model"
    ws.append(["parameter", "value", "notes"])
    ws.append(["hardware_cost_thb", "", "Fill after choosing hardware"])
    ws.append(["consumable_cost_per_case_thb", "", "Electrodes/leads"])
    ws.append(["standard_workflow_minutes", "", "Measured in pilot"])
    ws.append(["ai_supported_workflow_minutes", "", "Measured in pilot"])
    ws.append(["doctor_review_rate", "", "Measured in pilot"])
    wb.save(DIRS["cost"] / "cost_per_screening_case.xlsx")
except Exception:
    write(DIRS["cost"] / "cost_per_screening_case_xlsx_NOTE.md", """
# XLSX Not Created

Install openpyxl if XLSX output is needed:

`python -m pip install openpyxl`
""")

# ---------------------------------------------------------------------
# 7) Clinical report templates
# ---------------------------------------------------------------------

write(DIRS["report"] / "clinical_case_report_template.html", """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CardioTwin-AI Clinical Case Report Template</title>
<style>
body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }
.warning { padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #ddd; padding: 8px; }
</style>
</head>
<body>
<h1>CardioTwin-AI Screening Report</h1>
<div class="warning">Research-use preliminary screening. Not final diagnosis.</div>
<h2>Case Information</h2>
<table>
<tr><th>Case ID</th><td>{{case_id}}</td></tr>
<tr><th>Signal Quality</th><td>{{sqi}}</td></tr>
<tr><th>Safety Profile</th><td>{{profile}}</td></tr>
</table>
<h2>AI Result</h2>
<p>{{ai_result}}</p>
<h2>Region Explanation</h2>
<p>{{region_explanation}}</p>
<h2>Recommendation</h2>
<p>{{recommendation}}</p>
<h2>Limitations</h2>
<p>AI output must be reviewed by qualified human reviewer, especially for high-risk, uncertain, or low-SQI cases.</p>
</body>
</html>
""")

write(DIRS["report"] / "screening_report_template.md", """
# CardioTwin-AI Screening Report Template

## Case

- Case ID:
- Date:
- Source:
- De-identified: yes/no

## Signal Quality

- SQI:
- Missing leads:
- Low-SQI flag:

## AI Screening Output

- Safety profile:
- Positive classes:
- Probabilities:
- Thresholds:
- Uncertain/abstain:

## Region Explanation

- Region:
- Reason:
- Confidence:

## Recommendation

- Routine review
- Doctor review
- Repeat ECG
- Urgent referral

## Boundary

This is preliminary screening support, not final diagnosis.
""")

write(DIRS["report"] / "doctor_review_form.md", """
# Doctor Review Form

## Case ID

## Reviewer

## AI Result Reviewed

- Yes / No

## Reviewer Interpretation

- NORM
- MI
- STTC
- CD
- HYP
- Other:

## Reviewer Action

- Agree with AI
- Override AI
- Repeat ECG
- Urgent referral
- Routine follow-up

## Notes
""")

# ---------------------------------------------------------------------
# 8) Product readiness / robustness templates
# ---------------------------------------------------------------------

write_csv(
    DIRS["product"] / "robustness_test_matrix.csv",
    [
        {"test": "baseline_wander", "description": "Add low-frequency drift", "metric": "performance_drop", "status": "planned"},
        {"test": "muscle_artifact_noise", "description": "Add high-frequency noise", "metric": "performance_drop", "status": "planned"},
        {"test": "missing_lead", "description": "Zero/mask one or more leads", "metric": "critical_miss_rate", "status": "planned"},
        {"test": "lead_reversal", "description": "Swap limb leads", "metric": "detection_or_performance_drop", "status": "planned"},
        {"test": "sampling_rate_mismatch", "description": "Run 100/250/500 Hz variants", "metric": "preprocessing_stability", "status": "planned"},
    ],
    ["test", "description", "metric", "status"],
)

write(DIRS["product"] / "data_privacy_checklist.md", """
# Data Privacy Checklist

- Use anonymized case IDs only.
- Do not store names, national IDs, phone numbers, addresses, or identifiable notes.
- Keep raw ECG data in controlled folder.
- Track who can access pilot data.
- Use separate mapping file if a clinical collaborator needs re-identification.
- Do not include identifiable information in exported reports.
""")

write(DIRS["product"] / "hardware_poc_safety_checklist.md", """
# Hardware POC Safety Checklist

Before connecting ECG hardware to any human subject:

1. Confirm electrical isolation.
2. Use battery power where appropriate.
3. Avoid unsafe mains-connected circuits.
4. Review electrode safety.
5. Confirm qualified supervision.
6. Obtain ethics/permission if collecting human data.
7. Start with simulator/replay mode before human acquisition.
""")

# ---------------------------------------------------------------------
# 9) Pitch / paper / demo pack
# ---------------------------------------------------------------------

write(DIRS["pitch"] / "executive_one_page.md", """
# CardioTwin-AI v3.0 Executive One-page Summary

## One-line Positioning

CardioTwin-AI is a low-cost, safety-calibrated, multi-scale ECG intelligence platform for preliminary screening, visual explanation, and evidence-based referral support.

## What It Does

- Analyzes 12-lead ECG records.
- Provides safety-calibrated AI predictions.
- Flags uncertain and low-quality cases.
- Visualizes region-level explanation with 3D/4D CardioTwin.
- Exports audit-ready reports.
- Adds BeatScope beat-level morphology benchmark evidence.

## Why It Matters

Many screening settings lack immediate access to expert ECG interpretation. CardioTwin-AI supports early review and referral decisions without claiming to replace clinicians.

## Current Evidence

- Frozen v2.7 12-lead record-level release.
- BeatScope v2.8 beat-level full add-on.
- External validation and stress-test artifacts.
- Research addendum and claim boundary documentation.

## Next Step

Run v3.0 locked external validation and 50–100 case prospective pilot.
""")

write(DIRS["pitch"] / "demo_script_90sec.md", """
# 90-second Demo Script

## 0–15 sec

CardioTwin-AI is a low-cost, safety-calibrated, multi-scale ECG intelligence platform for preliminary screening and referral support.

## 15–35 sec

Show 12-lead ECG dashboard and AI prediction with thresholds and uncertainty.

## 35–55 sec

Show 3D/4D visual explanation and region mapper.

## 55–70 sec

Show exportable case report and doctor-review workflow.

## 70–85 sec

Show BeatScope v2.8 benchmark: beat-level morphology intelligence.

## 85–90 sec

Close: research-use support tool, not final diagnosis; next step is prospective pilot.
""")

write(DIRS["pitch"] / "judge_qna.md", """
# Judge Q&A

## Is this replacing doctors?

No. It is preliminary screening and referral support. High-risk, uncertain, or low-SQI cases require human review.

## Why is it low-cost?

It uses software-based analysis, standard 12-lead ECG data, and commodity computing. Hardware cost analysis is handled separately and must be validated in pilot.

## What makes it different?

It combines record-level 12-lead screening, beat-level morphology benchmarking, safety calibration, 3D/4D visual explanation, and audit-ready evidence export.

## Is it clinically validated?

Not yet for deployment. It has public external validation and is now prepared for locked external validation and prospective pilot.
""")

write(DIRS["pitch"] / "paper_abstract_draft.md", """
# Paper Abstract Draft

CardioTwin-AI is a research-stage, safety-calibrated, multi-scale ECG intelligence platform designed for preliminary 12-lead ECG screening, visual explanation, and evidence-based referral support. The system combines frozen record-level 12-lead ECG inference, uncertainty-aware safety gating, 3D/4D visual explanation, exportable case reports, and BeatScope v2.8 beat-level morphology benchmarking. This v3.0 protocol package defines locked external validation on additional public ECG datasets and a prospective 50–100 case pilot workflow with doctor-in-the-loop review, risk management, cost-effectiveness templates, and claim-boundary documentation. The platform is intended for research-use screening support and does not provide final diagnosis.
""")

write(DIRS["pitch"] / "contribution_bullets.md", """
# Contribution Bullets

1. Safety-calibrated 12-lead ECG screening pipeline.
2. 3D/4D visual explanation with region-level uncertainty handling.
3. BeatScope v2.8 beat-level morphology benchmark add-on.
4. Locked external validation protocol for MIMIC-IV-ECG/KURIAS-style datasets.
5. Prospective pilot workflow with doctor-in-the-loop review.
6. Risk, cost, privacy, and claim-boundary documentation for real-world pilot readiness.
""")

write(DIRS["pitch"] / "figure_caption_bank.md", """
# Figure Caption Bank

## System Architecture

Overview of the CardioTwin-AI v3.0 workflow, connecting 12-lead ECG input, safety-calibrated AI inference, signal-quality gating, 3D/4D visual explanation, doctor-in-the-loop review, and evidence report export.

## Locked External Validation

Pre-specified locked evaluation pipeline for MIMIC-IV-ECG or KURIAS-ECG, including fixed model assets, label mapping, metrics, and failure-case review.

## Real-time Replay Demo

Research-use replay dashboard simulating live ECG streaming and AI-assisted screening without connecting hardware to a human subject.

## Doctor-in-the-loop Workflow

Review and adjudication pathway for high-risk, uncertain, and low-signal-quality ECG cases.
""")

# ---------------------------------------------------------------------
# 10) Readiness checker script for MIMIC/KURIAS
# ---------------------------------------------------------------------

write(DIRS["scripts"] / "run_locked_external_readiness_v30.py", r"""
from pathlib import Path
import json
from datetime import datetime, timezone

OUT = Path("artifacts/locked_external_validation_v30")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "MIMIC_IV_ECG": Path("data/raw/mimic_iv_ecg"),
    "KURIAS_ECG": Path("data/raw/kurias_ecg"),
}

def count_files(root: Path, suffixes):
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)

report = {
    "version": "locked_external_readiness_v30",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "datasets": {},
    "recommendation": "",
}

for name, root in DATASETS.items():
    info = {
        "path": str(root),
        "exists": root.exists(),
        "hea_count": count_files(root, {".hea"}),
        "mat_count": count_files(root, {".mat"}),
        "csv_count": count_files(root, {".csv"}),
        "dat_count": count_files(root, {".dat"}),
        "ready_for_adapter": False,
        "notes": "",
    }
    if info["exists"] and (info["hea_count"] > 0 or info["csv_count"] > 0 or info["dat_count"] > 0):
        info["ready_for_adapter"] = True
        info["notes"] = "Dataset files detected. Next step: implement dataset-specific waveform/report adapter and label mapping."
    else:
        info["notes"] = "Dataset not found or no recognizable files detected."
    report["datasets"][name] = info

if report["datasets"]["MIMIC_IV_ECG"]["ready_for_adapter"]:
    report["recommendation"] = "Proceed with MIMIC-IV-ECG adapter and locked label mapping."
elif report["datasets"]["KURIAS_ECG"]["ready_for_adapter"]:
    report["recommendation"] = "Proceed with KURIAS-ECG adapter and ontology mapping."
else:
    report["recommendation"] = "Place MIMIC-IV-ECG or KURIAS-ECG files under data/raw before running locked external validation."

(OUT / "locked_external_dataset_readiness_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(report, indent=2, ensure_ascii=False))
""")

# ---------------------------------------------------------------------
# 11) Package v3.0 pack
# ---------------------------------------------------------------------

pack_files = []
for folder in [
    DIRS["locked"], DIRS["prospective"], DIRS["human"], DIRS["realtime"],
    DIRS["risk"], DIRS["cost"], DIRS["pitch"], DIRS["report"], DIRS["product"],
]:
    for p in folder.rglob("*"):
        if p.is_file():
            pack_files.append(p)

pack_files.append(DIRS["apps"] / "streamlit_realtime_replay_v30.py")
pack_files.append(DIRS["scripts"] / "run_locked_external_readiness_v30.py")

manifest = {
    "project": "CardioTwin-AI",
    "version": "v3.0 Clinical Pilot & Real-world Validation Pack",
    "created_at_utc": created,
    "positioning": "low-cost, safety-calibrated, multi-scale ECG intelligence platform for preliminary screening, visual explanation, and evidence-based referral support",
    "relationship_to_previous_releases": {
        "v2.7": "frozen 12-lead record-level ECG AI release",
        "v2.8": "frozen BeatScope beat-level benchmark add-on",
        "v3.0": "clinical pilot readiness, locked external validation, real-time replay, risk/cost/workflow documentation",
    },
    "claim_boundary": "Research-use preliminary screening support. Not final diagnosis and not clinical deployment.",
    "files_indexed": len(pack_files),
    "files": [],
}

for p in sorted(pack_files):
    manifest["files"].append({
        "path": p.as_posix(),
        "size_bytes": int(p.stat().st_size),
        "sha256": sha256_file(p),
    })

write_json(ART / "v30_clinical_pilot_pack_manifest.json", manifest)

zip_path = RELEASE / "cardiotwin_v3_0_clinical_pilot_pack.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(pack_files):
        z.write(p, p.as_posix())
    z.write(ART / "v30_clinical_pilot_pack_manifest.json", "artifacts/v30_clinical_pilot_pack_manifest.json")

print("DONE: CardioTwin-AI v3.0 Clinical Pilot Pack created")
print("ZIP:", zip_path)
print("ZIP size MB:", f"{zip_path.stat().st_size / 1024 / 1024:.2f}")
print("Manifest:", ART / "v30_clinical_pilot_pack_manifest.json")
print("files_indexed:", manifest["files_indexed"])
print("Realtime app:", DIRS["apps"] / "streamlit_realtime_replay_v30.py")
print("Readiness script:", DIRS["scripts"] / "run_locked_external_readiness_v30.py")

from pathlib import Path
import json
import csv
from datetime import datetime, timezone

RELEASE = Path("artifacts/release_rc1")
MANIFEST = RELEASE / "cardiotwin_beatscope_v2_8_full_manifest.json"

ADDENDUM = RELEASE / "BEATSCOPE_V28_RESEARCH_ADDENDUM.md"
RESULTS = RELEASE / "BEATSCOPE_V28_RESULTS_TABLE.csv"
EXEC_SUMMARY = RELEASE / "BEATSCOPE_V28_EXECUTIVE_SUMMARY.md"
CLAIM_BOUNDARY = RELEASE / "BEATSCOPE_V28_CLAIM_BOUNDARY.md"

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def fmt(x, digits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def pct(x, digits=2):
    try:
        if x is None:
            return "NA"
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return str(x)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

m = read_json(MANIFEST)
h = m.get("headline_results", {})
run = m.get("run_summary", {})
transfer = run.get("transfer_summary", {})
scratch = transfer.get("scratch", {})
trans = transfer.get("transfer", {})
gains = transfer.get("gains", {})

created = datetime.now(timezone.utc).isoformat()

mit_model = h.get("mitbih_best_model")
mit_f1 = h.get("mitbih_macro_f1")
mit_auroc = h.get("mitbih_auroc_macro")
mit_auprc = h.get("mitbih_auprc_macro")

ptb_model = h.get("ptbdb_best_model")
ptb_f1 = h.get("ptbdb_macro_f1")
ptb_auroc = h.get("ptbdb_auroc_macro")
ptb_auprc = h.get("ptbdb_auprc_macro")

ba_gain = gains.get("balanced_accuracy_gain")
f1_gain = gains.get("macro_f1_gain")
auroc_gain = gains.get("auroc_macro_gain")
auprc_gain = gains.get("auprc_macro_gain")

# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------
rows = [
    {
        "section": "MIT-BIH beat-level benchmark",
        "task": "MIT-BIH Arrhythmia 5-class",
        "model": mit_model,
        "metric": "Macro-F1",
        "value": mit_f1,
        "interpretation": "Best beat-level arrhythmia morphology classifier by Macro-F1.",
    },
    {
        "section": "MIT-BIH beat-level benchmark",
        "task": "MIT-BIH Arrhythmia 5-class",
        "model": mit_model,
        "metric": "AUROC macro",
        "value": mit_auroc,
        "interpretation": "Strong discrimination across beat-level arrhythmia classes.",
    },
    {
        "section": "MIT-BIH beat-level benchmark",
        "task": "MIT-BIH Arrhythmia 5-class",
        "model": mit_model,
        "metric": "AUPRC macro",
        "value": mit_auprc,
        "interpretation": "Macro precision-recall performance under class imbalance.",
    },
    {
        "section": "PTBDB beat-level benchmark",
        "task": "PTBDB Normal vs Abnormal",
        "model": ptb_model,
        "metric": "Macro-F1",
        "value": ptb_f1,
        "interpretation": "Best binary beat-level normal/abnormal classifier by Macro-F1.",
    },
    {
        "section": "PTBDB beat-level benchmark",
        "task": "PTBDB Normal vs Abnormal",
        "model": ptb_model,
        "metric": "AUROC macro",
        "value": ptb_auroc,
        "interpretation": "Strong binary beat-level discrimination.",
    },
    {
        "section": "PTBDB beat-level benchmark",
        "task": "PTBDB Normal vs Abnormal",
        "model": ptb_model,
        "metric": "AUPRC macro",
        "value": ptb_auprc,
        "interpretation": "High precision-recall performance for abnormal heartbeat detection.",
    },
    {
        "section": "Transfer learning",
        "task": "MIT-BIH pretrain to PTBDB fine-tune",
        "model": "Inception1D transfer vs scratch",
        "metric": "Balanced accuracy gain",
        "value": ba_gain,
        "interpretation": "Transfer slightly improved balanced accuracy.",
    },
    {
        "section": "Transfer learning",
        "task": "MIT-BIH pretrain to PTBDB fine-tune",
        "model": "Inception1D transfer vs scratch",
        "metric": "Macro-F1 gain",
        "value": f1_gain,
        "interpretation": "Transfer did not improve Macro-F1; report honestly.",
    },
    {
        "section": "Transfer learning",
        "task": "MIT-BIH pretrain to PTBDB fine-tune",
        "model": "Inception1D transfer vs scratch",
        "metric": "AUROC gain",
        "value": auroc_gain,
        "interpretation": "Transfer slightly improved AUROC.",
    },
    {
        "section": "Transfer learning",
        "task": "MIT-BIH pretrain to PTBDB fine-tune",
        "model": "Inception1D transfer vs scratch",
        "metric": "AUPRC gain",
        "value": auprc_gain,
        "interpretation": "Transfer slightly improved AUPRC.",
    },
]

with RESULTS.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["section", "task", "model", "metric", "value", "interpretation"],
    )
    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------
exec_text = f"""# BeatScope v2.8 Executive Summary

Created: `{created}`

## One-line Summary

**BeatScope v2.8 extends CardioTwin-AI into a multi-scale ECG intelligence platform by adding a beat-level auxiliary benchmark for heartbeat morphology classification and transfer-learning evaluation.**

## Relationship to CardioTwin-AI v2.7

BeatScope v2.8 is an **add-on**, not a replacement for CardioTwin-AI v2.7.

- **CardioTwin-AI v2.7 RC1**: 12-lead record-level ECG screening, safety calibration, 3D/4D visual explanation, external validation, and case-report export.
- **BeatScope v2.8**: beat-level segmented ECG benchmark using MIT-BIH and PTBDB-derived heartbeat vectors.

The two result sets must be reported separately.

## Headline Results

| Benchmark | Best model | Macro-F1 | AUROC | AUPRC |
|---|---|---:|---:|---:|
| MIT-BIH 5-class beat-level | {mit_model} | {fmt(mit_f1)} | {fmt(mit_auroc)} | {fmt(mit_auprc)} |
| PTBDB binary beat-level | {ptb_model} | {fmt(ptb_f1)} | {fmt(ptb_auroc)} | {fmt(ptb_auprc)} |

## Transfer Learning Result

MIT-BIH pretraining followed by PTBDB fine-tuning showed **mixed transfer behavior**:

| Metric | Gain |
|---|---:|
| Balanced accuracy | {fmt(ba_gain)} |
| Macro-F1 | {fmt(f1_gain)} |
| AUROC | {fmt(auroc_gain)} |
| AUPRC | {fmt(auprc_gain)} |

Interpretation: MIT-BIH pretraining slightly improved balanced accuracy, AUROC, and AUPRC, but did not improve Macro-F1 over PTBDB training from scratch.

## Competition Framing

BeatScope v2.8 allows the project to be described as a **multi-scale ECG intelligence platform**:

1. Record-level 12-lead screening from CardioTwin-AI v2.7.
2. Beat-level morphology classification from BeatScope v2.8.
3. Safety-calibrated prediction and uncertainty-aware reporting.
4. 3D/4D visual explanation and exportable evidence packs.

## Claim Boundary

BeatScope is a beat-level auxiliary benchmark. It must not be mixed with the 12-lead record-level validation metrics from CardioTwin-AI v2.7.
"""

EXEC_SUMMARY.write_text(exec_text, encoding="utf-8")

# ---------------------------------------------------------------------
# Full research addendum
# ---------------------------------------------------------------------
addendum = f"""# BeatScope v2.8 Research Addendum

Created: `{created}`

## 1. Purpose

BeatScope v2.8 was created as an auxiliary beat-level benchmark branch for CardioTwin-AI.

The purpose is to evaluate whether the broader CardioTwin-AI project can support multi-scale ECG intelligence:

- **beat-level morphology classification**
- **binary normal/abnormal heartbeat screening**
- **transfer learning from arrhythmia morphology to abnormal heartbeat detection**
- **separate, claim-safe reporting from 12-lead record-level validation**

BeatScope does not modify the frozen CardioTwin-AI v2.7 RC1 release.

## 2. Why BeatScope Is Separate from v2.7

CardioTwin-AI v2.7 is a 12-lead record-level system. It works with 10-second ECG records and predicts PTB-XL-style superclasses such as NORM, MI, STTC, CD, and HYP.

BeatScope v2.8 is different. It uses segmented, preprocessed heartbeat vectors. Each example represents a single heartbeat segment, not a full 12-lead ECG record.

Therefore, BeatScope metrics are not merged with v2.7 metrics. They are reported as auxiliary beat-level benchmark evidence.

## 3. Dataset Scope

BeatScope v2.8 uses the Kaggle ECG Heartbeat Categorization dataset files:

- `mitbih_train.csv`
- `mitbih_test.csv`
- `ptbdb_normal.csv`
- `ptbdb_abnormal.csv`

The MIT-BIH benchmark is treated as a 5-class beat-level arrhythmia morphology task.

The PTBDB benchmark is treated as a binary beat-level normal/abnormal classification task.

## 4. Model Families

BeatScope v2.8 evaluates four model families:

1. Logistic Regression
2. Random Forest
3. CNN1D
4. Inception1D

The deep models were trained using CPU in this run. The output package includes model artifacts, confusion matrices, per-class reports, leaderboards, and a transfer-learning report.

## 5. MIT-BIH Beat-Level Benchmark Result

Best MIT-BIH model by Macro-F1:

| Model | Macro-F1 | AUROC macro | AUPRC macro |
|---|---:|---:|---:|
| {mit_model} | {fmt(mit_f1)} | {fmt(mit_auroc)} | {fmt(mit_auprc)} |

Interpretation:

The MIT-BIH result shows that BeatScope can perform high-quality beat-level arrhythmia morphology classification. Random Forest performed best in this full run, suggesting that strong classical models remain competitive on preprocessed beat-level vectors.

## 6. PTBDB Binary Beat-Level Benchmark Result

Best PTBDB model by Macro-F1:

| Model | Macro-F1 | AUROC macro | AUPRC macro |
|---|---:|---:|---:|
| {ptb_model} | {fmt(ptb_f1)} | {fmt(ptb_auroc)} | {fmt(ptb_auprc)} |

Interpretation:

The PTBDB result shows that Inception1D is highly effective for binary normal/abnormal beat-level classification. This supports the value of deep temporal convolution for heartbeat morphology analysis.

## 7. Transfer Learning Result

Transfer experiment:

`MIT-BIH pretraining -> PTBDB fine-tuning`

| Metric | Transfer gain |
|---|---:|
| Balanced accuracy | {fmt(ba_gain)} |
| Macro-F1 | {fmt(f1_gain)} |
| AUROC | {fmt(auroc_gain)} |
| AUPRC | {fmt(auprc_gain)} |

Interpretation:

Transfer learning showed mixed behavior. MIT-BIH pretraining slightly improved balanced accuracy, AUROC, and AUPRC on PTBDB, but did not improve Macro-F1 compared with PTBDB training from scratch.

This should be reported honestly as evidence that morphology pretraining may help ranking/discrimination metrics, while classification-threshold performance may still require task-specific fine-tuning.

## 8. Research Contribution

BeatScope v2.8 strengthens CardioTwin-AI in three ways:

1. It adds beat-level ECG morphology intelligence.
2. It provides a separate auxiliary benchmark branch that does not contaminate 12-lead validation claims.
3. It introduces transfer-learning evidence for heartbeat representation learning.

Together with CardioTwin-AI v2.7, the project can now be positioned as a multi-scale ECG intelligence platform.

## 9. Recommended Paper Wording

Suggested wording:

**BeatScope v2.8 was used as an auxiliary beat-level benchmark to evaluate heartbeat morphology classification and transfer-learning behavior. These results were reported separately from the CardioTwin-AI v2.7 12-lead record-level validation because the input unit, task definition, and clinical interpretation differ.**

## 10. Recommended Competition Wording

Suggested wording:

**CardioTwin-AI combines 12-lead record-level screening with BeatScope beat-level morphology intelligence, creating a multi-scale ECG AI platform with safety-calibrated prediction, visual explanation, and audit-ready evidence export.**

## 11. Limitations

1. BeatScope uses segmented and preprocessed heartbeat vectors.
2. BeatScope does not provide 12-lead anatomical region mapping.
3. BeatScope results should not be interpreted as patient-level diagnostic validation.
4. Transfer learning gains were mixed and should not be overclaimed.
5. External prospective clinical validation remains required before clinical deployment.

## 12. Claim Boundary

BeatScope v2.8 is a **research-use beat-level auxiliary benchmark**. It is not a diagnostic medical device and must not be mixed with CardioTwin-AI v2.7 12-lead record-level external validation metrics.

## 13. Generated Artifacts

- `cardiotwin_beatscope_v2_8_full_addon.zip`
- `cardiotwin_beatscope_v2_8_full_manifest.json`
- `heartbeat_benchmark_v28/heartbeat_dataset_summary.json`
- `heartbeat_benchmark_v28/mitbih_model_leaderboard.csv`
- `heartbeat_benchmark_v28/ptbdb_model_leaderboard.csv`
- `heartbeat_benchmark_v28/transfer_learning_report.html`
- `heartbeat_benchmark_v28/beat_benchmark_summary.md`
- `BEATSCOPE_V28_RESEARCH_ADDENDUM.md`
- `BEATSCOPE_V28_RESULTS_TABLE.csv`
- `BEATSCOPE_V28_EXECUTIVE_SUMMARY.md`
"""

ADDENDUM.write_text(addendum, encoding="utf-8")

# ---------------------------------------------------------------------
# Claim boundary card
# ---------------------------------------------------------------------
claim_text = f"""# BeatScope v2.8 Claim Boundary Card

Created: `{created}`

## Safe Claim

BeatScope v2.8 is an auxiliary beat-level ECG benchmark branch that evaluates heartbeat morphology classification and transfer-learning behavior using segmented heartbeat vectors.

## Do Not Claim

Do not claim that BeatScope is a 12-lead record-level validation dataset.

Do not merge BeatScope metrics with CardioTwin-AI v2.7 record-level external validation metrics.

Do not claim clinical diagnostic readiness from BeatScope results alone.

## Correct Relationship

- CardioTwin-AI v2.7: 12-lead record-level ECG screening and visual explanation.
- BeatScope v2.8: beat-level morphology benchmark and transfer-learning add-on.

## Best Short Description

CardioTwin-AI v2.7 and BeatScope v2.8 together form a multi-scale ECG intelligence research platform, combining record-level 12-lead screening with beat-level morphology analysis.
"""

CLAIM_BOUNDARY.write_text(claim_text, encoding="utf-8")

print("DONE: BeatScope v2.8 Research Addendum created")
print("Addendum:", ADDENDUM)
print("Results table:", RESULTS)
print("Executive summary:", EXEC_SUMMARY)
print("Claim boundary:", CLAIM_BOUNDARY)
print("MIT-BIH best:", mit_model, fmt(mit_f1), fmt(mit_auroc), fmt(mit_auprc))
print("PTBDB best:", ptb_model, fmt(ptb_f1), fmt(ptb_auroc), fmt(ptb_auprc))
print("Transfer gains:", gains)

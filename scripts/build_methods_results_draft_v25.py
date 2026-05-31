from pathlib import Path
import json
import pandas as pd

OUT = Path("artifacts/paper_ready_v25")
OUT.mkdir(parents=True, exist_ok=True)

RELEASE = Path("artifacts/release_rc1")

def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def safe(x, digits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def maybe_read_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

table1 = maybe_read_csv(OUT / "table1_model_comparison_with_cpsc2018.csv")
table2 = maybe_read_csv(OUT / "table2_cpsc2018_inceptiontime_per_class.csv")

manifest = read_json(RELEASE / "release_manifest.json")
safety = read_json(RELEASE / "deep_safety_v21" / "metrics_deep_safety_v21.json")
hyp = read_json(RELEASE / "hyp_improvement_v21" / "hyp_improvement_report.json")
region = read_json(RELEASE / "region_mapping_v23" / "region_mapper_v23_summary.json")

geo_deep = read_json(RELEASE / "georgia_external_inceptiontime_v21" / "georgia_deep_external_metrics.json")
cpsc_deep = read_json(RELEASE / "cpsc2018_external_inceptiontime_v21" / "georgia_deep_external_metrics.json")
cpsc_base = read_json(RELEASE / "cpsc2018_external_baseline_v21" / "cpsc2018_valid_label_metrics_baseline_v21.json")

high = manifest.get("high_level_metrics", {})
cpsc_valid = cpsc_deep.get("valid_label_metrics", {})
geo_valid = geo_deep.get("valid_label_metrics", {})

safety_profiles = safety.get("profiles", {})
safety_rows = []
for name, v in safety_profiles.items():
    safety_rows.append({
        "profile": name,
        "macro_f1_valid": v.get("macro_f1_valid"),
        "precision_valid": v.get("macro_precision_valid"),
        "sensitivity_valid": v.get("macro_sensitivity_valid"),
        "mean_ece_valid": v.get("mean_ece_valid"),
        "critical_miss_rate": v.get("critical_miss_rate"),
    })
safety_df = pd.DataFrame(safety_rows)

hyp_current = hyp.get("current_threshold_0p5", {})
hyp_best = hyp.get("best_f1_threshold", {})

model_table_md = table1.to_markdown(index=False) if not table1.empty else "_Model comparison table not found._"
cpsc_per_class_md = table2.to_markdown(index=False) if not table2.empty else "_CPSC per-class table not found._"
safety_md = safety_df.to_markdown(index=False) if not safety_df.empty else "_Safety profile table not found._"

draft = f"""# CardioTwin-AI 12L v2.5 Methods and Results Draft

## Working Title

**CardioTwin-AI 12L: A Low-Cost 12-Lead ECG-to-3D/4D Digital Twin Platform for Safety-Calibrated Preliminary Cardiac Screening**

## 1. Study Objective

This work presents **CardioTwin-AI 12L**, a research-stage platform that uses standard 12-lead electrocardiography signals as a low-equipment input for preliminary cardiac screening, safety-calibrated AI prediction, explainability, and 3D/4D region-level visualization.

The central goal is to evaluate whether a compact deep waveform model can improve cross-dataset ECG screening performance while preserving interpretability, safety boundaries, and low-cost deployment potential.

## 2. System Overview

The current CardioTwin-AI 12L v2.5 pipeline contains five major components:

1. **ECG data pipeline** for reading 12-lead waveform records and preparing fixed-duration input.
2. **AI classification module** using both a lightweight feature baseline and a deep InceptionTime waveform model.
3. **Safety calibration module** using threshold profiles for different operating modes.
4. **Region Mapper v2.3** for lead-to-region explanation with lateral-bias reduction.
5. **3D/4D dashboard** for waveform visualization, safety-calibrated screening output, region explanation, and exportable case reports.

The current safety-calibrated default model is:

`artifacts/models/inceptiontime_v21_safety.pt`

## 3. Data Sources

### 3.1 Internal dataset

Internal validation used the PTB-XL processed dataset with five PTB-XL-style superclasses:

- NORM
- MI
- STTC
- CD
- HYP

### 3.2 External datasets

External validation used two PhysioNet/CinC 2020 subsets:

1. **Georgia external v2.1**
2. **CPSC 2018 external v2.1**

Both external datasets used harmonization v2.1 to map compatible diagnosis codes into the five target superclasses. Classes with insufficient mapped positive support were excluded from valid-label macro metrics.

For CPSC 2018, valid mapped labels were:

- NORM
- STTC
- CD

MI and HYP had zero mapped positive support under the current harmonization and were therefore excluded from valid-label macro reporting.

## 4. Models

Two model families were evaluated.

### 4.1 Feature baseline

The feature baseline is a lightweight, CPU-efficient model based on extracted ECG waveform features. It serves as a low-cost reference model.

### 4.2 InceptionTime deep waveform model

The InceptionTime model uses 12-lead ECG waveform input and was selected as the main model because it outperformed the feature baseline on internal and external validation.

The current high-level best external result in the release manifest is:

- Best external model: **{high.get("best_external_model", "NA")}**
- Best external dataset: **{high.get("best_external_dataset", "NA")}**
- Best external valid AUROC: **{safe(high.get("best_external_valid_auroc"))}**
- Best external valid AUPRC: **{safe(high.get("best_external_valid_auprc"))}**
- Best external valid Macro-F1: **{safe(high.get("best_external_valid_macro_f1"))}**

## 5. Safety Calibration

Deep safety calibration used InceptionTime predictions and external validation labels to construct multiple threshold profiles:

- `screening`
- `balanced`
- `high_specificity`
- `hyp_focus`

The release recommends:

- Default safety profile: **{high.get("recommended_default_safety_profile", "screening")}**
- Demo/reporting profile: **{high.get("recommended_demo_profile", "balanced")}**

### Safety profile results

{safety_md}

The `screening` profile prioritizes sensitivity and lower critical miss rate. The `balanced` profile is better suited for model comparison and paper-ready demonstration because it balances precision, sensitivity, and Macro-F1.

## 6. Region Mapper v2.3

Region Mapper v2.3 was introduced to reduce lateral-region dominance in the 3D/4D visualization. Its key changes include:

- Normalizing region evidence by the number of leads per region.
- Allowing ambiguous cases to become `uncertain`.
- Applying class-aware priors for STTC, MI, CD, and HYP.
- Reducing lateral dominance caused by lead-count advantage.

Summary status:

`{region.get("recommendation", "Use region_mapper_v23 for dashboard and 3D/4D heatmap region assignment.")}`

The 3D/4D heart map is a pseudo-3D visual explanation of lead-region evidence. It is not patient-specific ECGI.

## 7. Results

### 7.1 Model comparison

{model_table_md}

### 7.2 Georgia external validation

On Georgia external v2.1, InceptionTime improved valid-label external performance over the feature baseline.

Georgia InceptionTime valid-label metrics:

- Valid AUROC: **{safe(geo_valid.get("macro_auroc_valid"))}**
- Valid AUPRC: **{safe(geo_valid.get("macro_auprc_valid"))}**
- Valid Macro-F1: **{safe(geo_valid.get("macro_f1_valid"))}**
- Valid sensitivity: **{safe(geo_valid.get("macro_sensitivity_valid"))}**

### 7.3 CPSC 2018 external validation

CPSC 2018 was used as a second external stress-test dataset. Download QC passed with 6,877 paired `.hea/.mat` records.

Under harmonization v2.1, valid-label macro was computed over NORM, STTC, and CD only.

Feature baseline CPSC valid-label metrics:

- Valid AUROC: **{safe(cpsc_base.get("macro_auroc_valid"))}**
- Valid AUPRC: **{safe(cpsc_base.get("macro_auprc_valid"))}**
- Valid Macro-F1: **{safe(cpsc_base.get("macro_f1_valid"))}**
- Valid sensitivity: **{safe(cpsc_base.get("macro_sensitivity_valid"))}**

InceptionTime CPSC valid-label metrics:

- Valid AUROC: **{safe(cpsc_valid.get("macro_auroc_valid"))}**
- Valid AUPRC: **{safe(cpsc_valid.get("macro_auprc_valid"))}**
- Valid Macro-F1: **{safe(cpsc_valid.get("macro_f1_valid"))}**
- Valid sensitivity: **{safe(cpsc_valid.get("macro_sensitivity_valid"))}**
- CPU latency: **{safe(cpsc_deep.get("inference_latency_ms_per_record"))} ms/record**

### 7.4 CPSC 2018 per-class InceptionTime results

{cpsc_per_class_md}

## 8. HYP Limitation and Threshold Recommendation

HYP remains one of the weakest externally supported classes.

At threshold 0.5:

- F1: **{safe(hyp_current.get("f1"))}**
- Precision: **{safe(hyp_current.get("precision"))}**
- Sensitivity: **{safe(hyp_current.get("sensitivity"))}**

Best HYP threshold by F1 analysis:

- Threshold: **{safe(hyp_best.get("threshold"))}**
- F1: **{safe(hyp_best.get("f1"))}**
- Precision: **{safe(hyp_best.get("precision"))}**
- Sensitivity: **{safe(hyp_best.get("sensitivity"))}**

HYP should therefore be reported with caution until additional external datasets provide stronger HYP support and morphology-specific features are improved.

## 9. MI Limitation

MI performance should not be over-claimed from CPSC 2018 because CPSC 2018 had zero mapped MI support under harmonization v2.1. Georgia also had limited MI support. Future work should target MI-rich datasets or improve harmonization coverage.

## 10. Interpretation

The results show that InceptionTime consistently outperforms the feature baseline under valid-label external testing. CPSC 2018 provides the strongest external AUROC result, but only for NORM, STTC, and CD due to zero mapped MI/HYP support.

The safety-calibrated dashboard improves research usability by separating model probabilities from thresholded screening flags. Region Mapper v2.3 improves visual reliability by returning `uncertain` when anatomical evidence is not sufficiently separated.

## 11. Limitations

1. This is a research-use preliminary screening prototype, not a final diagnostic medical device.
2. The 3D/4D heart map is a lead-region visualization, not patient-specific ECGI.
3. External validation depends on label harmonization quality.
4. CPSC 2018 valid-label metrics exclude MI and HYP due to zero mapped support.
5. HYP remains weak and requires additional morphology features and external support.
6. Prospective clinical validation has not yet been performed.
7. Real-time 12-lead hardware is not yet included in the validated pipeline.

## 12. Conclusion

CardioTwin-AI 12L v2.5 demonstrates a low-cost 12-lead ECG AI workflow with internal validation, two external validation datasets, safety-calibrated threshold profiles, Region Mapper v2.3, and a 3D/4D visual explanation dashboard.

The InceptionTime model is the recommended main model for continued development. The current release is suitable as a research evidence bundle for mentor review, system demonstration, and early paper preparation.

## 13. Claim Boundary

**Research-use only. Preliminary AI screening and visual explanation prototype. Not for final diagnosis, emergency triage, or autonomous clinical decision-making.**
"""

out_path = OUT / "CARDIOTWIN_METHODS_RESULTS_DRAFT_v25.md"
out_path.write_text(draft, encoding="utf-8")

print("Saved:", out_path)
print("Size bytes:", out_path.stat().st_size)

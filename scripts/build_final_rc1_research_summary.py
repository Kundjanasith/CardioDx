from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = "CardioTwin-AI 12L"
RELEASE_NAME = "v2.3 RC1"
RELEASE_TITLE = "CardioTwin-AI 12L v2.3 RC1 — External-Validated + Safety-Calibrated Deep ECG Screening Prototype"

ROOT = Path(".")
RELEASE = ROOT / "artifacts" / "release_rc1"
RELEASE.mkdir(parents=True, exist_ok=True)

SUMMARY_MD = RELEASE / "CARDIOTWIN_RC1_RESEARCH_SUMMARY.md"
RESULTS_CSV = RELEASE / "CARDIOTWIN_RC1_RESULTS_TABLE.csv"
MANIFEST_JSON = RELEASE / "release_manifest.json"
ZIP_PATH = RELEASE / "release_rc1.zip"


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def clean_float(x, digits: int = 4):
    try:
        if x is None:
            return "NA"
        if isinstance(x, str) and x.strip() == "":
            return "NA"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return str(x)


def clean_json_value(x):
    try:
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    except Exception:
        return x


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stat(path: Path):
    st = path.stat()
    return {
        "size_bytes": int(st.st_size),
        "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------
leaderboard = read_csv(RELEASE / "model_dataset_comparison_leaderboard.csv")

comparison_summary = read_json(RELEASE / "model_dataset_comparison_summary.json")

deep_safety = read_json(RELEASE / "deep_safety_v21" / "metrics_deep_safety_v21.json")
if not deep_safety:
    deep_safety = read_json(ROOT / "artifacts" / "deep_safety_v21" / "metrics_deep_safety_v21.json")

threshold_profiles = read_json(RELEASE / "deep_safety_v21" / "threshold_profiles_deep.json")
if not threshold_profiles:
    threshold_profiles = read_json(ROOT / "artifacts" / "deep_safety_v21" / "threshold_profiles_deep.json")

hyp_report = read_json(RELEASE / "hyp_improvement_v21" / "hyp_improvement_report.json")
if not hyp_report:
    hyp_report = read_json(ROOT / "artifacts" / "hyp_improvement_v21" / "hyp_improvement_report.json")

harmonization = read_json(RELEASE / "georgia_external_baseline_v21" / "harmonization_v21_summary.json")
if not harmonization:
    harmonization = read_json(ROOT / "artifacts" / "external_validation" / "georgia_true_eval" / "harmonization_v21_summary.json")

geo_base = read_json(RELEASE / "georgia_external_baseline_v21" / "georgia_external_metrics.json")
geo_deep = read_json(RELEASE / "georgia_external_inceptiontime_v21" / "georgia_deep_external_metrics.json")


# ---------------------------------------------------------------------
# Build results table CSV
# ---------------------------------------------------------------------
rows = []

def add_result(section, item, dataset="", model="", metric="", value="", notes=""):
    rows.append({
        "section": section,
        "item": item,
        "dataset": dataset,
        "model": model,
        "metric": metric,
        "value": value,
        "notes": notes,
    })


if not leaderboard.empty:
    for _, r in leaderboard.iterrows():
        model = str(r.get("model", ""))
        dataset = str(r.get("dataset", ""))
        add_result("model_comparison", "AUROC macro", dataset, model, "auroc_macro", clean_json_value(r.get("auroc_macro")), str(r.get("notes", "")))
        add_result("model_comparison", "AUPRC macro", dataset, model, "auprc_macro", clean_json_value(r.get("auprc_macro")), str(r.get("notes", "")))
        add_result("model_comparison", "Macro-F1", dataset, model, "macro_f1", clean_json_value(r.get("macro_f1")), str(r.get("notes", "")))
        add_result("model_comparison", "Macro sensitivity", dataset, model, "macro_sensitivity", clean_json_value(r.get("macro_sensitivity")), str(r.get("notes", "")))
        add_result("model_comparison", "Latency", dataset, model, "latency_ms_per_record", clean_json_value(r.get("latency_ms_per_record")), "ms/record if available")
        add_result("model_comparison", "Model size", dataset, model, "model_size_mb", clean_json_value(r.get("model_size_mb")), "MB")

gains = comparison_summary.get("gains", {})
for k, v in gains.items():
    add_result("model_gain", k, "Georgia external v2.1", "InceptionTime vs Feature baseline", k, clean_json_value(v), "External valid-label macro gain")

profiles = deep_safety.get("profiles", {})
for profile_name, p in profiles.items():
    add_result("deep_safety_profile", profile_name, "Georgia external v2.1 split", "InceptionTime", "macro_f1_valid", p.get("macro_f1_valid"), "Threshold profile evaluation")
    add_result("deep_safety_profile", profile_name, "Georgia external v2.1 split", "InceptionTime", "macro_precision_valid", p.get("macro_precision_valid"), "Threshold profile evaluation")
    add_result("deep_safety_profile", profile_name, "Georgia external v2.1 split", "InceptionTime", "macro_sensitivity_valid", p.get("macro_sensitivity_valid"), "Threshold profile evaluation")
    add_result("deep_safety_profile", profile_name, "Georgia external v2.1 split", "InceptionTime", "mean_ece_valid", p.get("mean_ece_valid"), "Calibration quality")
    add_result("deep_safety_profile", profile_name, "Georgia external v2.1 split", "InceptionTime", "critical_miss_rate", p.get("critical_miss_rate"), "Lower is safer for screening")

if hyp_report:
    add_result("hyp_improvement", "HYP current threshold", "Georgia external v2.1", "InceptionTime", "threshold_0.5_f1", hyp_report.get("current_threshold_0p5", {}).get("f1"), "Current fixed threshold")
    add_result("hyp_improvement", "HYP current threshold", "Georgia external v2.1", "InceptionTime", "threshold_0.5_sensitivity", hyp_report.get("current_threshold_0p5", {}).get("sensitivity"), "Current fixed threshold")
    add_result("hyp_improvement", "HYP best F1 threshold", "Georgia external v2.1", "InceptionTime", "best_f1_threshold", hyp_report.get("best_f1_threshold", {}).get("threshold"), "Recommended threshold analysis")
    add_result("hyp_improvement", "HYP best F1", "Georgia external v2.1", "InceptionTime", "best_f1", hyp_report.get("best_f1_threshold", {}).get("f1"), "Recommended threshold analysis")
    add_result("hyp_improvement", "HYP best sensitivity", "Georgia external v2.1", "InceptionTime", "best_sensitivity", hyp_report.get("best_f1_threshold", {}).get("sensitivity"), "Recommended threshold analysis")

if harmonization:
    add_result("harmonization", "Harmonization v2.1 include codes", "Georgia external v2.1", "", "n_include_codes", harmonization.get("n_include_codes"), "SNOMED-to-PTBXL superclass map")
    add_result("harmonization", "Harmonization v2.1 exclude-other codes", "Georgia external v2.1", "", "n_exclude_other_codes", harmonization.get("n_exclude_other_codes"), "Rhythm/axis/other not forced into 5-class task")
    add_result("harmonization", "Harmonization v2.1 review codes", "Georgia external v2.1", "", "n_review_codes", harmonization.get("n_review_codes"), "Manual review remaining")

pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)


# ---------------------------------------------------------------------
# Build Markdown summary
# ---------------------------------------------------------------------
created_at = datetime.now(timezone.utc).isoformat()

leaderboard_rows = []
if not leaderboard.empty:
    for _, r in leaderboard.iterrows():
        leaderboard_rows.append([
            r.get("model", ""),
            r.get("dataset", ""),
            r.get("validation_type", ""),
            r.get("label_scope", ""),
            clean_float(r.get("auroc_macro")),
            clean_float(r.get("auprc_macro")),
            clean_float(r.get("macro_f1")),
            clean_float(r.get("macro_sensitivity")),
            clean_float(r.get("latency_ms_per_record")),
            clean_float(r.get("model_size_mb")),
        ])

safety_rows = []
for name, p in profiles.items():
    safety_rows.append([
        name,
        clean_float(p.get("macro_f1_valid")),
        clean_float(p.get("macro_precision_valid")),
        clean_float(p.get("macro_sensitivity_valid")),
        clean_float(p.get("mean_ece_valid")),
        clean_float(p.get("abstain_rate_proxy")),
        clean_float(p.get("critical_miss_rate")),
    ])

hyp_current = hyp_report.get("current_threshold_0p5", {}) if hyp_report else {}
hyp_best = hyp_report.get("best_f1_threshold", {}) if hyp_report else {}

recommended_default = deep_safety.get("recommended_default_profile", "screening")
recommended_demo = "balanced"

summary = f"""# {RELEASE_TITLE}

Generated at: `{created_at}`

## 0. Executive Summary

This release consolidates the CardioTwin-AI 12L RC1 research artifacts into a single evidence-oriented package. The system uses low-cost 12-lead ECG as input and evaluates both an ultra-light feature baseline and a deep InceptionTime waveform model under internal PTB-XL validation and external PhysioNet/CinC 2020 Georgia validation.

**Recommended research model:** `InceptionTime`  
**Recommended default safety profile:** `{recommended_default}`  
**Recommended demo/reporting profile:** `{recommended_demo}`  
**Clinical boundary:** research-use preliminary screening and visual explanation only; not a final diagnosis system.

---

## 1. Model Comparison

{md_table(
    ["Model", "Dataset", "Validation", "Label scope", "AUROC", "AUPRC", "Macro-F1", "Sensitivity", "Latency ms/record", "Size MB"],
    leaderboard_rows
)}

Key external gain from InceptionTime over feature baseline:

- AUROC gain: `{clean_float(gains.get("external_valid_auroc_gain_inceptiontime_vs_feature"))}`
- AUPRC gain: `{clean_float(gains.get("external_valid_auprc_gain_inceptiontime_vs_feature"))}`
- Macro-F1 gain: `{clean_float(gains.get("external_valid_macro_f1_gain_inceptiontime_vs_feature"))}`

---

## 2. Internal vs External Validation

Internal PTB-XL validation provides the controlled development benchmark. Georgia external validation provides cross-dataset stress testing under a different data source and diagnosis-label structure.

Important interpretation:

- PTB-XL internal results estimate in-distribution model capability.
- Georgia external results estimate cross-dataset generalization after v2.1 SNOMED-to-PTBXL harmonization.
- Valid-label macro excludes classes with positive support below 20.
- MI is excluded from valid-label macro because Georgia has insufficient mapped MI support.

---

## 3. Georgia Harmonization v2.1

Harmonization v2.1 maps PhysioNet/CinC 2020 Georgia SNOMED codes into the five PTB-XL-style superclasses where clinically reasonable, while keeping rhythm/axis/voltage/artifact-like findings outside the 5-class evaluation when they do not fit.

Harmonization summary:

- Total diagnosis codes observed: `{harmonization.get("n_codes", "NA")}`
- Include codes: `{harmonization.get("n_include_codes", "NA")}`
- Exclude-other codes: `{harmonization.get("n_exclude_other_codes", "NA")}`
- Review codes remaining: `{harmonization.get("n_review_codes", "NA")}`
- Include record-weight by code count: `{harmonization.get("include_record_weight_by_code_count", "NA")}`
- Exclude-other record-weight by code count: `{harmonization.get("exclude_other_record_weight_by_code_count", "NA")}`

Mapping file:

`configs/cinc2020_to_ptbxl_superclass_map_v21.csv`

---

## 4. Deep Safety Profile

Deep safety calibration v2.1 uses InceptionTime predictions on Georgia external data and creates calibrated threshold profiles for different operating modes.

{md_table(
    ["Profile", "Macro-F1", "Precision", "Sensitivity", "ECE", "Abstain proxy", "Critical miss"],
    safety_rows
)}

### Recommended default: screening

The `screening` profile is recommended as the default because it prioritizes sensitivity and minimizes critical misses. It is appropriate for preliminary screening, where missing a potentially abnormal ECG is more harmful than over-alerting.

### Recommended demo/reporting profile: balanced

The `balanced` profile is recommended for demonstrations, reports, and model comparison because it provides the best balance between F1, precision, and sensitivity.

---

## 5. HYP Limitation and Threshold Recommendation

HYP remains the weakest external class and should be reported with caution.

Current HYP threshold 0.5:

- F1: `{clean_float(hyp_current.get("f1"))}`
- Precision: `{clean_float(hyp_current.get("precision"))}`
- Sensitivity: `{clean_float(hyp_current.get("sensitivity"))}`

Best HYP threshold from threshold analysis:

- Threshold: `{clean_float(hyp_best.get("threshold"))}`
- F1: `{clean_float(hyp_best.get("f1"))}`
- Precision: `{clean_float(hyp_best.get("precision"))}`
- Sensitivity: `{clean_float(hyp_best.get("sensitivity"))}`

Recommendation:

- Use HYP-specific thresholding rather than fixed threshold 0.5.
- Add voltage/morphology features such as LVH/RVH voltage proxies, QRS amplitude in V1/V5/V6, and axis-related evidence.
- Report HYP as lower-confidence unless SQI, calibration, and evidence alignment pass.

---

## 6. MI Limitation

MI is not considered stable for Georgia external evaluation because mapped MI positive support is below the stable reporting threshold.

Recommendation:

- Do not use Georgia v2.1 alone to claim MI external performance.
- Add CPSC 2018 and/or another MI-rich external subset.
- Keep MI in model output but mark MI external metrics as limited by low support.

---

## 7. Research-Use Boundary

This release is a research-stage prototype for preliminary screening and visual explanation from 12-lead ECG.

It is **not**:

- A final diagnostic medical device
- A replacement for clinician interpretation
- Patient-specific anatomical ECGI
- A clinically deployed safety-validated product

All reports should include:

> Research-use only. Preliminary AI screening and visual explanation prototype. Not for final diagnosis or emergency decision-making.

---

## 8. Next Work

Recommended next steps:

1. Region mapper lateral-bias fix
2. CPSC 2018 external validation
3. Deep safety calibration using additional external datasets
4. HYP morphology feature enhancement
5. MI support expansion using MI-rich external data
6. Prospective data pipeline
7. Dashboard integration of `inceptiontime_v21_safety.pt`
8. Paper-ready tables and figures

---

## 9. Key Release Artifacts

- `CARDIOTWIN_RC1_RESEARCH_SUMMARY.md`
- `CARDIOTWIN_RC1_RESULTS_TABLE.csv`
- `model_dataset_comparison_leaderboard.csv`
- `model_dataset_comparison_summary.json`
- `georgia_external_baseline_v21/`
- `georgia_external_inceptiontime_v21/`
- `deep_safety_v21/`
- `hyp_improvement_v21/`
- `release_manifest.json`
- `release_rc1.zip`

---

## 10. Claim Boundary

Georgia external metrics use the PhysioNet/CinC 2020 Georgia subset with v2.1 harmonization. Valid-label macro excludes labels with support < 20. Safety thresholds derived from Georgia external split are research-stage thresholds and should not be treated as clinical deployment thresholds.
"""

SUMMARY_MD.write_text(summary, encoding="utf-8")


# ---------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

manifest_files = []
for p in sorted(RELEASE.rglob("*")):
    if not p.is_file():
        continue
    if p.name == "release_rc1.zip":
        continue
    # Avoid temp/lock files
    if p.name.startswith("~$"):
        continue
    rel = p.relative_to(RELEASE).as_posix()
    st = safe_stat(p)
    manifest_files.append({
        "path": rel,
        **st,
    })

manifest = {
    "project": PROJECT,
    "release": RELEASE_NAME,
    "title": RELEASE_TITLE,
    "created_at_utc": created_at,
    "release_dir": str(RELEASE),
    "files_indexed": len(manifest_files),
    "files": manifest_files,
    "high_level_metrics": {
        "best_external_model": comparison_summary.get("best_external_model_by_valid_auroc", {}).get("model"),
        "best_external_dataset": comparison_summary.get("best_external_model_by_valid_auroc", {}).get("dataset"),
        "best_external_valid_auroc": comparison_summary.get("best_external_model_by_valid_auroc", {}).get("auroc_macro"),
        "best_external_valid_auprc": comparison_summary.get("best_external_model_by_valid_auprc", {}).get("auprc_macro"),
        "best_external_valid_macro_f1": comparison_summary.get("best_external_model_by_valid_macro_f1", {}).get("macro_f1"),
        "recommended_default_safety_profile": recommended_default,
        "recommended_demo_profile": recommended_demo,
    },
    "claim_boundary": "Research-use preliminary screening and visual explanation prototype. Not final diagnosis.",
    "notes": [
        "release_manifest.json is generated after summary/results tables.",
        "release_rc1.zip is generated after the manifest and is not listed inside the manifest index to avoid self-referential hashing.",
    ],
}

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Build ZIP
# ---------------------------------------------------------------------
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RELEASE.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == ZIP_PATH.resolve():
            continue
        if p.name.startswith("~$"):
            continue
        arcname = Path("release_rc1") / p.relative_to(RELEASE)
        z.write(p, arcname.as_posix())

print("DONE: Final RC1 Research Summary created")
print("Summary:", SUMMARY_MD)
print("Results:", RESULTS_CSV)
print("Manifest:", MANIFEST_JSON)
print("ZIP:", ZIP_PATH)
print("ZIP size MB:", f"{ZIP_PATH.stat().st_size / 1024 / 1024:.2f}")
